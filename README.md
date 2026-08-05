# FastAAI 1.5

Average amino acid identity between microbial genomes, estimated from tetramer
sketches of single-copy protein-coding genes.

This is FastAAI with its search engine rewritten in Rust. Same algorithm, but faster and lighter weight.

```
FASTA ──► pyrodigal ──► pyhmmer ──► best-hit filter ──► k-merise ──► inverted index
          └──────────── Python ────────────┘          └────────── Rust ──────────┘
```

Python keeps ingestion, gene prediction and HMM search — pyfastx, pyrodigal and
pyhmmer already wrap mature C and release the GIL, so porting them to Rust would
FFI back to the same libraries for nothing. Rust takes k-merisation, the inverted
index, the counting kernel and the AAI transform, which is where v1 was bound by
Python's parallelism.

## Performance

**Search**, 8 threads, wall clock less the ~0.22 s interpreter start both
versions pay. v1 was run through `db_query --in_memory --store_results`, its
fastest path:

Throughput is per thread, which is the figure that compares across machines and
thread counts.

| scale | pairs | v1 | v1.5 | v1 /s/thread | v1.5 /s/thread |
|---|---|---|---|---|---|
| 120 genomes | 14,400 | 1.15 s | 0.016 s | 1,565 | 112,500 |
| **2,943 genomes** | **8,661,249** | **21.84 s** | **1.587 s** | **49,572** | **682,203** |

**The ratio depends on which v1 baseline is fair.** FastAAI 1's published
in-memory figure is ~100k comparisons/s/thread; this machine measured v1 at
49,572, about half that. So:

| v1 baseline | speedup |
|---|---|
| v1 as measured here (49,572 /s/thread) | 13.8× |
| **v1 as published (~100,000 /s/thread)** | **6.8×** |

6.8× is the figure to quote until the v1 baseline is re-measured on hardware
where it reaches its published throughput. This is a laptop — 6 performance
cores plus 8 efficiency cores — and v1's thread scaling on it is not
established.

The 120-genome row is fixed cost rather than throughput: at that size v1's
per-query `TEMP TABLE` and `INNER JOIN` dominate and neither version is doing
enough work to measure.

Memory and disk at 2,943 genomes:

| | v1 | v1.5 |
|---|---|---|
| peak RSS | 1.01 GB | 116 MB index |
| database on disk | 508 MB | 116 MB |

**Preprocessing is unchanged and still dominates a cold run** — it is the same
Prodigal and the same HMMER, now in-process. Per genome, one thread each (the
pipeline parallelises across genomes, so `cpus=1` is the figure that composes):

| stage | median s/genome | share |
|---|---|---|
| predict (pyrodigal) | 1.79 | 64% |
| hmmsearch (pyhmmer) | 1.03 | 36% |

At 2,943 genomes that is roughly two CPU-hours of preprocessing against 1.6 s of
search. The speedup above is what you get on a database you already built —
which is the case FastAAI exists for. Build once, `--archive`, and query forever.

## Agreement with FastAAI 1

120 Firmicutes genomes, all-vs-all, against FastAAI 1 driven through its own
`aai_index` module.

![v1 vs v1.5 concordance](fastaai2/methods/concordance_v1_v15.png)

| | |
|---|---|
| pairs compared (off-diagonal) | 14,280 |
| shared-SCP differences | 0 |
| median \|Δ AAI\| | 0.0115 percentage points |
| max \|Δ AAI\| | 0.0627 percentage points |

v1 and v1.5 differ in one respect that reaches the k-mers: v1 admits the stop
symbol `*` and ambiguous residues `X`, v1.5 does not. See
**[`fastaai2/methods/`](fastaai2/methods/)** for the data and the harness.

## Install

```sh
cd fastaai2
maturin develop --release      # or: pip install .
```

## Use

```sh
# build a database
fastaai build /path/to/genomes --hmm models.hmm -d db/ --archive arch/

# query it — against itself, or against another database
fastaai query -q db/ -o aai.tsv
fastaai query -q queries/ -t db/ --hmm models.hmm -o aai.tsv
```

Results are written one block per query/target partition pair. A search whose
two sides each fit a single partition is the 1×1 case and lands in one file, so
`-o aai.tsv` behaves as expected; a larger search needs `-o` to name a directory
to receive the blocks. The full matrix is never held, which is what makes an
all-vs-all at GTDB scale expressible — 200k × 200k species is 40 billion pairs.

## Output format

**The TSV is FastAAI 1's, unchanged** — same columns, same names, same order, so
a parser written against v1 keeps working:

```
query  target  avg_jacc_sim  jacc_SD  num_shared_SCPs  poss_shared_SCPs  AAI_estimate
```

Every block file carries this header, so blocks concatenate into one valid v1
table. Numbers are rendered as `str(numpy.round(v, dp))`, matching v1 digit for
digit: Jaccard and its standard deviation to 4 decimal places, AAI to 2.

| | |
|---|---|
| `jacc_SD` | always present; reads `N/A` unless `--do_stdev` was given |
| `poss_shared_SCPs` | `min(query SCPs, target SCPs)` — a pair cannot share more markers than the poorer genome carries |
| `AAI_estimate` | `<30%` / `>90%` outside the regression's sensitivity band |
| a pair sharing no marker | `N/A` in every value column |

`--emit` narrows the columns (`jaccard` drops `AAI_estimate`, `aai` drops
`avg_jacc_sim`); the default emits the full v1 schema.

**`--output_style matrix`** writes a Q×T grid of AAI — one row per query, one
column per target, `query_genome` in the corner — with v1's `15.0` and `95.0`
standing in for the two categorical labels, since a cell cannot hold a string.

It is written per block exactly as the TSV is: each file is the Q×T grid for one
partition pair, not for the whole search, so it carries no size restriction.

Two deliberate departures from v1: a pair sharing no marker is `N/A` in the
matrix where v1 writes `0`, which cannot be told from a real measurement of
zero; and `poss_shared_SCPs` uses the `minimum` of v1's three bulk paths rather
than the `max` of its one scalar path.

**FastAAI 1 command lines still work.** `build_db`, `db_query`, `merge_db`,
`aai_index`, `single_query`, `multi_query` and `simple_query` are rerouted to
the new verbs, with arguments preserved where they still mean something and a
diagnostic where they do not.

```python
import fastaai

models = fastaai.ModelSet("models.hmm")
paths  = fastaai.find_genomes("/path/to/genomes")

records = fastaai.preprocess(paths, models, threads=8, archive_root="arch/")
db, skipped = fastaai.build_database(records, models)
res = fastaai.search(db, db, threads=8)

db = fastaai.build_from_archive("arch/")   # rebuild in seconds

res.jaccard   # (n, n) float64, NaN where no accession is shared
res.shared    # (n, n) uint32, accessions carried by both genomes
res.aai       # (n, n) float64, uncensored
```

## Notes

**The model set defines the accession list.** Accession IDs are positions in the
HMM file's order; there is no compiled-in Pfam set. Two databases may only be
compared when accession list, `k` and alphabet all match, and `search` refuses
otherwise — mismatched model sets produce structurally valid, biologically
meaningless output.

**Model identity is verified, not assumed.** Matching accession names and order
do not establish that two databases were built from the same HMMs: a model
revised between Pfam releases keeps its name, accession and position while
matching different proteins, which changes the SCP assignment and so the AAI
with nothing in the output looking wrong. Each database records a digest of the
models it was built from, and a comparison across differing digests is refused.

The digest is over each model's own parameters — emissions and transitions —
plus HMMER's `CKSUM` where the file carries one, so it is reproducible by any
implementation reading the same models and is defined for any SCP set. `CKSUM`
is optional in the format, which is why it is not the only ingredient. A
database built without an HMM set records no digest; that is reported as
unverifiable rather than treated as a conflict.

## Layout

| | |
|---|---|
| `fastaai2/` | the package — Rust engine, PyO3 bindings, Python pipeline |
| `fastaai2/methods/` | figures and measured timings behind the claims above |
| `prototype/` | the measurement harness; benchmarks every rejected alternative |
| `FASTAAI2_PLAN.md` | design record — each decision with the measurement that settled it |

## Status

Working end to end: on-disk partitioned databases, merge, archives, the
FastAAI 1 compatible CLI, and optional per-pair Jaccard standard deviation
(`--do_stdev`). 87 Python and 44 Rust tests.

Not yet packaged for bioconda.
