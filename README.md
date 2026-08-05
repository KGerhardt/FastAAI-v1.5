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
