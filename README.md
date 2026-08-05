# FastAAI 1.5

Average amino acid identity between microbial genomes, estimated from tetramer
sketches of single-copy protein-coding genes.

This is FastAAI with its search engine rewritten in Rust. Same algorithm, same
answers, and no SQLite: the database is a partitioned inverted index that is
memory-mapped rather than queried.

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

| scale | pairs | v1 | v1.5 | v1.5 pairs/s | speedup |
|---|---|---|---|---|---|
| 120 genomes | 14,400 | 1.15 s | 0.016 s | 922,339 | 74× |
| **2,943 genomes** | **8,661,249** | **21.84 s** | **1.587 s** | **5,457,298** | **14×** |

**14× is the number that describes the engine.** The two rows are not in
tension: at 120 genomes the pairwise work is trivial and v1's per-query
`TEMP TABLE` + `INNER JOIN` overhead is most of the runtime, so 74× is really a
measure of fixed cost. Both are reported rather than the flattering one.

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

## Does it change the answers?

No. 120 Firmicutes, all-vs-all, against FastAAI 1 driven through its own
`aai_index` module — 14,280 off-diagonal pairs, **0 shared-SCP differences**,
median |Δ AAI| 0.0115 percentage points.

![v1 vs v1.5 concordance](fastaai2/methods/concordance_v1_v15.png)

Panel A is the finding; panel B exists because at this level of agreement a
reader cannot tell 1e-4 from 1e-2 by eye on a 40–100% axis, and publishing A
alone would be the more flattering and less honest figure.

**The residual is one-directional because it is v1.5 correcting v1.** FastAAI 1
encodes the stop codon `*` and ambiguous residues `X` as if they were amino
acids. Its filtering k-merizer — `unique_kmers`, `fastaai.py:1421` — resolves
tetramers through a `kmer_index` lookup that would admit only permissible
symbols, but that table is never built and the function is never called; the
numpy transform that runs instead takes `ord()` of every character. `X` is the
damaging case: it arises from runs of `N`, so two unrelated genomes with
sequencing gaps share `XXXX` and accrue similarity from it — a false positive
that scales with assembly fragmentation.

Full method, and the harness that produced it, in **[`fastaai2/methods/`](fastaai2/methods/)**.

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

## Things that are easy to get wrong

**Jaccard is stored raw and AAI is uncensored.** FastAAI 1 wrote the strings
`"<30%"` and `">90%"`. At the evolutionary distances this tool exists to serve,
the interesting values sit near the bottom of the range, so censoring them
discards exactly the signal being sought. The regression is unbounded above, so
identical genomes report >100% — clamp at display time, not in storage.

**Unshared pairs are NaN, not 0.** "These genomes share no marker" and "these
genomes have AAI 0" are different statements.

**The model set defines the accession list.** Accession IDs are positions in the
HMM file's order; there is no compiled-in Pfam set. Two databases may only be
compared when accession list, `k` and alphabet all match, and `search` refuses
otherwise — mismatched model sets produce structurally valid, biologically
meaningless output.

**Best-hit resolution is a real choice** (`--filter`). Strict reciprocal best hit
(`rbh`) is FastAAI's stated intent and is harsher than either path shipped in v1.
The default `v1` reproduces v1 as actually executed. None is a superset of
another, and the choice changes SCP sets and therefore AAI.

**Thread counts are capped deliberately.** The counting kernel is memory-bound:
6.17× at 16 threads on a 6P+8E laptop, and negative beyond — 20 threads was
slower than 16. Nothing defaults to `available_parallelism()`.

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
(`--do_stdev`). 81 Python and 44 Rust tests.

Not yet packaged for bioconda.
