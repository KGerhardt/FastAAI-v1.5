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

## Goals:

The purposes of this codebase are: (1) a stable release for FastAAI v1 to serve as the basis for future bioconda installation. (2) A version of the program able to run within reasonable RAM limits for a personal computer on any size of database, while scaling effectively to any HPC platform. (3) It is a Single Copy Protein (SCP)-agnostic revision of the code, able to work with other sets of universal marker genes and not just the specific set selected in FastAAI v1, (4) Preparation for a FastAAI 2, intended to work as a complementary search engine inside GTDB-tk.

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
which is the case FastAAI exists for. Build once, keep the intermediates, and
query forever.

## Preprocessing, and the three ranks it stores

Because preprocessing is ~98% of a cold run, what gets kept between runs matters
more than what gets computed. Every stage writes something reusable, and each
rank is smaller than the one before it.

One worker handles one genome; the pool runs `--preprocess-threads` of them at
once. Both C libraries release the GIL, so this scales on threads rather than
processes — FastAAI 1 forked, and every worker reloaded the model set.

Everything a run produces lands under one root — `FastAAI/` in the working
directory, or wherever `--dir` says. Nothing is discarded and nothing is written
outside it: an HPC job is handed a directory and is expected to stay in it, so
there are no temporary directories anywhere in this codebase.

```text
  genome.fna.gz  (yours, named however you like)
       │
       │  pyrodigal ····································   1.79 s   64%
       ▼
   every protein ─────────────────►  <root>/proteins/<genome>.fasta
       │
       │  pyhmmer ······································   1.03 s   36%
       ▼
   every raw hit ─────────────────►  <root>/hmm_hits/<genome>.tsv
       │
       │  best-hit filter ······························    ~0 s
       ▼
   the SCPs that won ─────────────►  <root>/crystals/<genome>.crystal.fasta
       │
       │  then Rust, once, over the whole crystal set
       ▼
   k-merise, invert, partition ───►  <root>/database/<name>/
                                     5 s for 2,943 genomes

   a search adds ─────────────────►  <root>/results/block_qNNNNN_tNNNNN.tsv
```

One file per genome per rank, in standard formats. No tar, no single table
covering the collection, no nesting past that one level — a directory of
per-genome files can be listed, subsetted and inspected with ordinary tools, and
one genome can be deleted or re-run without touching another. Files are plain
text unless `--gzip` is given, which gzips each one as it is written; every
reader here accepts either.

Each rank is a place to re-enter the pipeline, which is the point of storing
them:

| start from | skips | per genome |
|---|---|---|
| genomes | nothing | ~2.8 s |
| proteins (`--input protein`) | prediction | ~1.0 s |
| stored proteins + hits | prediction and search | re-resolve only |
| crystals | everything | **~1.7 ms** |

| rank | per genome | 2,943 Firmicutes | survives a change of |
|---|---|---|---|
| proteins + raw hits | 543 KB | 1.7 GB | model set, filter — anything |
| crystals | 9.6 KB | 29 MB | nothing, but rebuilds in 5 s |
| the database itself | — | 117 MB | (the built artifact) |

**Crystals are the resolved SCPs** — one FASTA per genome holding just
the marker proteins that won their accession, each record labelled with the
genome, the originating gene call, the model-set fingerprint and the filter that
produced it. FastAAI 1 called these crystals and they work the same way here.

They are also *how* a database gets built. Each preprocessing worker writes its
own crystal and the build reads them back — one ingestion path rather than two.
Peak memory stops tracking the size of the collection as a result: a worker
drops a genome's sequences once its crystal is written, and the build streams
them one file at a time.

```sh
fastaai build genomes/ -d firm            # crystals written to FastAAI/crystals/
fastaai crystallize old_run/              # or from proteins and hits you already have
fastaai build FastAAI/crystals -d firm    # rebuild, no prediction or search
```

Measured on the 2,943 Firmicutes: crystallising a preprocessed collection gives
35 MB, and rebuilding a sealed 116 MB database from it takes **5 s** — against
roughly two CPU-hours to preprocess the same genomes. All 8,661,250 pairs agree
with the database built directly from the stored hits.

They are the smallest of the three artifacts — **smaller than the database built
from them** — which makes them what you ship, and they carry no ordering of
their own, so any subset builds a database comparable with any other subset from
the same models. Accession *names* are stored, not positions; positions are
assigned at build time from the model set, and a build refuses crystals whose
fingerprint disagrees with it.

This is also the answer to growing a database. There is no incremental append:
adding genomes to a sealed database would fragment it into one-genome partitions
and cost roughly 90× search throughput, so instead you keep crystals and rebuild
when you are ready. A rebuild of 48 genomes takes 0.75 s against 46 s to
preprocess them.

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

## What changed: the data store

Same numbers, different container. v1 kept a database in one SQLite file,
storing both directions of the k-mer mapping for every accession:

```text
FastAAI 1                                   one SQLite file
fastaai.db
├─ genome_index                             name → id, protein count
├─ genome_acc_kmer_counts                   (genome, accession) → count
├─ PF00380                    ── inverted   kmer   → genomes[]
├─ PF00380_genomes            ── forward    genome → kmers[]
├─ PF00410  /  PF00410_genomes
└─ …                                        2 tables per accession,
                                            244 tables for 122 SCPs

adding genomes:  INSERT … ON CONFLICT DO UPDATE SET genomes = genomes || (?)
                 read-modify-write of every posting list the genome touches
```

v1.5 keeps a directory, stores only the inverted direction, and cuts it into
partitions that are independent of one another:

```text
FastAAI 1.5                                 a directory
db/
├─ schema                                   k, alphabet, ordered accessions,
│                                           filter mode, model fingerprint
├─ manifest                                 genome → ordinal, partition,
│                                           local id, hash, name
├─ part.00000  ┐                            CSR per accession, direct-addressed
├─ part.00001  │  inverted index only       by k-mer id:
└─ part.NNNNN  ┘                            offsets[kmer]..offsets[kmer+1]
                                            slices postings
                                            ≤16,384 genomes, u16 local ids,
                                            sorted ascending

adding genomes:  write one new part file, rewrite the manifest
                 no existing partition is read, no posting list renumbered
```

| | FastAAI 1 | FastAAI 1.5 |
|---|---|---|
| directions stored | forward and inverted | inverted only — ~34 GB rather than ~95 GB at GTDB scale |
| genome ids | global | local to a partition, `u16` |
| cost of adding genomes | scales with the database | scales with the addition |
| resident while working | the database | one partition |

The forward tables exist only to build the inverted ones, and the k-mer join
reads both sides as inverted indexes, so nothing ever reads them back.

**There is no merge, and no incremental append.** Both existed and both are
gone, because both preserve whatever partitioning their inputs had — merging N
one-genome databases produced N one-genome partitions, measured at 797
pairs/s/thread against 72,265 for the same genomes built together. Combining
collections means putting their crystals in one directory and rebuilding, which
repacks the index properly and costs 5 s for 2,943 genomes.

## What changed: the output

v1 assembled the whole result in memory and wrote one TSV or one matrix. That is
the part that stops being representable first — 200k × 200k species is 40
billion cells, before any of it reaches disk.

v1.5 writes one file per (query partition × target partition) block:

```text
                     target partitions
                 t00000     t00001     t00002
               ┌──────────┬──────────┬──────────┐
        q00000 │   file   │   file   │   file   │
               ├──────────┼──────────┼──────────┤
        q00001 │   file   │   file   │   file   │   ← query partitions
               ├──────────┼──────────┼──────────┤
        q00002 │   file   │   file   │   file   │
               └──────────┴──────────┴──────────┘

  out/
  ├─ block_q00000_t00000.tsv
  ├─ block_q00000_t00001.tsv
  ├─ …
  └─ block_q00002_t00002.tsv

  A cell is bounded by the partition size, not by the size of either
  database. Nothing in it reaches outside its own two partitions, so
  cells compute in any order on any machine, a crash costs one cell,
  and a self-comparison computes only the upper triangle.
```

The same property makes growth incremental: adding genomes adds a row and a
column to the grid, and only those cells need computing. A search whose two
sides each fit one partition is simply the 1×1 case and lands in a single file,
so `-o aai.tsv` behaves the way it always did.

## Install

```sh
cd fastaai2
maturin develop --release      # or: pip install .
```

## Use

```sh
# build a database — bundled 122-SCP set, everything kept under FastAAI/
fastaai build /path/to/genomes -d firm

# put the run somewhere else, and gzip every file it writes
fastaai build /path/to/genomes -d firm --dir /scratch/run17 --gzip

# query it — against itself, or against another database
fastaai query -q FastAAI/database/firm                 # -> FastAAI/results/
fastaai query -q queries/ -t FastAAI/database/firm -o aai.tsv
fastaai query -q FastAAI/database/firm -o -            # -> stdout

# any other SCP set works; --hmm takes a file
fastaai build /path/to/genomes --hmm my_markers.hmm -d custom

# or one of the packaged sets, by name
fastaai build /path/to/genomes --hmm gtdb-bact -d gtdb
```

**The 122 SCPs FastAAI 1 shipped are bundled and used by default**, so an install
works without hunting for models. They are a default, not a fixture: `--hmm` takes
any HMM file, plain or gzipped, and the accession list, index and output all
follow from it. Every database records which model set built it, so defaults and
overrides cannot be mixed by accident.

GTDB's marker sets are packaged too, reachable by name because a file inside
site-packages is not a path anyone wants to type:

| `--hmm` | models | |
|---|---|---|
| *(omitted)* | 122 | FastAAI 1's SCPs — the default |
| `gtdb-bact` | 120 | GTDB bac120, bacteria |
| `gtdb-arch` | 53 | GTDB ar53, archaea |
| `gtdb-all` | 168 | the union of both — they share 5 markers, so this is not 173 |

Case and underscores are interchangeable (`GTDB_BACT` works). These are
assembled from Pfam and TIGRFAM rather than copied from GTDB-Tk, and the pinned
Pfam versions have since moved on, so they benchmark the engine but **do not
reproduce GTDB's trees** — see `python/fastaai/data/README.md`.

On 2,943 genomes scored against GTDB R232 taxonomy, retrieval is saturated for
all three: same-genus top-1 is 99.04% for the default set, 99.16% for bac120 and
98.79% for ar53 — the last on the seven of 53 archaeal markers a bacterium
carries. There is no accuracy reason to switch; the reason is shared
preprocessing with GTDB-Tk, whose identify step already runs these searches.

Results are written one block per query/target partition pair, so a search
spanning more than one pair needs `-o` to name a directory rather than a file —
see [what changed: the output](#what-changed-the-output).

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
| `AAI_estimate` | `<30%` / `>90%` outside the regression's sensitivity band; `100.0` for a genome against itself |
| a pair sharing no marker | `N/A` in every value column |

`--emit` narrows the columns (`jaccard` drops `AAI_estimate`, `aai` drops
`avg_jacc_sim`); the default emits the full v1 schema.

**`--output_style matrix`** writes a Q×T grid of AAI — one row per query, one
column per target, `query_genome` in the corner — with v1's `15.0` and `95.0`
standing in for the two categorical labels, since a cell cannot hold a string.

It is written per block exactly as the TSV is: each file is the Q×T grid for one
partition pair, not for the whole search, so it carries no size restriction.

**A genome against itself reads `100`** — `100.0` in both the matrix diagonal and
the TSV's `AAI_estimate`. Identity there is given by the comparison, not inferred
from it; the regression is fitted and unbounded above, so consulting it returns a
value past 100 that reports as the `>90%` sentinel, which is uncertainty about
something that is not uncertain.

Only a genome against *itself* is exempt. Two distinct genomes that happen to be
identical are a measurement that came out at the ceiling, and still read `95.0`
in the matrix and `>90%` in the TSV — equality of content is not identity.

Three deliberate departures from v1: a pair sharing no marker is `N/A` in the
matrix where v1 writes `0`, which cannot be told from a real measurement of
zero; `poss_shared_SCPs` uses the `minimum` of v1's three bulk paths rather than
the `max` of its one scalar path; and a genome against itself reports `100`
where v1 reports `>90%`.

**FastAAI 1 command lines still work.** `build_db`, `db_query`, `aai_index`,
`single_query`, `multi_query` and `simple_query` are rerouted to the new verbs,
with arguments preserved where they still mean something and a diagnostic where
they do not. `merge_db` is the one that no longer has a target: it exits saying
so and gives the crystal-and-rebuild replacement.

```python
import fastaai

# The whole thing, from whichever rank you have. Ranks combine.
db = fastaai.preprocess(genomes="/path/to/genomes", database="firm")
db = fastaai.preprocess(proteins=["a.faa", "b.faa"], crystals="other/crystals")

res = fastaai.search(db, db, threads=8)
```

Reading the result. The matrices stay public, but the questions people actually
ask have answers:

```python
res.queries, res.targets     # the genomes on each side, in row/column order
res.shape                    # (n_queries, n_targets)
res.scps("GCF_000007085.1")  # markers that genome carries

res.best_hit("GCF_000007085.1")        # Match(query, target, aai, jaccard, shared, poss_shared)
res.hits_for("GCF_000007085.1", k=5)   # its five nearest, best first
res.top_hits(k=5)                      # that, for every query

# Filtered iteration. Call it to filter, or iterate it for everything.
for m in res(query="any", min_aai=60, min_shared_frac=0.5):
    print(m.query, m.target, m.aai, m.shared_frac)

res.jaccard   # (n, n) float64, NaN where no accession is shared
res.shared    # (n, n) uint32, accessions carried by both genomes
res.aai       # (n, n) float64, uncensored
```

`query=` and `target=` take `"any"` for all of them, one name, or a collection
of names; the thresholds are inclusive. `min_shared_frac` is
`shared / poss_shared` — of the markers the poorer genome carries, the fraction
actually compared, which is what separates a genuinely distant pair from a pair
where one genome is a bad assembly.

A self-comparison's diagonal is a genome against itself, so it is **not** a
neighbour: `hits_for` and `best_hit` skip it unless you pass
`include_self=True`. Pairs sharing no marker are dropped rather than reported as
zero — no shared marker is an absence of evidence, not evidence of distance.

`res.to_tsv(path)` writes FastAAI 1's table — same columns, same names, same
order. The band and the rounding are the engine's own, exposed rather than
reimplemented, and the output is asserted byte-for-byte against
`Database.write_block`; two writers for one format is how the two drift. A
search too large to hold in memory writes its blocks straight from Rust
instead.

Or one step at a time. Each takes and returns **paths**, so a step can be run
for a thousand genomes across a cluster and the results gathered afterwards;
the steps are per-genome and single-threaded, and `preprocess` is what adds the
thread pool.

```python
prot = fastaai.genome_to_protein("g.fna.gz", "FastAAI/proteins")
hits = fastaai.protein_to_hmm(prot, "FastAAI/hmm_hits")          # --hmm set optional
cry  = fastaai.prot_hmm_to_crystal([(prot, hits)], "FastAAI/crystals")

db   = fastaai.build_database("FastAAI/crystals", save_to="FastAAI/database/firm")
res  = fastaai.search(db, db, threads=8)
```

`prot_hmm_to_crystal` takes `(protein_path, hmm_path)` pairs and accepts either
this package's hit tables or HMMER's own `--tblout`, so a caller who already ran
`hmmsearch` does not have to reformat anything.

Every function that needs models takes the same `models=` spec `--hmm` takes:
omit it for the bundled 122 SCPs, name a packaged set (`"gtdb-bact"`,
`"gtdb-arch"`, `"gtdb-all"`, case and underscores interchangeable), or give a
path to any HMM file. A `ModelSet` you already built passes straight through.

```python
fastaai.preprocess(genomes="g/", models="gtdb-bact")
fastaai.protein_to_hmm(prot, "FastAAI/hmm_hits", models="my_markers.hmm")
```


## Notes

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

Working end to end: on-disk partitioned databases, the three stored
preprocessing ranks, crystal-driven builds, the FastAAI 1 compatible CLI, and
optional per-pair Jaccard standard deviation (`--do_stdev`). 258 Python and 53
Rust tests.

Not yet packaged for bioconda.

## Dependencies

`pyrodigal`, `pyhmmer`, `pyfastx`, `numpy`. Only numpy is ours alone — none of
the other three pull it in.

**numpy 1.x and 2.x both work, and produce identical output.** The 2.0 removals
(`np.float_` and friends) split a lot of downstream code; this package restricts
itself to spellings valid under both, and `tests/test_numpy_compat.py` fails if
one of the removed aliases is reintroduced. The suite passes under 1.26.4 and
2.4.6, and a query gives byte-identical TSV, matrix and API results under each.

numpy is used in one place — `SearchResult`, the in-memory Python API. The CLI
does not touch it; formatting and output are Rust.

## Licence

GPL-3.0-or-later — see [LICENSE](LICENSE). This is not only a preference:
FastAAI 1.5 imports pyrodigal, which is GPL-3.0-or-later, so a compatible
licence is required rather than chosen.

The bundled HMMs in `fastaai2/python/fastaai/data/` are redistributed from
FastAAI 1 under the MIT licence and keep their original notice; see the README
beside them. FastAAI 1 itself remains at
https://github.com/cruizperez/FastAAI.
