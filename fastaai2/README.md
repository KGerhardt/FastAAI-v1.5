# FastAAI 2

Average amino acid identity between microbial genomes, estimated from tetramer
sketches of single-copy protein-coding genes.

```
FASTA ──► pyrodigal ──► pyhmmer ──► best-hit filter ──► k-merise ──► inverted index
          └──────────── Python ────────────┘          └────────── Rust ──────────┘
```

Python owns ingestion, gene prediction and HMM search — pyfastx, pyrodigal and
pyhmmer already wrap mature C and release the GIL, so a Rust port would FFI back
to the same libraries for nothing. Rust owns k-merisation, the inverted index,
the counting kernel and the AAI transform.

## Install

```sh
maturin develop --release      # or: pip install .
```

## Use

```sh
fastaai build /path/to/genomes --hmm models.hmm -d db/ --archive arch/
fastaai query -q db/ -o aai.tsv
```

FastAAI 1 command lines (`build_db`, `db_query`, `aai_index`, …) are rerouted to
these verbs.

```python
import fastaai

models = fastaai.ModelSet("models.hmm")
paths  = fastaai.find_genomes("/path/to/genomes")

# Preprocessing is >98% of a cold run — archive it so it is paid once.
records = fastaai.preprocess(paths, models, threads=8, archive_root="arch/")
db, skipped = fastaai.build_database(records, models)
res = fastaai.search(db, db, threads=8)

# Later: rebuild in seconds, no prediction or HMM search.
db = fastaai.build_from_archive("arch/")

res.jaccard   # (n, n) float64, NaN where no accession is shared
res.shared    # (n, n) uint32, accessions carried by both genomes
res.aai       # (n, n) float64, uncensored
```

## Design notes that are easy to get wrong

**The model set defines the accession list.** Accession IDs are positions in the
HMM file's order — there is no compiled-in Pfam set. Two databases may only be
compared when accession list, `k` and alphabet all match; `Database.search`
refuses otherwise, because mismatched model sets produce structurally valid,
biologically meaningless output.

**Jaccard is stored raw and AAI is uncensored.** FastAAI 1 wrote the strings
`"<30%"` and `">90%"`. The usable band runs from J ≈ 0.006 (AAI 30%) to J ≈ 0.843
(AAI 90%), so at the evolutionary distances this tool exists to serve the
interesting values sit near the bottom of the range — censoring them discards
exactly the signal being sought. The regression is also unbounded above, so
identical genomes report >100%. Clamp at display time, not in storage.

**Unshared pairs are NaN, not 0.** "These genomes share no marker" and "these
genomes have AAI 0" are different statements.

**Best-hit resolution is a real choice** (`--filter`). Strict reciprocal best hit
(`rbh`) is FastAAI's stated intent and is markedly harsher than either path
shipped in v1. The default `v1` reproduces v1 as actually executed and admits
some non-reciprocal hits; `v1_alt` is the ordering from v1's uncalled second
implementation. None is a superset of another — see `tests/test_filters.py` for
the discriminating cases. The choice changes SCP sets and therefore AAI.

**Thread counts are capped deliberately.** The counting kernel is memory-bound.
Measured scaling on a 6P+8E laptop: 6.17× at 16 threads and *negative* beyond —
20 threads was slower than 16. Nothing here defaults to `available_parallelism()`.

## Performance

Measured in the prototype (`../prototype/`), which implements and benchmarks the
alternatives behind each decision:

| | |
|---|---|
| counting kernel | ~0.53 ns per posting increment (~1.8 cycles) |
| vs FastAAI 1's numpy kernel | **6.5× per core**, with SQLite already removed from v1's side |
| real 2,943-genome index, 8 threads | **5.34M genome pairs/s** — 8.66M pairs in 1.6 s |

The kernel is a k-mer join reading both sides as inverted indexes. On real data it
is 1.50× the superseded per-query kernel, 1.85× with the symmetric upper-triangle
path; the per-query kernel is in git history.

Variants measured and rejected: L1 tiling (0.83×), delta+varint payloads (0.43×),
private accumulators (0.85×), software prefetch (0.73×), narrow counters (noise).
`bitpacking::BitPacker4x` wins at full partition size and high thread count
(1.18× faster, 1.36× smaller) and is the one deferred optimisation worth revisiting.

**Synthetic benchmarks pointed the wrong way three times in this project** — most
sharply on the join, which synthetic sequences rated 0.96× and real genomes 1.50×.
Random sequences have no k-mer sharing structure, and sharing is what the kernel
exploits. Benchmark against an archive of real genomes.

The single load-bearing invariant is that **posting lists are sorted by local
genome ID**, which makes the accumulator a monotone sweep rather than a random
scatter. Every variant that disturbed it lost.

## Status

Working end to end. On-disk partitioned databases, genome manifest, merge,
archives, the FastAAI 1 compatible CLI, and optional per-pair Jaccard standard
deviation. Partitioned at 16,384 genomes.

Not yet implemented: block scheduler and resumable block outputs — a database
larger than memory must currently be queried partition by partition. Not yet
packaged for bioconda. See `../FASTAAI2_PLAN.md`.
