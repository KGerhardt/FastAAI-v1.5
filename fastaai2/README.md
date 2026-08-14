# fastaai2 — the package

The Rust core, the PyO3 bindings and the Python package. **Installation and
usage are in the [repository README](../README.md);** this file is the
developer-facing side - the decisions that are easy to get wrong when reading
the code, and the measurements behind them.

```
FASTA ──► pyrodigal ──► pyhmmer ──► best-hit filter ──► k-merise ──► inverted index
          └──────────── Python ────────────┘          └────────── Rust ──────────┘
```

Python owns ingestion, gene prediction and HMM search - pyfastx, pyrodigal and
pyhmmer already wrap mature C and release the GIL, so a Rust port would FFI back
to the same libraries for nothing. Rust owns k-merisation, the inverted index,
the counting kernel and the AAI transform.

```sh
maturin develop --release      # or: pip install .
python -m pytest tests -q      # 279 Python tests
cargo test                     # 53 Rust tests
```

## Design notes that are easy to get wrong

**The model set defines the accession list.** Accession IDs are positions in the
HMM file's order - there is no compiled-in Pfam set. Two databases may only be
compared when accession list, `k` and alphabet all match; `Database.search`
refuses otherwise, because mismatched model sets produce structurally valid,
biologically meaningless output.

**`<30%` and `>90%` are results, not withheld numbers.** The usable band of the
Jaccard→AAI regression runs from J ≈ 0.006 (AAI 30%) to J ≈ 0.843 (AAI 90%).
Outside it the regression has no sensitivity left - it cannot separate 27% from
19% - so reporting a figure there would claim a precision the estimator does not
have. The AAI column emits the two categories as labels, as v1 did; matrix
output cannot hold a string and carries v1's `15.0` / `95.0` sentinels instead.
Zero Jaccard is labelled `<30%`, because `log(0)` otherwise lands it above the
ceiling.

**Storage is raw; the labels are applied at output.** The database keeps full
precision so it can be re-reported or fed downstream without prior rounding, and
`res.jaccard` / `res.aai` are raw floats. The regression is unbounded above, so
identical genomes compute past 100% - that is a fact about the fitted curve, and
it is exactly why the reported value is capped categorically at `>90%`.

**Unshared pairs are NaN, not 0, and are reported `N/A`.** "These genomes share
no marker" and "these genomes have AAI 0" are different statements, and neither
is `<30%`.

**Best-hit resolution is a real choice** (`--filter`). Strict reciprocal best hit
(`rbh`) is FastAAI's stated intent and is markedly harsher than either path
shipped in v1. The default `v1` reproduces v1 as actually executed and admits
some non-reciprocal hits; `v1_alt` is the ordering from v1's uncalled second
implementation. None is a superset of another - see `tests/test_filters.py` for
the discriminating cases. The choice changes SCP sets and therefore AAI.

**Crystals are the only route into an index.** There is no merge and no
incremental append. Both preserve whatever partitioning their inputs had, which
is the fragmentation trap: merging N one-genome databases produces N one-genome
partitions and costs roughly 90x search throughput. Combining collections means
putting their crystals together and rebuilding, which repacks.

**`--threads` defaults to 8 as a starting point, not a ceiling.** The only
scaling measurement so far is from a 6P+8E laptop - 6.17× at 16 threads, worse
beyond - and that machine is the wrong instrument for the question: past six
threads it schedules onto efficiency cores and past fourteen it is
oversubscribed. Neither result is a property of the kernel, and neither
transfers to a compute node with real memory bandwidth. Raise `--threads` to the
core count there; nothing in the code caps it. Proper scaling numbers on
server hardware are still to be measured.

**Preprocessing parallelises with processes, not threads.** pyfastx holds the
GIL through the parsing that dominates crystallisation, so threads come out
slower than serial there; processes measured 5-6× at 8 workers. Pool
initialisers build the HMM set once per worker rather than once per genome.

## Kernel measurements

Measured in the prototype (`../prototype/`), which implements and benchmarks the
alternatives behind each decision:

| | |
|---|---|
| counting kernel | ~0.53 ns per posting increment (~1.8 cycles) |
| vs FastAAI 1's numpy kernel | **6.5× per core**, with SQLite already removed from v1's side |
| real 2,943-genome index, 8 threads | **682k genome pairs/s/thread** - 8.66M pairs in 1.6 s |

The kernel is a k-mer join reading both sides as inverted indexes. On real data it
is 1.50× the superseded per-query kernel, 1.85× with the symmetric upper-triangle
path; the per-query kernel is in git history.

Variants measured and rejected: L1 tiling (0.83×), delta+varint payloads (0.43×),
private accumulators (0.85×), software prefetch (0.73×), narrow counters (noise).
`bitpacking::BitPacker4x` wins at full partition size and high thread count
(1.18× faster, 1.36× smaller) and is the one deferred optimisation worth revisiting.

**Synthetic benchmarks pointed the wrong way three times in this project** - most
sharply on the join, which synthetic sequences rated 0.96× and real genomes 1.50×.
Random sequences have no k-mer sharing structure, and sharing is what the kernel
exploits. Benchmark against a real genome collection.

The single load-bearing invariant is that **posting lists are sorted by local
genome ID**, which makes the accumulator a monotone sweep rather than a random
scatter. Every variant that disturbed it lost.

Output is written in bands rather than one whole block: a 16,384² block of
Jaccard, shared and stdev arrays is about 5 GiB, past the 2 GiB-per-thread budget
that set `PARTITION_SIZE` in the first place. Banding brings a full block to
roughly 1 GiB for under 1% in wall time.

## Deliberately not implemented

**Resumable block outputs.** A search is cheap enough - minutes of CPU for a
realistic query against GTDB's ~200k species clusters - that recomputing one
costs less than the machinery for not recomputing it.

**Database merge and append.** See the crystal note above.
