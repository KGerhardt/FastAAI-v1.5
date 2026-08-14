# Prototype — frozen

**This is not the engine.** The shipped engine is `../fastaai1_5/`. Nothing here is
built, imported or tested by the package, and none of it is maintained.

It is kept because it is the only runnable record of the alternatives that were
measured and rejected. Every performance claim in the READMEs and in
`../FASTAAI1_5_PLAN.md` was produced here, and deleting it would turn those numbers
into assertions nobody can check.

## What it measured

| variant | result |
|---|---|
| k-mer join (inverted x inverted) | **chosen** — 1.50x the per-query kernel on real data, 1.85x with the symmetric path |
| L1 tiling | 0.83x |
| delta + varint payloads | 0.43x |
| private per-thread accumulators | 0.85x |
| software prefetch | 0.73x |
| narrow (u16) counters | noise |
| `bitpacking::BitPacker4x` | 1.18x faster, 1.36x smaller at full partition size and high thread count — the one deferred win worth revisiting |

The single load-bearing invariant it established: **posting lists sorted by local
genome ID**, which makes the accumulator a monotone sweep rather than a random
scatter. Every variant that disturbed it lost.

It also established that **synthetic benchmarks pointed the wrong way three
times** — most sharply on the join, which synthetic sequences rated 0.96x and
real genomes 1.50x. Random sequences have no k-mer sharing structure, and sharing
is what the kernel exploits. `data.bin` is an extract of real genomes for that
reason.

## Two things to know before trusting a number from here

**The data predates the alphabet fix.** `data.bin` is encoded over 21 symbols
including the stop codon (`*ACDEFGHIKLMNPQRSTVWY`); the engine uses the 20
canonical amino acids. This does not affect the table above, which is entirely
*ratios* measured on identical data, but it does affect absolute figures — the
15.8% occupancy measurement in particular would move.

**The code has diverged.** `kernel.rs` here still contains the retired per-query
kernel alongside the join; `kmer.rs`, `index.rs` and `aai.rs` are all earlier,
smaller versions of their counterparts. Read `../fastaai1_5/src/` for how anything
actually works.

## Running it

```sh
cargo run --release --bin bench      # kernel variants
cargo run --release --bin codecbench # posting-list encodings
cargo run --release --bin accbench   # accumulator strategies
cargo run --release --bin verify     # correctness against a brute-force oracle
```

Zero dependencies, std only.
