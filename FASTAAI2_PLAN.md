# FastAAI 2 — Engine Design & Implementation Plan

Status: design draft, 2026-08-02
Reference implementation under review: `FastAAI/fastaai/fastaai.py` (FastAAI 1, 4882 lines)

---

## 0.0 Repository layout

| path | what it is |
|---|---|
| `fastaai2/` | **the clean codebase.** Rust core + PyO3 + Python package, maturin build. Working end to end. |
| `prototype/` | the measurement harness. Every benchmark behind a MEASURED section lives here; keep until the clean tree reproduces them. |
| `FastAAI/` | upstream FastAAI 1, for reference and equivalence checking. |

`fastaai2/` currently implements: dense k-merisation, the partition-local
inverted index, the counting kernel, the AAI transform, pyfastx/pyrodigal/pyhmmer
preprocessing, all three best-hit filter modes, and a CLI. 23 Rust + 32 Python
tests pass. Not yet implemented: on-disk partition format, genome manifest, block
scheduler, `BitPacker4x` payloads.

### Alphabet — 20 residues, stop excluded

`DEFAULT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"`, so `|A|` = 20 and `K` = 20⁴ =
**160,000**.

FastAAI 1 encoded the stop codon `*` as ASCII 42 and therefore included the one
tetramer that spans it. It carries no amino-acid information and can never be a
meaningful part of a Jaccard comparison, so it is excluded here. Because
out-of-alphabet residues break the k-mer window rather than aliasing onto a code,
excluding it simply means that single terminal window is not emitted — verified
in `kmer.rs::stop_codon_is_not_in_the_alphabet`.

The payoff is in the addressing structure, which is sized by `K` and not by
genome count:

| | `\|A\|` = 21 (v1) | `\|A\|` = 20 (shipped) |
|---|---|---|
| `K` | 194,481 | **160,000** |
| offsets per accession | 778 KB | **640 KB** |
| offsets per partition @122 acc | 94.9 MB | **78.1 MB** |

**17.7% off the fixed per-partition overhead.** At GTDB scale — 930k genomes in
15 partitions of 65,536 — that is 1.42 GB → **1.17 GB** of pure addressing.

> **This is a bug fix, not a preference — see §6.3.** v1's k-merizer takes
> `ord()` of every character with no symbol check, so it encodes `*` and `X` as
> if they were residues. The lookup-table k-merizer that would have filtered them
> is dead code whose index is never built. Measured cost of the correction:
> **mean 1.58×10⁻⁴, max 1.17×10⁻³ Jaccard**, SCP sets unchanged (§6.1).

**Real data is now available** at
`C:\Users\kenji\Desktop\fastaai2\ncbi_bacilliati\genome_collections` — five
collections spanning multiple phyla. Note `bacciliati_labeled_genomes` (6,622) is
the *union* of the other four renamed by `genome_mover.py`
(3,434 + 83 + 2,943 + 162 = 6,622), so ingesting both directories double-counts
every genome. **6,622 unique genomes, not 13,244.**

This replaces the 10 same-family example genomes and fixes all three seed-set
biases recorded in §0 at once. First run on 12 Firmicutes genomes produced
off-diagonal Jaccard **0.057–0.254** (AAI 41–55%) — the distant regime the
Xanthomonadaceae set never reached.

### Ingestion

pyfastx, via `pyfastx.Fastx` — **not** `pyfastx.Fasta`. Measured on 40 gzipped
genomes: `Fastx` 1.22 s, `Fasta(build_index=False)` 2.56 s, hand-rolled gzip
1.05 s. `Fastx` leaves no `.fxi` sidecars beside read-only data. It is ~0.86× a
hand-rolled parser, which is irrelevant — ingestion is ~0.03 s/genome against
~4.8 s/genome for prediction plus HMM search.

> **pyfastx requires a path.** Passing `bytes` **segfaults the interpreter** with
> no traceback; `BytesIO`/`StringIO` raise `MemoryError`; a `str` is treated as a
> filename. Decompress-to-memory-then-parse is therefore not available, and
> `ingest.read_fasta` rejects non-path input explicitly.

### HMM loading — never pass a path

The reverse holds for pyhmmer, and the effect is large. HMMER's own file reader is
slow and pyhmmer binds it faithfully, so a *path* takes that slow route. **Any
Python file-like object bypasses it.** Measured on the bundled 9.22 MB /
122-model set, median of 7:

| route | | |
|---|---|---|
| `HMMFile(path)` | 4,121 ms | HMMER's reader |
| `open(rb)` → `HMMFile(fileobj)` | 428 ms | fast route, default 8 KB buffer |
| **`open(rb, buffering=1MB)` → `HMMFile(fileobj)`** | **173 ms** | **shipped** |
| `read()` + `HMMFile(BytesIO(data))` | 154 ms | fastest, but holds the whole file |

**~24x**, and all routes yield identical models — same names, accessions and
lengths. The residual gap between 428 ms and 173 ms is pure small-read overhead
from Python's default buffer, not anything about the parse.

`search._load_hmms` ships the buffered stream rather than the marginally faster
full read: constant memory regardless of model-database size, and no
`MemoryError` fallback branch. `tests/test_models.py` guards equivalence so the
fast path cannot silently regress.

**`hmmpress` was considered and declined.** Pressed files would fix both problems
at once — `.h3f`/`.h3p` hold optimized profiles, so there is no per-call
re-optimization and no text parse — and the pressed reader is very fast. It is not
worth it here because **a small marker set is a FastAAI design invariant, not an
accident**: 122 models / 9.2 MB loads in 173 ms once per worker, so the remaining
saving is negligible against the cost of four derived files per model set, a cache
location that works when the model directory is read-only, and staleness
invalidation. Keep the decision tied to that invariant — a variant built on
thousands of models would flip it, and at that point the model-set fingerprint
(§3.2) is the natural cache key.

This matters more than a one-off 4 s suggests. FastAAI 1 calls
`load_hmm_from_file` inside `hmm_preproc_initializer` — i.e. **once per
multiprocessing worker** (`fastaai.py:653`, and again at `1823`, `3969`), so a
16-process pool burned ~66 s on model parsing alone.

### Optimized profiles — not ported, on performance grounds only

`pyhmmer.hmmsearch` re-optimizes its models on **every call**, so pre-building
`OptimizedProfile` objects once (v1's `optimize_models`, `fastaai.py:238`) is
strictly correct for a workload that searches once per genome, and it does remove
that repeated work.

**It is not ported because the saving is invisible**, not because it is unsafe.
Optimization is a small fraction of the ingest → predict → search loop and sits
below inter-run variability in aggregate. Nothing more than that.

> **v1's bare `except:` here is deliberate, and correct.** An earlier draft of
> this document called it a silent-failure hazard. That was wrong. The safety
> argument is:
>
> * If `.optimized()` **succeeds**, the model carried the information the profile
>   needs, so the pre-built profile is what `hmmsearch` would have constructed
>   anyway — identical behaviour, minus the repeat overhead.
> * If it **fails**, that is precisely because the model relies on defaults that
>   `.optimized()` will not fabricate. It raises rather than guessing, the
>   `except` catches it, and the raw models go to `hmmsearch`, which does have the
>   default-arbitration logic.
>
> The exception *is* the fallback signal. Success implies equivalence, so there is
> no path where a pre-optimized profile silently behaves differently. This is not
> a candidate source of v1/v2 divergence in the equivalence harness (§6), and the
> author's note that it "doesn't seem to be improving performance" was measuring
> the real thing, not a fallback.

---

## 0. Prototype status

A working prototype lives in `fastaai-rs/` (Rust, zero dependencies, std only),
built against v1's fixed 122-accession SCP set as a controlled reference.

**Verified** (`cargo run --release --bin verify`), on the 10 real example genomes
run through v1's actual pyrodigal/pyhmmer pipeline:

- the dense encoding is an exact bijection with v1's decimal-ASCII encoding —
  checked element-wise across 792 SCPs / 156,157 kmers, not sampled
- all kernel variants agree bit-for-bit
- Jaccard means match an independent pure-Python set-arithmetic reference to
  **5.6×10⁻¹⁶** (2 ULP) across all 100 genome pairs

**Benchmarked** (`cargo run --release --bin bench`), synthetic scale-up to a full
65,536-genome partition. Results are in §4.1 and §4.2, and they **overturned one
of this document's original design claims** — tiling was measured to be a net
loss and has been dropped. Sections marked MEASURED supersede the reasoning that
preceded them.

Reference numbers actually observed, replacing earlier estimates:

| quantity | earlier estimate | measured |
|---|---|---|
| alphabet | 25 symbols | **20** (residues only; `*` excluded, see below) |
| `K` = \|A\|⁴ | 390,625 | **160,000** |
| unique tetramers per SCP | ~500 | **median 159, mean 197** (≈ length − 3) ⚠ |
| SCPs per bacterial genome | 122 | **79–80** (rest are archaeal) |
| mean posting-list length @16k | — | **47.8** (33% are singletons) |

> ⚠ **Seed-set caveats — RESOLVED against 2,943 real Firmicutes.** Both were
> checked once real data existed, and both came out differently than assumed:
>
> 1. **SCP length — no correction needed.** Real: mean **194** tetramers/SCP,
>    median 145 (p90 407, max 2,155). The seed set's 197 was accurate. The
>    ~1.7–2.1× pessimism factor previously applied to every throughput
>    projection in this document was unwarranted and is **withdrawn** — the
>    uncorrected figures stand.
> 2. **Tetramer occupancy — far lower than either estimate.** Real: **15.8%**
>    overall, mean 18.2% per accession, median 12.6%; only 8 of 106 accessions
>    exceed 50%. Synthetic said 35.6% and was *overstating*. This reverses the
>    §3.5 encoding decision — see there.
>
> The one genuine seed-set limitation that remains is Jaccard range: the bundled
> genomes span 0.52–1.00 and never reach the distant regime, where real
> Firmicutes sit at a median of 0.089.

---

## 0.1 Scope

FastAAI estimates AAI between microbial genomes by:

1. Predicting proteins (prodigal)
2. Identifying single-copy protein-coding genes (SCPs) by HMM search
3. Kmerizing each SCP into tetramers
4. Computing Jaccard between matched-SCP pairs across genomes
5. Averaging Jaccard over shared SCPs and regressing to AAI

Step 4 is the hot path and the reason the tool exists. The algorithm — an inverted
index from (SCP, kmer) to the set of genomes carrying it, then counting genome-ID
occurrences to recover intersection cardinalities — is correct and is retained
unchanged. This plan replaces its *implementation*.

Two changes define v2:

- **The counting engine moves to Rust.** Preprocessing stays in Python on
  pyrodigal/pyhmmer.
- **The SCP set becomes a property of the database, not of the program.** v1's
  fixed 122-accession list is removed.

### Non-goals

- **No minimum-AAI threshold or prefilter mode.** Sketch-based pruning (MinHash
  etc.) would change the complexity class for high-identity queries, but
  FastAAI's value proposition is behavior at extreme evolutionary distance and on
  highly novel genomes. Every pair is computed exactly. This is a deliberate,
  closed decision.
- No change to the Jaccard→AAI regression.
- No reimplementation of prodigal or HMMER.

---

## 1. Audit of FastAAI 1

Findings from reading the current source. These need decisions before or during
the port.

### 1.1 Preprocessing is already pyrodigal/pyhmmer

No `subprocess` calls to `prodigal` or `hmmsearch` remain. The "revision for
simplicity and robustness" is largely a **deduplication** job, not a port.

### 1.2 There are two generations of HMM manager, and the newer one is dead

- `pyhmmer_manager` (`fastaai.py:211`) — instantiated at `653`, `1823`, `3969`.
  This is what actually runs.
- `new_pyhmmer_manager` (`fastaai.py:488`) — **no call sites anywhere.** Dead.

They are not equivalent. `filter_to_best_hits` applies the two dedup passes in
opposite order:

| | first pass | second pass |
|---|---|---|
| old (`392`) | unique by protein (`402`) | unique by accession (`406`) |
| new (`604`) | unique by accession (`614`) | unique by protein (`620`) |

Concrete divergence. Rows sorted by score descending:

```
(P1, accA, 100)
(P2, accA,  95)
(P2, accB,  90)
```

- old → `{P1: accA}` — one SCP; accB discarded entirely
- new → `{P1: accA, P2: accB}` — two SCPs

The new order recovers an SCP the old one throws away, at the cost of assigning
P2 to accB despite P2 scoring higher on accA. Both are defensible reciprocal-best-hit
tie-breaks. **They produce different kmer sets and therefore different AAI.**

> **ACTION:** pick one, document it as a version boundary, and record the choice
> in database metadata (§3.2). Do not let this drift silently — it changes
> published numbers.

### 1.3 `assign_domain` is uncalled, but it is not dead weight — **RESOLVED**

`assign_domain` (`fastaai.py:415`) performs bacterial/archaeal domain voting by
comparing the *fraction* of each domain's SCP set recovered
(`bacterial_fraction = domain_counts["Bacteria"] / voted_domain["Bacteria"]`,
`422-423`), then prunes `best_hits` to the winning domain's set (`430-437`). It
has **no call site**; `run_for_fastaai` (`472`) calls only
`filter_to_best_hits()`.

**It encodes a real empirical finding:** proportional bacterial/archaeal HMM
recovery from the v1 122-SCP set classifies domain with 100% accuracy on the
validation performed. That is a genuine capability and must not be discarded.

But it is a **routing** decision — which database to query against — not part of
the AAI computation. It belongs above the engine (§2.1), scoped to a model
collection that carries domain labels, rather than compiled into the tool.

> **RESOLVED:** relocate, do not delete. Remove `assign_domain` from the
> preprocessing path (it is already not running there); reimplement as the
> routing layer in §2.1. **Do not restore the SCP-pruning behavior** — see §2.1
> for why routing makes it unnecessary and why restoring it would change AAI
> relative to every number v1 has ever published.

### 1.4 Fixed accession list

`generate_accessions_index()` (`fastaai.py:1469`) hardcodes 122 Pfam accessions.
`find_hmm()` (`1397`) locates a bundled `Complete_SCG_DB.hmm`. Both are removed
in v2 (§3.2).

### 1.5 Hardcoded HMM thresholds

`bit_cutoffs="trusted"` at `335`, `337`, `549`. This **raises an exception on HMMs
that lack TC lines**, which arbitrary user-supplied models frequently do. Blocks
SCP-agnostic operation directly.

> **ACTION:** threshold policy becomes per-database metadata (§3.2).

### 1.6 `cpus=1` is hardcoded in every hmmsearch call

Because each call runs inside a `multiprocessing` worker. pyhmmer releases the
GIL, so once the process pool is gone this becomes a live tuning knob — and
preprocessing can be threaded rather than forked even in Python today.

### 1.7 Memory amplification in the counting step

Per query genome, in `one_work` (`fastaai.py:2787`):

```python
these_intersections = np.bincount(flatten_cached_targets(...), minlength=_nt)  # int64!
results.append(these_intersections)
results = np.vstack(results)                              # full copy
unions  = np.subtract(np.add(tgt_counts, ...), results)   # another
results = np.divide(results, unions)                      # another
```

For M = 20,000 targets and ~120 SCPs that is ~16 MB per intermediate, four or five
of them — roughly **70–80 MB of freshly allocated, zeroed and traversed memory to
produce one 20,000-element answer vector.** `np.bincount` always returns `int64`,
so counters bounded by protein length (a few thousand) occupy 8 bytes each.

`flatten_cached_targets` (`2733`) concatenates the posting lists into one array
purely so `bincount` can walk it.

> **CORRECTION (measured).** An earlier draft called this concatenation "dead
> work." It is not. Materializing the target-ID stream and re-reading it measures
> **1.02×** against counting from each slice in place (§4.6, variant M1) — a
> memcpy that stays in cache costs nothing. v1's real cost in this function is
> numpy allocation, `int64` counters, and Python overhead, not the concatenation
> itself. The allocation and width points in this section stand; the
> concatenation point does not.

### 1.8 SQLite in the hot path

`file_v_db_worker` (`fastaai.py:1846`, lines `1910-1929`) creates a `TEMP TABLE`,
`executemany`s ~500 kmers, commits, and runs an `INNER JOIN` — **per SCP per query
genome**, i.e. ~120 table creations per genome. A B-tree is being used to look up
a key that could be a direct array index.

### 1.9 Index growth is read-modify-write

`build_db` grows the index via
`INSERT ... ON CONFLICT(kmer) DO UPDATE SET genomes = genomes || (?)` (`1775`,
again at `3723`). Every add rewrites every touched posting list.

### 1.10 Process parallelism

Every pool (`2195`, `3206`, `3257`) is `multiprocessing` with fork + module
globals. `_tl` is `np.array(tl, dtype=object)` (`2719`) — an object array of
arrays, so traversal dirties refcount pages and defeats copy-on-write. Resident
memory scales with thread count; on spawn platforms (macOS, Windows) the entire
index is pickled per worker.

### 1.11 Output precision

`avg_jacc_sim` is rounded to 4 decimals before writing (`1985`, `2827`). Inverting
the AAI regression:

| AAI | Jaccard |
|---|---|
| 30% | ≈ 0.0057 |
| 65% | ≈ 0.4447 |
| 90% | ≈ 0.8430 |

At the distances FastAAI exists to serve, Jaccard runs ~0.006–0.05, so 4 decimals
leaves **two significant figures** — the quantization is harshest exactly in the
regime the tool advertises. AAI is additionally censored to the literal string
`"<30%"` below Jaccard ≈0.006 (`2328`, `2335`, `2338`), and clamped to `15` in the
matrix path (`2375`).

> **ACTION:** emit raw Jaccard at full precision (or scientific notation).
> Censoring is a display choice and should not be baked into stored numbers.

### 1.12 Output formatting will become the bottleneck

With no threshold mode the full N×M matrix is always computed. At 20k×20k that is
4×10⁸ cells, and the TSV path (`1999-2003`, `2840-2844`) formats each with Python
string concatenation in a loop.

---

## 2. Architecture

```
  FASTA ──► pyrodigal ──► pyhmmer ──► best-hit filter ──► kmerize ──► sealed partitions
            (Python, GIL-releasing C)                     └──────── Rust ────────┘

  query DB ×  target DB  ──►  tiled counting kernel  ──►  Jaccard ──► AAI ──► output
                              └──────────────── Rust ─────────────────────┘
```

**Hybrid, not rewrite.** Steps 1–2 are already pyrodigal/pyhmmer — Cython over C,
already releasing the GIL. There is no mature pure-Rust HMMER; a Rust port would
FFI back to the same C library for no gain. Rust replaces steps 3–5 only, exposed
via PyO3/maturin. The Python CLI and the ecosystem integration survive.

> **This boundary is contingent on that premise, which is being actively
> falsified.** A Rust reimplementation of HMMER's management layer is under
> development separately (parallel `hmmpress`, work stealing, shared memory,
> cache-level model control, database-backed output). If it lands, the boundary
> should move to Python-orchestrates / Rust-owns-everything-from-FASTA-in.
>
> The concrete win for FastAAI is **shared preprocessed sequence**. The current
> path materialises the same residues three times across two languages:
> prodigal → Python `dict` → `TextSequence.digitize()` → search → back to Python
> `str` → `.encode()` → Rust → k-merise. With prediction, search and k-merisation
> all Rust-side, prodigal output feeds the search and then the k-merizer with no
> Python object ever constructed, and one work-stealing scheduler can balance
> stages whose per-genome costs differ by three orders of magnitude.

**Expected payoff.** 10–50× on the on-disk query path (SQLite removal dominates),
5–15× on the in-memory path, near-linear thread scaling, and a large drop in peak
RSS. End-to-end depends on workload mix: build-heavy runs stay HMM-bound and move
~1.5–2×; large all-vs-all queries are where the real number is.

### 2.1 Routing is a layer above the engine

Domain assignment (§1.3) is the first instance of a general capability:
**given a genome and several candidate databases, pick the right one from which
markers were recovered.** v1's bacteria-vs-archaea vote is the shipped special
case of this, and it is worth keeping prominent — proportional recovery across the
two labeled SCP subsets classifies domain with 100% accuracy on the validation
performed.

In an SCP-agnostic tool this cannot be compiled in. It becomes a property of a
model collection:

- Models carry optional **group labels** in the metadata table (§3.2). The
  bundled default collection labels its 122 models Bacteria / Archaea; a custom
  collection can define its own groups for whatever routing its author needs.
- A **routing policy** records the rule — for the default, "compare the fraction
  of each labeled group recovered; route to the group with the higher fraction."
- Routing runs **after HMM search, before database selection**, on best-hits that
  already exist. It costs nothing extra and stays entirely in the Python layer.
  The Rust engine never sees it.

**Routing supersedes pruning, and this matters for equivalence.** v1's
`assign_domain` dropped cross-domain SCPs from `best_hits` before kmerization.
Routing to a domain-specific database achieves the same effect *structurally* —
that database's model set simply does not contain the other domain's markers, so
the query database built against it cannot carry them. No explicit pruning step,
no divergence from the un-pruned path v1 has always actually run (§1.3).

**One honest cost.** Routing needs a first pass with a collection broad enough to
discriminate, then a query database built against the *chosen* target database's
models. When the target's models are a literal subset of the routing collection
with identical thresholds — true by construction for the bundled bacteria/archaea
split — the first search's results can be filtered rather than re-searched. Routing
to an arbitrary third-party database requires a second HMM search. The bundled
default case is the fast one; document the general case as paying for generality.

---

## 3. Database format

### 3.1 One format, two roles

A *database* is a model metadata table plus a set of sealed partitions. **Query
and target are runtime roles, not types.** A query database is built by running
FASTA through the models extracted from the target database. All-vs-all is
DB-vs-itself.

This is what makes v2 SCP-agnostic: the target database is the sole authority for
model identity, model ordering, and search policy.

### 3.2 Model metadata table

Replaces `generate_accessions_index()` and `find_hmm()` entirely.

| field | purpose |
|---|---|
| model blobs | the HMMs themselves, concatenated + per-model offsets. pyhmmer loads from a byte buffer, so no temp files. |
| ordered accession names | position in this list **is** the accession ID. Local to the database. |
| threshold policy | per-model or per-database: `trusted` / `gathering` / `noise` / explicit bit score / E-value. Fixes §1.5. |
| best-hit filter semantics | which resolution order from §1.2 was used. |
| `k` | kmer length (default 4) |
| alphabet | ordered symbol list; `K = |alphabet|^k` (default 25 symbols → 390,625) |
| model group labels | *optional.* Per-model group membership for routing (§2.1). The bundled collection labels its models Bacteria / Archaea. |
| routing policy | *optional.* The rule applied over those groups. |
| **model-set fingerprint** | hash (BLAKE3) over concatenated model blobs **in order**, plus `k` and alphabet |

**The fingerprint deliberately excludes labels and routing policy.** Those do not
affect kmer sets, accession IDs, or any computed value — only which database a
genome is sent to. Including them would break comparability between databases that
are in fact fully comparable, and would make adding a label to the bundled
collection invalidate every existing database and every stored block.

**The fingerprint is a hard compatibility gate.** A query DB records the
fingerprint of the target DB it was derived from. Comparison refuses to run on
mismatch. Without this, comparing against a different model set yields numerically
valid, biologically meaningless AAI — a silent-wrong-answer failure mode that
v1 could not have, because its SCP set was compiled in.

### 3.3 Kmer encoding

v1 encodes tetramers as decimal concatenation of ASCII codes —
`ord(a)·10⁶ + ord(b)·10⁴ + ord(c)·10² + ord(d)` (`fastaai.py:1155`) — spreading
~160k real tetramers over a ~9×10⁷ sparse range and forcing a hash or B-tree
lookup.

v2 uses a dense ordinal encoding over the declared alphabet:

```
id = ((a·|A| + b)·|A| + c)·|A| + d        # range [0, |A|^k)
```

For the default 25-symbol alphabet, `K = 390,625`. Rolling encode is
`code = (code · |A| + lut[byte]) mod |A|^k`.

Dedup during kmerization uses a reusable `K`-bit bitset (48 KB), cleared via a
touched-list rather than a full memset — exact, allocation-free, O(len), replacing
`np.unique`'s sort.

### 3.4 Partitions

**`PARTITION_SIZE` = 16,384. `MAX_PARTITION` = 65,536 is a `u16` limit, not a
target.**

The operating size is set by RAM, not by ID width. Most HPC platforms now bill
**2 GiB per thread with no way to request RAM and cores independently**, so with
one partition in flight per thread the whole partition — index *plus working
space* — has to live inside 2 GiB. At today's measured density (15,561 posting
entries/genome, `|A|` = 20):

| P | index | offsets share | free of 2 GiB | parts @930k | total offsets |
|---|---|---|---|---|---|
| 8,192 | 0.31 G | 23.1% | 1.69 G | 114 | 8.29 G |
| **16,384** | **0.56 G** | **13.1%** | **1.44 G** | 57 | 4.14 G |
| 32,768 | 1.04 G | 7.0% | 0.96 G | 29 | 2.11 G |
| 65,536 | 2.01 G | 3.6% | **−0.01 G** | 15 | 1.09 G |

**The `u16` cap is already over budget before any working space**, which settles
it: 65,536 cannot be the operating size. 16,384 leaves 1.44 GiB for the
accumulator, query k-mer sets, output buffering and preprocessing slack.

**The rising offsets column is a storage cost, not a resident one.** Streaming
means one partition's 78.1 MB is live regardless of how many partitions exist —
57 partitions costs 4.14 GB on disk and 78.1 MB in RAM. Trading cheap bytes for
expensive ones is the right direction, and the headline is worth stating plainly:
**~30 GB uncompressed (~15 GB compressed) encodes 900k genomes.**

Genome IDs remain **local `u16` within a partition**; a partition is
self-describing and posting lists never reference anything outside it.

Sizing pressures:

- **Upper bound:** `u16` local IDs cap a partition at 65,536 genomes. This halves
  the size of the largest and hottest structure in the system.
- **Lower pressure (not a bound):** a dense per-accession offsets array is
  direct-addressed by kmer ID — `(K+1) × u32` = **1.56 MB per accession per
  partition, independent of genome count.** Short partitions do not amortize it,
  which is precisely why short partitions use the sparse encoding (§3.5). The
  adaptive encoding is what makes variable-size partitions viable; the two
  features are load-bearing for each other.

> **Do not size partitions to fit L1.** If partitions were 8,192 genomes, a
> 1M-genome database at 122 accessions would carry 122 partitions ×
> 122 accessions × 1.56 MB ≈ **23 GB of offsets, mostly zeros.** L1 residency is
> obtained logically instead (§4.1).

At the 65,536 cap: ~190 MB of offsets per partition (at 122 accessions) amortized
over ~4×10⁹ posting entries at ~2 bytes ≈ 8 GB of payload — ~2% overhead. A 20k
genome database is one partition and degenerates gracefully.

What partition-local IDs buy:

1. `u16` posting entries (2× on the dominant memory stream)
2. **Append-only builds** — partitions are immutable once sealed. Adding genomes
   writes a new partition. Kills the read-modify-write merge of §1.9 outright.
3. **Merge is manifest-only.** Combining two databases concatenates their
   partition sets and merges their manifests. **No posting list is ever
   renumbered or rewritten**, because no partition's IDs depend on anything
   outside itself. Merge cost is O(genomes), not O(index).
4. **Partitions are relocatable.** Copy, ship, archive, or drop one without
   renumbering. Deleting a partition leaves no hole to compact around.
5. **Out-of-core** — databases larger than RAM stream partition by partition.
6. Bounds the standard-deviation path's working set (§4.3).

Everything the kernel needs is partition-local and sized by `n_local`, not by
total genome count:

| per-partition structure | type | purpose |
|---|---|---|
| inverted index | kmer → `u16` local IDs, delta-coded | target role (§3.6) |
| forward index | genome → sorted kmer IDs per accession, delta-coded | query role |
| per-accession kmer counts | `u16[n_local]` | the `+|T|` term of the union |
| per-accession presence | bitmap, `n_local` bits | shared-SCP count |
| local → manifest ordinal | `u32[n_local]` | output placement (§3.4.1) |

Both index directions are stored. v1 already does this — `{acc}` (kmer → genomes,
`1571`) and `{acc}_genomes` (genome → kmers, `1666`) — because a partition acting
as *query* needs forward kmer lists while a partition acting as *target* needs the
inverted index, and deriving either from the other is prohibitive. Roughly doubles
storage; both directions delta-code well (forward lists are ~500 sorted IDs drawn
from `K`, so deltas average ~780 → ~2 bytes varint).

**Compaction** is the price of append-only. Many incremental builds accumulate
many short partitions, each paying fixed per-accession overhead. A `compact`
operation rewrites *k* short partitions into one sealed partition and rewrites
only the manifest rows for the affected genomes. This is a named, explicit
operation — never implicit during a build.

### 3.4.0 Two ID schemes — build both, measure

With `PARTITION_SIZE` fixed and a power of two, the global index is recoverable
by arithmetic:

```
partition = id >> 14        local = id & 0x3FFF        (P = 16,384)
```

**This does not conflict with manifest-only merge (§3.4), which is the
non-obvious part.** Posting lists hold *local* IDs `0..P-1` and never change.
Merging concatenates partition lists and assigns new partition *numbers*;
`global = partition_number * P + local` is **derived, never stored**, so it
shifts on merge while no posting list is touched.

| | arithmetic mapping | independent partition tables |
|---|---|---|
| forward + reverse map | free, two shifts | table lookup |
| partition sizes | must be uniform | any |
| seal on a build boundary | no — pad or compact | yes |
| merge cost | manifest only | manifest only |
| relocate / drop a partition | renumbers everything after it | free |

The lookup cost is not a discriminator: `(partition, local) → ordinal` is used for
*output placement*, in the same ~0.06%-of-runtime territory as the offsets lookups
(§4.6). The real trade is **uniform partitions** versus **seal-anywhere**.

> **Status: both to be implemented and compared.** `kernel::partition_offsets`
> computes offsets from actual partition sizes, so the current code is correct
> under either scheme and a short final partition works unchanged.
> `kernel::partitioning_does_not_change_results` asserts the invariant that
> matters — results identical across every chunking (1, 2, 3, 4, 5, 8, 9) and
> thread count.

### 3.4.1 Genome manifest

One global table, the only structure indexed across all partitions:

| field | type | purpose |
|---|---|---|
| ordinal | `u32` | **canonical output order** — row/column index in the AAI matrix |
| genome name | string | user-facing identity |
| partition | `u32` | which partition holds it |
| local ID | `u16` | ID within that partition |
| content hash | `u64` | duplicate detection across incremental builds |

The ordinal is explicit and stored, **not** derived from `(partition, local)`.
Output ordering must survive compaction, partition deletion, and merge, none of
which preserve partition or local ID.

The kernel produces results indexed by `(partition, local)`; writing output needs
the reverse direction, so each partition carries its own `u32[n_local]` local →
ordinal array (table above). That is a contiguous, sequential lookup during
output — no hashing, no global map consulted in any inner loop.

The manifest is small — for 1M genomes, tens of MB dominated by name strings — and
is loaded whole. It is the only part of the database that is rewritten in place.

### 3.5 Per-accession index encoding — adaptive

**This is required by SCP-agnosticism, not an optimization.** With the model set
under user control, both the number of accessions and `K` are variable. A 1,000-model
database costs 1.56 GB of offsets per partition; `k=5` raises `K` to 9.7M and the
dense array to 39 MB per accession per partition. Dense direct addressing cannot
be assumed.

Each `(partition, accession)` therefore carries a one-byte encoding tag chosen at
seal time by observed density:

- **Dense** — `u32[K+1]` direct-addressed. O(1) lookup, no branches. Default for
  well-populated accessions at `k=4`.
- **Sparse** — sorted list of present kmer IDs + parallel offsets, binary-searched.
  For rare accessions, large `K`, and **any short partition** (§3.4).

A query only ever looks up its own ~500 kmers per accession, so the branch on
encoding is per-accession, not per-kmer.

Three independent forces demand this, which is why it is a requirement rather
than a tuning option: variable model-set size, variable `K`, and variable
partition size.

> **Alphabet note.** Every measurement in this section was taken with `|A|` = 21
> (v1's alphabet, which included the stop `*`). The shipped alphabet is now the
> **20 amino acids only** — see §0.0 — so `K` drops 194,481 → 160,000 and every
> offsets figure below scales by **0.823**: 94.9 MB → **78.1 MB per partition**.
> The ratios and crossover points are unchanged, since both sides scale together.

**MEASURED.** With `|A|` = 21, `K` = 194,481 and 122 accessions, the dense offsets
array is **94.9 MB per partition, flat**, regardless of genome count:

| partition size | posting payload (`u16`) | dense offsets | overhead |
|---|---|---|---|
| 10 genomes (real) | 0.3 MB | 94.9 MB | **316×** |
| 16,384 | 363 MB | 94.9 MB | 26% |
| 65,536 (cap) | 1.45 GB | 94.9 MB | 6.5% |

**Occupancy decides the encoding, and the break-even is exact.** For accession `a`
in a partition, let `f` = fraction of the `K` tetramer slots that are occupied:

| structure | bytes per accession | at `f` = 0.5 |
|---|---|---|
| dense — `u32[K+1]` direct-addressed | `4(K+1)` = **778 KB**, flat | 778 KB |
| naive sparse — sorted (kmer ID, offset) pairs | `8fK` | 778 KB |
| bitmap + rank — occupancy bitmap + offsets | `K/8 + 4fK` | **413 KB** |

Naive sparse breaks even against dense at **exactly `f` = 0.5**, and is strictly
worse above it.

> **MEASURED — real occupancy is 15.8%, so sparse wins decisively.** Over 2,943
> Firmicutes with corrected gene calls: mean 18.2% per accession, median 12.6%,
> only 8 of 106 accessions above 50%. At `f` ≈ 0.16 naive sparse costs ~0.32×
> dense and bitmap+rank ~0.29×, so the 78.1 MB per-partition offsets array can be
> roughly **3× smaller** — and the saving is largest exactly where offsets
> dominate.
>
> **This reverses the conclusion recorded immediately below**, which assumed the
> >50% occupancy estimate and concluded dense won outright. Caveat: one phylum at
> 2,943 genomes. Occupancy rises with diversity and count, but the synthetic
> extrapolation to 35.6% is now the least trustworthy of the three figures, so
> the crossover must be measured on real multi-phylum data before the format is
> frozen — not projected. Real databases are expected to run **well north of 50% tetramer
occupancy** — some tetramers common, some rare, most occupied — so at the default
`k` = 4 the naive sparse encoding is the wrong tool for a full partition and
**dense is correct at the cap**.

Measured occupancy in the synthetic scale-up (which saturates artificially low —
it descends from 10 same-family genomes, see the §0 caveat):

| genomes | occupancy | mean list length |
|---|---|---|
| 4,096 | 11.6% | 23.3 |
| 16,384 | 22.6% | 47.8 |
| 65,536 | 35.6% | 121.5 |

**Bitmap + rank beats dense at every occupancy below ~0.97**, and its extra cost is
a popcount on lookup — which §4.6 measures at 0.06% of all operations, i.e. free.
That makes it the one encoding that serves short partitions, full partitions, and
large `K` alike, and it is the recommended implementation of the §3.5 tag rather
than a dense/naive-sparse pair.

The overhead being managed here is real but bounded at the cap: 95 MB of offsets
against 2.05 GB of postings is 4.6%. It is short partitions (316× at 10 genomes)
and large `K` (`k` = 5 → 39 MB per accession dense) where the choice actually bites.

### 3.6 Posting list encoding

Within a `(partition, accession, kmer)`, genome local-IDs are stored **sorted
ascending**. Two properties follow:

- A tile of the ID space is a contiguous run — tiling needs no index (§4.1).
- Deltas are small on exactly the lists that dominate bandwidth. A common tetramer
  present in most of a 65,536-genome partition has deltas of 1–3; a rare tetramer
  has large deltas but a tiny list.

**MEASURED — the compression works; the decode does not pay.** Delta+varint hits
exactly the predicted **1.00 bytes/entry** (a 2× win over `u16`, 4× over `u32`),
but the kernel is *2.3× slower* single-threaded and still 1.5× slower at 20
threads (§4.1, variant C).

The reason is that we are not bandwidth-bound where it matters. At 1 thread the
`u16` sweep moves ~5.7 GB/s of posting bytes — nowhere near DRAM. Trading ALU for
bandwidth is a straight loss there. Even at 20 threads, where the sweep reaches
~37 GB/s of actual `u16` traffic and DRAM does become the limit, hand-rolled
varint decoding loses: its byte-at-a-time loop is a serial dependency chain with
an unpredictable branch per byte, and that costs more than the bandwidth it saves.

**MEASURED — five encodings compared, all verified against the raw-`u16` baseline
before timing.** Full 65,536-genome partition, 1.03×10⁹ posting entries.

| encoding | bytes/entry | size vs `u16` | 1 thread | 20 threads |
|---|---|---|---|---|
| raw `u32` | 4.00 | 0.50× | — | — |
| **raw `u16`** (baseline) | 2.00 | 1.00× | 1.00× | 1.00× |
| delta + varint, scalar (hand-rolled) | 1.09 | **1.83×** | 0.43× | 0.66× |
| [`stream-vbyte`](https://crates.io/crates/stream-vbyte), scalar path | 1.33 | 1.50× | 0.09× | 0.19× |
| **[`bitpacking`](https://crates.io/crates/bitpacking) `BitPacker4x`** | 1.47 | **1.36×** | 0.83× | **1.18×** |
| hybrid bitmap + `u16` (hand-rolled) | 1.51 | 1.32× | 0.49× | 0.89× |

> **DECISION: `bitpacking` `BitPacker4x` over sorted 128-blocks, `u16` tail.**
> At production scale it is **1.18× faster *and* 1.36× smaller** than raw `u16` —
> the only encoding that wins on both axes. Raw `u16` remains the right choice for
> single-threaded or small deployments, where `BitPacker4x` costs 0.83×.

The crossover is the whole story. At one thread, bandwidth is free and every codec
pays pure decode cost. At 20 threads on a 2 GB index the memory system becomes the
constraint and SIMD-decoded compression turns positive. It only appears at *both*
full partition size and high thread count — at 16,384 genomes `BitPacker4x` reaches
just 0.97× even at 20 threads.

Notes on the losers:

- **The hand-rolled scalar varint compresses best (1.09 B/entry) and is still a
  loss.** Byte-at-a-time decoding is a serial dependency chain with an
  unpredictable branch per byte; compression ratio was never the problem.
- **`stream-vbyte` was not fairly tested.** Its SIMD paths require nightly
  (`feature(portable_simd)`); only the scalar fallback runs on stable, and at
  0.09× that fallback is not what the crate is for. The SIMD-codec hypothesis was
  nonetheless tested fairly *through `bitpacking`*, which is a port of Lemire's
  simdcomp using native SSE3/AVX2 intrinsics on stable — and which is the better
  fit for sorted sequences anyway. [`svbyte`](https://docs.rs/svbyte) offers Stream
  VByte on stable but its cursor-over-reader API is the wrong granularity for
  5.4M short lists.
- **The bitmap hybrid does not pay.** Walking set bits via `trailing_zeros` is
  slower than a linear `u16` read, and only 34% of entries live in lists dense
  enough to qualify.

Sortedness is retained regardless — it is required by `compress_sorted`, and §4.1
measures it as independently positive.

### 3.7 File layout

Whole-file `mmap`. Zero-copy startup, one resident copy regardless of thread
count, OS page cache handles residency. Replaces both the SQLite hot path and the
per-accession pickle staging at `2725`.

---

## 4. Query kernel

### 4.0 MEASURED — the k-mer join is the kernel, on real data

> **This supersedes the "query batching 0.96x" result.** That measurement was made
> on synthetic sequences and pointed the wrong way.

Full Firmicutes index (2,943 real genomes, 169.2 MB, 8 threads):

| kernel | time | vs per-query | pairs/s |
|---|---|---|---|
| per-query (gather posting lists per query genome) | 2.909 s | 1.00x | 2,977,289 |
| **join, block = 128** | **2.029 s** | **1.43x** | **4,268,690** |
| join, block = 256 | 2.468 s | 1.18x | 3,509,402 |
| join, block = 512 | 4.325 s | 0.67x | 2,002,673 |
| join, block = 1024 | 4.086 s | 0.71x | 2,119,792 |

Output bit-identical at every block size.

**Why synthetic data got this wrong.** Uniform random sequences give a flat
posting-list distribution with no sharing structure, so each `T[k]` read serves
almost no query genomes and the join's index reuse is worthless. Real genomes
share conserved tetramers heavily — 33% singleton lists with a long tail — so
within a query block many genomes hit the same `k` and `T[k]` is read **once per
block** instead of once per query genome. The reuse that the batching experiment
measured as worthless was real; the data could not exhibit it.

**Block size is an L2 question, as always here.** At 128 the accumulator is 1.5 MB;
at 512 it is 6 MB against 1.25 MB of L2 per core, and the cliff between them is
visible in the table.

> **DECISION: the k-mer join is the only kernel.** It is faster on real data
> *and* it reads both sides as inverted indexes, so a targets-only database is
> closed under search — one stored format, no forward index, no transpose, no
> privileged query database (§3.4).
>
> Outstanding: `join_threaded` allocates the `n_acc * kspace * 4` = 78 MB cursor
> array **per thread**, so 8 threads costs 624 MB against a 2 GiB budget. Threads
> must share one cursor array by splitting the k-mer space rather than the query
> range, or seed lazily.

### 4.1 MEASURED — tiling does not help, and sortedness is why

> **This section replaces an earlier design claim that was wrong.** The prototype
> in `fastaai-rs/` implements and measures every variant. Numbers below are from a
> synthetic 65,536-genome partition (i7-12700H, 48 KB L1d P-core / 32 KB E-core,
> 1.25 MB L2/core), 8 queries, 2.98×10⁹ posting increments. All kernels verified
> bit-identical.

| kernel | 1 thread | 20 threads | vs `u32` baseline |
|---|---|---|---|
| A untiled `u32`, sorted | 0.48 ns/inc | 0.09 ns/inc | 1.00× |
| **A16 untiled `u16`, sorted** | **0.35 ns/inc** | **0.05 ns/inc** | **1.36× / 1.69×** |
| SH untiled `u16`, *shuffled* | 0.72 ns/inc | 0.12 ns/inc | 0.67× / 0.75× |
| B tiled `u16` (tile 8192) | 0.58 ns/inc | 0.07 ns/inc | 0.83× / 1.27× |
| C tiled delta+varint | 1.12 ns/inc | 0.14 ns/inc | 0.43× / 0.68× |

**Tiling loses at every thread count.** It is never better than the plain untiled
`u16` sweep, and single-threaded it is worse than the `u32` baseline.

The reason is that the premise was false. Posting lists are stored **sorted by
local genome ID**, so `cnt[g] += 1` walks the accumulator *monotonically* — a
sequential sweep the hardware prefetcher handles, not a random scatter. Tiling was
a mechanism for bounding the footprint of an access pattern that is already
sequential; it adds per-tile loop re-entry and per-kmer cursor bookkeeping and buys
nothing.

**Sortedness is the optimization that tiling was reaching for, and it is worth
about 2×** — the shuffled-posting control (identical data, order destroyed within
each run) costs 2.05× single-threaded and 2.26× at 20 threads. Sortedness was
already required for delta coding and cursor sweeps; it turns out to be load-bearing
on its own.

**The partition cap makes tiling structurally unnecessary.** At the 65,536 cap a
`u32` accumulator is 262 KB — comfortably inside a 1.25 MB L2. Tiling could only
matter once the accumulator exceeds L2, i.e. beyond ~300,000 genomes in one flat
index, which §3.4 already forbids. Partitioning and tiling were redundant answers
to the same problem, and partitioning is the one that pays.

> **DECISION:** ship the untiled `u16` sweep over sorted posting lists. Drop
> tiling. Keep §3.4's partition cap and §3.6's sortedness requirement — both are
> confirmed, for different reasons than originally argued.

### 4.2 MEASURED — what the engine is actually worth

Against v1's own in-memory kernel (`fastaai.py:2787`), benchmarked **generously**:
SQLite removed entirely, index prebuilt in memory as `parse_accession` leaves it,
only the counting arithmetic timed.

| | ns/increment | increments/s |
|---|---|---|
| v1 numpy kernel, no SQLite | 2.30 | 435 M |
| Rust untiled `u16`, 1 thread | 0.35 | 2,839 M |
| Rust untiled `u16`, 20 threads | 0.05 | 18,318 M |

**6.5× per core on the counting arithmetic alone**, before any of v1's SQL,
allocation, or `vstack` overhead is counted, and before thread scaling. v1's
on-disk path additionally pays the per-SCP `TEMP TABLE` + `INNER JOIN` of §1.8,
which this comparison excludes entirely.

Thread scaling for the chosen kernel: 2,839 → 8,994 → 18,318 M inc/s at 1 / 4 / 20
threads (3.2× at 4, 6.5× at 20). Sublinearity at 20 threads is DRAM bandwidth, not
lock contention — the index is immutable and shared.

### 4.3 Tile width — obsolete

Retained only to record that the question is closed. With tiling dropped there is
no tile width to choose, and therefore no load-time CPU probe and no risk of
freezing an L1-specific constant into the format. The heterogeneous L1d on the
benchmark machine (48 KB P-core, 32 KB E-core) would have made that choice
genuinely awkward.

### 4.3 Accumulators and precision

- **Intersection counters: `u16`.** Bounded by protein length (a few thousand),
  nowhere near 65,535. This is what makes a tile fit L1.
- **Jaccard accumulator: `f64`.** Do *not* use `f32`. At the distances that matter
  Jaccard runs ~0.006–0.05, and summing ~120 such terms in `f32` puts rounding
  error uncomfortably close to reported precision. The accumulator is one vector
  per query — 512 KB per partition at `f64`. Free.
- **Standard deviation.** v1 materializes the full (nSCP × M) Jaccard matrix
  (`1948`, `2820`). Per partition this is (nSCP × 65,536), which stays in L2/L3;
  or use Welford streaming and never materialize it. Partitioning fixes this path
  as a side effect.

### 4.3.1 Off-the-shelf crates — surveyed, none applicable to the accumulator

Checked at the suggestion that a counting-accumulator crate might already exist.
It does not, and the survey explains why:

- **The accumulator is a direct array index.** At 0.35 ns/increment it is ~1 cycle
  on this CPU. There is no abstraction to buy. The `histogram` /
  [`b2histogram`](https://docs.rs/b2histogram) family are latency/quantile
  structures with bucketing logic — strictly more work than `cnt[g] += 1`.
- **Sorted arrays are already the right structure.** The Roaring authors' own
  benchmarks state that for set intersection *"the fastest data structure for this
  problem is the sorted array."* [`roaring`](https://docs.rs/roaring) would help if
  we needed pairwise set intersection, but the inverted-index formulation needs
  *counting across many lists*, which is not a Roaring-native operation.
- **The real prior art is in posting-list compression**, not accumulation:
  Stream VByte, Elias-Fano, Partitioned Elias-Fano, Binary Interpolative Coding
  (see [Techniques for Inverted Index Compression](https://arxiv.org/pdf/1908.10598)).
  That is the §3.6 open question, and the only place a crate is likely to win.
- [`superintervals`](https://crates.io/crates/superintervals) solves interval
  overlap, not multiset counting — not applicable.

### 4.3.2 MEASURED — accumulator variants, all negative

Do not re-litigate these. Best-of-3, three independent runs, 16,384 genomes; all
variants verified bit-identical to production first.

| variant | vs production | why it loses |
|---|---|---|
| `u16` narrow counters | 0.98× / 1.02× / 0.98× | **noise, not a win** — an earlier single run showed 1.06× and was wrong |
| 2 private accumulators | 0.86× | see below |
| 4 private accumulators | 0.85× | see below |
| software prefetch, dist 8 | 0.73× | access pattern is already sequential; prefetch instructions are pure overhead |
| software prefetch, dist 32 | 0.74× | as above |
| `f32` vectorised fold | 0.89× | cheaper divide, but adds a second pass over an `n`-wide scratch array |
| count-only, fold removed | 1.03–1.15× | **the fold is only ~10% of runtime** — that is the ceiling on optimising everything downstream of counting |

**Why private accumulators lose is the interesting part.** They are the technique
the vectorisation literature recommends for histograms, because scatter-increment
cannot be auto-vectorised when two lanes may target the same bin — and
[AVX2 has no conflict detection at all](https://www.intel.com/content/www/us/en/developer/articles/technical/improve-vectorization-performance-using-intel-advanced-vector-extensions-512.html).
Private per-lane histograms trade memory for conflict-freedom.

But **our posting lists are sorted**, so indices within a list are strictly
increasing and there are *zero* conflicts to eliminate. The technique has nothing
to fix here, and you pay its cost — 2–4× accumulator footprint plus extra fold
work — for nothing. Sortedness (§4.1) is again doing the work.

At ~0.53 ns/increment ≈ 1.8 cycles for a load-modify-store with a data-dependent
address, the loop is at the scalar hardware limit on this ISA.

**Two things untested, and not claimed either way:**

1. **AVX-512 `vpconflictd`** — the one instruction designed for this loop. Alder
   Lake client silicon has AVX-512 fuse-disabled, so it could not be measured
   here. The literature is pessimistic for pure-histogram loops (*"overhead is
   notorious where the only computation is histogram update"*), but it is
   untested on a Xeon or Zen 4/5.
2. **GPU.** Hardware scatter-add atomics plus an order of magnitude more memory
   bandwidth is a different architecture rather than a micro-optimisation, and
   the block structure of §4.6 maps onto it cleanly. This is the only remaining
   large multiplier identified; everything on this CPU is within ~10% of
   achievable.

### 4.4 Loop nest — superseded by §4.1

Retained for the record. With tiling dropped the nest is simply accession-outer,
posting-sweep-inner, and the "which nest?" question is closed. The original
analysis follows.

Original recommendation — accession outer, tile inner:

```
for partition p:
  for query genome q:
    for accession a in q:
      init ~500 cursors                       # ~4 KB, L1
      for tile t in p:
        cnt[B] = 0                            # u16, L1-resident
        for each kmer of q in a:
          advance cursor through tile, cnt[local_id - tile_base] += 1
        jsum[t] += cnt / (tgt_count[a] + |q_a| - cnt)    # f64, sequential access
        shared[t] += present[a]
    emit AAI for q over partition p
```

Cursor state stays at ~4 KB; `jsum` is 512 KB but accessed strictly sequentially
as tiles advance, so it streams through L2 rather than thrashing.

The alternative nest (tile outer, accession inner) needs cursors for every
accession live at once — ~500 KB — in exchange for a tile-width `jsum`. **Benchmark
both.** This is the one structural choice in the kernel that is not obvious from
first principles.

### 4.5 MEASURED — scheduling and thread scaling

**Throughput in the operational unit.** 4 partitions × 16,384 genomes (2.05 GB
resident postings + 380 MB offsets), 60 query genomes, 6.12×10⁹ increments:

| threads | pairs/s | pairs/s/thread | scaling |
|---|---|---|---|
| 1 | 1,086,260 | **1,086,260** | 0.99× |
| 2 | 2,125,076 | 1,062,538 | 1.94× |
| 4 | 3,585,757 | 896,439 | 3.27× |
| 6 | 4,626,323 | 771,054 | 4.22× |
| 8 | 5,028,259 | 628,532 | 4.59× |
| 12 | 5,755,662 | 479,638 | 5.25× |
| **16** | **6,754,284** | 422,143 | **6.17×** |
| 20 | 6,280,820 | 314,041 | 5.73× |

**~1.09M genome pairs/s/thread single-threaded.** That is ~1,600–2,000 posting
increments per pair.

**Two things the curve says.** Scaling is near-linear only to 2 threads, and
**20 threads is slower than 16** — past the 6 P-cores (12 SMT threads) the
remaining logical CPUs are Gracemont E-cores and SMT siblings contending for the
same memory controller. The kernel is DRAM-bound, so the thread count should be a
*configurable cap, defaulting well below logical-core count*, not `available_parallelism()`.
Spawning 20 workers here costs 7% versus 16.

**Scheduling: block-parallel wins; target-major is not needed.**

| schedule | pairs/s |
|---|---|
| A block-parallel — threads on *different* target partitions concurrently | 7,927,791 |
| B target-major — all threads on the *same* target partition | 6,954,494 |

I had expected B to win by sharing the target index in cache. It does not. At
512 MB per partition against a 24 MB L3 there is no reuse to preserve either way,
so B only adds a barrier per partition and the stragglers that come with it.

> **DECISION: one work-stolen thread per `(query block × target partition)`, no
> ordering constraint on which target a thread picks up.** This is the simpler
> design and it measures faster. Inner-loop parallelism (splitting a single block
> across threads) is needed only as a wrap-up mode when fewer blocks remain than
> threads — and, note, as the *primary* mode for small databases, where a 3-partition
> database yields only 6 symmetric blocks and cannot fill a thread pool at all.

**Projected all-vs-all wall clock** at 6.75M pairs/s (16 threads, this laptop),
counting `N(N−1)/2` pairs with symmetry exploited. The right-hand column corrects
for the SCP-length caveat in §0 (~330 tetramers/SCP rather than the seed set's 197):

| genomes | pairs | measured seed set | corrected (~330/SCP) |
|---|---|---|---|
| 20,000 | 2.0×10⁸ | 30 s | ~50 s |
| 65,536 | 2.1×10⁹ | 5.3 min | ~9 min |
| 100,000 | 5.0×10⁹ | 12 min | ~20 min |
| 1,000,000 | 5.0×10¹¹ | ~21 h | **~35 h** |

Both columns are for a 2-channel laptop. The kernel is DRAM-bound, so a server
with 8–12 memory channels should push the knee in the scaling table considerably
further right; these are floors, not ceilings.

### 4.5.2 What the SCP-length caveat changes — and what it does not

Increments per pair = (query kmers per SCP) × (mean posting-list length). Doubling
distinct kmers per SCP doubles total index entries *and* the count of distinct
`(accession, kmer)` slots, so mean list length is roughly unchanged and increments
per pair scale linearly. Measured ns/increment was flat across every configuration
tested, so **pairs/s scales as 1/(kmers per SCP)** — an absolute-throughput
correction only.

Every structural conclusion is invariant or strengthened:

| finding | effect of ~2× more kmers/SCP |
|---|---|
| tiling loses (§4.1) | **invariant** — accumulator width is set by genomes per partition, not kmers/SCP |
| `u16` beats `u32` | **strengthened** — larger index, more bandwidth pressure |
| sortedness positive | invariant |
| `BitPacker4x` crossover (§3.6) | **strengthened, and arrives earlier** — the crossover is driven by index size vs memory bandwidth, and the index roughly doubles |
| thread knee below core count (§4.5) | **strengthened, knee moves lower** — more bandwidth demand per thread |
| block-parallel beats target-major | invariant — no cache reuse at either size |

So the caveat costs accuracy in the headline rate and in nothing else. The one
place it would matter for design is if it pushed a *small* deployment across the
`BitPacker4x` crossover; §4.5.1's policy reads partition size and thread count, and
should read index bytes instead, which captures both effects.

### 4.5.1 Search size changes the right configuration

The measured crossovers do not agree on a single best build, so the engine should
select by workload rather than ship one setting:

| | small search | large search |
|---|---|---|
| payload | raw `u16` (`BitPacker4x` is 0.83× at 1 thread) | `BitPacker4x` (1.18× at 65k/20t) |
| partition index | sparse (94.9 MB dense offsets swamp a small partition, §3.5) | dense |
| parallelism | inner-loop within a block | block-parallel, work-stolen |
| thread cap | cores | well below logical cores |

Both payload encodings are already required to coexist behind the
per-`(partition, accession)` encoding tag of §3.5, so this costs no extra
machinery — only a policy that reads partition size and thread count.

### 4.6 Execution model: the unit of work is a block

A `(query partition × target partition)` comparison produces a rectangular R×C
block of the final matrix, and **that block is strictly localizable** — every term
the kernel needs lives inside the two partitions:

| term | source |
|---|---|
| `\|∩\|` | target inverted index |
| `\|Q_a\|` | query partition kmer counts |
| `\|T_a\|` | target partition kmer counts |
| shared SCP count | query accession set × target presence bitmap |
| the mean | over shared SCPs *for that pair only* — no global normalization |

Nothing reaches outside the pair. There is no global reduction anywhere in the
Jaccard→AAI path, so the global manifest is an **assembly-time artifact, not a
build-time dependency.** A worker needs exactly two partition files and the model
fingerprint.

What follows from that:

1. **Embarrassingly parallel with zero coordination.** Each block is an
   independent work unit writing its own output file. No shared output buffer, no
   locking, no contention — which converts the output bottleneck (§1.12) from a
   contention problem into a pure throughput problem, and every writer streams
   sequentially.
2. **Distribution is trivial.** Ship partition pairs to nodes; each returns a
   block. No shared index, no inter-node communication, no global state.
3. **Blocks are checkpoints.** A crashed or killed run recomputes only missing
   blocks. Long all-vs-all runs stop being all-or-nothing.
4. **Incremental growth is incremental work.** Adding partition `P_new` to an
   existing database requires only the new row and column of blocks —
   `P_new × all`, `all × P_new`, `P_new × P_new`. **Existing blocks are never
   recomputed.** For a database that grows over time this is the difference
   between hours and minutes, and v1 has no equivalent.
5. **Symmetry is a clean 2×.** AAI is exactly symmetric: Jaccard is symmetric, the
   shared-SCP set is symmetric, the mean and regression are deterministic
   functions of it, and best-hit filtering happens per genome at build time,
   before any comparison. For DB-vs-itself, compute only blocks `i ≤ j` with the
   diagonal blocks triangular, and mirror at assembly. (No symmetry to exploit
   when query and target are genuinely different databases.)

**Blocks are streamed, never materialized.** A full 65,536² block is 8.6 GB at
`int16`. The kernel already produces one query genome's complete result vector at
a time, so a block is appended row by row and never held whole.

**The one piece of global state that cannot be deferred is the model fingerprint**
(§3.2). Genome IDs can be reconciled after the fact; accession IDs cannot. A block
computed from partitions built against different model sets is garbage that looks
structurally valid. Check the fingerprint before computing a block, not before
assembling one.

---

## 5. Output

**The native output is the block set plus the manifest, not a dense matrix.**
Assembly into a single dense file is an *export* step, not a required final phase.
This follows from §4.6 — blocks are already the natural unit, and forcing a global
gather at the end would reintroduce exactly the serialization the block model
removes. Many downstream uses (clustering, nearest-neighbour search, thresholded
extraction) consume blocks directly and never need the dense form.

- Each block is a binary file: header (query partition, target partition, R, C,
  fingerprint) followed by row-major values. Sequential append during compute,
  memory-mappable for reading.
- Export to dense TSV or Arrow is a separate pass driven by the manifest ordinals.
  Use `itoa`/`ryu` into a large `BufWriter` — never Python string concatenation
  per cell (§1.12).
- **Store raw Jaccard at full precision** (§1.11), not the rounded 4-decimal value.
  AAI censoring at `<30%` / `>90%` becomes a display-layer decision applied at
  export, never a property of stored values. Storing the censored form discards
  precisely the signal the tool exists to provide.

---

## 6. Validation — DONE, and it found a bug on each side

Run: `fastaai2/tests/equivalence_v1.py <v1_results_dir> <archive_dir>`.
120 Firmicutes genomes spanning the collection, 14,400 pairs, v1 driven through
its own `aai_index` module.

### 6.1 Result

| | bug-compatible (21-symbol) | shipped (20-symbol) |
|---|---|---|
| pairs compared | 14,400 | 14,400 |
| **shared-SCP mismatches** | **0** | **0** |
| max \|ΔJaccard\| | **6.4×10⁻⁵** | 1.17×10⁻³ |
| mean \|ΔJaccard\| | **2.5×10⁻⁵** | 1.58×10⁻⁴ |
| within v1's 4-decimal output | **99.64%** | 21.94% |
| `(genome, accession)` k-mer counts exact | **9,430 / 9,433 (99.97%)** | — |

**The engine is validated.** Given identical inputs it reproduces FastAAI 1 to
within v1's own output precision, with identical SCP sets. All three residual
k-mer-count differences are accounted for by mechanism (§6.3), not absorbed into
a tolerance.

The right-hand column is the *measured* size of correcting v1's k-merizer bug:
**mean 1.58×10⁻⁴, max 1.17×10⁻³ Jaccard, SCP sets untouched.**

### 6.2 Bug found in FastAAI 2 — translation-table selection

`predict_proteins` chose whichever genetic code gave higher coding density.
Table 4 reassigns UGA from stop to tryptophan, so genes run through codons table
11 would terminate on and density is *almost always* marginally higher. Result:
**table 4 was chosen for 72.8% of 2,943 Firmicutes genomes** — the mycoplasma
code applied to ordinary Firmicutes.

v1 has hysteresis (`fastaai.py:803`): table 11 is the incumbent and an
alternative must beat it by **>10%** to win. Ported as
`predict.TABLE_SWITCH_MARGIN`, with `select_table` factored out so the rule is
unit-testable without running Prodigal (`tests/test_predict.py`).

The failure mode is what matters: gene calls changed (2,585 vs 2,674 proteins on
one genome, only 1,692 sequences shared), every SCP set downstream was wrong, and
**the output looked entirely plausible** — sensible AAI values, correct symmetry,
a believable distribution. Three full runs passed over it. Only comparison against
a reference implementation exposed it.

> **This invalidates the AAI distribution reported from the first full Firmicutes
> run** (median 44.6%, 55% of pairs below AAI 45%): those genes were called under
> the wrong genetic code. Engine figures — 5.41M pairs/s, symmetry, the join's
> 1.85× — are unaffected, as they do not depend on which proteins go in.

### 6.3 Bug found in FastAAI 1 — the k-merizer encodes any byte

v1 has **two** k-merizers:

- `unique_kmers` (`fastaai.py:1421`) resolves tetramers through a `kmer_index`
  lookup, which by construction admits only permissible symbols. This is the
  intended design. **`kmer_index` is never defined and the function is never
  called** — invoking it raises `NameError`.
- `unique_kmer_simple_key` (`fastaai.py:1139`, called at `1233`) is what runs: a
  numpy transform taking `ord()` of every character with no symbol check.

The numpy rewrite dropped the filtering the lookup table provided. Two consequences,
both confirmed against v1's own `genome_acc_kmer_counts`:

**`*` (stop).** Every protein ends in one, so every SCP carries one spurious
tetramer. Over 4,021 stored counts the 21-symbol encoding matches 4,019 exactly
while 20-symbol is short by exactly one on 4,017 — the window spanning the stop.

**`X` (ambiguous residue).** Rarer but worse. Emulating v1's encoding reproduced
its counts exactly on all three residual cases:

| accession | length | `X` count | ours | v1 | v1-emulated |
|---|---|---|---|---|---|
| PF05833.11 | 487 | 1 | 477 | 481 | **481** |
| PF01351.18 | 299 | 50 | 242 | 249 | **249** |
| PF02601.15 | 439 | 45 | 384 | 391 | **391** |

`X` comes from runs of `N` in an assembly. v1 emits `XXXX`, `LXXX` and so on as
real tetramers — and **two unrelated genomes with N-runs share `XXXX`, scoring
similarity from assembly gaps.** That is a false-positive mechanism, not just
noise.

FastAAI 2 treats both as out-of-alphabet: the k-mer window breaks rather than
aliasing onto a valid code (`kmer.rs::kmers`). So all three residual differences
are **v1 being wrong and v2 being right**, which is why they are one-directional.

> Any tool reading v1 databases needs to know this: stored k-mer sets include
> tetramers spanning stops and ambiguity codes.

### 6.4 What generalises

Both bugs were invisible in isolation and neither would have been found by more
benchmarking. Each produced plausible output. The harness is cheap to re-run
against an archive (§0.0) and should gate any change to preprocessing.

The residual is now fully attributed rather than tolerated — the distinction
that separates "validated" from "close enough".

## 7. Open questions

1. **§1.2 best-hit filter order** — which semantics is canonical for v2? Blocks
   the equivalence harness, since it determines expected divergence from v1.
2. ~~**§1.3** — confirm `assign_domain`'s removal was intentional.~~ **Resolved:**
   it encodes a real finding (100% domain accuracy from proportional recovery) and
   is relocated to the routing layer (§2.1), not deleted. Its SCP-pruning behavior
   is not restored.
3. ~~**§4.4 loop nest** — decide by benchmark.~~ **Resolved by measurement:**
   tiling dropped entirely (§4.1), so there is no nest to choose.
3b. ~~**§3.6 payload encoding.**~~ **Resolved by measurement:** `bitpacking`
   `BitPacker4x`, 1.18× faster and 1.36× smaller at production scale. Raw `u16`
   as the small-deployment fallback; both should be supported behind the
   per-`(partition, accession)` encoding tag that §3.5 already requires.
4. Should `k` and the alphabet actually be user-configurable, or fixed at 4/25 and
   merely *recorded* in metadata? Configurable is more general and costs the
   adaptive encoding of §3.5; recorded-only still needs the fingerprint.
5. ~~Partition sealing policy.~~ **Resolved:** IDs reset per partition, 65,536 is
   a cap not a size, seal at build boundary or cap. Short partitions are made
   affordable by sparse encoding (§3.5) and reclaimed by explicit compaction
   (§3.4).
6. Compaction trigger policy — manual only, or advisory ("this database has 340
   partitions averaging 900 genomes; run `compact`")? Never implicit during a
   build, but a query that touches 340 partition headers instead of 5 pays for it.
   Note compaction invalidates every block that touched the affected partitions
   (§4.6.4), so it trades stored-result reuse for query speed.
7. Does dense export ever become the default, or is the block set always the
   deliverable with export opt-in? Affects what "run FastAAI" prints by default
   and how much of v1's UX survives.

---

## 8. Sequencing

1. Settle §7.1 and §7.2 (decisions, not code)
2. Write the format spec: header, model table, genome manifest, partition layout,
   encodings, ID assignment, sealing and compaction semantics
3. Converter from v1 `.db` + equivalence harness — **before** the kernel
4. Kmerizer in Rust (dense encoding + bitset dedup), validated against
   `unique_kmer_simple_key`
5. Counting kernel for a single `(partition × partition)` block; benchmark `u32`
   untiled vs `u16` tiled vs `u16` tiled + delta
6. Block scheduler — symmetry, resume-from-existing-blocks, incremental
   row/column for new partitions
7. Wire through PyO3/maturin; delete `new_pyhmmer_manager`,
   `generate_accessions_index`, `find_hmm`, and `assign_domain` (relocated, §2.1)
8. Threaded preprocessing (drop the process pools; `cpus=N` in pyhmmer)
9. Routing layer + labeled bundled collection (§2.1) — independent of 4–6, can
   proceed in parallel
10. Export path (blocks + manifest → dense TSV/Arrow)
