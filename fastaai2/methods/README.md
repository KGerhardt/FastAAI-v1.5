# Methods

Reproducible artifacts behind claims made in the top-level README.

## `concordance_v1_v15.{tsv,png}`

Does FastAAI 1.5 change the answers? 120 Firmicutes genomes, all-vs-all against
FastAAI 1 driven through its own `aai_index` module, 14,400 pairs.

![v1 vs v1.5 concordance](concordance_v1_v15.png)

    python plot_concordance.py [concordance_v1_v15.tsv] [out.png]

| | |
|---|---|
| pairs compared (off-diagonal) | 14,280 |
| **shared-SCP differences** | **0** |
| median \|Δ AAI\| | 0.0115 percentage points |
| max \|Δ AAI\| | 0.0627 percentage points |

**Two panels, because one would mislead.** Panel A puts v1 against v1.5 on the
identity line: every point sits on it, which is the finding, but at this
agreement a reader cannot tell 1e-4 from 1e-2 by eye on a 40–100% axis. Panel B
plots the residual at a scale where it is legible. Publishing A alone would be
the more flattering and less honest figure.

**The residual is v1.5 correcting v1, which is why it is one-directional.**
FastAAI 1 encodes the stop codon `*` and ambiguous residues `X` as if they were
amino acids. Its filtering k-merizer — `unique_kmers`, `fastaai.py:1421` —
resolves tetramers through a `kmer_index` lookup that would admit only
permissible symbols, but that table is never built and the function is never
called. The numpy transform that runs instead (`unique_kmer_simple_key`,
`fastaai.py:1139`) takes `ord()` of every character. Confirmed against v1's own
`genome_acc_kmer_counts`: 4,019 of 4,021 stored counts match a 21-symbol
encoding exactly.

`X` is the more damaging case. It arises from runs of `N` in an assembly, so two
unrelated genomes with sequencing gaps share `XXXX` and accrue similarity from
them — a false-positive mechanism that scales with assembly fragmentation, not a
rounding difference.

Self-comparisons are excluded: Jaccard is 1.0 in both versions, and the AAI
regression is unbounded above, so they land near 150% and compress the real
40–70% band into a corner.

## `timings_{preprocessing.tsv,search.txt}`

Measured, not extrapolated. Two earlier estimates of the search speedup — 3x
from a model of v1's kernel, 74x from a 120-genome run — both missed; the
number below comes from building a real FastAAI 1 database at 2,943 genomes
and querying it against itself.

    python timings.py <genome_dir> <models.hmm> [n_genomes] [threads]

**Preprocessing**, per genome, one thread each (the pipeline parallelises
across genomes, so `cpus=1` is the figure that composes). 24 Firmicutes:

| stage | median s/genome | share |
|---|---|---|
| predict (pyrodigal) | 1.79 | 64% |
| hmmsearch (pyhmmer) | 1.03 | 36% |

Unchanged in 1.5 beyond the move to in-process libraries — it is the same
Prodigal and the same HMMER, and it still dominates a cold run.

**Search**, 8 threads, wall clock less the ~0.22 s interpreter start both
versions pay:

| scale | pairs | v1 | v1.5 | v1.5 pairs/s | speedup |
|---|---|---|---|---|---|
| 120 genomes | 14,400 | 1.15 s | 0.016 s | 922,339 | 74x |
| **2,943 genomes** | **8,661,249** | **21.84 s** | **1.587 s** | **5,457,298** | **14x** |

**14x is the honest headline.** The two rows are not in tension: at 120
genomes the pairwise work is trivial and v1's per-query `TEMP TABLE` +
`INNER JOIN` overhead is most of the runtime, so 74x measures fixed cost.
At 2,943 genomes the work dominates.

v1 was run through `db_query --in_memory --store_results`, its fastest path.
At 2,943 genomes it peaked at 1.01 GB RSS against a 116 MB v1.5 index, and
its database occupies 508 MB on disk to v1.5's 116 MB.

## Reproducing

The v1 side needs FastAAI 1 and its bundled SCP HMMs. The v1.5 side rebuilds
from an archive of proteins and raw HMM hits, so re-filtering or re-scoring
costs seconds rather than repeating a half-hour preprocess:

    fastaai build <genomes> --hmm <models.hmm> --archive arch/ -d db/
    python ../tests/equivalence_v1.py <v1_results_dir> arch/
