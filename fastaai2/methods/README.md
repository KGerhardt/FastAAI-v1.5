# Methods

Reproducible artifacts behind claims made in the top-level README.

## `concordance_v1_v15.{tsv,png}`

Does FastAAI 1.5 change the answers? 120 Firmicutes genomes, all-vs-all against
FastAAI 1 driven through its own `aai_index` module, 14,400 pairs.

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

## Reproducing

The v1 side needs FastAAI 1 and its bundled SCP HMMs. The v1.5 side rebuilds
from an archive of proteins and raw HMM hits, so re-filtering or re-scoring
costs seconds rather than repeating a half-hour preprocess:

    fastaai build <genomes> --hmm <models.hmm> --archive arch/ -d db/
    python ../tests/equivalence_v1.py <v1_results_dir> arch/
