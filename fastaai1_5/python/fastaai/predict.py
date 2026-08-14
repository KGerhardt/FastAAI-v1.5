"""Gene prediction via pyrodigal.

pyrodigal wraps Prodigal's C and releases the GIL, so this is threaded rather
than forked. FastAAI 1 used a multiprocessing pool here, paying fork and pickle
costs for a library that never needed them.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import pyrodigal

from .ingest import read_fasta

#: Candidate genetic codes, **in priority order**. Table 11 (bacterial) is tried
#: first and is the default winner; table 4 (mycoplasma/spiroplasma) must clear
#: `TABLE_SWITCH_MARGIN` to displace it. Order is load-bearing — see below.
DEFAULT_TABLES: tuple[int, ...] = (11, 4)

#: An alternative table must beat the incumbent's coding density by this factor
#: to win, reproducing FastAAI 1 (`fastaai.py:803`).
#:
#: **This margin is not a tuning knob, it is a correctness requirement.** Table 4
#: reassigns UGA from stop to tryptophan, so genes run through codons that table
#: 11 would terminate on and coding density is *almost always* marginally higher.
#: Selecting on raw density therefore picks table 4 nearly every time: without
#: this margin, 72.8% of 2,943 Firmicutes genomes were called under the
#: mycoplasma code, which changed gene calls (2,585 vs 2,674 proteins on one
#: genome, only 1,692 sequences shared) and silently corrupted every downstream
#: SCP set. The output still looked entirely plausible.
TABLE_SWITCH_MARGIN = 1.1

#: Prodigal's own inter-sequence breaker, inserted so training does not run genes
#: across contig boundaries.
BREAKER = "TTAATTAATTAA"

#: Below this much training sequence, Prodigal's single-genome training is
#: unreliable and anonymous (metagenomic) mode is the honest fallback.
META_THRESHOLD = 20_000


def build_training_sequence(contigs: Sequence[tuple[str, str]]) -> str:
    """Concatenate contigs the way Prodigal does for training.

    A breaker goes *between* contigs and also after the last one when there is
    more than one, matching `fastaai.py:752`.
    """
    if not contigs:
        return ""
    if len(contigs) == 1:
        return contigs[0][1]
    return BREAKER.join(s for _, s in contigs) + BREAKER


def select_table(
    densities: Iterable[tuple[int, float]],
    margin: float = TABLE_SWITCH_MARGIN,
) -> int | None:
    """Pick a translation table from `(table, coding_density)` in priority order.

    The first entry is the incumbent; a later one wins only by beating it by
    *margin*. Factored out so the rule can be tested without running Prodigal.
    """
    winner: tuple[int, float] | None = None
    for table, density in densities:
        if winner is None or density > winner[1] * margin:
            winner = (table, density)
    return winner[0] if winner else None


def predict_proteins(
    path: os.PathLike | str,
    tables: Sequence[int] = DEFAULT_TABLES,
) -> tuple[dict[str, str], int | None]:
    """Predict proteins for one genome.

    Returns `(proteins, translation_table)`, mapping gene id to amino-acid
    sequence. *translation_table* is None when metagenomic mode was used.
    """
    contigs = list(read_fasta(path))
    if not contigs:
        return {}, None
    total_bp = sum(len(s) for _, s in contigs)
    training = build_training_sequence(contigs)

    if len(training) < META_THRESHOLD:
        finder = pyrodigal.GeneFinder(meta=True)
        proteins = {}
        for name, seq in contigs:
            for i, gene in enumerate(finder.find_genes(seq), 1):
                proteins[f"{name}_{i}"] = gene.translate()
        return proteins, None

    # Evaluate each candidate table, keeping results in priority order.
    evaluated: list[tuple[int, float, list]] = []
    for table in tables:
        finder = pyrodigal.GeneFinder(meta=False)
        try:
            finder.train(training, translation_table=table)
        except Exception:
            continue
        genes_by_contig = [(name, finder.find_genes(seq)) for name, seq in contigs]
        coding = sum(
            len(g.sequence()) for _, genes in genes_by_contig for g in genes
        )
        evaluated.append((table, coding / total_bp if total_bp else 0.0, genes_by_contig))

    if not evaluated:
        raise RuntimeError(f"gene prediction failed for {path}")

    chosen = select_table((t, d) for t, d, _ in evaluated)
    genes_by_contig = next(g for t, _, g in evaluated if t == chosen)

    proteins = {}
    for name, genes in genes_by_contig:
        for i, gene in enumerate(genes, 1):
            proteins[f"{name}_{i}"] = gene.translate()
    return proteins, chosen
