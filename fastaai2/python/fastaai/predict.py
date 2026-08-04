"""Gene prediction via pyrodigal.

pyrodigal wraps Prodigal's C and releases the GIL, so this is threaded rather
than forked. FastAAI 1 used a multiprocessing pool here, paying fork and pickle
costs for a library that never needed them.
"""

from __future__ import annotations

import os
from typing import Iterable

import pyrodigal

from .ingest import read_fasta

#: FastAAI 1 tried table 11 then 4 and kept whichever coded more densely.
DEFAULT_TABLES = (11, 4)

#: Below this, Prodigal's own docs say single-genome training is unreliable.
META_THRESHOLD = 20_000


def _coding_density(genes, total_bp: int) -> float:
    if total_bp == 0:
        return 0.0
    coded = sum(abs(g.end - g.begin) + 1 for g in genes)
    return coded / total_bp


def predict_proteins(
    path: os.PathLike | str,
    tables: Iterable[int] = DEFAULT_TABLES,
) -> tuple[dict[str, str], int | None]:
    """Predict proteins for one genome.

    Returns ``(proteins, translation_table)`` where *proteins* maps gene id to
    amino-acid sequence. *translation_table* is None when metagenomic mode was
    used, which happens for inputs too short to train on.
    """
    contigs = list(read_fasta(path))
    if not contigs:
        return {}, None
    total_bp = sum(len(s) for _, s in contigs)

    if total_bp < META_THRESHOLD:
        # Too little sequence to train; anonymous mode is the honest fallback.
        finder = pyrodigal.GeneFinder(meta=True)
        proteins = {}
        for name, seq in contigs:
            for i, gene in enumerate(finder.find_genes(seq), 1):
                proteins[f"{name}_{i}"] = gene.translate()
        return proteins, None

    # Train once per candidate table and keep the denser coding solution.
    joined = "TTAATTAATTAA".join(seq for _, seq in contigs)
    best = None
    for table in tables:
        finder = pyrodigal.GeneFinder(meta=False)
        try:
            finder.train(joined, translation_table=table)
        except Exception:
            continue
        genes_by_contig = [(name, finder.find_genes(seq)) for name, seq in contigs]
        density = _coding_density(
            [g for _, genes in genes_by_contig for g in genes], total_bp
        )
        if best is None or density > best[0]:
            best = (density, table, genes_by_contig)

    if best is None:
        raise RuntimeError(f"gene prediction failed for {path}")

    _, table, genes_by_contig = best
    proteins = {}
    for name, genes in genes_by_contig:
        for i, gene in enumerate(genes, 1):
            proteins[f"{name}_{i}"] = gene.translate()
    return proteins, table
