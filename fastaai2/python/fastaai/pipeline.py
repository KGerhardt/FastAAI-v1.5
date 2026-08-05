"""FASTA to AAI, end to end."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from . import _core
from .ingest import find_genomes, genome_name, looks_like_protein, read_proteins_fasta
from .predict import predict_proteins
from .archive import Archive
from .search import (DEFAULT_FILTER, FilterMode, ModelSet, resolve_hits,
                     search_hits)

#: Scaling is memory-bound and measured *negative* past ~16 threads on a 6P+8E
#: laptop, so never default to every logical core.
DEFAULT_SEARCH_THREADS = 8


@dataclass
class GenomeRecord:
    name: str
    scps: dict[str, str]
    translation_table: int | None
    error: str | None = None
    #: Full preprocessing output, retained only long enough to archive it.
    proteins: dict[str, str] | None = None
    hits: list | None = None

    def release(self) -> None:
        """Drop the bulky fields once archived — proteins are ~900 KB/genome."""
        self.proteins = None
        self.hits = None


def preprocess_one(
    path: os.PathLike | str,
    models: ModelSet,
    mode: FilterMode = DEFAULT_FILTER,
    input_kind: str = "auto",
) -> GenomeRecord:
    """Predict (if needed) and HMM-search one input.

    *input_kind* is `"genome"`, `"protein"`, or `"auto"` to guess from the
    extension. Protein input skips Prodigal entirely — which is the whole
    reference-build path, since GTDB ships predicted proteins and re-predicting
    600k genomes would be ~4 s each of pure waste. Prodigal remains mandatory
    for query genomes, which arrive as nucleotides.
    """
    name = genome_name(path)
    if input_kind == "auto":
        input_kind = "protein" if looks_like_protein(path) else "genome"
    try:
        if input_kind == "protein":
            proteins, table = read_proteins_fasta(path), None
        else:
            proteins, table = predict_proteins(path)
        hits = search_hits(proteins, models, cpus=1)
        assignment = resolve_hits(hits, mode)
        scps = {acc: proteins[prot] for prot, acc in assignment.items()}
        return GenomeRecord(name, scps, table, proteins=proteins, hits=hits)
    except Exception as exc:  # a bad genome must not abort the run
        return GenomeRecord(name, {}, None, error=f"{type(exc).__name__}: {exc}")


def preprocess(
    paths: Iterable[os.PathLike | str],
    models: ModelSet,
    mode: FilterMode = DEFAULT_FILTER,
    threads: int = 4,
    progress: Callable[[int, int, GenomeRecord], None] | None = None,
    archive_root=None,
    input_kind: str = "auto",
) -> list[GenomeRecord]:
    """Predict and HMM-search every genome. Order of *paths* is preserved.

    Threaded, not forked: pyrodigal and pyhmmer both release the GIL.

    With *archive_root*, proteins and raw hits are written as each genome
    finishes and then released, so peak memory stays flat and the run never has
    to be repeated.
    """
    paths = list(paths)
    out: list[GenomeRecord | None] = [None] * len(paths)
    archive = Archive(archive_root, models.accessions) if archive_root else None
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {
            pool.submit(preprocess_one, p, models, mode, input_kind): i
            for i, p in enumerate(paths)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            rec = fut.result()
            if archive is not None and rec.proteins is not None:
                archive.add(rec.name, rec.proteins, rec.hits or [],
                            rec.translation_table)
            rec.release()
            out[i] = rec
            done += 1
            if progress:
                progress(done, len(paths), rec)
    if archive is not None:
        archive.close()
    return [r for r in out if r is not None]


def build_from_archive(
    root,
    mode: FilterMode = DEFAULT_FILTER,
    only: set | None = None,
    k: int | None = None,
    alphabet: str | None = None,
) -> "_core.Database":
    """Rebuild a sealed database from an archive, with no prediction or search.

    Re-resolving under a different *mode* costs nothing, because the raw hits were
    stored rather than only the surviving SCPs.

    *only* restricts to a set of genome names. *k* and *alphabet* override the
    defaults — needed to reproduce FastAAI 1 exactly, which included the stop
    codon `*` in its 21-symbol alphabet (see equivalence harness).
    """
    from .archive import genome_names, read_hits, read_models, read_proteins

    accessions = read_models(root)
    acc_index = {a: i for i, a in enumerate(accessions)}
    order = {g: i for i, g in enumerate(genome_names(root))}

    resolved: dict[str, list[tuple[int, bytes]]] = {}
    for genome, hits in read_hits(root):
        if only is not None and genome not in only:
            continue
        assignment = resolve_hits(hits, mode)
        if not assignment:
            continue
        proteins = read_proteins(root, genome)
        payload = [
            (acc_index[acc], proteins[prot].encode())
            for prot, acc in assignment.items()
            if acc in acc_index and prot in proteins
        ]
        if payload:
            resolved[genome] = payload

    db = _core.Database(
        accessions,
        k if k is not None else _core.DEFAULT_K,
        alphabet if alphabet is not None else _core.DEFAULT_ALPHABET,
    )
    for genome in sorted(resolved, key=lambda g: order.get(g, 1 << 30)):
        db.add_genome(genome, resolved[genome])
    db.seal()
    return db


def build_database(
    records: Iterable[GenomeRecord],
    models: ModelSet,
    filter_mode: FilterMode = DEFAULT_FILTER,
) -> tuple["_core.Database", list[GenomeRecord]]:
    """K-merise and seal. Genomes with no SCPs are excluded and returned separately."""
    db = _core.Database(models.accessions)
    kept: list[GenomeRecord] = []
    skipped: list[GenomeRecord] = []
    for rec in records:
        if not rec.scps:
            skipped.append(rec)
            continue
        payload = [
            (models.acc_index[acc], seq.encode())
            for acc, seq in rec.scps.items()
            if acc in models.acc_index
        ]
        if not payload:
            skipped.append(rec)
            continue
        db.add_genome(rec.name, payload)
        kept.append(rec)
    if not kept:
        raise RuntimeError("no genome yielded a usable SCP set")
    db.seal()
    return db, skipped


@dataclass
class SearchResult:
    query_names: list[str]
    target_names: list[str]
    jaccard: np.ndarray  # (nq, nt) float64, NaN where nothing is shared
    shared: np.ndarray  # (nq, nt) uint32
    #: Standard deviation of Jaccard across shared accessions, or None when not
    #: requested. Spread matters independently of the mean: a pair at AAI 65%
    #: with tight agreement across markers is a different claim from the same
    #: mean carried by two markers at 0.9 and the rest near 0.02.
    stdev: np.ndarray | None = None

    @property
    def aai(self) -> np.ndarray:
        """AAI percentages. Uncensored — the fit extrapolates past 100 for
        near-identical genomes, and below 30% it is unreliable rather than wrong."""
        out = np.full(self.jaccard.shape, np.nan)
        ok = np.isfinite(self.jaccard) & (self.jaccard > 0)
        j = self.jaccard[ok]
        x = np.power(-0.2607023 * np.log(j), 1.0 / 3.435)
        out[ok] = (1.810741 * np.exp(-x) - 0.3087057) * 100.0
        return out


#: Query-blocking width. The accumulator is `block * n_target`; 128 keeps it in
#: L2 on typical hardware, and the measured cliff between 256 and 512 is that
#: accumulator leaving cache.
DEFAULT_BLOCK = 128


def search(
    query: "_core.Database",
    target: "_core.Database",
    threads: int = DEFAULT_SEARCH_THREADS,
    block: int = DEFAULT_BLOCK,
    stdev: bool = False,
) -> SearchResult:
    """Search via the k-mer join. Passing the same database twice takes the
    symmetric upper-triangle path automatically.

    *stdev* adds the spread of Jaccard across shared accessions. It costs one
    more output-width array — another 69 MB at 2,943 genomes — so it is off by
    default rather than always paid for.
    """
    jb, sb, nq, nt, qb = query.search(target, block, threads, stdev)
    jac = np.frombuffer(jb, dtype=np.float64).reshape(nq, nt)
    sh = np.frombuffer(sb, dtype=np.uint32).reshape(nq, nt)
    sd = np.frombuffer(qb, dtype=np.float64).reshape(nq, nt) if qb is not None else None
    return SearchResult(query.genome_names, target.genome_names, jac, sh, sd)
