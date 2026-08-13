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
    #: accession -> the gene call that won it. Provenance for crystals: which
    #: predicted protein a stored SCP sequence came from. Small — one short
    #: string per SCP — so it survives `release()`.
    scp_proteins: dict[str, str] | None = None

    def release(self) -> None:
        """Drop the bulky fields once archived — proteins are ~900 KB/genome."""
        self.proteins = None
        self.hits = None


def preprocess_one(
    path: os.PathLike | str,
    models: ModelSet,
    mode: FilterMode = DEFAULT_FILTER,
    input_kind: str = "auto",
    crystal_root=None,
    compress: bool = False,
) -> GenomeRecord:
    """Predict (if needed) and HMM-search one input.

    *input_kind* is `"genome"`, `"protein"`, or `"auto"` to guess from the
    extension. Protein input skips Prodigal entirely — which is the whole
    reference-build path, since GTDB ships predicted proteins and re-predicting
    600k genomes would be ~4 s each of pure waste. Prodigal remains mandatory
    for query genomes, which arrive as nucleotides.

    With *crystal_root* the worker writes its own crystal, which is the point at
    which this genome stops needing to be held: formatting (and compression, if
    asked for) happens on the worker thread instead of the collector, and the
    caller can drop the sequences immediately rather than carrying every
    genome's SCPs until the build.
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
        origins = {acc: prot for prot, acc in assignment.items()}
        if crystal_root is not None:
            from . import crystal
            crystal.write(crystal_root, name, scps, models.fingerprint, mode,
                          table, origins, compress)
        return GenomeRecord(name, scps, table, proteins=proteins, hits=hits,
                            scp_proteins=origins)
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
    crystal_root=None,
    compress: bool = False,
) -> list[GenomeRecord]:
    """Predict and HMM-search every genome. Order of *paths* is preserved.

    Threaded, not forked: pyrodigal and pyhmmer both release the GIL.

    With *archive_root*, proteins and raw hits are written as each genome
    finishes and then released, so peak memory stays flat and the run never has
    to be repeated.

    **With *crystal_root*, each worker writes its own crystal and the returned
    records come back with `scps` cleared.** That is the point of it: the SCPs
    are on disk, so nothing needs to hold every genome's sequences until the
    build, and peak memory stops depending on how many genomes there are. Build
    from the crystal directory (`build_from_crystals`) rather than from these
    records — `build_database` would find nothing in them.
    """
    paths = list(paths)
    out: list[GenomeRecord | None] = [None] * len(paths)
    archive = (Archive(archive_root, models.accessions, models.fingerprint,
                       compress)
               if archive_root else None)
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {
            pool.submit(preprocess_one, p, models, mode, input_kind,
                        crystal_root, compress): i
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
            if crystal_root is not None:
                # Written by the worker and now on disk, so the SCPs need not be
                # carried for the rest of the run. This is what keeps peak
                # memory independent of collection size.
                rec.scps = {}
                rec.scp_proteins = None
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
    from .archive import (genome_names, read_fingerprint, read_hits, read_models,
                          read_proteins)

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
    # Carried through so a rebuild is as verifiable as the original build. An
    # archive written before fingerprints existed yields "", meaning unknown.
    db.models = read_fingerprint(root)
    return db


def build_from_crystals(
    source,
    models: ModelSet,
    k: int | None = None,
    alphabet: str | None = None,
) -> "_core.Database":
    """Build a sealed database from crystals — no prediction, no HMM search.

    **Accession order comes from *models*, never from the crystals.** Accession
    IDs are positions in a list, so deriving that list from whichever accessions
    happen to appear would make the schema depend on which genomes were included:
    two subsets of one collection would number their shared markers differently
    and refuse to be compared. Taking the order from the model set means any
    subset builds a database comparable with any other.

    The crystals' recorded fingerprint must match *models* for the same reason
    the engine checks it anywhere else — mismatched marker sets produce
    well-formed, meaningless AAI.
    """
    from . import crystal

    acc_index = {a: i for i, a in enumerate(models.accessions)}
    db = _core.Database(
        models.accessions,
        k if k is not None else _core.DEFAULT_K,
        alphabet if alphabet is not None else _core.DEFAULT_ALPHABET,
    )

    prov = None
    checked = False
    n = 0
    # Streamed, so peak memory is one crystal file rather than the collection.
    for genome, scps, prov in crystal.iter_genomes(source):
        if not checked:
            # The fingerprint is uniform across a run — `iter_genomes` raises on
            # any disagreement — so it only needs testing against the model set
            # once, on the first genome that carries one.
            stored = prov.as_dict()["models"]
            if stored and stored != models.fingerprint:
                raise ValueError(
                    "these crystals were not built with this model set\n"
                    f"  crystals:  {stored}\n"
                    f"  model set: {models.fingerprint}\n"
                    "Pass --hmm naming the models the crystals were made with."
                )
            checked = bool(stored)
        unknown = set(scps) - set(acc_index)
        if unknown:
            raise ValueError(
                f"crystals reference {len(unknown)} accession(s) absent from the "
                f"model set, e.g. {sorted(unknown)[:3]}"
            )
        payload = [(acc_index[a], seq.encode()) for a, seq in sorted(scps.items())]
        if payload:
            db.add_genome(genome, payload)
            n += 1

    if not n:
        raise RuntimeError(f"no genome in {source} yielded a usable SCP set")
    db.seal()
    fields = prov.as_dict() if prov is not None else {}
    db.models = fields.get("models") or models.fingerprint
    if fields.get("filter"):
        db.filter_mode = fields["filter"]
    return db


def crystallize_archive(root, out, models: ModelSet,
                        mode: FilterMode = DEFAULT_FILTER,
                        compress: bool = False) -> int:
    """Emit crystals from an existing archive, without re-running anything.

    The archive already holds proteins and raw hits, so this is a resolve and a
    write — the point being that a collection preprocessed before crystals
    existed does not have to be preprocessed again.
    """
    from . import crystal
    from .archive import read_fingerprint, read_hits, read_proteins

    fingerprint = read_fingerprint(root) or models.fingerprint
    n = 0
    for genome, hits in read_hits(root):
        assignment = resolve_hits(hits, mode)
        if not assignment:
            continue
        proteins = read_proteins(root, genome)
        scps = {acc: proteins[prot] for prot, acc in assignment.items()
                if prot in proteins}
        origins = {acc: prot for prot, acc in assignment.items()
                   if prot in proteins}
        if crystal.write(out, genome, scps, fingerprint, mode, None, origins,
                         compress):
            n += 1
    return n


def build_database(
    records: Iterable[GenomeRecord],
    models: ModelSet,
    filter_mode: FilterMode = DEFAULT_FILTER,
) -> tuple["_core.Database", list[GenomeRecord]]:
    """K-merise and seal. Genomes with no SCPs are excluded and returned separately."""
    db = _core.Database(models.accessions)
    # Records which models these k-mers came from, so a later comparison can
    # verify it rather than trust matching accession names.
    db.models = models.fingerprint
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
        raise RuntimeError(
            "no genome yielded a usable SCP set. If these records came from "
            "`preprocess(crystal_root=...)` their SCPs are on disk by design — "
            "build from the crystal directory with `build_from_crystals`."
        )
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
