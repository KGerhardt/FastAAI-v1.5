"""The public API — one function per step, and one that runs the whole thing.

The pipeline is a chain of ranks, each a directory of per-genome files (see
`layout.py`). These functions are that chain, one per arrow::

    genome ──genome_to_protein──► protein ──protein_to_hmm──► hits
                                            │
                     prot_hmm_to_crystal ◄──┘
                              │
                              ▼
                           crystal ──build_database──► database ──search──► AAI

Every step takes and returns **paths**, because the ranks are files on disk and
that is what makes them composable: the output of one step is the input to the
next with nothing held in between, and a step can be run for a thousand genomes
across a cluster and gathered afterwards. `preprocess` is the convenience form
that runs the whole chain from whichever rank you happen to have.

Steps are per-genome and single-threaded; `preprocess` does the parallelism.
That split is deliberate — a caller distributing work across nodes wants the
unit of work, not a thread pool it has to fight.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from . import crystal as _crystal
from . import layout
from .archive import HITS_COLUMNS, _name_line, _read_name_line
from .ingest import genome_name
from .predict import predict_proteins
from .search import DEFAULT_FILTER, FilterMode, Hit, ModelSet, resolve_hits, search_hits


def _models(models) -> ModelSet:
    """A ModelSet from a ModelSet, a path, a keyword, or None for the default."""
    return models if isinstance(models, ModelSet) else ModelSet(models)


def genome_to_protein(genome, out_dir, *, compress: bool = False,
                      name: str | None = None) -> Path:
    """Predict genes for one genome. Returns the protein FASTA written.

    The translation table is chosen per genome and recorded in the crystal
    later; v1's >10% hysteresis applies, without which table 4 wins on most
    genomes for no biological reason.
    """
    name = name or genome_name(genome)
    proteins, table = predict_proteins(genome)
    return layout.write_text(
        Path(out_dir) / f"{layout.safe(name)}{layout.FASTA_EXT}",
        "".join(f">{p}\n{s}\n" for p, s in proteins.items()),
        compress,
    )


def protein_to_hmm(protein, out_dir, models=None, *, compress: bool = False,
                   name: str | None = None, cpus: int = 1) -> Path:
    """HMM-search one protein file. Returns the hit table written.

    Every included hit is written, not only the winners: which hit survives
    depends on the best-hit filter, and storing the raw table means the filter
    can be changed later without searching again.
    """
    from .ingest import read_proteins_fasta

    name = name or genome_name(protein)
    ms = _models(models)
    hits = search_hits(read_proteins_fasta(protein), ms, cpus=cpus)
    return layout.write_text(
        Path(out_dir) / f"{layout.safe(name)}{layout.TABLE_EXT}",
        _name_line(name) + HITS_COLUMNS
        + "".join(f"{h.protein}\t{h.accession}\t{h.score:.4f}\n" for h in hits),
        compress,
    )


def read_hit_table(path) -> tuple[str | None, list[Hit]]:
    """Read a hit table, ours or HMMER's own.

    Ours is `protein / accession / score` under a `# genome=` line. HMMER's
    `--tblout` is whitespace-delimited with the target name first, the query
    accession third and the full-sequence score sixth. Accepting both means a
    caller who already ran hmmsearch does not have to reformat anything, which
    is the point of taking a protein/HMM pair at all.
    """
    genome, hits = None, []
    with layout.open_text(path) as fh:
        first = fh.readline()
        genome = _read_name_line(first)
        if genome is None and first.strip() and not first.startswith("#"):
            fh.seek(0)
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith("protein\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                prot, acc, score = parts
            else:
                # HMMER tblout: target, accession, query, accession, E, score, ...
                f = line.split()
                if len(f) < 6:
                    continue
                prot, acc, score = f[0], (f[3] if f[3] != "-" else f[2]), f[5]
            try:
                hits.append(Hit(prot, acc, float(score)))
            except ValueError:
                continue
    return genome, hits


def _pair_to_crystal(args):
    """One `(protein, hmm)` pair. Module level and taking plain values so a
    worker process can receive it."""
    import pyfastx

    protein_path, hmm_path, out_dir, fingerprint, filter_mode, compress = args
    named, hits = read_hit_table(hmm_path)
    name = named or genome_name(protein_path)
    assignment = resolve_hits(hits, filter_mode)
    if not assignment:
        return None
    wanted = set(assignment)

    scps, origins = {}, {}
    for prot, seq in pyfastx.Fastx(str(protein_path)):
        if prot in wanted:
            acc = assignment[prot]
            scps[acc] = seq
            origins[acc] = prot

    path = _crystal.write(out_dir, name, scps, fingerprint, filter_mode, None,
                          origins, compress)
    return None if path is None else os.fspath(path)


def prot_hmm_to_crystal(pairs: Iterable[tuple], out_dir, models=None, *,
                        filter_mode: FilterMode = DEFAULT_FILTER,
                        compress: bool = False, processes: int = 1) -> list[Path]:
    """Resolve `(protein_path, hmm_path)` pairs into crystals.

    The cheap step: everything expensive already happened. Read the hits, take
    the best-hit assignment, stream the protein FASTA once and keep only the
    sequences that won an accession — a genome costs its SCPs rather than its
    whole proteome.

    Returns the crystals written, skipping any genome that recovered nothing.

    Pairs are independent and each writes its own file, so this scales with
    `processes` — measured 5-6x at 8. Not with threads: pyfastx holds the GIL
    through the parsing that dominates, so threads come out *slower* than
    serial. Default 1, so a job never oversubscribes its allocation.
    """
    ms = _models(models)
    work = [(os.fspath(a), os.fspath(b), os.fspath(out_dir), ms.fingerprint,
             filter_mode, compress) for a, b in ((p[0], p[1]) for p in pairs)]
    if processes <= 1:
        out = [_pair_to_crystal(w) for w in work]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=processes) as pool:
            out = list(pool.map(_pair_to_crystal, work, chunksize=16))
    return [Path(p) for p in out if p is not None]


def build_database(crystals, models=None, *, k: int | None = None,
                   alphabet: str | None = None, only: set | None = None,
                   save_to=None):
    """Build a sealed database from crystals. Returns the `Database`.

    Crystals are the only route into an index — there is no merge and no
    incremental append, because both preserve whatever partitioning their inputs
    had. Combining collections means putting their crystals together and
    calling this.

    With *save_to*, the database is also written there.
    """
    from .pipeline import build_from_crystals

    db = build_from_crystals(crystals, _models(models), k, alphabet, only)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        db.save(str(save_to))
    return db


def preprocess(genomes=None, proteins=None, PH_tups=None, crystals=None,
               directory=layout.DEFAULT_ROOT, database="database", *,
               models=None, filter_mode: FilterMode = DEFAULT_FILTER,
               threads: int = 4, compress: bool = False, save: bool = True,
               progress=None):
    """Any input to a database in one call.

    Every argument is a rank, and they combine: genomes are predicted and
    searched, proteins are searched, protein/HMM pairs are resolved, and
    crystals are taken as they are. Everything converges on *directory*'s
    crystal set, which is what gets built.

        db = fastaai.preprocess(genomes="genomes/", database="firm")

    *directory* is the output root, `FastAAI/` by default, and it is filled in
    exactly as the CLI fills it: `proteins/`, `hmm_hits/`, `crystals/` and
    `database/`. Nothing is written outside it and nothing is discarded.

    *database* is a name placed in `<directory>/database/`; a path with a
    separator is taken literally. Pass `save=False` to build without writing it.
    """
    from .pipeline import preprocess_paths

    site = layout.Layout(directory, compress)
    ms = _models(models)

    genomes = _as_list(genomes)
    proteins = _as_list(proteins)
    pairs = list(PH_tups or [])

    if genomes or proteins:
        # Prediction and search are the expensive ranks and the ones worth
        # parallelising; they write proteins/, hmm_hits/ and crystals/ as they go.
        for kind, paths in (("genome", genomes), ("protein", proteins)):
            if not paths:
                continue
            site.sub(layout.PROTEINS, create=True)
            site.sub(layout.HMM_HITS, create=True)
            site.sub(layout.CRYSTALS, create=True)
            preprocess_paths(
                paths, ms, mode=filter_mode, threads=threads, progress=progress,
                archive_root=site.root, input_kind=kind,
                crystal_root=site.crystals, compress=compress,
            )

    if pairs:
        prot_hmm_to_crystal(pairs, site.sub(layout.CRYSTALS, create=True), ms,
                            filter_mode=filter_mode, compress=compress)

    sources = [site.crystals] if _crystal.crystal_paths(site.crystals) else []
    for extra in _as_list(crystals):
        # Crystals given by the caller are read where they are rather than
        # copied — they are already the durable artifact.
        if os.fspath(extra) not in {os.fspath(s) for s in sources}:
            sources.append(Path(extra))
    if not sources:
        raise ValueError(
            "no input: pass genomes, proteins, PH_tups or crystals"
        )

    paths = [str(p) for src in sources for p in _crystal.crystal_paths(src)]
    if not paths:
        raise ValueError("no crystals were produced or found")

    from .pipeline import build_from_crystals_paths

    db = build_from_crystals_paths(paths, ms)
    db.filter_mode = filter_mode
    if save:
        dest = site.database_path(database)
        dest.parent.mkdir(parents=True, exist_ok=True)
        db.save(str(dest))
    return db


def _as_list(value) -> list:
    """One path, a directory of them, or an iterable — all to a list of paths."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        p = Path(value)
        if p.is_dir():
            from .ingest import find_genomes

            found = find_genomes(p)
            return list(found) if found else []
        return [p]
    return [Path(v) for v in value]
