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

Each step comes in two forms: a **unit** that does one genome, and a **driver**
that runs the unit over many in parallel — `genome_to_protein` and
`genomes_to_proteins`, `protein_to_hmm` and `proteins_to_hmms`. The unit is
single-threaded and returns a path; the driver owns the parallelism.

`all_steps` is the whole chain for one genome — predict, search, resolve, write
— **with nothing returned between stages**. That is the shape the parallelism
wants: one worker owns one genome from FASTA to crystal, so no intermediate
result crosses a process boundary and there is no funnel to coordinate through.
`preprocess` is its driver.

**Processes, not threads.** The work is already thousands of independent
per-file units, so a shared interpreter buys nothing and costs a serial
fraction. `multiprocessing.Pool.imap_unordered` throughout: results as they
finish, which is what progress reporting wants, with order restored by index
where a function promises it.
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


def _indexed(args):
    """Run one unit, carrying its position so order survives `imap_unordered`.

    The worker is passed by reference rather than closed over — module-level
    functions pickle as a name, closures do not pickle at all.
    """
    i, fn, payload = args
    return i, fn(payload)


def _run(fn, work: list, processes: int, chunksize: int = 1,
         initializer=None, initargs: tuple = ()) -> list:
    """Map *fn* over *work*, in this process or a pool, preserving order.

    `imap_unordered` rather than `map`: results come back as they finish, so a
    long genome cannot hold up the reporting of the hundred that finished behind
    it, and memory does not accumulate a completed-but-unyielded backlog.

    *initializer* runs once per worker. That is where anything expensive and
    reusable is built — the model set above all, which is 9.2 MB of HMM text to
    parse. Built there it costs once per process; built per task it would cost
    every genome, which is the mistake FastAAI 1 made (a 16-way pool spent ~66 s
    parsing models). It also keeps the spec out of the task payload, so nothing
    is pickled per item that could be sent once.
    """
    if not work:
        return []
    if processes <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(w) for w in work]

    import multiprocessing as mp

    out: list = [None] * len(work)
    tagged = [(i, fn, w) for i, w in enumerate(work)]
    with mp.Pool(processes=processes, initializer=initializer,
                 initargs=initargs) as pool:
        for i, result in pool.imap_unordered(_indexed, tagged, chunksize=chunksize):
            out[i] = result
    return out


def _models(models) -> ModelSet:
    """A ModelSet from a ModelSet, a path, a keyword, or None for the default."""
    return models if isinstance(models, ModelSet) else ModelSet(models)


def _predict_unit(args):
    from .predict import predict_proteins

    genome, out_dir, compress, name = args
    proteins, _table = predict_proteins(genome)
    return os.fspath(layout.write_text(
        Path(out_dir) / f"{layout.safe(name)}{layout.FASTA_EXT}",
        "".join(f">{p}\n{s}\n" for p, s in proteins.items()), compress))


def _hmm_unit(args):
    from .ingest import read_proteins_fasta

    protein, out_dir, compress, name, cpus = args
    ms = _worker_models()
    hits = search_hits(read_proteins_fasta(protein), ms, cpus=cpus)
    return os.fspath(layout.write_text(
        Path(out_dir) / f"{layout.safe(name)}{layout.TABLE_EXT}",
        _name_line(name) + HITS_COLUMNS
        + "".join(f"{h.protein}\t{h.accession}\t{h.score:.4f}\n" for h in hits),
        compress))


def _init_models(spec) -> None:
    """Build this process's model set. Runs once, as a pool initializer.

    A `ModelSet` holds pyhmmer objects and cannot be pickled, so it is rebuilt
    from its spec here rather than sent. Once per worker, before any task: the
    per-genome cost of having models available is then zero.
    """
    global _WORKER_MODELS
    _WORKER_MODELS = _models(spec)


def _worker_models() -> ModelSet:
    global _WORKER_MODELS
    try:
        return _WORKER_MODELS
    except NameError:  # a direct call, outside any pool
        _WORKER_MODELS = _models(None)
        return _WORKER_MODELS


def genome_to_protein(genome, out_dir, *, compress: bool = False,
                      name: str | None = None) -> Path:
    """Predict genes for one genome. Returns the protein FASTA written.

    The translation table is chosen per genome and recorded in the crystal
    later; v1's >10% hysteresis applies, without which table 4 wins on most
    genomes for no biological reason.
    """
    return Path(_predict_unit((os.fspath(genome), os.fspath(out_dir), compress,
                               name or genome_name(genome))))


def genomes_to_proteins(genomes: Iterable, out_dir, *, compress: bool = False,
                        processes: int = 1) -> list[Path]:
    """`genome_to_protein` over many, in parallel. Returns paths in input order."""
    work = [(os.fspath(g), os.fspath(out_dir), compress, genome_name(g))
            for g in genomes]
    return [Path(p) for p in _run(_predict_unit, work, processes)]


def protein_to_hmm(protein, out_dir, models=None, *, compress: bool = False,
                   name: str | None = None, cpus: int = 1) -> Path:
    """HMM-search one protein file. Returns the hit table written.

    Every included hit is written, not only the winners: which hit survives
    depends on the best-hit filter, and storing the raw table means the filter
    can be changed later without searching again.
    """
    _init_models(models.spec if isinstance(models, ModelSet) else models)
    return Path(_hmm_unit((os.fspath(protein), os.fspath(out_dir), compress,
                           name or genome_name(protein), cpus)))


def proteins_to_hmms(proteins: Iterable, out_dir, models=None, *,
                     compress: bool = False, processes: int = 1,
                     cpus: int = 1) -> list[Path]:
    """`protein_to_hmm` over many, in parallel. Returns paths in input order."""
    spec = models.spec if isinstance(models, ModelSet) else models
    work = [(os.fspath(p), os.fspath(out_dir), compress, genome_name(p), cpus)
            for p in proteins]
    return [Path(p) for p in _run(_hmm_unit, work, processes,
                                  initializer=_init_models, initargs=(spec,))]


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
    # Only the fingerprint reaches a worker, not the model set: this step reads
    # hit tables that already name their accessions, so no HMM is consulted and
    # there is nothing per-process to build.
    ms = _models(models)
    work = [(os.fspath(a), os.fspath(b), os.fspath(out_dir), ms.fingerprint,
             filter_mode, compress) for a, b in ((p[0], p[1]) for p in pairs)]
    out = _run(_pair_to_crystal, work, processes, chunksize=16)
    return [Path(p) for p in out if p is not None]


def describe_database(db) -> dict:
    """A database's metadata, as plain values."""
    from . import _core

    if not hasattr(db, "genome_names"):
        db = _core.open_database(os.fspath(db))
    return {
        "genomes": db.n_genomes,
        "partitions": db.n_partitions,
        "partition_size": db.partition_size,
        "partition_genomes": list(db.partition_genomes),
        "accessions": len(db.accession_names),
        "k": db.k,
        "alphabet": db.alphabet,
        "filter_mode": db.filter_mode,
        "models": db.models,
        "source": db.source,
        "schema_key": db.schema_key(),
        "index_bytes": db.index_bytes(),
        "occupancy": db.occupancy(),
    }


def dump_database(db, out_dir, *, orientation: str = "genome",
                  full: bool = False) -> dict:
    """Write a database out as text. Returns the paths written.

    The stored form is packed binary because that is what makes it fast and
    small; this is the readable view of the same thing:

        schema.txt        k, alphabet, filter, fingerprint, sizes
        accessions.tsv    accession id -> name, id being its position
        genomes.tsv       genome, partition, local id, markers carried
        by_genome.tsv     one row per (genome, accession)
        by_kmer.json      accession -> k-mer -> genomes, nested

    *orientation* picks which index file to write — `"genome"`, `"kmer"`, or
    `"both"`. They answer different questions: by-genome is what a genome
    contains and lines up with its crystal, by-kmer is the CSR as stored and
    shows which genomes share a k-mer. By-kmer needs no transpose and is the
    cheaper of the two.

    By-kmer is JSON rather than a table because a table repeats the partition
    and accession on every row, and at index scale that repetition is most of
    the file. Nested, each appears once and a k-mer's genomes are ordinals into
    the `genomes` list at the top. Partition stays an outer level because a
    posting list is partition-local — the same k-mer id in two partitions is two
    independent lists — and keeping it lets the file stream out one partition at
    a time instead of accumulating.

    *full* adds the member column — every k-mer id, or every genome name — which
    reconstructs the index exactly and is correspondingly large. Left off, each
    row carries a count, which is what "what is in here" needs.
    """
    from . import _core

    if not hasattr(db, "genome_names"):
        db = _core.open_database(os.fspath(db))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = describe_database(db)

    (out / "schema.txt").write_text(
        "\n".join(f"{k}\t{v}" for k, v in info.items() if k != "partition_genomes")
        + f"\npartition_genomes\t{','.join(map(str, info['partition_genomes']))}\n")

    (out / "accessions.tsv").write_text(
        "id\taccession\n"
        + "".join(f"{i}\t{a}\n" for i, a in enumerate(db.accession_names)))

    # Which partition a genome sits in, and its local id, are derivable from the
    # partition sizes — they are how a posting list is addressed.
    counts = db.scp_counts()
    rows, g = [], 0
    for pi, n in enumerate(db.partition_genomes):
        for local in range(n):
            rows.append(f"{db.genome_names[g]}\t{pi}\t{local}\t{counts[g]}\n")
            g += 1
    (out / "genomes.tsv").write_text("genome\tpartition\tlocal_id\tn_markers\n"
                                     + "".join(rows))

    if orientation not in ("genome", "kmer", "both"):
        raise ValueError("orientation must be 'genome', 'kmer' or 'both', "
                         f"not {orientation!r}")

    written = {"schema": out / "schema.txt",
               "accessions": out / "accessions.tsv",
               "genomes": out / "genomes.tsv"}
    rows = {}
    for which in (("genome", "kmer") if orientation == "both" else (orientation,)):
        dest = out / (f"by_{which}.json" if which == "kmer"
                      else f"by_{which}.tsv")
        rows[which] = db.write_dump(str(dest), which, full)
        written[f"by_{which}"] = dest
    written["rows"] = rows
    return written


def all_steps(genome, directory=layout.DEFAULT_ROOT, models=None, *,
              filter_mode: FilterMode = DEFAULT_FILTER, compress: bool = False,
              input_kind: str = "auto"):
    """One genome, all the way to a crystal, in this process.

    Predict, search, resolve, write — with **nothing returned between stages**.
    The intermediate proteins and hits go straight to their files; only a small
    record comes back. That is what makes this the right unit to parallelise:
    a worker owns a genome from FASTA to crystal, so no sequence crosses a
    process boundary and there is no collector to funnel through.

    Running the three steps as three parallel passes would be the same work with
    two extra synchronisation points and the intermediates read back off disk.
    """
    from .pipeline import preprocess_one

    site = layout.Layout(directory, compress)
    site.sub(layout.PROTEINS, create=True)
    site.sub(layout.HMM_HITS, create=True)
    site.sub(layout.CRYSTALS, create=True)
    return preprocess_one(genome, _models(models), filter_mode, input_kind,
                          site.crystals, compress, site.root)


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
               directory=layout.DEFAULT_ROOT, database=None, *,
               models=None, filter_mode: FilterMode = DEFAULT_FILTER,
               processes: int = 4, compress: bool = False, save: bool = True,
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

    The database is written to `<directory>/database/` unless *database* names
    something beneath it, or gives an absolute path. Pass `save=False` to build
    without writing it at all.
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
                paths, ms, mode=filter_mode, processes=processes, progress=progress,
                archive_root=site.root, input_kind=kind,
                crystal_root=site.crystals, compress=compress,
            )

    if pairs:
        prot_hmm_to_crystal(pairs, site.sub(layout.CRYSTALS, create=True), ms,
                            filter_mode=filter_mode, compress=compress,
                            processes=processes)

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
