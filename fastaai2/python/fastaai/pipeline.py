"""FASTA to AAI, end to end."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, NamedTuple

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
    #: Counts, which are what a worker reports when the data itself went to
    #: disk. Sized in bytes rather than megabytes, so they cross a process
    #: boundary for free.
    n_proteins: int = 0
    n_hits: int = 0
    n_scps: int = 0

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
    archive_root=None,
) -> GenomeRecord:
    """Predict (if needed), HMM-search and resolve one genome, writing its ranks.

    *input_kind* is `"genome"`, `"protein"`, or `"auto"` to guess from the
    extension. Protein input skips Prodigal entirely — which is the whole
    reference-build path, since GTDB ships predicted proteins and re-predicting
    600k genomes would be ~4 s each of pure waste. Prodigal remains mandatory
    for query genomes, which arrive as nucleotides.

    **The worker writes its own files.** With *archive_root* it writes the
    proteins and the hit table; with *crystal_root*, the crystal. Everything a
    genome produces is on disk before this returns, and the record it hands back
    carries counts rather than sequence. That is what makes the unit of work
    shippable to another process — a collector that had to receive the proteins
    would spend more time unpickling them than predicting them.
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

        if archive_root is not None:
            from .archive import write_genome
            write_genome(archive_root, name, proteins, hits, compress)
        if crystal_root is not None:
            from . import crystal
            crystal.write(crystal_root, name, scps, models.fingerprint, mode,
                          table, origins, compress)

        rec = GenomeRecord(name, scps, table)
        rec.n_proteins, rec.n_hits = len(proteins), len(hits)
        if archive_root is not None or crystal_root is not None:
            # On disk already; carrying it further is the cost this avoids.
            rec.scps = {}
            rec.n_scps = len(scps)
        else:
            rec.n_scps = len(scps)
            rec.proteins, rec.hits, rec.scp_proteins = proteins, hits, origins
        return rec
    except Exception as exc:  # a bad genome must not abort the run
        return GenomeRecord(name, {}, None, error=f"{type(exc).__name__}: {exc}")


def _preprocess_indexed(args):
    """`_preprocess_task` carrying its position, so `imap_unordered` can report
    genomes as they finish while the returned list stays in input order."""
    i, work = args
    return i, _preprocess_task(work)


def _init_worker(spec) -> None:
    """Build this process's model set. Runs once, as a pool initializer.

    A `ModelSet` holds pyhmmer objects and cannot be pickled, so it is rebuilt
    from its spec rather than sent — and rebuilt *here*, before any task, so
    having models available costs a genome nothing. FastAAI 1 reloaded them per
    task and a 16-process pool spent ~66 s on parsing alone.
    """
    global _WORKER_MODELS
    _WORKER_MODELS = ModelSet(spec)


def _preprocess_task(args) -> GenomeRecord:
    """A worker's whole job. Module level and over plain values, so it pickles."""
    path, mode, input_kind, crystal_root, compress, archive_root = args
    global _WORKER_MODELS
    try:
        ms = _WORKER_MODELS
    except NameError:  # serial path, or a direct call
        ms = _WORKER_MODELS = ModelSet(None)
    return preprocess_one(path, ms, mode, input_kind, crystal_root, compress,
                          archive_root)


def preprocess_paths(
    paths: Iterable[os.PathLike | str],
    models: ModelSet,
    mode: FilterMode = DEFAULT_FILTER,
    processes: int = 4,
    progress: Callable[[int, int, GenomeRecord], None] | None = None,
    archive_root=None,
    input_kind: str = "auto",
    crystal_root=None,
    compress: bool = False,
) -> list[GenomeRecord]:
    """Predict and HMM-search every genome. Order of *paths* is preserved.

    **Processes, not threads.** pyrodigal and pyhmmer do release the GIL, but
    that only removes an obstacle — it does not make N threads in one
    interpreter competitive with N independent single-threaded workers, and a
    real share of the work here is plain Python that holds the GIL anyway:
    building the protein dict, resolving best hits, formatting the crystal.
    The task decomposes into thousands of small, independent, single-threaded
    units, which is what processes are for. Measured on the crystal step alone,
    threads ran at 0.65-0.90x of serial while processes reached 5-6x.

    Measured on 16 genomes at 8 workers, threads and processes come out level
    for *these two stages* — 5.32x against 5.44x — because pyrodigal and pyhmmer
    dominate and do release the GIL. Processes are used anyway, and the reason
    is design rather than that number: the work is already decomposed into
    thousands of independent per-file units, so a shared interpreter buys
    nothing and costs a funnel. Each worker writes its own ranks and returns
    counts, so there is no single writer to coordinate through, no sequence
    crossing a boundary, and no serial fraction to run into Amdahl at high
    worker counts. Threaded, the same 16 genomes peaked at 1.37 GB in one heap;
    per-process peaks are independent.
    """
    paths = list(paths)
    if not paths:
        return []

    work = [(os.fspath(p), mode, input_kind,
             None if crystal_root is None else os.fspath(crystal_root),
             compress,
             None if archive_root is None else os.fspath(archive_root))
            for p in paths]

    # The two files that describe the run, written once by the parent rather
    # than raced over by every worker.
    if archive_root is not None:
        Archive(archive_root, models.accessions, models.fingerprint, compress)

    workers = max(1, processes)
    out: list[GenomeRecord | None] = [None] * len(paths)
    if workers == 1:
        _init_worker(models.spec)
        for i, w in enumerate(work):
            out[i] = _preprocess_task(w)
            if progress:
                progress(i + 1, len(paths), out[i])
    else:
        import multiprocessing as mp

        # chunksize 1: a genome is seconds of work, so handing them out one at a
        # time costs nothing and keeps a slow genome from stranding a chunk.
        tagged = list(enumerate(work))
        with mp.Pool(processes=workers, initializer=_init_worker,
                     initargs=(models.spec,)) as pool:
            done = 0
            for i, rec in pool.imap_unordered(_preprocess_indexed, tagged,
                                              chunksize=1):
                out[i] = rec
                done += 1
                if progress:
                    progress(done, len(paths), rec)
    return [r for r in out if r is not None]


def build_from_crystals(
    source,
    models: ModelSet,
    k: int | None = None,
    alphabet: str | None = None,
    only: set | None = None,
) -> "_core.Database":
    """Build a sealed database from crystals — the only route into an index.

    Parsing happens in Rust: needletail reads each file, the sequence is
    k-merised and dropped, and no sequence becomes a Python object. Files are
    read one at a time, so peak memory is one crystal rather than the
    collection.

    *only* restricts to a set of genome names — the programmatic form of copying
    a subset of the files. *k* and *alphabet* override the defaults, which is
    what reproducing FastAAI 1 needs: it included the stop codon `*` in a
    21-symbol alphabet.

    **Accession order comes from *models*, never from the crystals.** Accession
    IDs are positions in a list, so deriving that list from whichever accessions
    happen to appear would make the schema depend on which genomes were
    included: two subsets of one collection would number their shared markers
    differently and refuse to be compared. Taking the order from the model set
    means any subset builds a database comparable with any other.
    """
    from . import crystal

    paths = [str(p) for p in crystal.crystal_paths(source)]
    if not paths:
        raise ValueError(f"no crystals ({crystal.SUFFIX}) found at {source}")
    return build_from_crystals_paths(paths, models, k, alphabet, only)


def build_from_crystals_paths(
    paths: list[str],
    models: ModelSet,
    k: int | None = None,
    alphabet: str | None = None,
    only: set | None = None,
) -> "_core.Database":
    """`build_from_crystals` over an explicit file list.

    Separate because a caller may be combining crystals from several places —
    which is how collections are combined, now that there is no merge.
    """
    db = _core.build_from_crystals(list(paths), models.accessions, k, alphabet, only)

    # Checked here rather than in Rust because the comparison is against a
    # ModelSet, which is a Python object. Everything below the file — parsing,
    # k-merisation, the index — never crosses back.
    if db.models and db.models != models.fingerprint:
        raise ValueError(
            "these crystals were not built with this model set\n"
            f"  crystals:  {db.models}\n"
            f"  model set: {models.fingerprint}\n"
            "Pass --hmm naming the models the crystals were made with."
        )
    if not db.models:
        db.models = models.fingerprint
    return db


def _crystallize_one(args) -> int:
    """One genome, p+h -> crystal. Module level and taking plain values so it
    can be sent to a worker process; a closure over a `ModelSet` could not be
    pickled, and does not need to be — only the fingerprint string is used."""
    import pyfastx

    from . import crystal, layout
    from .archive import read_hits_for

    root, genome, out, fingerprint, mode, compress = args
    path = layout.find(os.path.join(root, layout.PROTEINS), layout.safe(genome),
                       layout.FASTA_EXT)
    if path is None:
        return 0
    assignment = resolve_hits(read_hits_for(root, genome), mode)
    if not assignment:
        return 0
    wanted = set(assignment)

    scps: dict[str, str] = {}
    origins: dict[str, str] = {}
    for name, seq in pyfastx.Fastx(str(path)):
        if name in wanted:
            acc = assignment[name]
            scps[acc] = seq
            origins[acc] = name

    return 1 if crystal.write(out, genome, scps, fingerprint, mode, None,
                              origins, compress) else 0


def crystallize_archive(root, out, models: ModelSet,
                        mode: FilterMode = DEFAULT_FILTER,
                        compress: bool = False, processes: int = 1) -> int:
    """Turn stored proteins and hits into crystals — the one operator between
    those ranks.

    Everything expensive already happened, so this is a resolve and a write:
    read the hits, take the best-hit assignment, stream the protein FASTA once
    and keep only the sequences that won an accession. Nothing is predicted and
    nothing is searched.

    Streamed with `pyfastx.Fastx`, which is the sequential reader — `Fasta`
    builds an index and can leave `.fxi` sidecars beside read-only data. Only
    the winning proteins are retained, so a genome costs its SCPs rather than
    its whole proteome; the previous version read every protein into a dict
    first, which is what used to make this the slow step.

    **Parallelise with processes, not threads.** Each genome is independent and
    writes its own file, so the work is embarrassingly parallel — but pyfastx
    holds the GIL through the parsing that dominates, so threads serialise on it
    and pay scheduling overhead on top. Measured over 1,500 units:

                        serial    threads x8    processes x8
        local, plain     1320       859 (0.65x)   6966 (5.28x)
        local, gzipped    196       176 (0.90x)   1230 (6.29x)

    Default is 1 so a job never oversubscribes the cores it was allocated.
    Raise it and it scales: the 2,943-genome Firmicutes collection takes 19 s
    serial from gzipped files, ~3 s at 8 processes.
    """
    from . import layout
    from .archive import genome_names, read_fingerprint

    fingerprint = read_fingerprint(root) or models.fingerprint
    prot_dir = os.path.join(os.fspath(root), layout.PROTEINS)

    # Driven by the hit files, which carry each genome's true name. Deriving it
    # from a filename would silently rename any genome `layout.safe` rewrote.
    names = genome_names(root)
    work = [(os.fspath(root), g, os.fspath(out), fingerprint, mode, compress)
            for g in names]
    if processes <= 1:
        return sum(_crystallize_one(w) for w in work)

    import multiprocessing as mp

    # Chunked because the per-item cost is milliseconds and the dispatch is not.
    with mp.Pool(processes=processes) as pool:
        return sum(pool.imap_unordered(_crystallize_one, work, chunksize=16))


def _select(spec, names: list[str]) -> list[int]:
    """A side selector to row indices.

    `"any"` and None mean all of them — the default, so a caller filtering only
    on thresholds does not have to name every genome.
    """
    if spec is None or spec == "any":
        return list(range(len(names)))
    wanted = [spec] if isinstance(spec, str) else list(spec)
    index = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise KeyError(f"not in this result: {missing[:3]}")
    return [index[w] for w in wanted]


class Match(NamedTuple):
    """One pair, as a caller reads it.

    Distinct from `search.Hit`, which is a protein against a model. This is a
    genome against a genome, and the fields are the ones the TSV reports.
    """

    query: str
    target: str
    aai: float
    jaccard: float
    shared: int
    #: Markers the poorer of the two genomes carries — the ceiling on `shared`.
    #: Zero only when the counts were unavailable.
    poss_shared: int = 0

    @property
    def shared_frac(self) -> float:
        """`shared / poss_shared`: how much of what *could* be compared was.

        The number that separates "these genomes are distant" from "one of them
        is a poor assembly". A pair at AAI 60 over 8 of 78 possible markers is a
        different claim from the same AAI over 76, and only this tells them
        apart.
        """
        return self.shared / self.poss_shared if self.poss_shared else float("nan")


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
    #: Accessions carried by each genome. `poss_shared_SCPs` is the smaller of
    #: the two: a pair cannot share more markers than the poorer genome has.
    #: Carried so this object can write the v1 table; the engine's own writer
    #: computes the same thing per block.
    query_scps: list[int] | None = None
    target_scps: list[int] | None = None
    #: True when a database was searched against itself, where the diagonal is
    #: a genome against itself. Identity there is given, not estimated.
    selfcmp: bool = False

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

    # --- reading the result ---------------------------------------------------
    #
    # The matrices are the honest representation and stay public, but almost
    # every caller wants one of three things: the neighbours of a genome, the
    # best hit for each genome, or a table. Without these each of them writes
    # the same index-juggling, and the self-pair is the part they get wrong —
    # in a self-comparison the diagonal is the genome against itself at 100,
    # which is not a neighbour.

    def _self_pair(self, qi: int, ti: int) -> bool:
        return self.query_names[qi] == self.target_names[ti]

    @property
    def queries(self) -> list[str]:
        """The genomes on the query side, in row order."""
        return self.query_names

    @property
    def targets(self) -> list[str]:
        """The genomes on the target side, in column order."""
        return self.target_names

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.query_names), len(self.target_names)

    def scps(self, genome: str) -> int:
        """Markers *genome* carries. Raises if it is in neither side."""
        if self.query_scps is not None and genome in self.query_names:
            return self.query_scps[self.query_names.index(genome)]
        if self.target_scps is not None and genome in self.target_names:
            return self.target_scps[self.target_names.index(genome)]
        raise KeyError(f"{genome!r} is not in this result, or counts are absent")

    def _poss(self, qi: int, ti: int) -> int:
        if self.query_scps is None or self.target_scps is None:
            return 0
        return min(self.query_scps[qi], self.target_scps[ti])

    def _match(self, qi: int, ti: int, aai, jac, shared) -> "Match":
        return Match(self.query_names[qi], self.target_names[ti],
                     float(aai[qi, ti]), float(jac[qi, ti]),
                     int(shared[qi, ti]), self._poss(qi, ti))

    def __call__(self, query="any", target="any", *, min_aai: float | None = None,
                 max_aai: float | None = None, min_shared: int | None = None,
                 min_shared_frac: float | None = None,
                 include_self: bool = False):
        """Iterate the pairs that pass a filter.

            for m in result(query="any", min_aai=60, min_shared_frac=0.5):
                ...

        *query* and *target* select a side: `"any"` (or None) for all of them, a
        genome name for one, or any collection of names for several. The rest
        are thresholds, and all of them are inclusive.

        `min_shared_frac` is `shared / poss_shared` — of the markers the poorer
        genome carries, the fraction actually compared. It is the filter that
        separates a genuinely distant pair from a pair where one genome is a bad
        assembly, and it needs the counts `search` records.

        Pairs sharing no marker never appear: no shared marker is an absence of
        evidence, not a measurement. Self-pairs are excluded unless asked for,
        for the same reason `best_hit` excludes them.
        """
        qsel = _select(query, self.query_names)
        tsel = _select(target, self.target_names)
        if min_shared_frac is not None and (self.query_scps is None
                                            or self.target_scps is None):
            raise RuntimeError(
                "min_shared_frac needs the per-genome marker counts, which this "
                "result does not carry; it was not produced by `search`"
            )

        aai, jac = self.aai, self.jaccard
        shared = np.asarray(self.shared)
        for qi in qsel:
            for ti in tsel:
                s = int(shared[qi, ti])
                if s == 0:
                    continue
                if not include_self and self._self_pair(qi, ti):
                    continue
                a = aai[qi, ti]
                if min_aai is not None and not (a >= min_aai):
                    continue
                if max_aai is not None and not (a <= max_aai):
                    continue
                if min_shared is not None and s < min_shared:
                    continue
                if min_shared_frac is not None:
                    poss = self._poss(qi, ti)
                    if not poss or s / poss < min_shared_frac:
                        continue
                yield self._match(qi, ti, aai, jac, shared)

    def __iter__(self):
        """Every pair that shares a marker, self-pairs excluded."""
        return self()

    def hits_for(self, query: str, *, k: int | None = None,
                 min_aai: float | None = None,
                 include_self: bool = False) -> list["Match"]:
        """Neighbours of one genome, best first.

        Pairs sharing no marker are dropped rather than reported as zero: no
        shared marker is an absence of evidence, not evidence of distance.
        """
        try:
            qi = self.query_names.index(query)
        except ValueError:
            raise KeyError(f"{query!r} is not a query in this result") from None

        aai = self.aai[qi]
        jac = self.jaccard[qi]
        shared = np.asarray(self.shared)[qi]

        order = np.argsort(-np.nan_to_num(aai, nan=-np.inf))
        out: list[Match] = []
        for ti in order:
            if not np.isfinite(aai[ti]):
                continue
            if not include_self and self._self_pair(qi, int(ti)):
                continue
            if min_aai is not None and aai[ti] < min_aai:
                break
            out.append(Match(query, self.target_names[ti], float(aai[ti]),
                             float(jac[ti]), int(shared[ti]),
                             self._poss(qi, int(ti))))
            if k is not None and len(out) >= k:
                break
        return out

    def top_hits(self, k: int = 5, *, min_aai: float | None = None,
                 include_self: bool = False) -> dict[str, list["Match"]]:
        """`hits_for` over every query. The candidate-reduction shape."""
        return {q: self.hits_for(q, k=k, min_aai=min_aai,
                                 include_self=include_self)
                for q in self.query_names}

    def best_hit(self, query: str, *, include_self: bool = False) -> "Match | None":
        """The single closest genome, or None if this query shares no marker
        with anything."""
        got = self.hits_for(query, k=1, include_self=include_self)
        return got[0] if got else None

    def best_hits(self, *, include_self: bool = False) -> dict[str, "Match | None"]:
        return {q: self.best_hit(q, include_self=include_self)
                for q in self.query_names}

    def rows(self, *, include_self: bool = True):
        """Every pair, long format — the TSV's rows as objects."""
        aai, jac = self.aai, self.jaccard
        shared = np.asarray(self.shared)
        for qi, q in enumerate(self.query_names):
            for ti, t in enumerate(self.target_names):
                if not include_self and self._self_pair(qi, ti):
                    continue
                yield Match(q, t, float(aai[qi, ti]), float(jac[qi, ti]),
                            int(shared[qi, ti]), self._poss(qi, ti))

    def to_tsv(self, path, *, include_self: bool = True) -> "os.PathLike | str":
        """Write FastAAI 1's TSV — same columns, same names, same order.

        The band and the rounding are not reimplemented here: `_core.aai_label`
        and `_core.py_round` are the engine's own, exposed so this and the
        streaming writer cannot drift. Verified byte-for-byte against
        `Database.write_block`.

        For a result already in memory. A search too large to hold writes its
        blocks straight from Rust instead — see `cli.write_blocks`.
        """
        if self.query_scps is None or self.target_scps is None:
            raise RuntimeError(
                "this result carries no SCP counts, so poss_shared_SCPs cannot "
                "be written; it was not produced by `search`"
            )
        na = _core.NO_HIT
        jac, shared = self.jaccard, np.asarray(self.shared)
        sd = self.stdev
        with open(path, "w") as fh:
            fh.write("query\ttarget\tavg_jacc_sim\tjacc_SD\tnum_shared_SCPs"
                     "\tposs_shared_SCPs\tAAI_estimate\n")
            for qi, q in enumerate(self.query_names):
                for ti, t in enumerate(self.target_names):
                    if not include_self and self._self_pair(qi, ti):
                        continue
                    s, j = int(shared[qi, ti]), float(jac[qi, ti])
                    # v1 blanks every value column when a pair shares no
                    # accession: there is no measurement, not a measurement of
                    # zero.
                    if s == 0:
                        fh.write(f"{q}\t{t}\t{na}\t{na}\t{na}\t{na}\t{na}\n")
                        continue
                    jtxt = na if not np.isfinite(j) else _core.py_round(j, 4)
                    sdtxt = na
                    if sd is not None and np.isfinite(sd[qi, ti]):
                        sdtxt = _core.py_round(float(sd[qi, ti]), 4)
                    poss = min(self.query_scps[qi], self.target_scps[ti])
                    if self.selfcmp and qi == ti:
                        # Identity is given by the comparison, not inferred from
                        # it; the regression is not consulted.
                        aai = _core.py_round(_core.SELF_IDENTITY, 2)
                    else:
                        aai = _core.aai_label(_core.jaccard_to_aai(j), s, j)
                    fh.write(f"{q}\t{t}\t{jtxt}\t{sdtxt}\t{s}\t{poss}\t{aai}\n")
        return path


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
    # Carried so the result can write the v1 table. One extra pass over the
    # partitions against an N-squared search, and it is what stops the table
    # from having a second implementation.
    qc = query.scp_counts()
    tc = qc if target is query else target.scp_counts()
    return SearchResult(query.genome_names, target.genome_names, jac, sh, sd,
                        list(qc), list(tc), target is query)
