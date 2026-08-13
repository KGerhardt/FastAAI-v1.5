"""FASTA to AAI, end to end."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def preprocess_paths(
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
    build, and peak memory stops depending on how many genomes there are.
    Build from the crystal directory with `build_from_crystals`; the records are
    metadata for reporting, not a second copy of the data.
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


def crystallize_archive(root, out, models: ModelSet,
                        mode: FilterMode = DEFAULT_FILTER,
                        compress: bool = False) -> int:
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
    first.
    """
    import pyfastx

    from . import crystal, layout
    from .archive import genome_names, read_fingerprint, read_hits_for

    fingerprint = read_fingerprint(root) or models.fingerprint
    prot_dir = os.path.join(os.fspath(root), layout.PROTEINS)
    n = 0

    # Driven by the hit files, which carry each genome's true name. Deriving it
    # from a filename would silently rename any genome `layout.safe` rewrote.
    for genome in genome_names(root):
        path = layout.find(prot_dir, layout.safe(genome), layout.FASTA_EXT)
        if path is None:
            continue
        assignment = resolve_hits(read_hits_for(root, genome), mode)
        if not assignment:
            continue
        wanted = set(assignment)

        scps: dict[str, str] = {}
        origins: dict[str, str] = {}
        for name, seq in pyfastx.Fastx(str(path)):
            if name in wanted:
                acc = assignment[name]
                scps[acc] = seq
                origins[acc] = name

        if crystal.write(out, genome, scps, fingerprint, mode, None, origins,
                         compress):
            n += 1
    return n


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
                             float(jac[ti]), int(shared[ti])))
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
                            int(shared[qi, ti]))

    def to_tsv(self, path, *, include_self: bool = True) -> "os.PathLike | str":
        """Write these values as a table. **This is not FastAAI 1's TSV.**

        It is the numbers this object holds, with its own column names to make
        the difference visible. The v1-compatible table has two columns that are
        not here — `jacc_SD` and `poss_shared_SCPs` — and applies a reporting
        band that reports a self-pair as 100 and anything above 90 as `>90%`.
        That band and its rounding live in Rust (`report.rs`) with their own
        tests, so reproducing them here would be a second implementation to keep
        in step, and the numbers would drift the first time one changed.

        For the v1 table, let the engine write it: `fastaai query`, or
        `Database.write_block`, which is what the CLI calls.

        `aai` here is the uncensored fit, so a self-pair reads as its
        extrapolated value rather than 100.
        """
        with open(path, "w") as fh:
            fh.write("query\ttarget\tjaccard\tshared\taai\n")
            for h in self.rows(include_self=include_self):
                j = "" if not np.isfinite(h.jaccard) else f"{h.jaccard:.6f}"
                a = "" if not np.isfinite(h.aai) else f"{h.aai:.4f}"
                fh.write(f"{h.query}\t{h.target}\t{j}\t{h.shared}\t{a}\n")
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
    return SearchResult(query.genome_names, target.genome_names, jac, sh, sd)
