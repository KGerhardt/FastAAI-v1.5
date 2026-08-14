"""Command line entry point.

Two verbs replace FastAAI 1's seven modules:

    fastaai build   inputs -> a database
    fastaai query   database x database -> AAI

plus two that compute nothing:

    fastaai crystallize   proteins + hits -> crystals
    fastaai inspect       a database -> readable text
    fastaai reshape       block results -> v1's output shapes

There is no merge. Combining sealed databases preserved each donor's
partitioning, which fragments the index; putting crystals together and
rebuilding repacks it instead.

That is not a reduction in capability. Query and target databases are the same
format and the k-mer join reads both sides as inverted indexes, so v1's modules
were combinatorial variations on one operation: `aai_index` is `query A A`,
`single_query` is `query A B` where both hold one genome, `multi_query` is a
build followed by a query.

**FastAAI 1 command lines still work.** They are rerouted with arguments
preserved. Where a v1 flag no longer has a meaning it is reported rather than
silently dropped — a flag that changes the output columns must never be ignored
quietly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import _core, crystal, layout
from . import archive as archive_module
from .ingest import find_genomes
from .pipeline import (
    DEFAULT_BLOCK,
    DEFAULT_SEARCH_THREADS,
    build_from_crystals,
    crystallize_archive,
    preprocess_paths,
)
from .search import DEFAULT_FILTER, MODEL_SETS, ModelSet, model_set_key

#: v1 module names, so `fastaai build_db ...` keeps working.
LEGACY_MODULES = (
    "build_db", "merge_db", "simple_query", "db_query",
    "single_query", "multi_query", "aai_index",
)

#: v1 flags with no counterpart. Accepted, then reported — not silently dropped.
RETIRED = {
    "in_memory": "a partition is always worked on in memory; this is the only mode now",
    "store_results": "results are assembled in memory regardless",
    "compress": "archived proteins are gzipped already",
}


def _log(quiet):
    return (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))


def _is_database(path) -> bool:
    return (Path(path) / "schema").exists()


def _describe_models(models, spec, log) -> None:
    """Report which models were loaded, and say so in the user's own terms.

    A keyword names a set the user cannot see on disk, so echoing the keyword
    back proves nothing. Reporting the count — and, for a union, how many
    markers the two sets shared — is what shows the set that was actually built.
    """
    key = model_set_key(spec)
    if key is not None:
        shared = (f", {len(models.shared)} shared between them and kept once"
                  if models.shared else "")
        log(f"{len(models)} models from the packaged {MODEL_SETS[key][1]} set{shared}")
    elif spec is None:
        log(f"{len(models)} models from the bundled default set")
    else:
        log(f"{len(models)} models from {spec}")
    if not models.has_trusted:
        log("  note: models lack TC cutoffs; using default inclusion thresholds")


def _load_or_build(source, models, args, log) -> "_core.Database":
    """Accept a database directory, an archive, crystals, or raw sequence input."""
    p = Path(source)
    if _is_database(p):
        db = _core.open_database(str(p))
        log(f"  opened {p}: {db.n_genomes} genomes, {db.n_partitions} partition(s)")
        return db
    # Crystals first: when a root holds both ranks, the resolved SCPs are the
    # cheaper and more direct route to the same database.
    if crystal.looks_like_crystals(p):
        # Crystals carry the model set that made them, so a build needs the
        # matching models for their ordering — the bundled default if --hmm
        # was not given, which is also what made them in the common case.
        try:
            db = build_from_crystals(p, models if models is not None else ModelSet())
        except ValueError as e:
            # Mismatched models or mixed provenance. A refusal, not a crash.
            raise SystemExit(f"{p}: {e}") from None
        log(f"  built {p} from {len(crystal.crystal_paths(p))} crystals: "
            f"{db.n_genomes} genomes")
        db.source = str(p)
        return db
    if archive_module.looks_like_archive(p):
        # Stored proteins and hits are an earlier rank, not a second way to
        # build. Resolve them into crystals beside themselves, then build from
        # those — one route into an index, and the crystals stay.
        dest = p / layout.CRYSTALS
        n = crystallize_archive(p, dest, models if models is not None else ModelSet(),
                                mode=args.filter, compress=args.compress,
                                processes=args.preprocess_processes)
        log(f"  resolved {n} crystals from stored proteins and hits into {dest}/")
        db = build_from_crystals(dest, models if models is not None else ModelSet())
        log(f"  built {db.n_genomes} genomes")
        db.filter_mode = args.filter
        db.source = str(p)
        return db

    if models is None:
        # No --hmm: fall back to the model set shipped with the package, so an
        # install builds a database without hunting for models.
        models = ModelSet()
        log(f"  using the bundled model set ({len(models)} SCPs)")
    paths = find_genomes(p)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no FASTA files found under {source}")
    log(f"  preprocessing {len(paths)} inputs from {p}")

    t0 = time.perf_counter()
    last = [0.0]

    def progress(done, total, rec):
        if rec.error:
            log(f"    ! {rec.name}: {rec.error}")
        now = time.perf_counter()
        if now - last[0] > 2.0 or done == total:
            last[0] = now
            log(f"    {done}/{total} ({now - t0:.0f}s)")

    # Every rank is kept, always, under one root. Preprocessing is ~98% of a
    # cold run, so discarding it is the expensive choice — and the database is
    # built by reading the crystals back, so there is one ingestion path rather
    # than a stored copy and a separate in-memory one.
    site = layout.Layout(args.root, args.compress)
    site.sub(layout.PROTEINS, create=True)
    site.sub(layout.HMM_HITS, create=True)
    site.sub(layout.CRYSTALS, create=True)
    log(f"  writing to {site.root}/")

    records = preprocess_paths(
        paths, models, mode=args.filter, processes=args.preprocess_processes,
        progress=progress, archive_root=site.root, input_kind=args.input_kind,
        crystal_root=site.crystals, compress=site.compress,
    )
    failed = [r for r in records if r.error]
    written = len(crystal.crystal_paths(site.crystals))
    empty = len(records) - len(failed) - written
    if empty:
        log(f"  {empty} inputs excluded for having no usable SCPs")
    if not written:
        raise SystemExit("no input yielded a usable SCP set")
    log(f"  {written} crystals in {site.crystals}/")

    db = build_from_crystals(site.crystals, models)
    db.filter_mode = args.filter
    db.source = str(p)
    return db


def block_name(qi: int, ti: int, style: str = "tsv") -> str:
    suffix = "matrix" if style == "matrix" else "tsv"
    return f"block_q{qi:05d}_t{ti:05d}.{suffix}"


def write_blocks(query, target, out_path, threads, block, stdev, emit,
                 style="tsv", quiet=False):
    """Write results as one file per (query partition x target partition).

    The only output path. A search whose two sides each fit one partition is
    simply the 1x1 case and lands in a single file — there is no separate
    whole-matrix route, because the whole matrix is what stops being
    representable first: 200k x 200k species is 40 billion pairs.

    With one block, *out_path* is that file (`-` for stdout). With more, it is a
    directory that receives them. Each block is bounded by the partition size
    rather than by the size of either database.

    Rust computes and writes each block: a full self-block is 268M rows, which
    is not something to format one row at a time from Python.
    """
    grid = query.n_partitions * target.n_partitions
    rows = compute = 0

    # A path that is already a directory receives block files, however few there
    # are. Without this the 1x1 case would try to write a file on top of
    # `results/`, and the layout of a run's output would depend on how many
    # partitions it happened to have.
    if grid == 1 and out_path and out_path != "-" and Path(out_path).is_dir():
        dest = Path(out_path) / block_name(0, 0, style)
        rows, compute = query.write_block(target, 0, 0, str(dest), block,
                                          threads, stdev, emit, style)
        if not quiet:
            print(f"  {dest.name}  {rows:,} rows", file=sys.stderr)
        return rows, compute

    if grid == 1:
        rows, compute = query.write_block(target, 0, 0, out_path or "-", block,
                                          threads, stdev, emit, style)
        if not quiet:
            print(f"  {rows:,} rows", file=sys.stderr)
        return rows, compute

    if not out_path or out_path == "-":
        raise SystemExit(
            f"this search is {query.n_partitions} x {target.n_partitions} blocks; "
            "give -o a directory to write them into"
        )
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for qi in range(query.n_partitions):
        for ti in range(target.n_partitions):
            dest = out_dir / block_name(qi, ti, style)
            n, secs = query.write_block(target, qi, ti, str(dest), block, threads,
                                        stdev, emit, style)
            rows += n
            compute += secs
            if not quiet:
                print(f"  {dest.name}  {n:,} rows", file=sys.stderr)

    if not quiet:
        print(f"{grid} blocks written ({rows:,} rows)", file=sys.stderr)
    return rows, compute


def _common(p):
    p.add_argument("--hmm", metavar="FILE|SET",
                   help="HMM file defining the SCP model set (plain or gzipped), "
                        "or a packaged set: " + ", ".join(MODEL_SETS)
                        + " (default: the bundled 122-SCP set)")
    p.add_argument("--threads", type=int, default=DEFAULT_SEARCH_THREADS,
                   help=f"threads for the counting kernel (default {DEFAULT_SEARCH_THREADS})")
    p.add_argument("--processes", dest="preprocess_processes", type=int, default=4,
                   help="worker processes for preprocessing — prediction, HMM "
                        "search and crystallising (default 4). Processes, not "
                        "threads: each genome is an independent unit that "
                        "writes its own files")
    p.add_argument("--filter", choices=("v1", "v1_alt", "rbh"), default=DEFAULT_FILTER,
                   help="best-hit resolution")
    p.add_argument("--input", dest="input_kind",
                   choices=("auto", "genome", "protein"), default="auto",
                   help="input type (default: guess from extension); protein input "
                        "skips gene prediction")
    p.add_argument("--dir", dest="root", metavar="DIR", default=layout.DEFAULT_ROOT,
                   help=f"output root, created if needed (default: "
                        f"{layout.DEFAULT_ROOT}/ here). A build fills "
                        f"proteins/, hmm_hits/, crystals/ and database/; a "
                        f"search adds results/")
    p.add_argument("--gzip", dest="compress", action="store_true",
                   help="gzip each file as it is written (default: plain text)")
    p.add_argument("--limit", type=int, help="use only the first N inputs")
    p.add_argument("--quiet", action="store_true")
    for flag in RETIRED:
        p.add_argument(f"--{flag}", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--do_stdev", action="store_true",
                   help="also report the standard deviation of Jaccard across shared SCPs")
    p.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fastaai",
        description="Average amino acid identity from single-copy protein tetramers.",
        epilog="FastAAI 1 command lines (build_db, db_query, aai_index, ...) still work.",
    )
    sub = p.add_subparsers(dest="command")

    b = sub.add_parser("build", help="build a database from genomes or proteins")
    b.add_argument("inputs", help="FASTA file or directory")
    b.add_argument("-d", "--database", default=None,
                   help="by default the database is written to <dir>/database/. "
                        "Give a name to add levels beneath that, or an absolute "
                        "path to put it elsewhere")
    b.add_argument("--source", default="", help="provenance label, e.g. 'GTDB R232 bac120'")
    _common(b)

    q = sub.add_parser("query", help="query one database against another")
    q.add_argument("-q", "--query", required=True,
                   help="query database, archive, or sequences")
    q.add_argument("-t", "--target",
                   help="target database (default: query against itself, upper triangle)")
    q.add_argument("-o", "--output",
                   help="explicit output path, or - for stdout. Default is "
                        "<dir>/results/, which is a directory of block files")
    q.add_argument("--output_style", choices=("tsv", "matrix"), default="tsv")
    q.add_argument("--emit", choices=("aai", "jaccard", "both"), default="both")
    _common(q)

    r = sub.add_parser("reshape",
                       help="turn block results into v1's output shapes")
    r.add_argument("results", help="results directory, or a single block file")
    r.add_argument("--per-genome", metavar="DIR",
                   help="write one TSV per query genome, that genome against "
                        "everything")
    r.add_argument("--matrix", metavar="FILE",
                   help="write the AAI-only matrix")
    r.add_argument("--gzip", dest="compress", action="store_true")
    r.add_argument("--quiet", action="store_true")

    i = sub.add_parser("inspect",
                       help="write a database out as readable text")
    i.add_argument("database", help="database directory")
    i.add_argument("-o", "--output", help="directory for the text files "
                                          "(default: <database>/../inspect)")
    i.add_argument("--by", dest="orientation", default="both",
                   choices=("genome", "kmer", "both"),
                   help="genome: what each genome contains, which lines up with "
                        "its crystal. kmer: the CSR as stored, which genomes "
                        "share a k-mer (default: both)")
    i.add_argument("--full", action="store_true",
                   help="list every member rather than a count, reconstructing "
                        "the index exactly. Large: tens of millions of rows at "
                        "GTDB scale")
    i.add_argument("--quiet", action="store_true")

    c = sub.add_parser("crystallize",
                       help="emit crystals from an existing archive")
    c.add_argument("source", help="root holding proteins/ and hmm_hits/")
    c.add_argument("-o", "--output",
                   help="crystal directory (default: <source>/crystals/)")
    c.add_argument("--gzip", dest="compress", action="store_true",
                   help="gzip each crystal as it is written")
    c.add_argument("--hmm", metavar="FILE|SET",
                   help="models the archive was searched with (default: bundled)")
    c.add_argument("--filter", choices=("v1", "v1_alt", "rbh"), default=DEFAULT_FILTER,
                   help="best-hit resolution to bake into the crystals")
    c.add_argument("--processes", type=int, default=1,
                   help="worker processes; scales 5-6x at 8. Processes, not "
                        "threads: pyfastx holds the GIL (default 1)")
    c.add_argument("--quiet", action="store_true")

    return p


def cmd_build(args) -> int:
    log = _log(args.quiet)
    models = ModelSet(args.hmm) if args.hmm else None  # resolved on demand
    if models:
        _describe_models(models, args.hmm, log)

    db = _load_or_build(args.inputs, models, args, log)
    if args.source:
        db.source = args.source
    dest = layout.Layout(args.root, args.compress).database_path(args.database)
    dest.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(dest))
    parts, _, man = db.stored_bytes(str(dest))
    log(f"wrote {dest}: {db.n_genomes} genomes, {db.n_partitions} partition(s), "
        f"{parts / 1e6:.1f} MB index + {man / 1e3:.0f} KB manifest")
    return 0


def cmd_query(args) -> int:
    log = _log(args.quiet)

    # Flags that change the output columns must never be dropped quietly.
    if args.output_style == "matrix":
        if args.do_stdev:
            raise SystemExit("--do_stdev has no matrix representation: a cell holds "
                             "one value. Use the default --output_style tsv.")
        if args.emit != "both":
            raise SystemExit(f"--emit {args.emit} is not available with "
                             "--output_style matrix, which writes AAI only. "
                             "Use --output_style tsv.")

    models = ModelSet(args.hmm) if args.hmm else None  # resolved on demand
    if models:
        _describe_models(models, args.hmm, log)
    qdb = _load_or_build(args.query, models, args, log)
    same = not args.target or args.target == args.query
    tdb = qdb if same else _load_or_build(args.target, models, args, log)

    # Results are an output like any other and default into the run's root. An
    # explicit -o still wins, including `-` for stdout.
    out_path = args.output
    if not out_path:
        site = layout.Layout(args.root, args.compress)
        out_path = str(site.sub(layout.RESULTS, create=True))
        log(f"  results to {out_path}/")

    pairs = qdb.n_genomes * tdb.n_genomes
    t0 = time.perf_counter()
    try:
        _rows, compute = write_blocks(
            qdb, tdb, out_path, threads=args.threads, block=DEFAULT_BLOCK,
            stdev=args.do_stdev, emit=args.emit, style=args.output_style,
            quiet=args.quiet)
    except ValueError as e:
        # The engine refuses to compare databases built from different model
        # sets, which is the guard working. Reaching it is easy now that a set
        # is one word on the command line, so it needs to read as a decision
        # rather than as a crash.
        if "schema mismatch" not in str(e):
            raise
        raise SystemExit(
            f"{e}\n"
            f"  query  {args.query}: {len(qdb.accession_names)} accessions, "
            f"models {qdb.models[:16]}\n"
            f"  target {args.target or args.query}: {len(tdb.accession_names)} "
            f"accessions, models {tdb.models[:16]}\n"
            "Databases built from different model sets hold different accession "
            "lists, so their AAI is not comparable. Rebuild one with the other's "
            "--hmm."
        ) from None
    dt = time.perf_counter() - t0
    # Throughput is the kernel's, per thread — not aggregated over the job and
    # not diluted by formatting and disk. Per thread is what compares across
    # machines and thread counts, and is the convention FastAAI 1's published
    # figures use; the write time is reported beside it rather than inside it.
    kernel = compute if compute else dt
    per_thread = pairs / max(kernel, 1e-9) / max(args.threads, 1)
    tail = f", write {dt - kernel:.2f}s" if compute else ""
    log(f"search {kernel:.2f}s ({per_thread:,.0f} pairs/s/thread, "
        f"{args.threads} threads){tail}"
        f"{' [symmetric, upper triangle]' if same else ''}")
    return 0


def cmd_reshape(args) -> int:
    """Block results back into the shapes v1 wrote."""
    from . import reshape

    log = _log(args.quiet)
    if not args.per_genome and not args.matrix:
        raise SystemExit("nothing to do: pass --per-genome DIR, --matrix FILE, "
                         "or both")
    blocks = reshape.block_files(args.results)
    log(f"{args.results}: {len(blocks)} block file(s)")
    if args.per_genome:
        written = reshape.per_genome(args.results, args.per_genome,
                                     compress=args.compress)
        log(f"  {len(written)} per-genome TSVs in {args.per_genome}/")
    if args.matrix:
        dest = reshape.to_matrix(args.results, args.matrix,
                                 compress=args.compress)
        log(f"  matrix written to {dest}")
    return 0


def cmd_inspect(args) -> int:
    """The readable view of a packed binary database."""
    from .api import dump_database

    log = _log(args.quiet)
    db = _core.open_database(args.database)
    out = Path(args.output) if args.output else Path(args.database).parent / "inspect"
    written = dump_database(db, out, orientation=args.orientation, full=args.full)
    log(f"{args.database}: {db.n_genomes} genomes, {db.n_partitions} partition(s), "
        f"{len(db.accession_names)} accessions")
    log(f"  wrote {out}/")
    for key, path in written.items():
        if key != "rows":
            n = written["rows"].get(key.replace("by_", ""))
            log(f"    {path.name}" + (f"  {n:,} rows" if n else ""))
    return 0


def cmd_crystallize(args) -> int:
    log = _log(args.quiet)
    models = ModelSet(args.hmm) if args.hmm else ModelSet()
    _describe_models(models, args.hmm, log)
    out = args.output or (Path(args.source) / layout.CRYSTALS)
    n = crystallize_archive(args.source, out, models, mode=args.filter,
                            compress=args.compress, processes=args.processes)
    log(f"wrote {n} crystals to {out}")
    log(f"  build from them with: fastaai build {out}")
    return 0


def _reroute(argv: list[str]) -> list[str]:
    """Translate a FastAAI 1 command line into the new surface."""
    module, rest = argv[0], argv[1:]

    def opt(*names):
        for n in names:
            if n in rest:
                i = rest.index(n)
                if i + 1 < len(rest):
                    return rest[i + 1]
        return None

    def pair(flag, value):
        """A flag only when it has a value.

        `opt` returns None for an absent flag, and putting that None into argv
        makes argparse report "expected one argument" against a flag the user
        never typed. Optional v1 arguments — `-o`, or `-t` on a self-query —
        must simply not appear.
        """
        return [flag, value] if value is not None else []

    def carried():
        out = []
        for n in ("--threads", "--filter", "--output_style", "--limit", "--archive", "--hmm"):
            v = opt(n)
            if v is not None:
                out += [n, v]
        for n in ("--verbose", "--quiet", "--do_stdev", "--in_memory",
                  "--store_results", "--compress"):
            if n in rest:
                out.append(n)
        return out

    if opt("-m", "--hmms") or opt("-qh", "--query_hmms") or opt("-th", "--target_hmms"):
        raise SystemExit(
            "precomputed HMMER tables (-m/-qh/-th) were an input in FastAAI 1. FastAAI 2 "
            "does not read them; supply genomes or proteins, or reuse an --archive, which "
            "stores raw hits and can be re-filtered without re-searching."
        )

    # Anything the translation does not consume would otherwise vanish, because
    # the new command line is built from scratch rather than edited. A dropped
    # flag that changed v1's behaviour must be reported, never ignored.
    unsupported = {
        "--create_query_db": "build the query set into a database with `fastaai build`",
        "--query_db_name": "build the query set into a database with `fastaai build`",
        "--query_output": "query and target outputs are one file; run the two "
                          "directions separately if both are wanted",
        "--target_output": "query and target outputs are one file; run the two "
                           "directions separately if both are wanted",
    }
    for flag, advice in unsupported.items():
        if flag in rest:
            raise SystemExit(f"{flag} is not supported: {advice}.")

    genomes, proteins = opt("-g", "--genomes"), opt("-p", "--proteins")
    inp = genomes or proteins
    kind = ["--input", "protein"] if (proteins and not genomes) else []

    out = opt("-o", "--output")
    if module == "build_db":
        return ["build", inp] + pair("-d", opt("-d", "--database") or out) \
            + kind + carried()
    if module == "aai_index":
        return ["query"] + pair("-q", inp) + pair("-o", out) + kind + carried()
    if module == "db_query":
        return ["query"] + pair("-q", opt("-q", "--query")) \
            + pair("-t", opt("-t", "--target")) + pair("-o", out) + carried()
    if module == "simple_query":
        return ["query"] + pair("-q", inp) + pair("-t", opt("--target")) \
            + pair("-o", out) + kind + carried()
    if module == "multi_query":
        qi = opt("--query_genomes") or opt("--query_proteins") or opt("--query_database")
        ti = opt("--target_genomes") or opt("--target_proteins") or opt("--target_database")
        k = ["--input", "protein"] if opt("--query_proteins") else []
        return ["query"] + pair("-q", qi) + pair("-t", ti) + pair("-o", out) \
            + k + carried()
    if module == "single_query":
        qi = opt("-qg", "--query_genome") or opt("-qp", "--query_protein")
        ti = opt("-tg", "--target_genome") or opt("-tp", "--target_protein")
        k = ["--input", "protein"] if opt("-qp", "--query_protein") else []
        return ["query"] + pair("-q", qi) + pair("-t", ti) + pair("-o", out) \
            + k + carried()
    if module == "merge_db":
        # Merging sealed databases is gone. It preserved each donor's
        # partitioning, which is the fragmentation trap: merging N one-genome
        # databases produced N one-genome partitions and cost ~90x search
        # throughput. Rebuilding from crystals repacks instead, so the
        # replacement is strictly better — but it is not a flag translation,
        # and silently doing something else would be worse than saying so.
        raise SystemExit(
            "merge_db is not supported: FastAAI 2 does not merge sealed "
            "databases.\n"
            "Merging kept each donor's partitioning, which fragments the index "
            "and costs search throughput.\n"
            "Put the crystals together and rebuild instead — the result is one "
            "cleanly partitioned database:\n"
            "  cp a/crystals/*.crystal.fasta b/crystals/*.crystal.fasta all/\n"
            "  fastaai build all/ -d combined"
        )

    raise SystemExit(f"unknown module {module!r}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in LEGACY_MODULES:
        module = argv[0]
        argv = [a for a in _reroute(argv) if a is not None]
        print(f"note: `{module}` is a FastAAI 1 module, rerouted to "
              f"`fastaai {' '.join(argv[:3])} ...`", file=sys.stderr)

    args = build_parser().parse_args(argv)
    if not getattr(args, "command", None):
        build_parser().print_help()
        return 1

    quiet = getattr(args, "quiet", False)
    for flag, why in RETIRED.items():
        if getattr(args, flag, False) and not quiet:
            print(f"note: --{flag} no longer applies — {why}", file=sys.stderr)
    return {"build": cmd_build, "query": cmd_query,
            "crystallize": cmd_crystallize,
            "inspect": cmd_inspect,
            "reshape": cmd_reshape}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
