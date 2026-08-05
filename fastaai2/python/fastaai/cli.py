"""Command line entry point.

Three verbs replace FastAAI 1's seven modules:

    fastaai build   inputs -> a database
    fastaai query   database x database -> AAI
    fastaai merge   databases -> one database

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

import numpy as np

from . import _core
from .ingest import find_genomes
from .pipeline import (
    DEFAULT_BLOCK,
    DEFAULT_SEARCH_THREADS,
    build_database,
    build_from_archive,
    preprocess,
    search,
)
from .search import DEFAULT_FILTER, ModelSet

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


def _load_or_build(source, models, args, log) -> "_core.Database":
    """Accept a database directory, an archive, or raw sequence input."""
    p = Path(source)
    if _is_database(p):
        db = _core.open_database(str(p))
        log(f"  opened {p}: {db.n_genomes} genomes, {db.n_partitions} partition(s)")
        return db
    if (p / "manifest.tsv").exists() and (p / "proteins").exists():
        db = build_from_archive(p, mode=args.filter)
        log(f"  rebuilt {p} from archive: {db.n_genomes} genomes")
        db.filter_mode = args.filter
        db.source = str(p)
        return db

    if models is None:
        raise SystemExit(f"{source} is not a database; --hmm is required to build one")
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

    records = preprocess(
        paths, models, mode=args.filter, threads=args.preprocess_threads,
        progress=progress, archive_root=args.archive, input_kind=args.input_kind,
    )
    db, skipped = build_database(records, models, filter_mode=args.filter)
    if skipped:
        log(f"  {len(skipped)} inputs excluded for having no usable SCPs")
    db.filter_mode = args.filter
    db.source = str(p)
    return db


#: The band the Jaccard->AAI regression has sensitivity across. Outside it the
#: estimate cannot support a specific figure, so the output says so categorically
#: rather than printing a number that would assert precision it does not have.
#: These are labels about the limits of the estimator, not display preferences,
#: and they are not optional (v1 fastaai.py:2327-2338).
AAI_FLOOR = 30.0
AAI_CEILING = 90.0
LABEL_BELOW = "<30%"
LABEL_ABOVE = ">90%"

#: A matrix cell cannot hold a string, so the two categories carry v1's sentinel
#: values there instead (v1 README: "reports these categorical estimates with
#: 15.0 and 95.0 AAI, respectively").
MATRIX_BELOW = 15.0
MATRIX_ABOVE = 95.0


def aai_label(aai, shared, jaccard) -> str:
    """Format one AAI estimate, categorically where the regression cannot resolve.

    Zero Jaccard is the case that needs naming: `log(0)` sends it to the top of
    the regression, so it must be caught before the ceiling test or two genomes
    with nothing in common report as >90%. v1 carries the same correction.

    No shared markers stays `NA`. "These genomes share no SCP" is not a claim
    that their AAI is below 30% — it is the absence of a measurement.
    """
    if shared == 0 or (jaccard is not None and np.isnan(jaccard)):
        return "NA"
    if (jaccard is not None and jaccard == 0) or np.isnan(aai) or aai < AAI_FLOOR:
        return LABEL_BELOW
    if aai > AAI_CEILING:
        return LABEL_ABOVE
    return f"{aai:.2f}"


def aai_matrix_value(aai, shared, jaccard) -> str:
    """Matrix-format counterpart to `aai_label`, using v1's numeric sentinels."""
    label = aai_label(aai, shared, jaccard)
    if label == "NA":
        return "NA"
    if label == LABEL_BELOW:
        return f"{MATRIX_BELOW:.1f}"
    if label == LABEL_ABOVE:
        return f"{MATRIX_ABOVE:.1f}"
    return label


def block_name(qi: int, ti: int) -> str:
    return f"block_q{qi:05d}_t{ti:05d}.tsv"


def write_blocks(query, target, out_path, threads, block, stdev, emit, quiet=False):
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

    if grid == 1:
        rows, compute = query.write_block(target, 0, 0, out_path or "-", block,
                                          threads, stdev, emit)
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
            dest = out_dir / block_name(qi, ti)
            n, secs = query.write_block(target, qi, ti, str(dest), block, threads,
                                        stdev, emit)
            rows += n
            compute += secs
            if not quiet:
                print(f"  {dest.name}  {n:,} rows", file=sys.stderr)

    if not quiet:
        print(f"{grid} blocks written ({rows:,} rows)", file=sys.stderr)
    return rows, compute


def _write(res, out_path, style, emit):
    """Matrix output only.

    TSV goes through `write_blocks`, which is the single writer for it — one
    formatter, in Rust, for every TSV this program emits. A matrix is a
    different shape rather than a different format: one dense array of every
    pair, so it exists only for searches small enough to hold one.
    """
    assert style == "matrix", style
    aai, shared, jacc = res.aai, res.shared, res.jaccard
    fh = open(out_path, "w") if out_path else sys.stdout
    try:
        fh.write("\t" + "\t".join(res.target_names) + "\n")
        for i, qn in enumerate(res.query_names):
            row = [aai_matrix_value(aai[i, j], shared[i, j], jacc[i, j])
                   for j in range(len(res.target_names))]
            fh.write(qn + "\t" + "\t".join(row) + "\n")
    finally:
        if out_path:
            fh.close()


def _common(p):
    p.add_argument("--hmm", help="HMM file defining the SCP model set")
    p.add_argument("--threads", type=int, default=DEFAULT_SEARCH_THREADS,
                   help=f"threads for the counting kernel (default {DEFAULT_SEARCH_THREADS})")
    p.add_argument("--preprocess-threads", type=int, default=4,
                   help="threads for gene prediction and HMM search (default 4)")
    p.add_argument("--filter", choices=("v1", "v1_alt", "rbh"), default=DEFAULT_FILTER,
                   help="best-hit resolution")
    p.add_argument("--input", dest="input_kind",
                   choices=("auto", "genome", "protein"), default="auto",
                   help="input type (default: guess from extension); protein input "
                        "skips gene prediction")
    p.add_argument("--archive", metavar="DIR",
                   help="persist proteins and raw HMM hits for later rebuilds")
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
    b.add_argument("-d", "--database", required=True, help="output database directory")
    b.add_argument("--source", default="", help="provenance label, e.g. 'GTDB R232 bac120'")
    _common(b)

    q = sub.add_parser("query", help="query one database against another")
    q.add_argument("-q", "--query", required=True,
                   help="query database, archive, or sequences")
    q.add_argument("-t", "--target",
                   help="target database (default: query against itself, upper triangle)")
    q.add_argument("-o", "--output",
                   help="output TSV, or a directory when the search spans more "
                        "than one partition pair (default: stdout)")
    q.add_argument("--output_style", choices=("tsv", "matrix"), default="tsv")
    q.add_argument("--emit", choices=("aai", "jaccard", "both"), default="both")
    _common(q)

    m = sub.add_parser("merge", help="merge databases into one")
    m.add_argument("-o", "--output", required=True, help="output database directory")
    m.add_argument("inputs", nargs="+", help="databases to merge")
    m.add_argument("--quiet", action="store_true")
    return p


def cmd_build(args) -> int:
    log = _log(args.quiet)
    models = ModelSet(args.hmm) if args.hmm else None
    if models:
        log(f"{len(models)} models from {args.hmm}")
        if not models.has_trusted:
            log("  note: models lack TC cutoffs; using default inclusion thresholds")

    db = _load_or_build(args.inputs, models, args, log)
    if args.source:
        db.source = args.source
    db.save(args.database)
    parts, _, man = db.stored_bytes(args.database)
    log(f"wrote {args.database}: {db.n_genomes} genomes, {db.n_partitions} partition(s), "
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

    models = ModelSet(args.hmm) if args.hmm else None
    qdb = _load_or_build(args.query, models, args, log)
    same = not args.target or args.target == args.query
    tdb = qdb if same else _load_or_build(args.target, models, args, log)

    pairs = qdb.n_genomes * tdb.n_genomes
    t0 = time.perf_counter()
    compute = None

    if args.output_style == "matrix":
        # A matrix is one dense array of every pair, so it exists only for
        # searches small enough to hold one.
        if qdb.n_partitions * tdb.n_partitions > 1:
            raise SystemExit(
                f"--output_style matrix needs the whole result in memory, and this "
                f"search is {qdb.n_partitions} x {tdb.n_partitions} partitions. "
                "Use the default TSV output."
            )
        res = search(qdb, tdb, threads=args.threads, stdev=args.do_stdev)
        _write(res, args.output, args.output_style, args.emit)
    else:
        _rows, compute = write_blocks(
            qdb, tdb, args.output, threads=args.threads, block=DEFAULT_BLOCK,
            stdev=args.do_stdev, emit=args.emit, quiet=args.quiet)

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


def cmd_merge(args) -> int:
    log = _log(args.quiet)
    written, skipped, parts = _core.merge_databases(args.output, args.inputs)
    log(f"merged {len(args.inputs)} databases -> {args.output}")
    log(f"  {written} genomes, {parts} partitions, {skipped} duplicates skipped")
    log("  no posting list was read or renumbered; ordinals were reassigned, so any "
        "stored result matrix keyed to the old order is invalidated")
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

    if opt("-m", "--hmms"):
        raise SystemExit(
            "-m/--hmms took precomputed HMMER tables as input. FastAAI 2 does not read "
            "them; supply genomes or proteins, or reuse an --archive, which stores raw "
            "hits and can be re-filtered without re-searching."
        )

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
        donors = [rest[i + 1] for i, a in enumerate(rest) if a in ("-d", "--donors")]
        recipient = opt("-r", "--recipient")
        return ["merge", "-o", recipient, recipient] + donors

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
    return {"build": cmd_build, "query": cmd_query, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
