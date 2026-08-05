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


def _write(res, out_path, style, emit):
    has_sd = res.stdev is not None
    fh = open(out_path, "w") if out_path else sys.stdout
    try:
        if style == "matrix":
            aai = res.aai
            fh.write("\t" + "\t".join(res.target_names) + "\n")
            for i, qn in enumerate(res.query_names):
                row = ["NA" if np.isnan(v) else f"{v:.2f}" for v in aai[i]]
                fh.write(qn + "\t" + "\t".join(row) + "\n")
            return

        aai = res.aai if emit in ("aai", "both") else None
        cols = ["query", "target", "shared_scps"]
        if emit in ("jaccard", "both"):
            cols.append("jaccard")
        if has_sd:
            cols.append("jaccard_sd")
        if emit in ("aai", "both"):
            cols.append("aai")
        fh.write("\t".join(cols) + "\n")
        for i, qn in enumerate(res.query_names):
            for j, tn in enumerate(res.target_names):
                row = [qn, tn, str(res.shared[i, j])]
                if emit in ("jaccard", "both"):
                    v = res.jaccard[i, j]
                    row.append("NA" if np.isnan(v) else f"{v:.10g}")
                if has_sd:
                    v = res.stdev[i, j]
                    row.append("NA" if np.isnan(v) else f"{v:.6g}")
                if emit in ("aai", "both"):
                    v = aai[i, j]
                    row.append("NA" if np.isnan(v) else f"{v:.4f}")
                fh.write("\t".join(row) + "\n")
    finally:
        if out_path:
            fh.close()


def _common(p):
    p.add_argument("--hmm", help="HMM file defining the SCP model set")
    p.add_argument("--threads", type=int, default=DEFAULT_SEARCH_THREADS,
                   help=f"threads for the counting kernel (default {DEFAULT_SEARCH_THREADS}); "
                        "scaling is memory-bound and goes negative past ~16")
    p.add_argument("--preprocess-threads", type=int, default=4,
                   help="threads for gene prediction and HMM search (default 4)")
    p.add_argument("--filter", choices=("v1", "v1_alt", "rbh"), default=DEFAULT_FILTER,
                   help="best-hit resolution; decides which protein each accession gets")
    p.add_argument("--input", dest="input_kind",
                   choices=("auto", "genome", "protein"), default="auto",
                   help="input type (default: guess from extension); protein input "
                        "skips gene prediction, ~80%% of preprocessing")
    p.add_argument("--archive", metavar="DIR",
                   help="persist proteins and raw HMM hits so preprocessing need never repeat")
    p.add_argument("--limit", type=int, help="use only the first N inputs")
    p.add_argument("--quiet", action="store_true")
    for flag in RETIRED:
        p.add_argument(f"--{flag}", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--do_stdev", action="store_true",
                   help="also report the standard deviation of Jaccard across shared "
                        "SCPs; costs one more output-width array, so off by default")
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
    q.add_argument("-o", "--output", help="output TSV (default: stdout)")
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
    models = ModelSet(args.hmm) if args.hmm else None
    qdb = _load_or_build(args.query, models, args, log)
    same = not args.target or args.target == args.query
    tdb = qdb if same else _load_or_build(args.target, models, args, log)

    t0 = time.perf_counter()
    res = search(qdb, tdb, threads=args.threads, stdev=args.do_stdev)
    dt = time.perf_counter() - t0
    pairs = len(res.query_names) * len(res.target_names)
    log(f"search {dt:.2f}s ({pairs / max(dt, 1e-9):,.0f} pairs/s)"
        f"{' [symmetric, upper triangle]' if same else ''}")

    _write(res, args.output, args.output_style, args.emit)
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

    if module == "build_db":
        return ["build", inp, "-d", opt("-d", "--database") or opt("-o", "--output")] \
            + kind + carried()
    if module == "aai_index":
        return ["query", "-q", inp, "-o", opt("-o", "--output")] + kind + carried()
    if module == "db_query":
        return ["query", "-q", opt("-q", "--query"), "-t", opt("-t", "--target"),
                "-o", opt("-o", "--output")] + carried()
    if module == "simple_query":
        return ["query", "-q", inp, "-t", opt("--target"),
                "-o", opt("-o", "--output")] + kind + carried()
    if module == "multi_query":
        qi = opt("--query_genomes") or opt("--query_proteins") or opt("--query_database")
        ti = opt("--target_genomes") or opt("--target_proteins") or opt("--target_database")
        k = ["--input", "protein"] if opt("--query_proteins") else []
        return ["query", "-q", qi, "-t", ti, "-o", opt("-o", "--output")] + k + carried()
    if module == "single_query":
        qi = opt("-qg", "--query_genome") or opt("-qp", "--query_protein")
        ti = opt("-tg", "--target_genome") or opt("-tp", "--target_protein")
        k = ["--input", "protein"] if opt("-qp", "--query_protein") else []
        return ["query", "-q", qi, "-t", ti, "-o", opt("-o", "--output")] + k + carried()
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
