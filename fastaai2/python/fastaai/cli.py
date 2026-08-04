"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .ingest import find_genomes
from .pipeline import DEFAULT_SEARCH_THREADS, build_database, preprocess, search
from .search import DEFAULT_FILTER, ModelSet


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fastaai",
        description="Average amino acid identity from single-copy protein tetramers.",
    )
    p.add_argument("genomes", help="FASTA file or directory of FASTA files")
    p.add_argument("--hmm", required=True, help="HMM file defining the SCP model set")
    p.add_argument("-o", "--output", help="output TSV (default: stdout)")
    p.add_argument(
        "--preprocess-threads",
        type=int,
        default=4,
        help="threads for prediction and HMM search (default: 4)",
    )
    p.add_argument(
        "--search-threads",
        type=int,
        default=DEFAULT_SEARCH_THREADS,
        help=(
            f"threads for the counting kernel (default: {DEFAULT_SEARCH_THREADS}). "
            "Scaling is memory-bound; more than ~16 is usually slower, not faster."
        ),
    )
    p.add_argument(
        "--filter",
        choices=("v1", "v1_alt", "rbh"),
        default=DEFAULT_FILTER,
        help=(
            "best-hit resolution. 'v1' reproduces FastAAI 1 as shipped (default). "
            "'rbh' is strict reciprocal best hit — the stated intent, and harsher. "
            "These give different SCP sets and therefore different AAI."
        ),
    )
    p.add_argument("--limit", type=int, help="use only the first N genomes")
    p.add_argument("--archive", metavar="DIR",
                   help="persist proteins and raw HMM hits so the run need never "
                        "repeat; re-searching a different model set then needs no "
                        "gene prediction, and re-filtering needs no search")
    p.add_argument("--from-archive", metavar="DIR",
                   help="rebuild from an archive, skipping prediction and search")
    p.add_argument(
        "--emit",
        choices=("aai", "jaccard", "both"),
        default="both",
        help="values to write (default: both; raw Jaccard is stored uncensored)",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))

    if args.from_archive:
        from .pipeline import build_from_archive
        t0 = time.perf_counter()
        db = build_from_archive(args.from_archive, mode=args.filter)
        log(f"rebuilt {db.n_genomes} genomes from archive in "
            f"{time.perf_counter()-t0:.1f}s ({db.n_partitions} partitions, "
            f"{db.index_bytes()/1e6:.0f} MB, filter={args.filter})")
        res = search(db, db, threads=args.search_threads)
        return _write(args, res, log)

    paths = find_genomes(args.genomes)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no FASTA files found under {args.genomes}", file=sys.stderr)
        return 1

    models = ModelSet(args.hmm)
    log(f"{len(models)} models from {args.hmm}")
    if not models.has_trusted:
        log("  note: models lack TC cutoffs; using default inclusion thresholds")
    log(f"{len(paths)} genomes, filter={args.filter}")

    t0 = time.perf_counter()
    last = [0.0]

    def progress(done: int, total: int, rec):
        if rec.error:
            log(f"  ! {rec.name}: {rec.error}")
        now = time.perf_counter()
        if now - last[0] > 2.0 or done == total:
            last[0] = now
            log(f"  preprocessed {done}/{total} ({now - t0:.0f}s)")

    records = preprocess(
        paths, models, mode=args.filter,
        threads=args.preprocess_threads, progress=progress,
        archive_root=args.archive,
    )
    if args.archive:
        from .archive import size_report
        sz = size_report(args.archive)
        log(f"archive: {args.archive}  proteins {sz['proteins_mb']:.0f} MB, "
            f"hits {sz['hits_mb']:.0f} MB")
    t_pre = time.perf_counter() - t0
    failed = [r for r in records if r.error]
    log(f"preprocessing done in {t_pre:.1f}s ({len(failed)} failed)")

    db, skipped = build_database(records, models, filter_mode=args.filter)
    if skipped:
        log(f"{len(skipped)} genomes excluded for having no usable SCPs")
    log(f"index: {db.n_genomes} genomes, {db.index_bytes() / 1e6:.1f} MB")

    t1 = time.perf_counter()
    res = search(db, db, threads=args.search_threads)
    t_search = time.perf_counter() - t1
    pairs = db.n_genomes * db.n_genomes
    log(f"search done in {t_search:.2f}s ({pairs / max(t_search, 1e-9):,.0f} pairs/s)")

    return _write(args, res, log)


def _write(args, res, log):
    aai = res.aai if args.emit in ("aai", "both") else None
    out = open(args.output, "w") if args.output else sys.stdout
    try:
        cols = ["query", "target", "shared_scps"]
        if args.emit in ("jaccard", "both"):
            cols.append("jaccard")
        if args.emit in ("aai", "both"):
            cols.append("aai")
        out.write("\t".join(cols) + "\n")
        for i, qn in enumerate(res.query_names):
            for j, tn in enumerate(res.target_names):
                row = [qn, tn, str(res.shared[i, j])]
                if args.emit in ("jaccard", "both"):
                    v = res.jaccard[i, j]
                    row.append("NA" if np.isnan(v) else f"{v:.10g}")
                if args.emit in ("aai", "both"):
                    v = aai[i, j]
                    row.append("NA" if np.isnan(v) else f"{v:.4f}")
                out.write("\t".join(row) + "\n")
    finally:
        if args.output:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
