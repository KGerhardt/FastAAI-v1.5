"""Does the marker set change who your nearest neighbour is?

Aggregate correlation between marker sets is not the interesting measure. Two
sets will agree that unrelated genomes are unrelated, and at a median AAI of 44%
almost every pair is unrelated, so r is dominated by the grey zone. What matters
for FastAAI's actual job — reducing candidates before a slower, exact method —
is retrieval:

  * is the top hit the same genome under both marker sets?
  * how far down the ranking is the first genome of the same genus?
  * if you keep the top N candidates, how often is the right one in there?

That last one is the number that decides whether FastAAI can sit in front of
GTDB-Tk. A candidate-reduction step is only sound if recall is ~1.0; being fast
is worthless if the answer is not in the shortlist.

Truth is GTDB R232's own taxonomy, so a marker set is scored against the
classification it is meant to serve.

    python nearest_neighbours.py <db_dir> <taxonomy.tsv.gz> [out.md]
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np  # noqa: E402

import fastaai  # noqa: E402

SETS = ("fastaai_122", "bac120", "ar53")
RANKS = {"species": 6, "genus": 5, "family": 4}
TOP_N = (1, 5, 10, 50, 100)


def taxonomy(path: Path, names: list[str]) -> dict[str, list[str]]:
    want = {}
    for n in names:
        m = re.search(r"(GC[AF]_\d+\.\d+)", n)
        if m:
            want[m.group(1)] = n
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            gid, lineage = line.rstrip("\n").split("\t")
            bare = gid.split("_", 1)[1] if gid[:3] in ("RS_", "GB_") else gid
            if bare in want:
                out[want[bare]] = lineage.split(";")
    return out


def analyse(name: str, db_dir: Path, tax: dict[str, list[str]], threads: int):
    db = fastaai.open_database(str(db_dir / name))
    res = fastaai.search(db, db, threads=threads)
    n = db.n_genomes
    aai = np.asarray(res.aai).reshape(n, n).copy()
    names = res.query_names

    # A genome is not its own neighbour.
    np.fill_diagonal(aai, -np.inf)
    aai[~np.isfinite(aai)] = -np.inf

    # Rank every target per query, best first.
    order = np.argsort(-aai, axis=1)

    keep = [i for i, g in enumerate(names) if g in tax]
    out = {"name": name, "n": n, "labelled": len(keep), "names": names,
           "order": order, "aai": aai}

    for rank, idx in RANKS.items():
        label = np.array([tax[g][idx] if g in tax else None for g in names], dtype=object)
        # Only genomes with at least one other genome sharing the label can be
        # scored: with no congener present, failing to find one is not an error.
        counts: dict[str, int] = {}
        for g in keep:
            counts[label[g]] = counts.get(label[g], 0) + 1
        scorable = [i for i in keep if counts.get(label[i], 0) > 1]

        hits = {t: 0 for t in TOP_N}
        first_rank = []
        for i in scorable:
            row = order[i]
            same = label[row] == label[i]
            pos = np.flatnonzero(same)
            r = int(pos[0]) if len(pos) else None
            first_rank.append(r if r is not None else 10 ** 9)
            for t in TOP_N:
                if r is not None and r < t:
                    hits[t] += 1
        out[rank] = {
            "scorable": len(scorable),
            "recall": {t: hits[t] / len(scorable) if scorable else 0.0 for t in TOP_N},
            "median_first_rank": float(np.median(first_rank)) if first_rank else float("nan"),
        }
    return out


def main() -> int:
    db_dir = Path(sys.argv[1])
    tax_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    threads = 8

    available = [s for s in SETS if (db_dir / s / "schema").exists()]
    if not available:
        raise SystemExit(f"no databases under {db_dir}")

    first = fastaai.open_database(str(db_dir / available[0]))
    tax = taxonomy(tax_path, first.genome_names)
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"{first.n_genomes} genomes, {len(tax)} with GTDB R232 taxonomy\n")

    results = {}
    for name in available:
        results[name] = analyse(name, db_dir, tax, threads)

    for rank in RANKS:
        emit(f"## Same-{rank} recall at N candidates")
        emit()
        emit("| marker set | scorable | " + " | ".join(f"top {t}" for t in TOP_N)
             + " | median rank of first |")
        emit("|---" * (len(TOP_N) + 3) + "|")
        for name in available:
            r = results[name][rank]
            cells = " | ".join(f"{r['recall'][t] * 100:.2f}%" for t in TOP_N)
            emit(f"| {name} | {r['scorable']} | {cells} | {r['median_first_rank']:.0f} |")
        emit()

    emit("## Do the marker sets pick the same nearest neighbour?")
    emit()
    base = results[available[0]]
    common = base["names"]
    emit("| vs fastaai_122 | same top-1 | top-5 overlap | top-10 overlap |")
    emit("|---|---|---|---|")
    for name in available[1:]:
        other = results[name]
        idx = {g: i for i, g in enumerate(other["names"])}
        shared = [i for i, g in enumerate(common) if g in idx]
        same1 = ov5 = ov10 = 0
        for i in shared:
            j = idx[common[i]]
            a1 = common[base["order"][i][0]]
            b1 = other["names"][other["order"][j][0]]
            same1 += a1 == b1
            a5 = {common[k] for k in base["order"][i][:5]}
            b5 = {other["names"][k] for k in other["order"][j][:5]}
            ov5 += len(a5 & b5) / 5
            a10 = {common[k] for k in base["order"][i][:10]}
            b10 = {other["names"][k] for k in other["order"][j][:10]}
            ov10 += len(a10 & b10) / 10
        m = len(shared)
        emit(f"| {name} | {same1 / m * 100:.2f}% | {ov5 / m * 100:.2f}% | "
             f"{ov10 / m * 100:.2f}% |")

    if out_path:
        out_path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
