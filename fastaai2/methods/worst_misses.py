"""How bad are the worst misses?

Recall saturates: a same-genus genome is the top hit ~99% of the time and is in
the top 5 essentially always. That says the median case is solved and stops
being informative — which makes the tail the whole question. A candidate
reduction step is characterised by its failures, not its averages, because the
failures are what get silently dropped before the exact method ever sees them.

So this reports the shape of the tail rather than its centre:

  * the rank of the first correct hit at P50/P90/P99/max, not the median alone
  * every genome whose correct neighbour falls outside the top N
  * what those genomes retrieved instead, and how much SCP signal they carried

The last column is the one that matters operationally. If misses concentrate in
genomes with few markers, a shortlist can be widened for exactly those and the
failure mode is manageable. If misses are spread across well-covered genomes,
the ranking itself is unreliable and no shortlist size fixes it.

    python worst_misses.py <db_dir> <taxonomy.tsv.gz> [rank] [cutoff]
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
RANK_INDEX = {"species": 6, "genus": 5, "family": 4}


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


def tail(name: str, db_dir: Path, tax_path: Path, rank: str, cutoff: int, threads=8):
    db = fastaai.open_database(str(db_dir / name))
    res = fastaai.search(db, db, threads=threads)
    n = db.n_genomes
    names = res.query_names
    aai = np.asarray(res.aai).reshape(n, n).copy()
    scp = np.asarray(db.scp_counts())

    np.fill_diagonal(aai, -np.inf)
    aai[~np.isfinite(aai)] = -np.inf
    order = np.argsort(-aai, axis=1)

    tax = taxonomy(tax_path, names)
    idx = RANK_INDEX[rank]
    label = np.array([tax[g][idx] if g in tax else None for g in names], dtype=object)

    counts: dict[str, int] = {}
    for g in names:
        if g in tax:
            counts[tax[g][idx]] = counts.get(tax[g][idx], 0) + 1
    scorable = [i for i, g in enumerate(names)
                if g in tax and counts.get(label[i], 0) > 1]

    first = []
    for i in scorable:
        row = order[i]
        pos = np.flatnonzero(label[row] == label[i])
        first.append(int(pos[0]) if len(pos) else n)  # n = never found
    first = np.array(first)

    misses = [(scorable[k], int(first[k])) for k in np.argsort(-first)
              if first[k] >= cutoff]
    return {
        "name": name, "scorable": len(scorable), "first": first,
        "misses": misses, "names": names, "order": order, "aai": aai,
        "label": label, "scp": scp, "tax": tax,
    }


def main() -> int:
    db_dir = Path(sys.argv[1])
    tax_path = Path(sys.argv[2])
    rank = sys.argv[3] if len(sys.argv) > 3 else "genus"
    cutoff = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    available = [s for s in SETS if (db_dir / s / "schema").exists()]
    print(f"rank = {rank}, a miss is the first correct hit at position >= {cutoff}\n")

    for name in available:
        r = tail(name, db_dir, tax_path, rank, cutoff)
        f = r["first"]
        pct = [50, 90, 99, 99.9]
        qs = " ".join(f"P{p}={np.percentile(f, p):.0f}" for p in pct)
        print(f"## {name}  ({r['scorable']} scorable)")
        print(f"  rank of first correct hit: {qs}  max={f.max()}")
        print(f"  misses (>= {cutoff}): {len(r['misses'])} "
              f"({len(r['misses']) / r['scorable'] * 100:.2f}%)")

        med_scp = np.median(r["scp"])
        if r["misses"]:
            miss_scp = np.array([r["scp"][i] for i, _ in r["misses"]])
            print(f"  SCPs carried — all genomes median {med_scp:.0f}, "
                  f"missing genomes median {np.median(miss_scp):.0f}")
            print(f"\n  {'genome':<34} {'rank':>6} {'SCPs':>5}  {'top hit instead':<30} {'AAI':>6}")
            for i, pos in r["misses"][:12]:
                top = r["order"][i][0]
                lab_i = (r["label"][i] or "?")[3:]
                lab_t = (r["label"][top] or "?")[3:]
                shown = "never" if pos >= len(r["names"]) else str(pos)
                print(f"  {r['names'][i][:34]:<34} {shown:>6} {r['scp'][i]:>5}  "
                      f"{lab_i[:14]:<14} -> {lab_t[:13]:<13} {r['aai'][i, top]:>6.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
