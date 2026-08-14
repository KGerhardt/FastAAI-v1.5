"""Which HMMER score reproduces FastAAI 1's SCP sets?

Re-searches archived proteins under each candidate scoring rule and compares the
resulting shared-SCP counts against v1's. Needs no gene prediction, so it runs in
minutes on a subset rather than repeating a half-hour preprocess.

    python score_probe.py <v1_results_dir> <archive_dir> <hmm_file> [n_genomes]
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fastaai import _core  # noqa: E402
from fastaai.archive import read_models, read_proteins  # noqa: E402
from fastaai.ingest import genome_name  # noqa: E402
from fastaai.pipeline import search  # noqa: E402
from fastaai.search import ModelSet, resolve_hits, search_hits  # noqa: E402

CANDIDATES = ["domain_v1", "domain", "sequence"]


def load_v1(results_dir: Path):
    out = {}
    for f in sorted(results_dir.rglob("*_results.txt")):
        with open(f) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row["avg_jacc_sim"] == "N/A":
                    continue
                out[(genome_name(row["query"]), genome_name(row["target"]))] = (
                    float(row["avg_jacc_sim"]), int(row["num_shared_SCPs"]))
    return out


def main() -> int:
    v1_dir, archive, hmm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 40

    v1 = load_v1(v1_dir)
    genomes = sorted({q for q, _ in v1})[:limit]
    pairs = {(q, t): v for (q, t), v in v1.items() if q in set(genomes) and t in set(genomes)}
    print(f"v1: {len(pairs):,} pairs over {len(genomes)} genomes")

    models = ModelSet(hmm)
    accessions = read_models(archive)
    acc_index = {a: i for i, a in enumerate(accessions)}

    print("loading archived proteins ...")
    prots = {g: read_proteins(archive, g) for g in genomes}
    print(f"  {sum(len(p) for p in prots.values()):,} proteins\n")

    print(f"{'score rule':<14} {'search':>8} {'shared mismatch':>16} "
          f"{'max |dJ|':>11} {'mean |dJ|':>11} {'within 5e-5':>12}")
    for kind in CANDIDATES:
        t0 = time.perf_counter()
        db = _core.Database(accessions)
        for g in genomes:
            hits = search_hits(prots[g], models, cpus=8, score=kind)
            assign = resolve_hits(hits, "v1")
            payload = [(acc_index[a], prots[g][p].encode())
                       for p, a in assign.items() if a in acc_index]
            if payload:
                db.add_genome(g, payload)
        db.seal()
        t = time.perf_counter() - t0
        res = search(db, db, threads=8)
        idx = {n: i for i, n in enumerate(db.genome_names)}

        d, mism, n = [], 0, 0
        for (q, tt), (vj, vs) in pairs.items():
            if q not in idx or tt not in idx:
                continue
            i, j = idx[q], idx[tt]
            if np.isnan(res.jaccard[i, j]):
                continue
            d.append(abs(res.jaccard[i, j] - vj))
            mism += int(res.shared[i, j]) != vs
            n += 1
        d = np.asarray(d)
        print(f"{kind:<14} {t:>7.1f}s {mism:>9,}/{n:<6,} {d.max():>11.3e} "
              f"{d.mean():>11.3e} {(d <= 5e-5).mean()*100:>11.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
