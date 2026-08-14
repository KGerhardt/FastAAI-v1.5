"""One genome set, three marker sets.

Asks what changes when the SCP set is swapped — the thing FastAAI 1.5 made
possible and the thing a GTDB-keyed v2 depends on:

    fastaai_122   FastAAI 1's bundled set, the default
    bac120        GTDB's bacterial markers   (6 Pfam + 114 TIGRFAM)
    ar53          GTDB's archaeal markers   (12 Pfam +  41 TIGRFAM)

ar53 is included precisely because these are bacteria: it should recover few
markers and is the control for what a wrong-domain marker set looks like.

Runs against an archive, so gene prediction is not repeated — only HMM search,
which is the stage a marker swap actually changes.

    python bench_marker_sets.py <archive> <marker_dir> [n_genomes] [threads]
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np  # noqa: E402

import fastaai  # noqa: E402
from fastaai.archive import genome_names, read_proteins  # noqa: E402
from fastaai.search import ModelSet, resolve_hits, search_hits  # noqa: E402


def build(name, models, proteins, order, threads):
    """HMM search -> best hit -> sealed database, timing the stages separately."""
    t0 = time.perf_counter()
    scps = {}
    for g in order:
        hits = search_hits(proteins[g], models, cpus=threads)
        scps[g] = resolve_hits(hits, "v1")
    t_search = time.perf_counter() - t0

    t0 = time.perf_counter()
    db = fastaai.Database(models.accessions)
    db.models = models.fingerprint
    kept = 0
    for g in order:
        payload = [(models.acc_index[a], proteins[g][p].encode())
                   for p, a in scps[g].items()]
        if payload:
            db.add_genome(g, payload)
            kept += 1
    db.seal()
    db.filter_mode = "v1"
    t_build = time.perf_counter() - t0

    counts = db.scp_counts()
    return {
        "name": name,
        "models": len(models),
        "genomes": kept,
        "hmm_s": t_search,
        "build_s": t_build,
        "scps_median": statistics.median(counts) if counts else 0,
        "scps_min": min(counts) if counts else 0,
        "index_mb": db.index_bytes() / 1e6,
        "db": db,
    }


def main() -> int:
    archive = Path(sys.argv[1])
    markers = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    order = genome_names(archive)[:n]
    proteins = {g: read_proteins(archive, g) for g in order}
    print(f"{len(order)} genomes, {sum(len(p) for p in proteins.values()):,} proteins\n")

    sets = [
        ("fastaai_122", ModelSet()),
        ("bac120", ModelSet(str(markers / "bac120.hmm"))),
        ("ar53", ModelSet(str(markers / "ar53.hmm"))),
    ]

    results = []
    for name, ms in sets:
        r = build(name, ms, proteins, order, threads)
        t0 = time.perf_counter()
        res = fastaai.search(r["db"], r["db"], threads=threads)
        r["search_s"] = time.perf_counter() - t0
        pairs = r["genomes"] ** 2
        r["per_thread"] = pairs / max(r["search_s"], 1e-9) / threads
        jac = np.asarray(res.jaccard).reshape(r["genomes"], r["genomes"])
        off = ~np.eye(r["genomes"], dtype=bool)
        r["aai"] = np.asarray(res.aai).reshape(r["genomes"], r["genomes"])
        vals = r["aai"][off]
        r["aai_median"] = float(np.nanmedian(vals))
        r["shared_median"] = float(np.median(np.asarray(res.shared).reshape(
            r["genomes"], r["genomes"])[off]))
        r["unmeasured"] = float(np.mean(~np.isfinite(jac[off])) * 100)
        r["names"] = res.query_names
        results.append(r)

    w = 14
    print(f"{'marker set':<{w}} {'models':>7} {'SCPs/gen':>9} {'HMM s':>8} "
          f"{'index MB':>9} {'pairs/s/thr':>12} {'med AAI':>8} {'no est %':>9}")
    for r in results:
        print(f"{r['name']:<{w}} {r['models']:>7} {r['scps_median']:>9.0f} "
              f"{r['hmm_s']:>8.1f} {r['index_mb']:>9.1f} {r['per_thread']:>12,.0f} "
              f"{r['aai_median']:>8.2f} {r['unmeasured']:>9.2f}")

    print("\nAAI agreement between marker sets (off-diagonal pairs):")
    base = results[0]
    for other in results[1:]:
        common = [g for g in other["names"] if g in set(base["names"])]
        bi = {g: i for i, g in enumerate(base["names"])}
        oi = {g: i for i, g in enumerate(other["names"])}
        a = np.array([[base["aai"][bi[x], bi[y]] for y in common] for x in common])
        b = np.array([[other["aai"][oi[x], oi[y]] for y in common] for x in common])
        m = ~np.eye(len(common), dtype=bool) & np.isfinite(a) & np.isfinite(b)
        d = b[m] - a[m]
        r = np.corrcoef(a[m], b[m])[0, 1]
        print(f"  {other['name']:<12} vs fastaai_122 on {len(common)} genomes: "
              f"median Δ {np.median(d):+.2f}, IQR {np.percentile(d, 75) - np.percentile(d, 25):.2f}, "
              f"r = {r:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
