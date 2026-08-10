"""Build one database per marker set over a whole archive, and save them.

Separated from the analysis because HMM search over 2,943 genomes takes tens of
minutes per marker set and the nearest-neighbour questions want re-running.

    python build_marker_dbs.py <archive> <marker_dir> <outdir> [threads]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import fastaai  # noqa: E402
from fastaai.archive import genome_names, read_proteins  # noqa: E402
from fastaai.search import ModelSet, resolve_hits, search_hits  # noqa: E402


def main() -> int:
    archive, markers, out = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    out.mkdir(parents=True, exist_ok=True)

    order = genome_names(archive)
    print(f"{len(order)} genomes", flush=True)

    sets = {
        "fastaai_122": ModelSet(),
        "bac120": ModelSet(str(markers / "bac120.hmm")),
        "ar53": ModelSet(str(markers / "ar53.hmm")),
    }

    for name, ms in sets.items():
        dest = out / name
        if (dest / "schema").exists():
            print(f"{name}: already built, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        db = fastaai.Database(ms.accessions)
        db.models = ms.fingerprint
        for i, g in enumerate(order, 1):
            prots = read_proteins(archive, g)
            scps = resolve_hits(search_hits(prots, ms, cpus=threads), "v1")
            payload = [(ms.acc_index[a], prots[p].encode()) for p, a in scps.items()]
            if payload:
                db.add_genome(g, payload)
            if i % 250 == 0:
                el = time.perf_counter() - t0
                print(f"  {name}: {i}/{len(order)}  {el:.0f}s  "
                      f"(eta {el / i * (len(order) - i):.0f}s)", flush=True)
        db.seal()
        db.filter_mode = "v1"
        db.source = f"Firmicutes archive, {name}"
        db.save(str(dest))
        print(f"{name}: {db.n_genomes} genomes in {time.perf_counter() - t0:.0f}s "
              f"-> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
