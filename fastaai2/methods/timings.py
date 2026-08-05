"""Stage timings for the release README.

Measures the three stages separately rather than quoting a total, because the
split is the whole story: preprocessing dominates a cold run and the search
dominates a warm one, and only the second is what FastAAI 1.5 changes.

  predict      Prodigal via pyrodigal, per genome
  hmmsearch    pyhmmer against the SCP model set, per genome
  search       the AAI kernel, v1 against v1.5 on identical genomes

Prediction and HMM search are timed inside `preprocess_one`'s two calls rather
than by differencing a protein-input run against a genome-input one — the
difference would also absorb FASTA parsing and archive writes.

    python timings.py <genome_dir> <models.hmm> [n_genomes] [threads]
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fastaai.ingest import find_genomes  # noqa: E402
from fastaai.predict import predict_proteins  # noqa: E402
from fastaai.search import ModelSet, resolve_hits, search_hits  # noqa: E402


def main() -> int:
    gdir = Path(sys.argv[1])
    hmm = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    paths = find_genomes(gdir)[:n]
    if not paths:
        raise SystemExit(f"no genomes under {gdir}")

    t0 = time.perf_counter()
    models = ModelSet(hmm)
    t_models = time.perf_counter() - t0

    predict, hmmer, n_prot, n_scp = [], [], [], []
    for p in paths:
        t0 = time.perf_counter()
        proteins, _ = predict_proteins(p)
        predict.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        hits = search_hits(proteins, models, cpus=threads)
        hmmer.append(time.perf_counter() - t0)

        n_prot.append(len(proteins))
        n_scp.append(len(resolve_hits(hits, "v1")))

    def row(name, xs):
        return (name, statistics.median(xs), sum(xs))

    rows = [row("predict (pyrodigal)", predict), row("hmmsearch (pyhmmer)", hmmer)]
    total = sum(predict) + sum(hmmer)

    print(f"{len(paths)} genomes, cpus={threads} for HMM search")
    if threads != 1:
        print("  note: the pipeline gives each genome one thread and parallelises")
        print("        across genomes, so cpus=1 is the figure that composes")
    print(f"model set load: {t_models * 1000:.0f} ms ({len(models)} models)")
    print(f"proteins/genome: median {int(statistics.median(n_prot)):,}   "
          f"SCPs/genome: median {int(statistics.median(n_scp))}\n")
    print(f"{'stage':<24} {'median s/genome':>16} {'total s':>10} {'share':>8}")
    for name, med, tot in rows:
        print(f"{name:<24} {med:>16.2f} {tot:>10.1f} {tot / total * 100:>7.1f}%")
    print(f"{'preprocessing total':<24} {total / len(paths):>16.2f} {total:>10.1f}")

    out = Path(__file__).parent / "timings_preprocessing.tsv"
    with open(out, "w") as fh:
        fh.write("stage\tmedian_s_per_genome\ttotal_s\tn_genomes\tthreads\n")
        for name, med, tot in rows:
            fh.write(f"{name}\t{med:.4f}\t{tot:.2f}\t{len(paths)}\t{threads}\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
