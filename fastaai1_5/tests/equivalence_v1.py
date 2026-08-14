"""Equivalence harness: FastAAI 1.5 against FastAAI 1 on the same genomes.

Run as a script, not under pytest — it needs a v1 output file and a crystal
set. The crystals are published alongside the release as a tar.gz, so the
evidence behind this harness travels with the code rather than needing a 1.7 GB
collection of stored proteins regenerated locally.

    python equivalence_v1.py <v1_results_dir> <crystal_dir>

The v1-compatible run passes `alphabet=V1_ALPHABET` so the 21-symbol alphabet is
applied at k-merisation. Crystals hold sequences, not k-mers, so a single
crystal set serves both runs.

FastAAI 1.5 differs from v1 in one place that changes numbers: **v1.5 excludes the
stop codon `*` from the k-mer alphabet, and v1 does not.**

That is a *fix*, not a deviation. v1 has two k-merizers. `unique_kmers`
(fastaai.py:1421) indexes tetramers through a `kmer_index` lookup, which by
construction admits only permissible symbols — the intended behaviour. But
`kmer_index` is never defined and `unique_kmers` is never called; invoking it
raises NameError. The function that actually runs is `unique_kmer_simple_key`
(fastaai.py:1139, called at 1233), a numpy transform that takes `ord()` of every
character with no symbol check. When the numpy rewrite replaced the lookup table
it silently dropped the filtering, so every v1 database carries one spurious
tetramer per SCP — the window spanning the terminal stop.

Confirmed against v1's own `genome_acc_kmer_counts`: over 4,021 stored counts the
21-symbol encoding matches 4,019 exactly, while 20-symbol is short by exactly one
on 4,017 of them.

The harness therefore runs twice:

  bug-compatible   21-symbol alphabet. Reproduces v1 *including* its stop-codon
                   bug. Any difference here is a fault in FastAAI 1.5 — this is the
                   pass that proves the engine correct.
  shipped          20-symbol alphabet. The difference from the first pass is the
                   size of the correction, measured rather than assumed.

v1 rounds `avg_jacc_sim` to 4 decimals on output, so agreement can only be
asserted to ~5e-5. That is a limit of v1's output format, not of either engine.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fastaai.ingest import genome_name  # noqa: E402
from fastaai.pipeline import build_from_crystals, search  # noqa: E402
from fastaai.search import ModelSet  # noqa: E402

#: v1's effective alphabet: the 20 residues plus the stop codon its numpy
#: k-merizer never filtered out.
V1_ALPHABET = "*ACDEFGHIKLMNPQRSTVWY"
ROUNDING = 5e-5  # v1 writes 4 decimal places


def load_v1(results_dir: Path) -> dict[tuple[str, str], tuple[float, int]]:
    """v1 writes one `<query>_results.txt` per query genome."""
    out: dict[tuple[str, str], tuple[float, int]] = {}
    files = sorted(results_dir.rglob("*_results.txt"))
    if not files:
        raise SystemExit(f"no v1 results under {results_dir}")
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                j, n = row["avg_jacc_sim"], row["num_shared_SCPs"]
                if j == "N/A":
                    continue
                # v1 keys on the full filename; v1.5 strips FASTA and compression
                # suffixes. Normalise so the two agree on identity.
                out[(genome_name(row["query"]), genome_name(row["target"]))] = (
                    float(j), int(n))
    return out


def compare(label: str, v1: dict, res, names: list[str]) -> dict:
    idx = {n: i for i, n in enumerate(names)}
    diffs, shared_mismatch, missing, n = [], 0, 0, 0
    for (q, t), (vj, vs) in v1.items():
        if q not in idx or t not in idx:
            missing += 1
            continue
        i, jx = idx[q], idx[t]
        mj, ms = res.jaccard[i, jx], int(res.shared[i, jx])
        if np.isnan(mj):
            missing += 1
            continue
        diffs.append(abs(mj - vj))
        if ms != vs:
            shared_mismatch += 1
        n += 1
    d = np.asarray(diffs)
    stats = {
        "pairs": n,
        "unmatched": missing,
        "shared_mismatch": shared_mismatch,
        "max_abs": float(d.max()) if n else float("nan"),
        "mean_abs": float(d.mean()) if n else float("nan"),
        "within_rounding": float((d <= ROUNDING).mean() * 100) if n else float("nan"),
    }
    print(f"\n{label}")
    print(f"  pairs compared        : {stats['pairs']:,}")
    print(f"  unmatched / NaN       : {stats['unmatched']:,}")
    print(f"  shared-SCP mismatches : {stats['shared_mismatch']:,}")
    print(f"  max |delta jaccard|   : {stats['max_abs']:.3e}")
    print(f"  mean |delta jaccard|  : {stats['mean_abs']:.3e}")
    print(f"  within v1's rounding  : {stats['within_rounding']:.2f}%")
    return stats


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    v1_dir, crystals = Path(sys.argv[1]), Path(sys.argv[2])
    models = ModelSet()

    v1 = load_v1(v1_dir)
    subset = {q for q, _ in v1} | {t for _, t in v1}
    print(f"v1: {len(v1):,} pairs over {len(subset)} genomes")

    print("\nbuilding v1.5 with v1-compatible settings "
          f"(alphabet={V1_ALPHABET!r}, filter=v1) ...")
    db = build_from_crystals(crystals, models, alphabet=V1_ALPHABET, only=subset)
    compat = search(db, db, threads=8)
    a = compare("bug-compatible — any difference here is a FastAAI 1.5 fault", v1,
                compat, db.genome_names)

    print("\nbuilding v1.5 with shipped settings (20-symbol alphabet, filter=v1) ...")
    db2 = build_from_crystals(crystals, models, only=subset)
    shipped = search(db2, db2, threads=8)
    b = compare("shipped — difference here is the size of the stop-codon fix",
                v1, shipped, db2.genome_names)

    print("\n--- verdict ---")
    ok = a["max_abs"] <= ROUNDING and a["shared_mismatch"] == 0
    print(f"  engine reproduces FastAAI 1 bug-for-bug : {'YES' if ok else 'NO'}")
    print(f"    (limit of assertion is v1's own 4-decimal output rounding, "
          f"{ROUNDING:.0e})")
    print(f"  size of the stop-codon correction       : mean {b['mean_abs']:.2e}, "
          f"max {b['max_abs']:.2e} Jaccard")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
