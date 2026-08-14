"""v1 vs v1.5 AAI concordance figure.

  A  concordance — v1 against v1.5, with the identity line for reference.
  B  residual — v1.5 minus v1, at a scale where the deviation is legible.

Plotted in AAI rather than Jaccard: they are a transform of one another, and AAI
is the number people act on.

    python plot_concordance.py [concordance_v1_v15.tsv] [out.png]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Single series, so one hue plus recessive gray for reference marks; no legend —
# the title names the series. Values from the validated default palette.
BLUE = "#2a78d6"
INK = "#1a1a19"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"


def jaccard_to_aai(j):
    out = np.full(np.shape(j), np.nan, dtype=float)
    j = np.asarray(j, dtype=float)
    ok = np.isfinite(j) & (j > 0)
    x = np.power(-0.2607023 * np.log(j[ok]), 1.0 / 3.435)
    out[ok] = (1.810741 * np.exp(-x) - 0.3087057) * 100.0
    return out


def load(path):
    """Off-diagonal pairs only.

    A genome against itself is Jaccard 1.0 in both versions, so it carries no
    information about agreement — and the regression is unbounded above, putting
    it at AAI ~150 and compressing the real 40-70 band into a corner.
    """
    v1, v15, sh1, sh15, self_pairs = [], [], [], [], 0
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["query"] == r["target"]:
                self_pairs += 1
                continue
            v1.append(float(r["v1_jaccard"]))
            v15.append(float(r["v15_jaccard"]))
            sh1.append(int(r["v1_shared"]))
            sh15.append(int(r["v15_shared"]))
    return (np.array(v1), np.array(v15), np.array(sh1), np.array(sh15), self_pairs)


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else
               Path(__file__).parent / "concordance_v1_v15.tsv")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else
               Path(__file__).parent / "concordance_v1_v15.png")

    j1, j15, s1, s15, self_pairs = load(src)
    a1, a15 = jaccard_to_aai(j1), jaccard_to_aai(j15)
    keep = np.isfinite(a1) & np.isfinite(a15)
    a1, a15 = a1[keep], a15[keep]
    resid = a15 - a1

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(10.5, 4.9), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.26},
    )

    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK_2, labelsize=9, length=3, color=GRID)
        a.grid(True, color=GRID, linewidth=0.8, zorder=0)
        a.set_axisbelow(True)

    # ---- A: concordance -----------------------------------------------------
    lo, hi = min(a1.min(), a15.min()) - 1, max(a1.max(), a15.max()) + 1
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.5, zorder=1,
            solid_capstyle="round")
    ax.scatter(a1, a15, s=7, color=BLUE, alpha=0.28, linewidths=0, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("FastAAI 1 — AAI (%)", color=INK_2, fontsize=10)
    ax.set_ylabel("FastAAI 1.5 — AAI (%)", color=INK_2, fontsize=10)
    ax.set_title("A   Concordance", loc="left", color=INK, fontsize=11.5,
                 fontweight="bold", pad=10)
    ax.annotate("identity", xy=(lo + (hi - lo) * 0.72, lo + (hi - lo) * 0.72),
                xytext=(6, -14), textcoords="offset points",
                color=MUTED, fontsize=9)

    # ---- B: residual --------------------------------------------------------
    bx.axhline(0, color=MUTED, linewidth=1.5, zorder=1)
    bx.scatter(a1, resid, s=7, color=BLUE, alpha=0.28, linewidths=0, zorder=2)
    span = np.abs(resid).max() * 1.25
    bx.set_ylim(-span, span)
    bx.set_xlim(lo, hi)
    bx.set_xlabel("FastAAI 1 — AAI (%)", color=INK_2, fontsize=10)
    bx.set_ylabel("v1.5 − v1  (AAI percentage points)", color=INK_2, fontsize=10)
    bx.set_title("B   Residual", loc="left", color=INK, fontsize=11.5,
                 fontweight="bold", pad=10)

    med = float(np.median(np.abs(resid)))
    mx = float(np.abs(resid).max())
    bx.annotate(
        f"median |Δ| {med:.4f} pp\nmax |Δ| {mx:.4f} pp",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
        color=INK_2, fontsize=9,
    )

    mism = int((s1 != s15).sum())
    fig.suptitle(
        f"FastAAI 1 vs FastAAI 1.5   ·   {len(a1):,} pairs   ·   "
        f"{mism} shared-SCP differences",
        x=0.008, y=0.978, ha="left", color=INK, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.008, 0.015,
        f"120 Firmicutes genomes, all-vs-all; {self_pairs} self-comparisons excluded.",
        color=MUTED, fontsize=8.4, va="bottom",
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.175)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"  pairs {len(a1):,} ({self_pairs} self-comparisons excluded)")
    print(f"  median |Δ| {med:.2e} pp   max |Δ| {mx:.2e} pp")
    print(f"  shared-SCP differences: {mism}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
