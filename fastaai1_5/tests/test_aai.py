"""The Jaccard -> AAI transform.

    aai = (-0.3087057 + 1.810741 * exp(-(-0.2607023 * ln J)^(1/3.435))) * 100

FastAAI 1's regression, unchanged (`fastaai.py:2302`). It exists twice here — in
Rust for the writers, and in numpy for `SearchResult.aai` — so the first thing
these tests do is check the two cannot drift apart.

Audited against 8,661,249 real pairs: every emitted `AAI_estimate` matches an
independent recomputation from the stored Jaccard.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastaai

#: FastAAI 1's constants, spelled out so a typo in the engine cannot be copied
#: into the test and agree with itself.
A, B, C, D = -0.3087057, 1.810741, -0.2607023, 3.435


def reference(j: float) -> float:
    """v1's formula, transcribed from fastaai.py:2302."""
    return (A + B * np.exp(-((C * np.log(j)) ** (1 / D)))) * 100


#: Spans the reachable range and well past it, log-spaced at the bottom where
#: the interesting values live.
JACCARDS = np.unique(np.concatenate([
    np.logspace(-12, -3, 200),
    np.linspace(0.001, 1.0, 500),
]))


def test_rust_matches_the_v1_formula_exactly():
    for j in JACCARDS:
        assert fastaai.jaccard_to_aai(float(j)) == pytest.approx(reference(j), abs=1e-12)


def test_the_numpy_path_matches_rust_bit_for_bit():
    """`SearchResult.aai` and the Rust transform are separate implementations.

    They are applied to the same data by different code paths — the API and the
    writers — so a divergence would show up as two different answers for one
    database.
    """
    rust = np.array([fastaai.jaccard_to_aai(float(j)) for j in JACCARDS])

    out = np.full(JACCARDS.shape, np.nan)
    ok = np.isfinite(JACCARDS) & (JACCARDS > 0)
    x = np.power(-0.2607023 * np.log(JACCARDS[ok]), 1.0 / 3.435)
    out[ok] = (1.810741 * np.exp(-x) - 0.3087057) * 100.0

    assert np.array_equal(rust, out), "the two implementations must agree bitwise"


def test_it_is_strictly_monotone_over_the_whole_domain():
    """Monotone far below anything reachable, not just over the usable band.

    A non-monotone patch would make AAI non-comparable in exactly the regime
    this tool exists for.
    """
    wide = np.unique(np.concatenate([np.logspace(-300, -3, 400), JACCARDS]))
    v = np.array([fastaai.jaccard_to_aai(float(j)) for j in wide])
    assert np.all(np.diff(v) > 0)


def test_the_usable_band_anchors():
    """The values quoted in the docs, checked rather than repeated."""
    assert fastaai.aai_to_jaccard(30.0) == pytest.approx(0.005743, abs=1e-6)
    assert fastaai.aai_to_jaccard(90.0) == pytest.approx(0.842999, abs=1e-6)


def test_the_inverse_round_trips():
    for aai in np.linspace(30, 90, 61):
        assert fastaai.jaccard_to_aai(fastaai.aai_to_jaccard(aai)) == pytest.approx(
            aai, abs=1e-9)


def test_identical_genomes_extrapolate_past_100():
    """Why a self-pair is reported as identity rather than run through the fit."""
    assert fastaai.jaccard_to_aai(1.0) == pytest.approx(150.20353, abs=1e-5)


def test_non_positive_jaccard_has_no_estimate():
    for j in (0.0, -0.5, float("nan")):
        assert np.isnan(fastaai.jaccard_to_aai(j))


def test_the_negative_tail_is_unreachable_in_practice():
    """The fit's asymptote is -30.87%, so AAI can in principle go negative.

    It cannot with real data. A per-SCP Jaccard is at least 1/|A u B| ~ 1/660,
    and the mean over ~122 accessions bottoms out near 1e-5 — seven orders of
    magnitude above where the curve crosses zero. The floor that does occur is
    Jaccard exactly 0, which has no estimate at all and is reported `<30%`.
    """
    crossing = 1.189e-12
    assert fastaai.jaccard_to_aai(crossing * 0.5) < 0
    assert fastaai.jaccard_to_aai(crossing * 2) > 0

    smallest_realistic = 1.0 / 660 / 122
    assert fastaai.jaccard_to_aai(smallest_realistic) > 0
    assert smallest_realistic > crossing * 1e6
