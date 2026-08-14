"""numpy 1.x / 2.x compatibility.

numpy 2.0 removed a long list of aliases. `np.float_` in particular broke a lot
of downstream code, FastAAI 1 included, and the resulting split — code that runs
on one major but not the other — is a maintenance cost out of all proportion to
what the aliases bought.

So the package restricts itself to spellings valid in both. This is a guard
against reintroducing one, which is easy to do by reflex: `np.NaN` and `np.nan`
look interchangeable, and only one of them still exists.

Verified beyond this file: the full suite passes under numpy 1.26.4 and 2.4.6,
and a query produces byte-identical TSV, matrix and API output under both.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "python" / "fastaai"

#: Removed in numpy 2.0. Not exhaustive — these are the ones with a surviving
#: lookalike, so they are the ones reached for by accident.
REMOVED_IN_2 = [
    "float_", "complex_", "longfloat", "singlecomplex", "cfloat", "longcomplex",
    "string_", "unicode_",
    "NaN", "NAN", "Inf", "Infinity", "infty", "NINF", "PINF", "NZERO", "PZERO",
    "round_", "product", "cumproduct", "alltrue", "sometrue",
    "int0", "uint0", "bool8", "object0", "str0", "void0",
    "mat", "source", "lookfor", "issctype", "asfarray", "byte_bounds",
    "set_string_function", "deprecate", "safe_eval", "trapz", "in1d",
    "row_stack", "msort", "find_common_type", "obj2sctype", "sctype2char",
]


def _sources():
    return sorted(PACKAGE.rglob("*.py"))


def test_the_package_has_python_sources_to_scan():
    """Guards the guard: a bad path would make every scan below vacuous."""
    assert len(_sources()) >= 5


@pytest.mark.parametrize("alias", REMOVED_IN_2)
def test_no_numpy_alias_removed_in_2_0(alias):
    pattern = re.compile(rf"\bnp\.{re.escape(alias)}\b|\bnumpy\.{re.escape(alias)}\b")
    offenders = [
        f"{f.name}:{i}"
        for f in _sources()
        for i, line in enumerate(f.read_text().split("\n"), 1)
        if pattern.search(line) and not line.strip().startswith("#")
    ]
    assert not offenders, f"np.{alias} was removed in numpy 2.0; used at {offenders}"


def test_every_numpy_symbol_used_exists_in_this_numpy():
    """Whichever major is installed, everything the package reaches for is there.

    Catches the reverse case too — a spelling that only exists in 2.x would fail
    this run under 1.x.
    """
    used = set()
    for f in _sources():
        used.update(re.findall(r"\bnp\.([A-Za-z_][A-Za-z0-9_]*)", f.read_text()))
    missing = sorted(s for s in used if not hasattr(np, s))
    assert not missing, f"not present in numpy {np.__version__}: {missing}"


def test_the_values_we_depend_on_behave_the_same_in_both_majors():
    """The specific numpy behaviours the AAI transform rests on.

    NEP 50 changed scalar promotion in 2.0; these are the operations
    `SearchResult.aai` performs, pinned to their expected results.
    """
    jac = np.array([1.0, 0.5, 0.0, np.nan], dtype=np.float64)
    ok = np.isfinite(jac) & (jac > 0)
    assert ok.tolist() == [True, True, False, False]

    x = np.power(-0.2607023 * np.log(jac[ok]), 1.0 / 3.435)
    aai = (1.810741 * np.exp(-x) - 0.3087057) * 100.0
    assert aai.dtype == np.float64
    # Measured identical under numpy 1.26.4 and 2.4.6, to six decimal places.
    assert aai[0] == pytest.approx(150.20353, abs=1e-5), \
        "Jaccard 1.0 extrapolates past 100 — the reason a self-pair is reported as identity"
    assert aai[1] == pytest.approx(67.742838, abs=1e-5)


def test_frombuffer_gives_the_same_view_in_both_majors():
    """How Rust's buffers reach Python. A dtype or copy change here would be
    silent and would corrupt every result."""
    raw = np.array([0.25, 0.5], dtype=np.float64).tobytes()
    view = np.frombuffer(raw, dtype=np.float64).reshape(1, 2)
    assert view.shape == (1, 2)
    assert view.dtype == np.float64
    assert view.tolist() == [[0.25, 0.5]]
