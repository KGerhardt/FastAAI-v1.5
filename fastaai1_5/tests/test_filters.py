"""Best-hit resolution semantics.

The three modes are genuinely different and none is a superset of another. These
cases are the discriminating ones, worked out against FastAAI 1's actual code
paths, and they exist so the choice cannot drift silently.
"""

import pytest

from fastaai.search import Hit, _resolve


def resolve(rows, mode):
    return _resolve([Hit(p, a, s) for p, a, s in rows], mode)


# Case A: the losing protein has a free fallback accession.
CASE_A = [("P1", "accA", 100.0), ("P2", "accA", 95.0), ("P2", "accB", 90.0)]

# Case B: the accession's globally best protein was claimed elsewhere.
CASE_B = [("P1", "accB", 100.0), ("P1", "accA", 90.0), ("P2", "accA", 80.0)]

# Case D: a three-way chain.
CASE_D = [
    ("P1", "accA", 100.0), ("P1", "accB", 95.0), ("P2", "accB", 90.0),
    ("P2", "accC", 85.0), ("P3", "accC", 80.0),
]


def test_case_a_modes_disagree():
    assert resolve(CASE_A, "v1") == {"P1": "accA"}
    assert resolve(CASE_A, "v1_alt") == {"P1": "accA", "P2": "accB"}
    assert resolve(CASE_A, "rbh") == {"P1": "accA"}


def test_case_b_reverses_which_mode_is_harsher():
    """v1 is harsher in case A but more permissive in case B — neither dominates."""
    assert resolve(CASE_B, "v1") == {"P1": "accB", "P2": "accA"}
    assert resolve(CASE_B, "v1_alt") == {"P1": "accB"}
    assert resolve(CASE_B, "rbh") == {"P1": "accB"}


def test_strict_rbh_is_markedly_harsher():
    assert len(resolve(CASE_D, "v1")) == 3
    assert len(resolve(CASE_D, "v1_alt")) == 2
    assert len(resolve(CASE_D, "rbh")) == 1


def test_v1_admits_a_non_reciprocal_hit():
    """In case B, accA's best protein is P1 (90) yet v1 assigns P2 (80) to it.
    This is why the shipped v1 path is not strict reciprocal best hit."""
    out = resolve(CASE_B, "v1")
    assert out["P2"] == "accA"
    assert "P2" not in resolve(CASE_B, "rbh")


def test_rbh_is_a_true_intersection_of_argmaxes():
    for case in (CASE_A, CASE_B, CASE_D):
        out = resolve(case, "rbh")
        best_prot, best_acc = {}, {}
        for p, a, s in sorted(case, key=lambda r: -r[2]):
            best_prot.setdefault(p, a)
            best_acc.setdefault(a, p)
        for p, a in out.items():
            assert best_prot[p] == a and best_acc[a] == p


def test_uncontended_hits_survive_every_mode():
    rows = [("P1", "accA", 100.0), ("P2", "accB", 90.0)]
    want = {"P1": "accA", "P2": "accB"}
    for mode in ("v1", "v1_alt", "rbh"):
        assert resolve(rows, mode) == want, mode


def test_ties_resolve_deterministically():
    """FastAAI 1 used an unstable argsort then reversed it, so equal bit scores
    gave non-reproducible assignments. Resolution here must be stable."""
    rows = [("P1", "accA", 100.0), ("P2", "accA", 100.0)]
    first = resolve(rows, "v1")
    for _ in range(20):
        assert resolve(list(reversed(rows)), "v1") == first


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        resolve(CASE_A, "nonsense")
