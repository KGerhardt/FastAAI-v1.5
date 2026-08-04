"""Tests for the Rust core through its Python boundary.

Correctness properties here are the ones that would fail silently if broken:
symmetry, self-identity, unshared pairs, and schema compatibility.
"""

import math

import numpy as np
import pytest

import fastaai
from fastaai import _core


# --------------------------------------------------------------------- k-mers

def test_kmerize_is_sorted_unique_and_dense():
    out = fastaai.kmerize(b"MKVLAAT")
    assert out == sorted(set(out))
    # 20^4 space, so every code is addressable as an index
    assert all(0 <= c < 20 ** 4 for c in out)


def test_kmerize_window_count():
    seq = b"ACDEFGHIK"
    assert len(fastaai.kmerize(seq)) == len(seq) - 4 + 1


def test_unknown_residue_breaks_window_not_silently_aliased():
    # 'X' is outside the default alphabet.
    assert fastaai.kmerize(b"AAAAXAAAA") == fastaai.kmerize(b"AAAA")


def test_kmerize_rejects_bad_alphabet():
    with pytest.raises(ValueError):
        fastaai.kmerize(b"AAAA", 4, "AAB")  # duplicate symbol


# ------------------------------------------------------------------- database

def make_db(genomes, accessions=("acc0", "acc1")):
    db = fastaai.Database(list(accessions))
    for name, scps in genomes.items():
        db.add_genome(name, [(i, s.encode()) for i, s in scps])
    db.seal()
    return db


def test_self_comparison_is_exactly_one():
    db = make_db({
        "g0": [(0, "MKVLAATTGGHH"), (1, "PQRSTVWYACDE")],
        "g1": [(0, "MKVLAATTGGHH"), (1, "MMMMMMMMMMMM")],
    })
    res = fastaai.search(db, db, threads=1)
    assert res.jaccard[0, 0] == 1.0
    assert res.jaccard[1, 1] == 1.0


def test_result_is_symmetric():
    db = make_db({
        "g0": [(0, "MKVLAATTGGHH"), (1, "PQRSTVWYACDE")],
        "g1": [(0, "MKVLAATTGGHY"), (1, "PQRSTVWYACDA")],
        "g2": [(0, "WWWWWWWWWWWW")],
    })
    res = fastaai.search(db, db, threads=1)
    assert np.allclose(res.jaccard, res.jaccard.T, equal_nan=True)
    assert (res.shared == res.shared.T).all()


def test_unshared_pair_is_nan_not_zero():
    """A pair sharing no accession must be distinguishable from AAI 0."""
    db = make_db({
        "g0": [(0, "MKVLAATTGGHH")],
        "g1": [(1, "PQRSTVWYACDE")],
    })
    res = fastaai.search(db, db, threads=1)
    assert res.shared[0, 1] == 0
    assert math.isnan(res.jaccard[0, 1])
    assert math.isnan(res.aai[0, 1])


def test_threading_does_not_change_results():
    db = make_db({
        f"g{i}": [(0, "MKVLAATTGGHH" + "A" * i), (1, "PQRSTVWYACDE" + "C" * i)]
        for i in range(12)
    })
    base = fastaai.search(db, db, threads=1)
    for t in (2, 4, 16):
        got = fastaai.search(db, db, threads=t)
        assert np.allclose(got.jaccard, base.jaccard, equal_nan=True), f"threads={t}"
        assert (got.shared == base.shared).all(), f"threads={t}"


def test_shared_counts_only_mutually_present_accessions():
    db = make_db({
        "both": [(0, "MKVLAATTGGHH"), (1, "PQRSTVWYACDE")],
        "one": [(0, "MKVLAATTGGHH")],
    })
    res = fastaai.search(db, db, threads=1)
    assert res.shared[0, 0] == 2
    assert res.shared[0, 1] == 1


# --------------------------------------------------------------- guard rails

def test_duplicate_accession_in_one_genome_is_rejected():
    db = fastaai.Database(["acc0"])
    with pytest.raises(ValueError):
        db.add_genome("g", [(0, b"MKVLAATT"), (0, b"PQRSTVWY")])


def test_out_of_range_accession_is_rejected():
    db = fastaai.Database(["acc0"])
    with pytest.raises(ValueError):
        db.add_genome("g", [(5, b"MKVLAATT")])


def test_cannot_add_after_seal():
    db = make_db({"g0": [(0, "MKVLAATTGGHH")]})
    with pytest.raises(RuntimeError):
        db.add_genome("g1", [(0, b"MKVLAATT")])


def test_searching_unsealed_target_is_rejected():
    db = fastaai.Database(["acc0"])
    db.add_genome("g", [(0, b"MKVLAATTGG")])
    other = make_db({"g0": [(0, "MKVLAATTGGHH")]})
    with pytest.raises(RuntimeError):
        other.search(db, 1)


def test_schema_mismatch_is_rejected():
    """Different accession lists must not compare — IDs are database-local, so
    the output would be structurally valid and biologically meaningless."""
    a = make_db({"g0": [(0, "MKVLAATTGGHH")]}, accessions=("acc0", "acc1"))
    b = make_db({"g0": [(0, "MKVLAATTGGHH")]}, accessions=("other0", "other1"))
    with pytest.raises(ValueError):
        a.search(b, 1)


def test_empty_database_cannot_seal():
    db = fastaai.Database(["acc0"])
    with pytest.raises(RuntimeError):
        db.seal()


# --------------------------------------------------------------------- AAI

def test_aai_round_trips():
    for aai in (35.0, 50.0, 65.0, 80.0, 90.0):
        j = fastaai.aai_to_jaccard(aai)
        assert abs(fastaai.jaccard_to_aai(j) - aai) < 1e-6


def test_aai_band_matches_design_notes():
    assert abs(fastaai.aai_to_jaccard(30.0) - 0.0057) < 5e-4
    assert abs(fastaai.aai_to_jaccard(65.0) - 0.4447) < 5e-4
    assert abs(fastaai.aai_to_jaccard(90.0) - 0.8430) < 5e-4


def test_aai_is_uncensored():
    """Storing censored values would discard precision exactly where FastAAI is
    meant to be informative."""
    assert fastaai.jaccard_to_aai(1.0) > 100.0
    assert math.isnan(fastaai.jaccard_to_aai(0.0))


# ------------------------------------------------- targets-only databases

def test_seal_drops_forward_index_by_default():
    """A stored partition needs only the inverted index — ~36% of full size, and
    the difference between 34 GB and 95 GB at GTDB scale."""
    db = fastaai.Database(["a0", "a1"])
    db.add_genome("g0", [(0, b"MKVLAATTGGHH"), (1, b"PQRSTVWYACDE")])
    db.seal()
    assert db.is_sealed
    assert not db.has_forward


def test_keep_forward_retains_it():
    db = fastaai.Database(["a0"])
    db.add_genome("g0", [(0, b"MKVLAATTGGHH")])
    db.seal(keep_forward=True)
    assert db.has_forward


def test_targets_only_database_can_still_query():
    """The forward index is rebuilt from the inverted one on demand, so a
    targets-only artifact is closed under search."""
    genomes = {
        "g0": [(0, "MKVLAATTGGHHWWYY"), (1, "PQRSTVWYACDEFGHI")],
        "g1": [(0, "MKVLAATTGGHHWWYA"), (1, "PQRSTVWYACDEFGHA")],
        "g2": [(0, "CCCCDDDDEEEEFFFF")],
    }

    kept = fastaai.Database(["a0", "a1"])
    dropped = fastaai.Database(["a0", "a1"])
    for name, scps in genomes.items():
        payload = [(i, s.encode()) for i, s in scps]
        kept.add_genome(name, payload)
        dropped.add_genome(name, payload)
    kept.seal(keep_forward=True)
    dropped.seal()

    a = fastaai.search(kept, kept, threads=1)
    b = fastaai.search(dropped, dropped, threads=1)
    assert np.allclose(a.jaccard, b.jaccard, equal_nan=True)
    assert (a.shared == b.shared).all()


def test_scp_counts_survive_dropping_the_forward_index():
    genomes = {"both": [(0, "MKVLAATTGGHH"), (1, "PQRSTVWYACDE")],
               "one": [(0, "MKVLAATTGGHH")]}
    kept = fastaai.Database(["a0", "a1"])
    dropped = fastaai.Database(["a0", "a1"])
    for name, scps in genomes.items():
        payload = [(i, s.encode()) for i, s in scps]
        kept.add_genome(name, payload)
        dropped.add_genome(name, payload)
    kept.seal(keep_forward=True)
    dropped.seal()
    assert kept.scp_counts() == dropped.scp_counts() == [2, 1]
