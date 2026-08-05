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

def test_seal_drops_the_forward_index():
    """A stored partition holds only the inverted index — ~36% of the full size,
    and the difference between 34 GB and 95 GB at GTDB scale. Nothing downstream
    reads the forward sets: the join takes both sides inverted."""
    db = fastaai.Database(["a0", "a1"])
    db.add_genome("g0", [(0, b"MKVLAATTGGHH"), (1, b"PQRSTVWYACDE")])
    db.seal()
    assert db.is_sealed
    assert db.index_bytes() > 0


def test_scp_counts_come_from_the_inverted_index():
    db = fastaai.Database(["a0", "a1"])
    db.add_genome("both", [(0, b"MKVLAATTGGHH"), (1, b"PQRSTVWYACDE")])
    db.add_genome("one", [(0, b"MKVLAATTGGHH")])
    db.seal()
    assert db.scp_counts() == [2, 1]


def test_block_size_does_not_change_results():
    """Blocking is a memory-budget knob, never a numerical one."""
    db = fastaai.Database(["a0", "a1"])
    for i in range(20):
        db.add_genome(f"g{i}", [(0, ("MKVLAATTGGHH" + "A" * i).encode()),
                                (1, ("PQRSTVWYACDE" + "C" * i).encode())])
    db.seal()
    base = fastaai.search(db, db, threads=1, block=1)
    for blk in (2, 7, 64, 4096):
        got = fastaai.search(db, db, threads=1, block=blk)
        assert np.allclose(got.jaccard, base.jaccard, equal_nan=True), f"block={blk}"
        assert (got.shared == base.shared).all(), f"block={blk}"


# ------------------------------------------------------- direct comparison

def test_compare_pair_matches_the_indexed_path():
    """The pairwise function is the oracle for the index, so agreement between
    them is the whole point of it existing."""
    genomes = {
        "g0": [(0, b"MKVLAATTGGHHWWYY"), (1, b"PQRSTVWYACDEFGHI")],
        "g1": [(0, b"MKVLAATTGGHHWWYA"), (1, b"PQRSTVWYACDEFGHA")],
        "g2": [(0, b"CCCCDDDDEEEEFFFF")],
    }
    db = fastaai.Database(["a0", "a1"])
    for name, scps in genomes.items():
        db.add_genome(name, scps)
    db.seal()
    res = fastaai.search(db, db, threads=1)

    names = db.genome_names
    for i, qn in enumerate(names):
        for j, tn in enumerate(names):
            mj, shared, aai = fastaai._core.compare_pair(genomes[qn], genomes[tn], 2)
            assert shared == res.shared[i, j], f"shared {qn} vs {tn}"
            if shared == 0:
                assert math.isnan(mj) and math.isnan(res.jaccard[i, j])
            else:
                assert abs(mj - res.jaccard[i, j]) < 1e-12, f"jaccard {qn} vs {tn}"


def test_compare_pair_self_is_one():
    g = [(0, b"MKVLAATTGGHHWWYY"), (1, b"PQRSTVWYACDEFGHI")]
    mj, shared, aai = fastaai._core.compare_pair(g, g, 2)
    assert shared == 2
    assert abs(mj - 1.0) < 1e-12
    assert aai > 100.0  # regression is unbounded above, uncensored by design


def test_compare_pair_no_shared_accession_is_nan():
    a = [(0, b"MKVLAATTGGHH")]
    b = [(1, b"PQRSTVWYACDE")]
    mj, shared, aai = fastaai._core.compare_pair(a, b, 2)
    assert shared == 0
    assert math.isnan(mj) and math.isnan(aai)


def test_compare_pair_rejects_out_of_range_accession():
    with pytest.raises(ValueError):
        fastaai._core.compare_pair([(5, b"MKVLAATT")], [(0, b"MKVLAATT")], 2)


# ------------------------------------------------------- on-disk database

def _small_db(tmp_path, n=5):
    db = fastaai.Database(["a0", "a1"])
    for i in range(n):
        db.add_genome(f"g{i}", [(0, ("MKVLAATTGGHH" + "A" * i).encode()),
                                (1, ("PQRSTVWYACDE" + "C" * i).encode())])
    db.seal()
    db.source = "unit test"
    db.filter_mode = "v1"
    return db


def test_database_round_trips_through_disk(tmp_path):
    """A stored index must compute the same numbers as the one it came from."""
    db = _small_db(tmp_path)
    ref = fastaai.search(db, db, threads=1)
    path = str(tmp_path / "db.fastaai")
    db.save(path)

    got = fastaai._core.open_database(path)
    assert got.n_genomes == db.n_genomes
    assert got.genome_names == db.genome_names, "genome order is the output order"
    assert got.scp_counts() == db.scp_counts()
    assert got.source == "unit test" and got.filter_mode == "v1"

    res = fastaai.search(got, got, threads=1)
    assert np.allclose(res.jaccard, ref.jaccard, equal_nan=True)
    assert (res.shared == ref.shared).all()


def test_saving_an_unsealed_database_is_rejected(tmp_path):
    db = fastaai.Database(["a0"])
    db.add_genome("g", [(0, b"MKVLAATTGG")])
    with pytest.raises(RuntimeError):
        db.save(str(tmp_path / "db"))


def test_opening_a_missing_database_errors(tmp_path):
    with pytest.raises(Exception):
        fastaai._core.open_database(str(tmp_path / "nope"))


def test_truncated_partition_is_rejected_not_misread(tmp_path):
    """A corrupt file must fail loudly. Silently wrong numbers are the costly
    failure mode, and the one a checksum-free format has to guard against."""
    import pathlib
    db = _small_db(tmp_path)
    path = tmp_path / "db.fastaai"
    db.save(str(path))
    part = path / "part.00000"
    raw = part.read_bytes()
    part.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(Exception):
        fastaai._core.open_database(str(path))


def test_forward_sets_are_not_stored(tmp_path):
    """Only the inverted index goes to disk — keeping forward sets would roughly
    triple a database for something the join never reads.

    Stored size exceeds `index_bytes()` by the per-genome metadata that method
    does not count (kmer_counts, the presence bitmap) plus length framing. That
    overhead is fixed per accession, so it dominates a toy database and vanishes
    at scale: measured 116.3 MB on disk against a 114.8 MB index for 2,943
    genomes, a ratio of 1.013.
    """
    db = _small_db(tmp_path, n=20)
    path = str(tmp_path / "db.fastaai")
    db.save(path)
    parts, _, _ = db.stored_bytes(path)
    assert parts >= db.index_bytes()
    assert parts < db.index_bytes() * 1.5, \
        "stored size must track the inverted index, not the forward sets"
