"""Streamed partitions and per-block output.

Both exist for the same reason: a database at GTDB scale neither fits in RAM nor
produces a result matrix that does. Partitions are therefore read per block, and
the output is written per block, so peak footprint is bounded by PARTITION_SIZE
rather than by the size of the database.

The multi-partition path is exercised here because PARTITION_SIZE is 16,384 —
nothing smaller reaches it, and until these tests it was never crossed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import fastaai
from fastaai.cli import block_name, main


def _scps(i: int):
    return [(0, b"MKVLAATTGGHHIKLMNPQRS" + bytes([65 + i % 20]) * 3),
            (1, b"PQRSTVWYACDEFGHIKLMNP" + bytes([65 + (i * 7) % 20]) * 3)]


def _db(n: int, start: int = 0):
    db = fastaai.Database(["a0", "a1"])
    for i in range(start, start + n):
        db.add_genome(f"g{i}", _scps(i))
    db.seal()
    db.filter_mode = "v1"
    return db


@pytest.fixture(scope="module")
def multipart(tmp_path_factory):
    """A database that genuinely spans two partitions."""
    path = tmp_path_factory.mktemp("mp") / "db"
    _db(16_500).save(str(path))
    return str(path)


def test_a_database_over_partition_size_really_splits(multipart):
    db = fastaai.open_database(multipart)
    assert db.n_partitions == 2
    assert db.partition_genomes == [16_384, 116]


def test_opening_a_database_does_not_load_its_partitions(multipart):
    db = fastaai.open_database(multipart)
    assert db.is_streamed
    # Reported footprint is what the index *would* cost resident; the point is
    # that it is available without having paid it.
    assert db.index_bytes() > 0


def test_results_cross_the_partition_boundary_correctly(multipart):
    """The boundary at 16,384 is where a local u16 genome ID wraps.

    Checked against `compare_pair`, which shares no machinery with the index.
    """
    target = fastaai.open_database(multipart)
    qi = [0, 8_000, 16_000]
    res = fastaai.search(_query(qi), target, threads=2)
    jac = np.asarray(res.jaccard).reshape(len(qi), target.n_genomes)
    for row, g in enumerate(qi):
        for col in (0, 5_000, 16_383, 16_384, 16_499):
            expect, _shared, _aai = fastaai.compare_pair(_scps(g), _scps(col), 2)
            got = jac[row, col]
            assert got == pytest.approx(expect, abs=1e-12), f"q{g} vs g{col}"


def _query(indices):
    db = fastaai.Database(["a0", "a1"])
    for i in indices:
        db.add_genome(f"q{i}", _scps(i))
    db.seal()
    db.filter_mode = "v1"
    return db


def test_a_streamed_search_matches_a_resident_one(tmp_path):
    resident = _db(300)
    path = tmp_path / "db"
    resident.save(str(path))
    streamed = fastaai.open_database(str(path))
    assert streamed.is_streamed and not resident.is_streamed

    a = fastaai.search(resident, resident, threads=2)
    b = fastaai.search(streamed, streamed, threads=2)
    assert np.allclose(a.jaccard, b.jaccard, equal_nan=True)
    assert (a.shared == b.shared).all()


def test_blocks_reassemble_into_the_full_matrix(multipart):
    """Each block is final, not a partial sum awaiting a reduction step."""
    db = fastaai.open_database(multipart)
    q = _query([0, 8_000, 16_000])
    full = np.asarray(fastaai.search(q, db, threads=2).jaccard)
    full = full.reshape(3, db.n_genomes)

    offs = np.cumsum([0] + list(db.partition_genomes))
    for ti in range(db.n_partitions):
        jb, sb, r, c, _ = q.search_block(db, 0, ti, threads=2)
        blk = np.frombuffer(jb, dtype=np.float64).reshape(r, c)
        ref = full[:, offs[ti]:offs[ti] + c]
        assert np.allclose(blk, ref, equal_nan=True)


def test_a_self_block_carries_its_own_diagonal_and_mirror(tmp_path):
    db = _db(40)
    jb, _sb, r, c, _ = db.search_block(db, 0, 0, threads=1)
    blk = np.frombuffer(jb, dtype=np.float64).reshape(r, c)
    assert np.allclose(blk, blk.T, equal_nan=True)          # mirrored
    assert np.isfinite(np.diag(blk)).all()                  # diagonal filled


def test_block_files_agree_with_the_single_file_output(tmp_path):
    db = _db(6)
    p = str(tmp_path / "db")
    db.save(p)

    single = tmp_path / "all.tsv"
    main(["query", "-q", p, "-o", str(single), "--quiet"])
    blocks = tmp_path / "blocks"
    main(["query", "-q", p, "--blocks", str(blocks), "--quiet"])

    def rows(paths):
        out = {}
        for path in paths:
            with open(path) as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    out[(r["query"], r["target"])] = (r["aai"], r["jaccard"])
        return out

    assert rows([single]) == rows(sorted(blocks.glob("*.tsv")))


def test_an_existing_block_is_not_recomputed(tmp_path):
    db = _db(6)
    p = str(tmp_path / "db")
    db.save(p)
    blocks = tmp_path / "blocks"
    main(["query", "-q", p, "--blocks", str(blocks), "--quiet"])

    dest = blocks / block_name(0, 0)
    dest.write_text("SENTINEL\n")           # would be overwritten if recomputed
    main(["query", "-q", p, "--blocks", str(blocks), "--quiet"])
    assert dest.read_text() == "SENTINEL\n"

    # ...unless resume is refused, which must rewrite it.
    main(["query", "-q", p, "--blocks", str(blocks), "--no-resume", "--quiet"])
    assert dest.read_text() != "SENTINEL\n"


def test_a_partial_block_never_looks_complete(tmp_path):
    """Resume keys on file existence, so writes must be atomic."""
    db = _db(6)
    p = str(tmp_path / "db")
    db.save(p)
    blocks = tmp_path / "blocks"
    main(["query", "-q", p, "--blocks", str(blocks), "--quiet"])
    assert list(blocks.glob("*.part")) == []


@pytest.mark.parametrize("extra", [
    ["--do_stdev"],
    ["--emit", "jaccard"],
])
def test_matrix_refuses_flags_it_cannot_represent(tmp_path, extra):
    """A flag that changes the output columns must never be dropped quietly."""
    db = _db(4)
    p = str(tmp_path / "db")
    db.save(p)
    with pytest.raises(SystemExit):
        main(["query", "-q", p, "-o", str(tmp_path / "out"),
              "--output_style", "matrix", "--quiet"] + extra)
