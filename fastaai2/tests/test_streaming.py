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



# --- output ------------------------------------------------------------------
#
# There is one writer. A search whose two sides each fit a single partition is
# the 1x1 case and lands in one file; anything larger lands in a directory of
# blocks. No resume: the search is cheap enough that recomputing it beats the
# machinery for not recomputing it.

def _saved(tmp_path, name, n, start=0):
    p = tmp_path / name
    _db(n, start).save(str(p))
    return str(p)


def _rows(paths):
    out = []
    for path in paths:
        with open(path) as fh:
            out.extend(csv.DictReader(fh, delimiter="\t"))
    return out


def test_a_single_partition_search_writes_one_file(tmp_path):
    p = _saved(tmp_path, "db", 6)
    out = tmp_path / "aai.tsv"
    main(["query", "-q", p, "-o", str(out), "--quiet"])
    assert out.is_file()
    rows = _rows([out])
    assert len(rows) == 36
    assert set(rows[0]) == {"query", "target", "shared_scps", "jaccard", "aai"}


def test_a_multi_block_search_writes_a_directory_of_blocks(tmp_path, multipart):
    q = _saved(tmp_path, "q", 3, start=90_000)
    out = tmp_path / "blocks"
    main(["query", "-q", q, "-t", multipart, "-o", str(out), "--quiet"])
    assert sorted(f.name for f in out.glob("*.tsv")) == [
        block_name(0, 0), block_name(0, 1)
    ]


def test_multi_block_output_covers_every_pair_exactly_once(tmp_path, multipart):
    """Blocks partition the result: no pair missing, none written twice."""
    q = _saved(tmp_path, "q", 3, start=90_000)
    out = tmp_path / "blocks"
    main(["query", "-q", q, "-t", multipart, "-o", str(out), "--quiet"])

    rows = _rows(sorted(out.glob("*.tsv")))
    target = fastaai.open_database(multipart)
    assert len(rows) == 3 * target.n_genomes
    assert len({(r["query"], r["target"]) for r in rows}) == len(rows)

    # ...and the values are the ones the API computes.
    ref = fastaai.search(fastaai.open_database(q), target, threads=2)
    jac = np.asarray(ref.jaccard).reshape(3, target.n_genomes)
    index = {n: i for i, n in enumerate(ref.query_names)}
    tindex = {n: i for i, n in enumerate(ref.target_names)}
    for r in rows[::997]:                       # stride: 49,500 rows is plenty
        got = r["jaccard"]
        want = jac[index[r["query"]], tindex[r["target"]]]
        assert got == ("NA" if np.isnan(want) else f"{want:.10g}")


def test_a_multi_block_search_refuses_to_write_to_one_file(tmp_path, multipart):
    """Silently concatenating blocks into stdout would defeat the point."""
    q = _saved(tmp_path, "q", 3, start=90_000)
    with pytest.raises(SystemExit):
        main(["query", "-q", q, "-t", multipart, "--quiet"])


def test_matrix_refuses_a_search_it_cannot_hold(tmp_path, multipart):
    q = _saved(tmp_path, "q", 3, start=90_000)
    with pytest.raises(SystemExit):
        main(["query", "-q", q, "-t", multipart, "-o", str(tmp_path / "m"),
              "--output_style", "matrix", "--quiet"])


def test_a_partial_result_never_looks_complete(tmp_path):
    p = _saved(tmp_path, "db", 6)
    out = tmp_path / "aai.tsv"
    main(["query", "-q", p, "-o", str(out), "--quiet"])
    assert list(tmp_path.glob("*.part*")) == []


def test_concurrent_writers_do_not_share_a_temp_file(tmp_path):
    """Two processes may be told to write the same block.

    With a shared temp name they would interleave rows into one file and rename
    the mixture into place — corruption indistinguishable from a finished block.
    """
    import multiprocessing as mp

    p = _saved(tmp_path, "db", 40)
    out = tmp_path / "aai.tsv"

    def worker(_i):
        main(["query", "-q", p, "-o", str(out), "--quiet"])

    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(i,)) for i in range(4)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(60)
    assert all(pr.exitcode == 0 for pr in procs)

    rows = _rows([out])
    assert len(rows) == 40 * 40
    assert len({(r["query"], r["target"]) for r in rows}) == 40 * 40
    assert list(tmp_path.glob("*.part*")) == []


def test_output_across_two_distinct_databases(tmp_path):
    qa = _saved(tmp_path, "q", 10)
    tb = _saved(tmp_path, "t", 12, start=500)
    out = tmp_path / "aai.tsv"
    main(["query", "-q", qa, "-t", tb, "-o", str(out), "--quiet"])
    assert len(_rows([out])) == 120


@pytest.mark.parametrize("emit,cols", [
    ("aai", ["query", "target", "shared_scps", "aai"]),
    ("jaccard", ["query", "target", "shared_scps", "jaccard"]),
    ("both", ["query", "target", "shared_scps", "jaccard", "aai"]),
])
def test_emit_selects_the_columns(tmp_path, emit, cols):
    p = _saved(tmp_path, "db", 6)
    out = tmp_path / f"{emit}.tsv"
    main(["query", "-q", p, "-o", str(out), "--emit", emit, "--quiet"])
    assert open(out).readline().rstrip("\n").split("\t") == cols


def test_stdev_adds_a_column_matching_the_api(tmp_path):
    p = _saved(tmp_path, "db", 8)
    out = tmp_path / "sd.tsv"
    main(["query", "-q", p, "-o", str(out), "--do_stdev", "--quiet"])
    rows = _rows([out])
    assert "jaccard_sd" in rows[0]

    db = fastaai.open_database(p)
    ref = fastaai.search(db, db, threads=1, stdev=True)
    sd = np.asarray(ref.stdev).reshape(8, 8)
    idx = {n: i for i, n in enumerate(ref.query_names)}
    for r in rows:
        want = sd[idx[r["query"]], idx[r["target"]]]
        assert r["jaccard_sd"] == ("NA" if np.isnan(want) else f"{want:.6g}")


def test_stdout_is_the_default_for_a_single_block(tmp_path, capfd):
    p = _saved(tmp_path, "db", 4)
    main(["query", "-q", p, "--quiet"])
    out = capfd.readouterr().out
    assert out.startswith("query\ttarget\tshared_scps")
    assert len(out.strip().split("\n")) == 1 + 16


def test_saving_a_streamed_database_reproduces_it(tmp_path):
    """The streamed path copies partition files rather than re-serialising."""
    src = tmp_path / "src"
    _db(30).save(str(src))
    streamed = fastaai.open_database(str(src))
    dest = tmp_path / "dest"
    streamed.save(str(dest))

    reopened = fastaai.open_database(str(dest))
    assert reopened.n_genomes == 30
    assert np.allclose(fastaai.search(reopened, reopened, threads=1).jaccard,
                       fastaai.search(streamed, streamed, threads=1).jaccard,
                       equal_nan=True)


def test_saving_a_streamed_database_onto_itself_is_not_destructive(tmp_path):
    p = tmp_path / "db"
    _db(20).save(str(p))
    db = fastaai.open_database(str(p))
    db.save(str(p))
    assert fastaai.open_database(str(p)).n_genomes == 20


def test_merge_spans_partitions(tmp_path, multipart):
    other = tmp_path / "other"
    _db(100, start=99_000).save(str(other))
    out = tmp_path / "merged"
    written, _skipped, parts = fastaai.merge_databases(str(out), [multipart, str(other)])
    assert written == 16_600
    merged = fastaai.open_database(str(out))
    assert merged.n_genomes == 16_600
    assert merged.n_partitions == parts


def test_a_block_outside_the_grid_is_refused(tmp_path):
    p = _saved(tmp_path, "db", 5)
    db = fastaai.open_database(p)
    with pytest.raises(ValueError):
        db.search_block(db, 0, 99)
