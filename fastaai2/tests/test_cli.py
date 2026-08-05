"""CLI surface and FastAAI 1 compatibility.

v1 command lines must keep working, and a v1 flag that no longer applies must be
reported rather than silently dropped — quietly ignoring `--do_stdev` would
remove an output column the caller explicitly asked for.
"""

import numpy as np
import pytest

import fastaai
from fastaai import _core
from fastaai.cli import (
    LEGACY_MODULES,
    _reroute,
    build_parser,
    main,
)


def _db(tmp_path, name, n=4, start=0):
    db = fastaai.Database(["a0", "a1"])
    for i in range(start, start + n):
        db.add_genome(f"g{i}", [(0, ("MKVLAATTGGHH" + "A" * i).encode()),
                                (1, ("PQRSTVWYACDE" + "C" * i).encode())])
    db.seal()
    db.filter_mode = "v1"
    p = str(tmp_path / name)
    db.save(p)
    return p, db


# ----------------------------------------------------------------- rerouting

def test_every_v1_module_reroutes():
    """None may fall through to 'unknown module'."""
    for m in LEGACY_MODULES:
        assert m in ("build_db", "merge_db", "simple_query", "db_query",
                     "single_query", "multi_query", "aai_index")


def test_build_db_becomes_build():
    got = _reroute(["build_db", "-g", "/in", "-d", "/out", "--threads", "8"])
    assert got[:4] == ["build", "/in", "-d", "/out"]
    assert "--threads" in got and "8" in got, "arguments are carried, not dropped"


def test_aai_index_becomes_a_self_query():
    got = _reroute(["aai_index", "-g", "/in", "-o", "/out.tsv"])
    assert got[:2] == ["query", "-q"]
    assert "-t" not in got, "self-query has no separate target"


def test_db_query_maps_query_and_target():
    got = _reroute(["db_query", "-q", "/a", "-t", "/b", "-o", "/o"])
    assert got == ["query", "-q", "/a", "-t", "/b", "-o", "/o"]


def test_protein_inputs_set_the_input_kind():
    got = _reroute(["build_db", "-p", "/prots", "-d", "/out"])
    assert "--input" in got and got[got.index("--input") + 1] == "protein"


def test_single_query_maps_both_sides():
    got = _reroute(["single_query", "-qg", "/q.fna", "-tg", "/t.fna", "-o", "/o"])
    assert got == ["query", "-q", "/q.fna", "-t", "/t.fna", "-o", "/o"]


def test_merge_db_keeps_the_recipient_in_the_merge():
    got = _reroute(["merge_db", "-r", "/recip", "-d", "/donor"])
    assert got[0] == "merge"
    assert got.count("/recip") == 2, "recipient is both destination and an input"
    assert "/donor" in got


def test_hmm_table_input_is_refused_not_ignored():
    """v1 accepted precomputed HMMER tables. Silently ignoring -m would search
    from scratch and produce different SCPs without saying so."""
    with pytest.raises(SystemExit, match="HMMER tables"):
        _reroute(["build_db", "-g", "/in", "-m", "/hmms", "-d", "/out"])


def test_unknown_module_is_an_error():
    with pytest.raises(SystemExit):
        _reroute(["not_a_module", "-g", "/in"])


# --------------------------------------------------------------- retired flags

def test_do_stdev_fills_the_jacc_sd_column(tmp_path):
    """v1 always emits jacc_SD; --do_stdev decides whether it holds a number."""
    p, _ = _db(tmp_path, "a", n=3)
    out = tmp_path / "o.tsv"
    main(["query", "-q", p, "--do_stdev", "-o", str(out), "--quiet"])
    header, first = out.read_text().splitlines()[:2]
    assert header.split("\t")[3] == "jacc_SD"
    assert first.split("\t")[3] != "N/A"


def test_stdev_is_not_computed_by_default(tmp_path):
    """It costs another output-width array, so it must be opt-in.

    The column still appears — v1's schema is fixed — but reads N/A.
    """
    p, _ = _db(tmp_path, "a", n=3)
    out = tmp_path / "o.tsv"
    main(["query", "-q", p, "-o", str(out), "--quiet"])
    header, first = out.read_text().splitlines()[:2]
    assert header.split("\t")[3] == "jacc_SD"
    assert first.split("\t")[3] == "N/A"


def test_retired_flags_are_reported(tmp_path, capsys):
    p, _ = _db(tmp_path, "a")
    main(["query", "-q", p, "--in_memory", "-o", str(tmp_path / "o.tsv")])
    assert "in_memory" in capsys.readouterr().err


# --------------------------------------------------------------------- merge

def test_merge_combines_distinct_databases(tmp_path):
    a, _ = _db(tmp_path, "a", n=3, start=0)
    b, _ = _db(tmp_path, "b", n=3, start=100)
    out = str(tmp_path / "m")
    written, skipped, parts = _core.merge_databases(out, [a, b])
    assert (written, skipped) == (6, 0)
    m = _core.open_database(out)
    assert m.n_genomes == 6
    assert sorted(m.genome_names) == sorted([f"g{i}" for i in [0, 1, 2, 100, 101, 102]])


def test_merge_deduplicates_by_content(tmp_path):
    a, _ = _db(tmp_path, "a", n=3)
    out = str(tmp_path / "m")
    written, skipped, _ = _core.merge_databases(out, [a, a])
    assert (written, skipped) == (3, 3), "same genomes twice must not double"


def test_merge_preserves_results(tmp_path):
    a, dba = _db(tmp_path, "a", n=3, start=0)
    b, _ = _db(tmp_path, "b", n=3, start=100)
    out = str(tmp_path / "m")
    _core.merge_databases(out, [a, b])
    m = _core.open_database(out)

    ra = fastaai.search(dba, dba, threads=1)
    rm = fastaai.search(m, m, threads=1)
    idx = {n: i for i, n in enumerate(m.genome_names)}
    for i, q in enumerate(dba.genome_names):
        for j, t in enumerate(dba.genome_names):
            assert np.isclose(rm.jaccard[idx[q], idx[t]], ra.jaccard[i, j], equal_nan=True)


def test_merge_refuses_incompatible_schemas(tmp_path):
    a, _ = _db(tmp_path, "a")
    other = fastaai.Database(["different0", "different1"])
    other.add_genome("x", [(0, b"MKVLAATTGG")])
    other.seal()
    b = str(tmp_path / "b")
    other.save(b)
    with pytest.raises(ValueError, match="accession"):
        _core.merge_databases(str(tmp_path / "m"), [a, b])


def test_merge_needs_inputs(tmp_path):
    with pytest.raises(ValueError):
        _core.merge_databases(str(tmp_path / "m"), [])


# ------------------------------------------------------------------- output

def test_matrix_and_tsv_styles(tmp_path):
    p, _ = _db(tmp_path, "a", n=3)
    tsv, mat = tmp_path / "o.tsv", tmp_path / "o.mat"
    main(["query", "-q", p, "-o", str(tsv), "--quiet"])
    main(["query", "-q", p, "-o", str(mat), "--output_style", "matrix", "--quiet"])
    assert tsv.read_text().startswith(
        "query\ttarget\tavg_jacc_sim\tjacc_SD\tnum_shared_SCPs\t"
        "poss_shared_SCPs\tAAI_estimate")
    lines = mat.read_text().splitlines()
    assert lines[0].startswith("query_genome\t"), "v1's matrix corner label"
    assert len(lines) == 4, "header plus one row per genome"


def test_query_without_target_is_a_self_comparison(tmp_path):
    p, _ = _db(tmp_path, "a", n=3)
    out = tmp_path / "o.tsv"
    main(["query", "-q", p, "-o", str(out), "--quiet"])
    rows = out.read_text().splitlines()[1:]
    assert len(rows) == 9, "3 x 3 pairs"


def test_parser_exposes_three_verbs():
    p = build_parser()
    sub = [a for a in p._actions if a.dest == "command"][0]
    assert set(sub.choices) == {"build", "query", "merge"}


# --- AAI reporting band -------------------------------------------------------
#
# The labels themselves are unit-tested in src/report.rs, which owns the
# formatting. These go through the writer, which is what a user actually sees.

def _band_db(tmp_path):
    """Three genomes spanning the band: identical, divergent, and disjoint."""
    db = fastaai.Database(["a0", "a1", "a2"])
    db.add_genome("same_a", [(0, b"MKVLAATTGGHHIKLMNPQRS"), (1, b"PQRSTVWYACDEFGHIKLMN")])
    db.add_genome("same_b", [(0, b"MKVLAATTGGHHIKLMNPQRS"), (1, b"PQRSTVWYACDEFGHIKLMN")])
    db.add_genome("far_c", [(2, b"WWWWCCCCYYYYFFFFMMMM")])
    db.seal()
    db.filter_mode = "v1"
    p = str(tmp_path / "band")
    db.save(p)
    return p


def _cells(path, col):
    out = {}
    for line in path.read_text().splitlines()[1:]:
        f = line.split("\t")
        out[(f[0], f[1])] = f[col]
    return out


def test_the_band_is_reported_categorically(tmp_path):
    p = _band_db(tmp_path)
    out = tmp_path / "o.tsv"
    main(["query", "-q", p, "-o", str(out), "--quiet"])
    aai = _cells(out, 6)
    assert aai[("same_a", "same_b")] == ">90%"


def test_a_pair_sharing_nothing_is_not_a_low_score(tmp_path):
    p = _band_db(tmp_path)
    out = tmp_path / "o.tsv"
    main(["query", "-q", p, "-o", str(out), "--quiet"])
    row = _cells(out, 6)
    assert row[("same_a", "far_c")] == "N/A"
    # ...and every value column agrees it was not measured.
    for col in (2, 3, 4, 5, 6):
        assert _cells(out, col)[("same_a", "far_c")] == "N/A"


def test_matrix_carries_v1_sentinels_since_a_cell_cannot_hold_a_string(tmp_path):
    p = _band_db(tmp_path)
    out = tmp_path / "o.matrix"
    main(["query", "-q", p, "-o", str(out), "--output_style", "matrix", "--quiet"])
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == ["query_genome", "same_a", "same_b", "far_c"]
    cells = dict(zip(lines[0].split("\t")[1:], lines[1].split("\t")[1:]))
    assert cells["same_b"] == "95.0"
    assert cells["far_c"] == "N/A", "v1 writes 0; that cannot be told from a measurement"


def test_absent_optional_v1_flags_do_not_reach_argv():
    """`opt` returns None for a flag the user did not type.

    Passing that None through produced "expected one argument" against a flag
    that was never on the command line.
    """
    for argv in (["db_query", "-q", "A"],
                 ["aai_index", "-g", "G"],
                 ["single_query", "-qg", "A", "-tg", "B"]):
        assert None not in _reroute(argv), argv


def test_a_v1_self_query_without_target_still_runs(tmp_path):
    path, _db_obj = _db(tmp_path, "db", n=4)
    main(["db_query", "-q", path, "-o", str(tmp_path / "out.tsv")])
    assert (tmp_path / "out.tsv").is_file()


def test_donor_file_is_read_not_dropped(tmp_path):
    """v1 took a file listing donors. Dropping it merged nothing, silently."""
    a, _ = _db(tmp_path, "a", n=3, start=0)
    b, _ = _db(tmp_path, "b", n=3, start=100)
    listing = tmp_path / "donors.txt"
    listing.write_text(f"{a}\n{b}\n")
    got = _reroute(["merge_db", "-r", a, "--donor_file", str(listing)])
    assert a in got and b in got


def test_merge_without_donors_is_an_error(tmp_path):
    a, _ = _db(tmp_path, "a")
    with pytest.raises(SystemExit, match="donor"):
        _reroute(["merge_db", "-r", a])


@pytest.mark.parametrize("flag", ["--create_query_db", "--query_db_name",
                                  "--query_output", "--target_output"])
def test_unsupported_v1_flags_are_reported(flag):
    with pytest.raises(SystemExit, match="not supported"):
        _reroute(["db_query", "-q", "/a", "-t", "/b", flag, "x"])


@pytest.mark.parametrize("flag", ["-qh", "-th", "--query_hmms", "--target_hmms"])
def test_hmm_table_flags_are_all_refused(flag):
    with pytest.raises(SystemExit, match="HMMER tables"):
        _reroute(["single_query", flag, "/tbl", "-o", "/o"])
