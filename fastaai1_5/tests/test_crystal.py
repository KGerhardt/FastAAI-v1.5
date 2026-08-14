"""Crystals — the resolved-SCP rank.

The properties worth guarding are not the file format but what the format is
*for*: a crystal set must rebuild the same database its genomes built directly,
must survive being split and concatenated, and must refuse to mix model sets.
"""

import gzip
import os

import pytest

pyhmmer = pytest.importorskip("pyhmmer")

from fastaai import crystal
from fastaai.pipeline import build_from_crystals
from fastaai.search import ModelSet

FP = "a" * 64
OTHER_FP = "b" * 64


def _write(root, genome, scps, fp=FP, mode="v1", table=11, origins=None):
    return crystal.write(root, genome, scps, fp, mode, table, origins)


def test_render_is_deterministic():
    """Same genome and models must give a byte-identical file, whatever order
    HMMER reported hits in — otherwise crystals cannot be checksummed."""
    a = crystal.render("g", {"PF2": "MKV", "PF1": "AAA"}, FP, "v1")
    b = crystal.render("g", {"PF1": "AAA", "PF2": "MKV"}, FP, "v1")
    assert a == b
    assert a.index("PF1") < a.index("PF2"), "accessions in sorted order"


def test_header_round_trips():
    line = crystal.header("PF00380.26", "GCF_1", "prot_7", FP, "v1", 11)
    acc, fields = crystal.parse_header(line)
    assert acc == "PF00380.26"
    assert fields == {"genome": "GCF_1", "protein": "prot_7",
                      "models": FP, "filter": "v1", "table": "11"}


def test_genome_with_no_scps_writes_nothing(tmp_path):
    """An empty file is indistinguishable from a truncated write."""
    assert _write(tmp_path, "empty", {}) is None
    assert crystal.crystal_paths(tmp_path) == []


def test_read_groups_by_genome_not_by_file(tmp_path):
    """Grouping by the genome= field is what makes crystals concatenable."""
    _write(tmp_path, "g1", {"PF1": "AAAA"})
    _write(tmp_path, "g2", {"PF1": "CCCC"})

    merged = tmp_path / f"merged{crystal.SUFFIX}"
    parts = sorted(tmp_path.glob(f"*{crystal.SUFFIX}"))
    merged.write_text("".join(q.read_text() for q in parts))
    for q in parts:
        os.remove(q)

    genomes, prov = crystal.read(tmp_path)
    assert set(genomes) == {"g1", "g2"}, "one file, two genomes"
    assert prov["models"] == FP


def test_sequences_survive_wrapping(tmp_path):
    """Sequences are wrapped for readability; they must reassemble exactly."""
    seq = "MKVLAATTGG" * 25
    _write(tmp_path, "g", {"PF1": seq})
    genomes, _ = crystal.read(tmp_path)
    assert genomes["g"]["PF1"] == seq


def test_mixed_model_sets_are_refused(tmp_path):
    _write(tmp_path, "g1", {"PF1": "AAAA"}, fp=FP)
    _write(tmp_path, "g2", {"PF1": "CCCC"}, fp=OTHER_FP)
    with pytest.raises(ValueError, match="disagree on the model set"):
        crystal.read(tmp_path)


def test_mixed_filter_modes_are_refused(tmp_path):
    _write(tmp_path, "g1", {"PF1": "AAAA"}, mode="v1")
    _write(tmp_path, "g2", {"PF1": "CCCC"}, mode="rbh")
    with pytest.raises(ValueError, match="disagree on the best-hit filter"):
        crystal.read(tmp_path)


def test_record_without_genome_field_is_rejected(tmp_path):
    """A plain protein FASTA is not a crystal, and must not be read as one."""
    (tmp_path / f"bogus{crystal.SUFFIX}").write_text(">PF1 some protein\nMKVL\n")
    with pytest.raises(ValueError, match="not a FastAAI crystal"):
        crystal.read(tmp_path)


def test_empty_source_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no crystals"):
        crystal.read(tmp_path)


def test_build_refuses_a_different_model_set(tmp_path):
    """The guard that stops two marker sets producing meaningless AAI."""
    models = ModelSet()
    _write(tmp_path, "g1", {models.accessions[0]: "MKVLAATTGG"}, fp=OTHER_FP)
    with pytest.raises(ValueError, match="not built with this model set"):
        build_from_crystals(tmp_path, models)


def test_build_refuses_unknown_accessions(tmp_path):
    models = ModelSet()
    _write(tmp_path, "g1", {"NOT_A_REAL_ACCESSION": "MKVLAATTGG"},
           fp=models.fingerprint)
    with pytest.raises(ValueError, match="absent from the model set"):
        build_from_crystals(tmp_path, models)


def test_accession_ids_come_from_the_model_set_not_the_crystals(tmp_path):
    """Two subsets must number their shared markers identically.

    If positions were derived from whichever accessions appeared, a subset
    missing one marker would shift every later ID and the two databases would
    refuse to compare — silently turning a subset into an incomparable island.
    """
    models = ModelSet()
    a0, a1, a2 = models.accessions[0], models.accessions[1], models.accessions[2]

    full = tmp_path / "full"
    part = tmp_path / "part"
    _write(full, "g1", {a0: "MKVLAATTGG", a1: "AAAACCCCGG", a2: "WWWWYYYYVV"},
           fp=models.fingerprint)
    _write(part, "g2", {a0: "MKVLAATTGG", a2: "WWWWYYYYVV"},
           fp=models.fingerprint)

    db_full = build_from_crystals(full, models)
    db_part = build_from_crystals(part, models)
    assert db_full.accession_names == db_part.accession_names
    assert db_full.schema_key() == db_part.schema_key()


@pytest.mark.parametrize("genome", [
    "my genome v2",            # spaces: header fields are whitespace-delimited
    "strain=A",                # an '=' inside a value
    "GCF_000007085.1_ASM708v1_genomic",   # the ordinary case, unchanged
])
def test_awkward_genome_names_round_trip(tmp_path, genome):
    """Names come from filenames. A space once truncated the name silently, and
    the truncation became the genome's identity everywhere downstream."""
    _write(tmp_path, genome, {"PF1": "MKVLAATTGG"})
    genomes, _ = crystal.read(tmp_path)
    assert list(genomes) == [genome]


def test_ordinary_names_are_not_escaped(tmp_path):
    """Encoding must stay invisible for the names people actually have."""
    line = crystal.header("PF00380.26", "GCF_000007085.1_ASM708v1_genomic",
                          "NC_003869.1_604", "a" * 64, "v1", 11)
    assert "%" not in line


def test_one_genome_split_across_files_is_refused(tmp_path):
    """Streaming bounds memory by the largest file, so a genome scattered over
    several files cannot be reassembled. Refuse it rather than add it twice."""
    for i, seq in enumerate(("AAAA", "CCCC")):
        (tmp_path / f"part{i}{crystal.SUFFIX}").write_text(
            crystal.render("same_genome", {f"PF{i}": seq}, FP, "v1"))
    with pytest.raises(ValueError, match="also appears in an earlier file"):
        list(crystal.iter_genomes(tmp_path))


def test_streaming_and_materialising_agree(tmp_path):
    for g in ("g1", "g2", "g3"):
        _write(tmp_path, g, {"PF1": "AAAA", "PF2": "CCCC"})
    streamed = {g: s for g, s, _ in crystal.iter_genomes(tmp_path)}
    materialised, _ = crystal.read(tmp_path)
    assert streamed == materialised
