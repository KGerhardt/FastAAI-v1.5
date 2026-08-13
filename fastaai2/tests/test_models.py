"""HMM model loading.

The in-memory route exists because HMMER's own file reader is slow and pyhmmer
binds it faithfully — measured 4337 ms vs 135 ms on the bundled 122-model set.
These tests guard the two things that would make that optimisation wrong: models
must come out identical, and the accession list must stay positional.
"""

import io
import os
from pathlib import Path

import pytest

pyhmmer = pytest.importorskip("pyhmmer")

from fastaai.search import ModelSet, _load_hmms

BUNDLED = Path(
    "/mnt/c/Users/kenji/Desktop/kenji_side_hustles/fastaai2/FastAAI/fastaai/"
    "00.Libraries/01.SCG_HMMs/Complete_SCG_DB.hmm"
)

needs_hmm = pytest.mark.skipif(not BUNDLED.exists(), reason="bundled HMM set not present")


@needs_hmm
def test_in_memory_load_matches_path_load():
    """The fast route must be a pure optimisation, not a different parse."""
    fast = _load_hmms(str(BUNDLED))
    with pyhmmer.plan7.HMMFile(str(BUNDLED)) as fh:
        slow = list(fh)

    assert len(fast) == len(slow)
    assert [h.name for h in fast] == [h.name for h in slow]
    assert [h.accession for h in fast] == [h.accession for h in slow]
    assert [h.M for h in fast] == [h.M for h in slow]


@needs_hmm
def test_accession_index_is_positional():
    """Accession IDs are positions in the model file's order — there is no
    compiled-in Pfam list, so this mapping is the whole schema."""
    m = ModelSet(BUNDLED)
    assert len(m) > 0
    assert all(m.acc_index[a] == i for i, a in enumerate(m.accessions))
    assert len(set(m.accessions)) == len(m.accessions), "accessions must be unique"


@needs_hmm
def test_trusted_cutoffs_are_detected_not_assumed():
    """`bit_cutoffs='trusted'` raises on models lacking TC lines. The bundled set
    has them; the point is that the code checks rather than assuming."""
    m = ModelSet(BUNDLED)
    assert isinstance(m.has_trusted, bool)
    assert m.bit_cutoffs in ("trusted", None)
    assert (m.bit_cutoffs == "trusted") == m.has_trusted


def test_missing_file_raises():
    with pytest.raises((OSError, ValueError)):
        ModelSet("/nonexistent/models.hmm")


def test_empty_model_file_is_rejected(tmp_path):
    # pyhmmer raises EOFError here, which is not an OSError subclass.
    empty = tmp_path / "empty.hmm"
    empty.write_bytes(b"")
    with pytest.raises((ValueError, OSError, EOFError)):
        ModelSet(empty)


def test_garbage_model_file_is_rejected(tmp_path):
    junk = tmp_path / "junk.hmm"
    junk.write_bytes(b"this is not an HMM\n" * 10)
    with pytest.raises(Exception):
        ModelSet(junk)


# --- Packaged model sets -----------------------------------------------------
#
# `--hmm` names a file, which makes a set shipped inside site-packages awkward
# to reach. These keywords resolve to the packaged files wherever they landed.
# What needs guarding is not the lookup but the union: bac120 and ar53 share
# markers, and a repeated accession silently breaks the positional schema.

from fastaai.search import MODEL_SETS, ModelSet, model_set_key, resolve_model_spec


def _packaged_present() -> bool:
    return all(os.path.exists(p) for k in MODEL_SETS for p in resolve_model_spec(k))


needs_packaged = pytest.mark.skipif(
    not _packaged_present(), reason="packaged GTDB model sets not present"
)

EXPECTED = {"gtdb-bact": 120, "gtdb-arch": 53, "gtdb-all": 168}


@needs_packaged
@pytest.mark.parametrize("key,count", EXPECTED.items())
def test_packaged_set_loads_with_expected_size(key, count):
    assert len(ModelSet(key)) == count


@needs_packaged
def test_gtdb_all_is_the_union_not_the_concatenation():
    """The 5 markers bac120 and ar53 share appear once.

    Concatenation would give 173 entries with 5 accessions twice. Accession IDs
    are positions in this list and `acc_index` is keyed by accession, so the
    duplicate makes one position unreachable and lets a single protein occupy
    two slots — valid-looking output, wrong numbers.
    """
    bact, arch, both = ModelSet("gtdb-bact"), ModelSet("gtdb-arch"), ModelSet("gtdb-all")
    assert set(both.accessions) == set(bact.accessions) | set(arch.accessions)
    assert len(both) == len(bact) + len(arch) - 5
    assert both.shared and len(both.shared) == 5
    # First occurrence wins, so bac120 keeps its order and its accession IDs.
    assert both.accessions[: len(bact)] == bact.accessions


@needs_packaged
@pytest.mark.parametrize("key", list(MODEL_SETS))
def test_packaged_accessions_stay_unique(key):
    m = ModelSet(key)
    assert len(set(m.accessions)) == len(m.accessions)
    assert all(m.acc_index[a] == i for i, a in enumerate(m.accessions))


@needs_packaged
def test_every_packaged_set_fingerprints_differently():
    """Different marker sets must refuse to be compared, not mix quietly."""
    fps = {k: ModelSet(k).fingerprint for k in MODEL_SETS}
    fps["default"] = ModelSet().fingerprint
    assert len(set(fps.values())) == len(fps)


@needs_packaged
@pytest.mark.parametrize("alias", ["gtdb-bact", "gtdb_bact", "GTDB-BACT", "GTDB_Bact"])
def test_case_and_underscore_are_the_same_set(alias):
    assert ModelSet(alias).fingerprint == ModelSet("gtdb-bact").fingerprint


@pytest.mark.parametrize("spec", ["gtdb", "gtdb-bac", "gtdb-bacteria", "gtdbtk"])
def test_near_miss_names_the_real_sets(spec):
    """A typo must not be reported as a missing file — the set is installed."""
    with pytest.raises(SystemExit) as e:
        resolve_model_spec(spec)
    assert all(k in str(e.value) for k in MODEL_SETS)


def test_a_path_is_never_read_as_a_keyword(tmp_path):
    p = tmp_path / "gtdb-all"
    p.write_bytes(b"")
    assert model_set_key(str(p)) is None
    assert resolve_model_spec(str(p)) == [str(p)]


@needs_packaged
def test_keyword_wins_over_a_file_of_the_same_name(tmp_path, monkeypatch, capsys):
    """Otherwise `--hmm gtdb-bact` means different things in different
    directories, and the wrong one still produces a valid fingerprint."""
    monkeypatch.chdir(tmp_path)
    decoy = tmp_path / "gtdb-bact"
    decoy.write_bytes(b"not an HMM")

    assert resolve_model_spec("gtdb-bact") != [str(decoy)]
    assert len(ModelSet("gtdb-bact")) == 120
    # The shadowing is reported, not silently obeyed.
    assert "packaged model set" in capsys.readouterr().err
    # And the file stays reachable by an unambiguous path.
    assert resolve_model_spec(os.path.join(".", "gtdb-bact")) == [
        os.path.join(".", "gtdb-bact")
    ]
