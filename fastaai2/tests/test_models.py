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
