"""Model-set identity.

Accession names and their ordering do not establish that two databases were
built from the same HMMs. A model revised between Pfam releases keeps its name,
its accession and its position while matching different proteins — which changes
the SCP assignment, the k-mer sets and the AAI, with nothing in the output
looking wrong. The fingerprint is what makes that detectable.

It is a digest of each model's own identity — HMMER's `CKSUM`, model length and
names — rather than of the file, so it is reproducible by any implementation
reading the same models, and survives reformatting and HMMER version drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import fastaai
from fastaai.search import ModelSet

HMMS = Path(__file__).resolve().parents[2] / \
    "FastAAI/fastaai/00.Libraries/01.SCG_HMMs/Complete_SCG_DB.hmm"

pytestmark = pytest.mark.skipif(not HMMS.exists(), reason="v1 SCP HMMs not present")


@pytest.fixture(scope="module")
def models():
    return ModelSet(str(HMMS))


def _edit_first_cksum(text: str) -> str:
    """One model revised: same name, same accession, different alignment."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("CKSUM"):
            lines[i] = "CKSUM " + str((int(ln.split()[1]) + 1) % (2 ** 32))
            return "\n".join(lines)
    raise AssertionError("no CKSUM line found")


def test_fingerprint_is_stable_across_loads(models):
    assert ModelSet(str(HMMS)).fingerprint == models.fingerprint
    assert len(models.fingerprint) == 64


def test_a_revised_model_changes_the_fingerprint(tmp_path, models):
    """The case the fingerprint exists for, and the one names cannot catch."""
    bumped = tmp_path / "bumped.hmm"
    bumped.write_text(_edit_first_cksum(HMMS.read_text()))
    other = ModelSet(str(bumped))

    assert other.accessions == models.accessions, "names and order are unchanged"
    assert other.fingerprint != models.fingerprint


def test_a_different_model_count_changes_the_fingerprint(tmp_path, models):
    blocks = HMMS.read_text().split("//\n")
    subset = tmp_path / "subset.hmm"
    subset.write_text("//\n".join(blocks[:-2]) + "//\n")
    assert ModelSet(str(subset)).fingerprint != models.fingerprint


def test_reordering_changes_the_fingerprint(tmp_path, models):
    """Accession IDs are positions, so order is part of identity."""
    blocks = [b for b in HMMS.read_text().split("//\n") if b.strip()]
    rev = tmp_path / "rev.hmm"
    rev.write_text("//\n".join(reversed(blocks)) + "//\n")
    other = ModelSet(str(rev))
    assert sorted(other.accessions) == sorted(models.accessions)
    assert other.fingerprint != models.fingerprint


def _strip_cksum(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not l.startswith("CKSUM"))


def test_a_model_set_without_cksum_still_fingerprints(tmp_path, models):
    """CKSUM is optional in the HMM format.

    Resting identity on it alone would leave any file that omits one
    fingerprinting on its model names, which is no verification at all.
    """
    p = tmp_path / "nocksum.hmm"
    p.write_text(_strip_cksum(HMMS.read_text()))
    other = ModelSet(str(p))
    assert len(other.fingerprint) == 64
    assert other.fingerprint != models.fingerprint


def test_changed_parameters_are_caught_without_cksum(tmp_path, models):
    """The case CKSUM cannot cover: same names, same length, different model."""
    import re

    stripped = _strip_cksum(HMMS.read_text())
    a_path = tmp_path / "a.hmm"
    a_path.write_text(stripped)

    lines = stripped.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(\s+\d+\s+)(\d+\.\d+)(\s.*)$", line)
        if m:
            lines[i] = m.group(1) + f"{float(m.group(2)) + 0.5:.5f}" + m.group(3)
            break
    else:
        pytest.skip("no match-emission row found to perturb")
    b_path = tmp_path / "b.hmm"
    b_path.write_text("\n".join(lines))

    a, b = ModelSet(str(a_path)), ModelSet(str(b_path))
    assert a.accessions == b.accessions and len(a) == len(b)
    assert a.fingerprint != b.fingerprint


def _db(accessions, fingerprint, path):
    db = fastaai.Database(accessions)
    db.add_genome("g", [(0, b"MKVLAATTGGHHIKLMNPQRS")])
    db.seal()
    db.filter_mode = "v1"
    db.models = fingerprint
    db.save(str(path))
    return fastaai.open_database(str(path))


def test_databases_from_different_models_refuse_to_be_compared(tmp_path, models):
    bumped = tmp_path / "bumped.hmm"
    bumped.write_text(_edit_first_cksum(HMMS.read_text()))
    other = ModelSet(str(bumped))

    a = _db(models.accessions, models.fingerprint, tmp_path / "a")
    b = _db(other.accessions, other.fingerprint, tmp_path / "b")
    with pytest.raises(ValueError, match="model sets differ"):
        fastaai.search(a, b, threads=1)


def test_the_same_models_compare_freely(tmp_path, models):
    a = _db(models.accessions, models.fingerprint, tmp_path / "a")
    b = _db(models.accessions, models.fingerprint, tmp_path / "b")
    fastaai.search(a, b, threads=1)


def test_an_unknown_fingerprint_is_not_treated_as_a_mismatch(tmp_path, models):
    """A database built without an HMM set cannot be checked.

    Refusing it would assert a conflict there is no evidence for — the
    fingerprint reports disagreement, not absence of agreement.
    """
    known = _db(models.accessions, models.fingerprint, tmp_path / "known")
    unknown = _db(models.accessions, "", tmp_path / "unknown")
    fastaai.search(known, unknown, threads=1)
    fastaai.search(unknown, known, threads=1)


def test_the_fingerprint_survives_a_save_and_reopen(tmp_path, models):
    a = _db(models.accessions, models.fingerprint, tmp_path / "a")
    assert a.models == models.fingerprint


def test_an_archive_carries_the_fingerprint_into_a_rebuild(tmp_path, models):
    from fastaai.archive import Archive, read_fingerprint

    root = tmp_path / "arch"
    Archive(root, models.accessions, models.fingerprint).close()
    assert read_fingerprint(root) == models.fingerprint


def test_an_archive_without_a_fingerprint_reports_unknown(tmp_path, models):
    from fastaai.archive import Archive, read_fingerprint

    root = tmp_path / "arch"
    Archive(root, models.accessions).close()
    assert read_fingerprint(root) == ""
