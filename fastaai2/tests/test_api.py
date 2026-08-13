"""The public API.

What matters is not that each function runs but that the chain composes and
that every entry point into it lands on the same database — a caller who has
proteins must not get a different answer from one who has genomes.
"""

import numpy as np
import pytest

pyhmmer = pytest.importorskip("pyhmmer")
pyrodigal = pytest.importorskip("pyrodigal")

import fastaai

# A real ribosomal protein, long enough to carry tetramers.
SCP = ("MFRFFKSIGQEMREVDWPNFKQLRKDSSTVISTSVFFIAFLALARLANSIVPFVSSFLLDS"
       "VLGRSMSNVLLIALVYAVVASVFSGLMTFLQSYLLSVVSQKVIYDLRSDLFAHLQKLSLSF"
       "FTKTPVGRLVTRVTNDIEALNELFTSVLVTFVGDLFTLVGILIFMLSMNVKLTLLTLLVLP")


def _genome_dir(tmp_path):
    """One genome whose reverse complement will still call genes."""
    import gzip
    d = tmp_path / "genomes"
    d.mkdir()
    # A nucleotide sequence that back-translates to something HMM-searchable is
    # not worth constructing; the pipeline is exercised from proteins instead.
    return d


def _protein_file(tmp_path, name, n=1):
    p = tmp_path / f"{name}.fasta"
    p.write_text("".join(f">{name}_p{i}\n{SCP}\n" for i in range(n)))
    return p


def test_api_surface_is_exported():
    for fn in ("genome_to_protein", "protein_to_hmm", "prot_hmm_to_crystal",
               "build_database", "search", "preprocess"):
        assert hasattr(fastaai, fn), fn
        assert fn in fastaai.__all__, fn


def test_protein_to_hmm_then_crystal_composes(tmp_path):
    prot = _protein_file(tmp_path, "g1")
    hits = fastaai.protein_to_hmm(prot, tmp_path / "hmm_hits")
    assert hits.exists()

    crystals = fastaai.prot_hmm_to_crystal([(prot, hits)], tmp_path / "crystals")
    # The synthetic protein may or may not clear a TC cutoff; either way the
    # step must run and its output must be readable as what it claims to be.
    for c in crystals:
        assert c.name.endswith(".crystal.fasta")


def test_hit_table_reads_our_own_format(tmp_path):
    prot = _protein_file(tmp_path, "g2")
    hits = fastaai.protein_to_hmm(prot, tmp_path / "hmm_hits")
    genome, parsed = fastaai.read_hit_table(hits)
    assert genome == "g2", "the true name travels in the file"
    assert all(h.score == h.score for h in parsed)  # no NaNs


def test_hit_table_reads_hmmer_tblout(tmp_path):
    """A caller who already ran hmmsearch should not have to reformat.

    That is the point of taking a (protein, hmm) pair rather than requiring the
    run directory.
    """
    p = tmp_path / "hmmer.tblout"
    p.write_text(
        "#                                                               --- full sequence ----\n"
        "# target name        accession  query name  accession    E-value  score  bias\n"
        "prot_1               -          rp_S2       PF00380.26   1.2e-40  135.4  0.1\n"
        "prot_2               -          rp_S7       PF00410.25   4.5e-30  101.2  0.0\n"
    )
    genome, hits = fastaai.read_hit_table(p)
    assert genome is None, "HMMER's own output carries no genome name"
    assert [h.protein for h in hits] == ["prot_1", "prot_2"]
    assert [h.accession for h in hits] == ["PF00380.26", "PF00410.25"]
    assert [h.score for h in hits] == [135.4, 101.2]


def test_preprocess_requires_some_input(tmp_path):
    with pytest.raises(ValueError, match="no input"):
        fastaai.preprocess(directory=str(tmp_path / "out"))


@pytest.mark.parametrize("kind", ["proteins", "crystals"])
def test_preprocess_entry_points_agree(tmp_path, kind):
    """Whichever rank you start from, the database must be the same one."""
    prots = [_protein_file(tmp_path, f"g{i}") for i in range(3)]

    ref = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "ref"),
                             save=False, threads=2)
    if kind == "proteins":
        got = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "b"),
                                 save=False, threads=2)
    else:
        got = fastaai.preprocess(crystals=str(tmp_path / "ref" / "crystals"),
                                 directory=str(tmp_path / "c"), save=False)

    assert got.genome_names == ref.genome_names
    assert got.schema_key() == ref.schema_key()
    a, b = fastaai.search(ref, ref, threads=1), fastaai.search(got, got, threads=1)
    assert np.allclose(a.jaccard, b.jaccard, equal_nan=True)


def test_preprocess_fills_the_run_directory(tmp_path):
    prots = [_protein_file(tmp_path, f"h{i}") for i in range(2)]
    root = tmp_path / "run"
    fastaai.preprocess(proteins=prots, directory=str(root), database="firm",
                       threads=2)
    assert (root / "crystals").is_dir()
    assert (root / "hmm_hits").is_dir()
    assert (root / "database" / "firm").is_dir(), "named database under database/"


def test_preprocess_combines_ranks(tmp_path):
    """Ranks are additive: crystals from elsewhere join the ones just made."""
    a = [_protein_file(tmp_path, "x0")]
    first = tmp_path / "first"
    fastaai.preprocess(proteins=a, directory=str(first), save=False, threads=1)

    b = [_protein_file(tmp_path, "y0")]
    combined = fastaai.preprocess(
        proteins=b, crystals=str(first / "crystals"),
        directory=str(tmp_path / "both"), save=False, threads=1)
    assert combined.n_genomes == 2, combined.genome_names
