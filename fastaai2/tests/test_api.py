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


# --- reading a result ---------------------------------------------------------

def _tiny_result(tmp_path):
    prots = [_protein_file(tmp_path, f"r{i}") for i in range(3)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "res"),
                            save=False, threads=2)
    return db, fastaai.search(db, db, threads=1)


def test_self_pair_is_not_a_neighbour(tmp_path):
    """The diagonal of a self-comparison is the genome against itself. Reporting
    it as the closest relative is the mistake this interface exists to stop."""
    db, res = _tiny_result(tmp_path)
    q = res.query_names[0]
    assert all(m.target != q for m in res.hits_for(q))
    assert res.hits_for(q, include_self=True)[0].target == q


def test_hits_are_ordered_best_first(tmp_path):
    db, res = _tiny_result(tmp_path)
    for q in res.query_names:
        aai = [m.aai for m in res.hits_for(q)]
        assert aai == sorted(aai, reverse=True)


def test_best_hit_agrees_with_hits_for(tmp_path):
    db, res = _tiny_result(tmp_path)
    for q in res.query_names:
        top = res.hits_for(q, k=1)
        assert res.best_hit(q) == (top[0] if top else None)


def test_k_and_min_aai_bound_the_list(tmp_path):
    db, res = _tiny_result(tmp_path)
    q = res.query_names[0]
    assert len(res.hits_for(q, k=1)) <= 1
    assert all(m.aai >= 1e9 for m in res.hits_for(q, min_aai=1e9)) or \
        res.hits_for(q, min_aai=1e9) == []


def test_unknown_query_is_a_keyerror(tmp_path):
    db, res = _tiny_result(tmp_path)
    with pytest.raises(KeyError):
        res.hits_for("not_a_genome")


def test_rows_covers_the_matrix(tmp_path):
    db, res = _tiny_result(tmp_path)
    n = len(res.query_names) * len(res.target_names)
    assert sum(1 for _ in res.rows()) == n


@pytest.mark.parametrize("stdev", [False, True])
def test_to_tsv_matches_the_engine_byte_for_byte(tmp_path, stdev):
    """Two writers for one format is how the two drift.

    The band, the rounding and the self-pair rule are the engine's own
    (`_core.aai_label`, `_core.py_round`, `_core.SELF_IDENTITY`), so this asserts
    the Python path and the streaming Rust path produce the same bytes rather
    than merely the same numbers.
    """
    prots = [_protein_file(tmp_path, f"t{i}") for i in range(3)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "tt"),
                            save=False, threads=2)
    res = fastaai.search(db, db, threads=1, stdev=stdev)

    mine = tmp_path / f"py_{stdev}.tsv"
    theirs = tmp_path / f"rust_{stdev}.tsv"
    res.to_tsv(mine)
    db.write_block(db, 0, 0, str(theirs), 128, 1, stdev, "both", "tsv")
    assert mine.read_text() == theirs.read_text()


def test_to_tsv_has_v1s_columns(tmp_path):
    db, res = _tiny_result(tmp_path)
    out = tmp_path / "r.tsv"
    res.to_tsv(out)
    assert out.read_text().splitlines()[0] == (
        "query\ttarget\tavg_jacc_sim\tjacc_SD\tnum_shared_SCPs"
        "\tposs_shared_SCPs\tAAI_estimate")


def test_a_self_pair_reports_as_identity_not_an_estimate(tmp_path):
    """The regression is unbounded above, so feeding it a self-comparison
    returns something past 100 that then reads as `>90%`."""
    db, res = _tiny_result(tmp_path)
    out = tmp_path / "self.tsv"
    res.to_tsv(out)
    q = res.query_names[0]
    row = next(ln for ln in out.read_text().splitlines()
               if ln.startswith(f"{q}\t{q}\t"))
    assert row.endswith("\t100.0")


def test_poss_shared_is_the_poorer_genome(tmp_path):
    db, res = _tiny_result(tmp_path)
    counts = dict(zip(res.query_names, res.query_scps))
    out = tmp_path / "p.tsv"
    res.to_tsv(out)
    for ln in out.read_text().splitlines()[1:]:
        f = ln.split("\t")
        if f[5] == "N/A":
            continue
        assert int(f[5]) == min(counts[f[0]], counts[f[1]])


@pytest.mark.parametrize("spec", [None, "gtdb-bact", "GTDB_ARCH"])
def test_model_spec_forms_are_accepted_everywhere(spec, tmp_path):
    """--hmm's spellings work in the API too: the default, a packaged set by
    name, or a path."""
    from fastaai.api import _models
    assert len(_models(spec)) > 0


def test_model_spec_accepts_a_path_and_a_modelset():
    from fastaai.api import _models
    default = _models(None)
    assert len(_models(default.path)) == len(default)
    assert _models(default) is default


# --- metadata and iteration ---------------------------------------------------

def test_queries_and_targets_name_the_sides(tmp_path):
    db, res = _tiny_result(tmp_path)
    assert res.queries == res.query_names
    assert res.targets == res.target_names
    assert res.shape == (len(res.queries), len(res.targets))


def test_scps_reports_marker_counts(tmp_path):
    db, res = _tiny_result(tmp_path)
    for g in res.queries:
        assert res.scps(g) > 0
    with pytest.raises(KeyError):
        res.scps("not_a_genome")


def test_iteration_skips_self_and_empty_pairs(tmp_path):
    db, res = _tiny_result(tmp_path)
    got = list(res)
    assert all(m.query != m.target for m in got)
    assert all(m.shared > 0 for m in got), "a pair with no shared marker is no evidence"
    assert len(list(res(include_self=True))) > len(got)


def test_shared_frac_is_shared_over_possible(tmp_path):
    """The number that tells a distant pair from a poor assembly."""
    db, res = _tiny_result(tmp_path)
    for m in res:
        assert m.poss_shared > 0
        assert m.shared_frac == pytest.approx(m.shared / m.poss_shared)
        assert 0.0 < m.shared_frac <= 1.0


def test_filters_are_inclusive_and_compose(tmp_path):
    db, res = _tiny_result(tmp_path)
    everything = list(res)
    floor = min(m.aai for m in everything)

    assert len(list(res(min_aai=floor))) == len(everything), "inclusive"
    assert all(m.aai >= floor + 1 for m in res(min_aai=floor + 1))
    assert all(m.shared_frac >= 0.5 for m in res(min_shared_frac=0.5))
    assert all(m.shared >= 2 for m in res(min_shared=2))
    assert len(list(res(min_aai=floor, min_shared_frac=0.0))) == len(everything)


def test_selecting_a_side(tmp_path):
    db, res = _tiny_result(tmp_path)
    one = res.queries[0]
    assert {m.query for m in res(query=one)} == {one}
    assert {m.query for m in res(query=[one])} == {one}
    assert {m.target for m in res(target=one, include_self=True)} == {one}
    # "any" and None are the same thing: no filter.
    assert len(list(res(query="any"))) == len(list(res(query=None)))
    with pytest.raises(KeyError):
        list(res(query="nope"))


def test_min_shared_frac_needs_the_counts(tmp_path):
    """Without counts the fraction has no denominator, and silently returning
    everything would be worse than saying so."""
    db, res = _tiny_result(tmp_path)
    res.query_scps = None
    with pytest.raises(RuntimeError, match="min_shared_frac"):
        list(res(min_shared_frac=0.5))
