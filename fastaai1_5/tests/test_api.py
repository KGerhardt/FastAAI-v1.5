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
                             save=False, processes=2)
    if kind == "proteins":
        got = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "b"),
                                 save=False, processes=2)
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
                       processes=2)
    assert (root / "crystals").is_dir()
    assert (root / "hmm_hits").is_dir()
    assert (root / "database" / "firm").is_dir(), "named database under database/"


def test_preprocess_combines_ranks(tmp_path):
    """Ranks are additive: crystals from elsewhere join the ones just made."""
    a = [_protein_file(tmp_path, "x0")]
    first = tmp_path / "first"
    fastaai.preprocess(proteins=a, directory=str(first), save=False, processes=1)

    b = [_protein_file(tmp_path, "y0")]
    combined = fastaai.preprocess(
        proteins=b, crystals=str(first / "crystals"),
        directory=str(tmp_path / "both"), save=False, processes=1)
    assert combined.n_genomes == 2, combined.genome_names


# --- reading a result ---------------------------------------------------------

def _tiny_result(tmp_path):
    prots = [_protein_file(tmp_path, f"r{i}") for i in range(3)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "res"),
                            save=False, processes=2)
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
                            save=False, processes=2)
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


# --- parallel crystallising ---------------------------------------------------

def test_processes_give_identical_crystals(tmp_path):
    """Genomes are independent and write their own files, so the only thing to
    guard is that going wide changes nothing but the wall clock."""
    prots = [_protein_file(tmp_path, f"w{i}") for i in range(6)]
    hmms = [fastaai.protein_to_hmm(p, tmp_path / "hh") for p in prots]
    pairs = list(zip(prots, hmms))

    serial = fastaai.prot_hmm_to_crystal(pairs, tmp_path / "s", processes=1)
    wide = fastaai.prot_hmm_to_crystal(pairs, tmp_path / "p", processes=4)

    assert [p.name for p in serial] == [p.name for p in wide]
    for a, b in zip(sorted(serial), sorted(wide)):
        assert a.read_text() == b.read_text()


# --- unit / driver pairs ------------------------------------------------------

def test_every_step_has_a_unit_and_a_driver():
    assert hasattr(fastaai, "genome_to_protein") and hasattr(fastaai, "genomes_to_proteins")
    assert hasattr(fastaai, "protein_to_hmm") and hasattr(fastaai, "proteins_to_hmms")
    assert hasattr(fastaai, "all_steps") and hasattr(fastaai, "preprocess")


def test_driver_preserves_input_order(tmp_path):
    """`imap_unordered` yields as workers finish, so order is restored by index.
    Callers pair inputs with outputs positionally; losing that silently would be
    a nasty way to mislabel a genome."""
    prots = [_protein_file(tmp_path, f"o{i}") for i in range(5)]
    shuffled = [prots[3], prots[0], prots[4], prots[1]]
    out = fastaai.proteins_to_hmms(shuffled, tmp_path / "hh", processes=3)
    assert [p.stem for p in out] == [p.stem for p in shuffled]


def test_driver_output_matches_serial(tmp_path):
    prots = [_protein_file(tmp_path, f"m{i}") for i in range(4)]
    one = fastaai.proteins_to_hmms(prots, tmp_path / "s", processes=1)
    many = fastaai.proteins_to_hmms(prots, tmp_path / "p", processes=4)
    assert [a.read_text() for a in one] == [b.read_text() for b in many]


def test_all_steps_writes_every_rank_for_one_genome(tmp_path):
    """The whole chain in one process, nothing returned between stages."""
    prot = _protein_file(tmp_path, "solo")
    root = tmp_path / "one"
    rec = fastaai.all_steps(prot, root, input_kind="protein")

    assert rec.name == "solo" and rec.error is None
    assert (root / "proteins" / "solo.fasta").exists()
    assert (root / "hmm_hits" / "solo.tsv").exists()
    # The record carries counts, not sequence — that is what lets it cross a
    # process boundary cheaply.
    assert rec.n_proteins > 0 and rec.scps == {}


def test_models_are_built_once_per_worker_not_per_task(tmp_path, monkeypatch):
    """The pool initializer is what makes reusing HMM data free per genome.

    Built per task instead, every genome would re-parse 9.2 MB of HMM text —
    the mistake FastAAI 1 made, where a 16-process pool spent ~66 s on parsing
    alone.
    """
    import os
    from fastaai import api

    marks = tmp_path / "builds"
    marks.mkdir()
    real = api._models

    def counting(spec):
        m = real(spec)
        (marks / f"{os.getpid()}_{len(list(marks.iterdir()))}").write_text("x")
        return m

    monkeypatch.setattr(api, "_models", counting)

    prots = [_protein_file(tmp_path, f"init{i}") for i in range(6)]
    fastaai.proteins_to_hmms(prots, tmp_path / "hh", processes=3)

    builds = list(marks.iterdir())
    workers = {b.name.split("_")[0] for b in builds}
    assert len(builds) == len(workers), (
        f"{len(builds)} model builds across {len(workers)} workers — "
        "the initializer should build exactly one per process"
    )
    assert len(builds) < len(prots), "fewer builds than tasks"


def test_parallel_preprocessing_requires_an_output_directory(tmp_path):
    """A worker with nowhere to write must hand its proteins back, and they
    would be pickled across the process boundary — the cost the whole design
    avoids, paid silently. Refuse instead."""
    from fastaai.pipeline import preprocess_paths
    from fastaai.search import ModelSet

    prots = [_protein_file(tmp_path, "guard")]
    with pytest.raises(ValueError, match="needs somewhere to write"):
        preprocess_paths(prots, ModelSet(), processes=4)

    # Serial is fine: nothing crosses a boundary, so the data is free to keep.
    recs = preprocess_paths(prots, ModelSet(), processes=1, input_kind="protein")
    assert recs[0].proteins is not None


# --- inspecting a database ----------------------------------------------------

def test_dump_orientations_describe_the_same_index(tmp_path):
    """by_genome transposes the CSR, by_kmer emits it as stored. They are two
    views of one thing, so their posting-entry totals must agree."""
    prots = [_protein_file(tmp_path, f"d{i}") for i in range(3)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "dd"),
                            save=False, processes=2)

    out = tmp_path / "inspect"
    written = fastaai.dump_database(db, out, orientation="both", full=True)

    import json

    by_genome = sum(int(r.split("\t")[2])
                    for r in written["by_genome"].read_text().splitlines()[1:])
    doc = json.loads(written["by_kmer"].read_text())
    by_kmer = sum(len(v) for part in doc["partitions"]
                  for acc in part["accessions"].values() for v in acc.values())
    assert by_genome == by_kmer


def test_dump_writes_the_metadata_files(tmp_path):
    prots = [_protein_file(tmp_path, f"e{i}") for i in range(2)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "ee"),
                            save=False, processes=1)
    written = fastaai.dump_database(db, tmp_path / "insp", orientation="genome")

    for key in ("schema", "accessions", "genomes", "by_genome"):
        assert written[key].exists(), key
    assert "by_kmer" not in written, "only the requested orientation is written"


def test_by_kmer_is_valid_json_and_nests_away_the_repetition(tmp_path):
    """A flat table repeats partition and accession on every row; at index scale
    that repetition is most of the file."""
    import json

    prots = [_protein_file(tmp_path, f"j{i}") for i in range(3)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "jj"),
                            save=False, processes=1)
    written = fastaai.dump_database(db, tmp_path / "js", orientation="kmer",
                                    full=True)

    doc = json.loads(written["by_kmer"].read_text())
    assert doc["genomes"] == db.genome_names, "names appear once, at the top"
    assert doc["members"] is True
    assert len(doc["partitions"]) == db.n_partitions

    # Every posting is an ordinal into that list, not a repeated name.
    for part in doc["partitions"]:
        for kmers in part["accessions"].values():
            for members in kmers.values():
                assert all(0 <= g < db.n_genomes for g in members)


def test_by_kmer_counts_only_when_not_full(tmp_path):
    import json

    prots = [_protein_file(tmp_path, "k0")]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "kk"),
                            save=False, processes=1)
    written = fastaai.dump_database(db, tmp_path / "kc", orientation="kmer")
    doc = json.loads(written["by_kmer"].read_text())
    assert doc["members"] is False
    counts = [v for p in doc["partitions"] for a in p["accessions"].values()
              for v in a.values()]
    assert counts and all(isinstance(v, int) for v in counts)

    genomes = written["genomes"].read_text().splitlines()
    assert genomes[0] == "genome\tpartition\tlocal_id\tn_markers"
    assert len(genomes) == 1 + db.n_genomes


def test_dump_rejects_an_unknown_orientation(tmp_path):
    prots = [_protein_file(tmp_path, "f0")]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "ff"),
                            save=False, processes=1)
    with pytest.raises(ValueError, match="orientation"):
        fastaai.dump_database(db, tmp_path / "x", orientation="sideways")


def test_describe_database_reports_the_schema(tmp_path):
    prots = [_protein_file(tmp_path, f"g{i}") for i in range(2)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "gg"),
                            save=False, processes=1)
    info = fastaai.describe_database(db)
    assert info["genomes"] == db.n_genomes
    assert info["k"] == db.k
    assert info["models"] == db.models
    assert info["alphabet"] == db.alphabet


def test_database_lands_directly_in_the_database_directory(tmp_path):
    """`<root>/database/` *is* the database — no name level in between, so the
    path can be handed straight back to a query."""
    prots = [_protein_file(tmp_path, f"p{i}") for i in range(2)]
    root = tmp_path / "flat"
    fastaai.preprocess(proteins=prots, directory=str(root), processes=1)

    assert (root / "database" / "schema").exists()
    assert not (root / "database" / "database").exists(), "no stutter"
    assert fastaai.open_database(str(root / "database")).n_genomes == 2


def test_naming_a_database_adds_levels_beneath(tmp_path):
    prots = [_protein_file(tmp_path, "q0")]
    root = tmp_path / "named"
    fastaai.preprocess(proteins=prots, directory=str(root), database="alt/run2",
                       processes=1)
    assert (root / "database" / "alt" / "run2" / "schema").exists()


def test_an_absolute_database_path_escapes_the_root(tmp_path):
    prots = [_protein_file(tmp_path, "r0")]
    elsewhere = tmp_path / "elsewhere" / "db"
    fastaai.preprocess(proteins=prots, directory=str(tmp_path / "root2"),
                       database=str(elsewhere), processes=1)
    assert (elsewhere / "schema").exists()


# --- reshaping block results --------------------------------------------------

def _fake_blocks(tmp_path, nq=3, splits=((0, 2), (2, 5))):
    """Hand-built blocks for one query partition over several target blocks.

    A search large enough to produce a real grid needs >16,384 genomes, so the
    interleaving is exercised on constructed files instead. The property under
    test is ordering: a query's row must come out with its targets in block
    order, not shuffled.
    """
    out = tmp_path / "blocks"
    out.mkdir()
    head = ("query\ttarget\tavg_jacc_sim\tjacc_SD\tnum_shared_SCPs"
            "\tposs_shared_SCPs\tAAI_estimate")
    for ti, (lo, hi) in enumerate(splits):
        lines = [head]
        for q in range(nq):
            for t in range(lo, hi):
                lines.append(f"q{q}\tt{t}\t0.1\tN/A\t50\t60\t{40 + t}.00")
        (out / f"block_q00000_t{ti:05d}.tsv").write_text("\n".join(lines) + "\n")
    return out


def test_per_genome_gathers_a_query_across_its_target_blocks(tmp_path):
    from fastaai import reshape

    blocks = _fake_blocks(tmp_path)
    written = reshape.per_genome(blocks, tmp_path / "pg")
    assert len(written) == 3

    rows = [l.split("\t") for l in
            (tmp_path / "pg" / "q0.tsv").read_text().splitlines()[1:]]
    assert [r[0] for r in rows] == ["q0"] * 5, "one genome per file"
    assert [r[1] for r in rows] == ["t0", "t1", "t2", "t3", "t4"], \
        "targets in block order, blocks in target-partition order"


def test_matrix_from_blocks_spans_every_target_block(tmp_path):
    from fastaai import reshape

    blocks = _fake_blocks(tmp_path)
    dest = reshape.to_matrix(blocks, tmp_path / "m.matrix")
    lines = dest.read_text().splitlines()
    assert lines[0].split("\t")[1:] == ["t0", "t1", "t2", "t3", "t4"]
    assert len(lines) == 1 + 3
    assert lines[1].split("\t")[0] == "q0"


def test_matrix_maps_the_categorical_labels(tmp_path):
    """A TSV cell can read `>90%`; a matrix cell holds a number, so v1's
    sentinels are used. The mapping is the engine's, not a copy."""
    from fastaai import reshape

    out = tmp_path / "b"
    out.mkdir()
    head = ("query\ttarget\tavg_jacc_sim\tjacc_SD\tnum_shared_SCPs"
            "\tposs_shared_SCPs\tAAI_estimate")
    (out / "block_q00000_t00000.tsv").write_text(
        head + "\nq0\tt0\t0.9\tN/A\t50\t60\t>90%\n"
             + "q0\tt1\tN/A\tN/A\tN/A\tN/A\tN/A\n"
             + "q0\tt2\t0.01\tN/A\t2\t60\t<30%\n")
    cells = reshape.to_matrix(out, tmp_path / "m2").read_text().splitlines()[1].split("\t")
    assert cells[1:] == ["95.0", "N/A", "15.0"]


def test_reshape_rejects_a_non_result_file(tmp_path):
    from fastaai import reshape

    bad = tmp_path / "block_q00000_t00000.tsv"
    bad.write_text("not\ta\tresult\n1\t2\t3\n")
    with pytest.raises(ValueError, match="not a FastAAI result TSV"):
        reshape.per_genome(bad, tmp_path / "out")


def test_reshaped_matrix_matches_the_engines_own(tmp_path):
    """Two ways to a matrix - the engine writing one directly, and reshaping its
    TSV - must agree, or the label mapping has drifted."""
    prots = [_protein_file(tmp_path, f"mm{i}") for i in range(3)]
    db = fastaai.preprocess(proteins=prots, directory=str(tmp_path / "mmr"),
                            save=False, processes=1)
    from fastaai import reshape

    native = tmp_path / "native.matrix"
    tsv = tmp_path / "block_q00000_t00000.tsv"
    db.write_block(db, 0, 0, str(native), 128, 1, False, "both", "matrix")
    db.write_block(db, 0, 0, str(tsv), 128, 1, False, "both", "tsv")

    reshaped = reshape.to_matrix(tsv, tmp_path / "reshaped.matrix")
    assert reshaped.read_text() == native.read_text()
