//! PyO3 bindings.
//!
//! The boundary is deliberately coarse: Python owns FASTA ingestion, gene
//! prediction and HMM search (pyfastx / pyrodigal / pyhmmer, all of which already
//! release the GIL and wrap mature C), and hands Rust the finished best-hit
//! protein sequences. Rust owns k-merisation, the inverted index, counting and
//! the AAI transform.
//!
//! Rewriting prediction or HMM search in Rust would gain nothing — there is no
//! mature pure-Rust HMMER, so it would FFI back to the same C library.

pub mod aai;
pub mod index;
pub mod kernel;
pub mod kmer;
pub mod pairwise;
pub mod store;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use index::Partition;
use kmer::{Alphabet, Kmerizer};
use store::{ManifestEntry, Schema};

/// A set of genomes k-merised against one accession list.
///
/// Build with `add_genome`, then `seal()` to construct the inverted index.
/// Query and target databases are the same type — the role is decided at call
/// time, so all-vs-all is simply `db.search(db)`.
#[pyclass]
pub struct Database {
    accessions: Vec<String>,
    /// Best-hit resolution used to build this database. Recorded because it
    /// decides which protein each accession gets, and therefore comparability.
    filter_mode: String,
    /// Free-text provenance, e.g. "GTDB R232 bac120".
    source: String,
    manifest: Vec<ManifestEntry>,
    alphabet_str: String,
    k: usize,
    kspace: usize,
    kmerizer: Kmerizer,
    names: Vec<String>,
    /// `sets[genome][accession]` — the forward index, retained because a database
    /// acting as *query* needs k-mer lists while a database acting as *target*
    /// needs the inverted index.
    sets: Vec<Vec<Vec<u32>>>,
    partitions: Vec<Partition>,
}

impl Database {
    /// Not a `#[pymethod]`: `Schema` is a Rust type with no Python conversion.
    fn schema_struct(&self) -> Schema {
        Schema {
            k: self.k,
            alphabet: self.alphabet_str.clone(),
            accessions: self.accessions.clone(),
            filter_mode: self.filter_mode.clone(),
            source: self.source.clone(),
        }
    }
}

#[pymethods]
impl Database {
    #[new]
    #[pyo3(signature = (accessions, k = kmer::DEFAULT_K, alphabet = kmer::DEFAULT_ALPHABET))]
    fn new(accessions: Vec<String>, k: usize, alphabet: &str) -> PyResult<Self> {
        if accessions.is_empty() {
            return Err(PyValueError::new_err("accession list is empty"));
        }
        let alpha = Alphabet::new(alphabet.as_bytes(), k).map_err(PyValueError::new_err)?;
        let kspace = alpha.kspace as usize;
        Ok(Database {
            accessions,
            filter_mode: String::new(),
            source: String::new(),
            manifest: Vec::new(),
            alphabet_str: alphabet.to_string(),
            k,
            kspace,
            kmerizer: Kmerizer::new(alpha),
            names: Vec::new(),
            sets: Vec::new(),
            partitions: Vec::new(),
        })
    }

    /// Add one genome. `scps` maps accession index to that accession's best-hit
    /// protein sequence. Duplicate accessions within a genome are rejected —
    /// silently keeping the last would corrupt results invisibly.
    fn add_genome(&mut self, name: &str, scps: Vec<(usize, Vec<u8>)>) -> PyResult<()> {
        if !self.partitions.is_empty() {
            return Err(PyRuntimeError::new_err("database is sealed; cannot add genomes"));
        }

        let n_acc = self.accessions.len();
        let mut per_acc: Vec<Vec<u32>> = vec![Vec::new(); n_acc];
        for (acc, seq) in scps {
            if acc >= n_acc {
                return Err(PyValueError::new_err(format!(
                    "accession index {acc} out of range (have {n_acc})"
                )));
            }
            if !per_acc[acc].is_empty() {
                return Err(PyValueError::new_err(format!(
                    "genome {name:?} supplied accession {acc} twice"
                )));
            }
            per_acc[acc] = self.kmerizer.kmers(&seq);
        }

        self.names.push(name.to_string());
        self.sets.push(per_acc);
        Ok(())
    }

    /// Build the inverted index, chunked into partitions of `PARTITION_SIZE`.
    /// Idempotent.
    ///
    /// The forward k-mer sets are dropped afterwards: a stored partition needs
    /// only the inverted index, which is ~36% of the full size and what keeps a
    /// partition inside a 2 GiB-per-thread budget with working space left over
    /// (0.56 GiB vs 1.55 GiB).
    ///
    /// Genome *i* lands in partition `i / PARTITION_SIZE` at local ID
    /// `i % PARTITION_SIZE`, so the global index is recoverable by arithmetic
    /// while posting lists store only `u16` local IDs.
    fn seal(&mut self) -> PyResult<()> {
        if !self.partitions.is_empty() {
            return Ok(());
        }
        if self.names.is_empty() {
            return Err(PyRuntimeError::new_err("cannot seal an empty database"));
        }
        // Manifest rows are built here, while the forward sets are still around:
        // the content hash is over them, and it is what makes duplicate detection
        // possible on a later merge.
        self.manifest.clear();
        for (ordinal, (name, sets)) in self.names.iter().zip(&self.sets).enumerate() {
            self.manifest.push(ManifestEntry {
                ordinal: ordinal as u32,
                partition: (ordinal / index::PARTITION_SIZE) as u32,
                local: (ordinal % index::PARTITION_SIZE) as u16,
                scp_count: sets.iter().filter(|v| !v.is_empty()).count() as u16,
                content_hash: store::content_hash(sets),
                name: name.clone(),
            });
        }
        for chunk in self.sets.chunks(index::PARTITION_SIZE) {
            let p = Partition::build(chunk, self.accessions.len(), self.kspace)
                .map_err(PyRuntimeError::new_err)?;
            self.partitions.push(p);
        }
        // The forward k-mer sets exist only to build the inverted index. Nothing
        // downstream reads them, and keeping them would roughly triple a stored
        // partition — 95 GB instead of 34 GB at GTDB scale.
        self.sets = Vec::new();
        self.sets.shrink_to_fit();
        Ok(())
    }

    #[getter]
    fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    #[getter]
    fn partition_size(&self) -> usize {
        index::PARTITION_SIZE
    }

    #[getter]
    fn n_genomes(&self) -> usize {
        self.names.len()
    }

    #[getter]
    fn genome_names(&self) -> Vec<String> {
        self.names.clone()
    }

    #[getter]
    fn accession_names(&self) -> Vec<String> {
        self.accessions.clone()
    }

    #[getter]
    fn is_sealed(&self) -> bool {
        !self.partitions.is_empty()
    }

    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    #[getter]
    fn alphabet(&self) -> String {
        self.alphabet_str.clone()
    }

    /// Number of accessions carried by each genome, in `genome_names` order.
    fn scp_counts(&self) -> Vec<usize> {
        // Presence flags in the inverted index carry this directly.
        let mut out = Vec::with_capacity(self.names.len());
        for p in &self.partitions {
            for g in 0..p.n_genomes {
                out.push(p.accs.iter().filter(|a| a.present[g]).count());
            }
        }
        out
    }

    /// Resident index footprint in bytes, or 0 if unsealed.
    fn index_bytes(&self) -> usize {
        self.partitions.iter().map(|p| p.index_bytes()).sum()
    }

    /// What a dense k-mer index would have cost, for comparison.
    fn dense_index_bytes(&self) -> usize {
        self.partitions.iter().map(|p| p.dense_index_bytes()).sum()
    }

    /// Fraction of the k-mer space occupied, averaged over accessions and
    /// partitions. Below 0.5, sparse storage is the smaller layout.
    fn occupancy(&self) -> f64 {
        if self.partitions.is_empty() {
            return 0.0;
        }
        self.partitions.iter().map(|p| p.occupancy()).sum::<f64>()
            / self.partitions.len() as f64
    }

    /// Resident bytes of the single largest partition — the figure that must fit
    /// a per-thread RAM budget when partitions are streamed one at a time.
    fn largest_partition_bytes(&self) -> usize {
        self.partitions.iter().map(|p| p.index_bytes()).max().unwrap_or(0)
    }

    /// Compatibility key over the accession list, k and alphabet.
    ///
    /// Two databases may only be compared when these match. Accession IDs are
    /// positions in a database-local list, so comparing across different model
    /// sets would yield structurally valid, biologically meaningless output.
    fn schema_key(&self) -> String {
        format!(
            "k={};alphabet={};accessions={}",
            self.k,
            self.alphabet_str,
            self.accessions.join(",")
        )
    }

    /// Best-hit filter recorded on this database.
    #[getter]
    fn filter_mode(&self) -> String { self.filter_mode.clone() }

    #[setter]
    fn set_filter_mode(&mut self, v: &str) { self.filter_mode = v.to_string(); }

    /// Free-text provenance, e.g. "GTDB R232 bac120".
    #[getter]
    fn source(&self) -> String { self.source.clone() }

    #[setter]
    fn set_source(&mut self, v: &str) { self.source = v.to_string(); }

    /// Write the database to *path* as a directory.
    ///
    /// Only the inverted index is stored — the forward k-mer sets exist to build
    /// it and nothing downstream reads them. Partitions are written as separate
    /// files so appending genomes later touches no existing partition.
    fn save(&self, path: &str) -> PyResult<()> {
        if self.partitions.is_empty() {
            return Err(PyRuntimeError::new_err("database is not sealed"));
        }
        let dir = std::path::Path::new(path);
        std::fs::create_dir_all(dir).map_err(to_py)?;
        store::write_schema(dir, &self.schema_struct()).map_err(to_py)?;
        store::write_manifest(dir, &self.manifest).map_err(to_py)?;
        for (i, p) in self.partitions.iter().enumerate() {
            store::write_partition(&dir.join(store::partition_file(i)), p).map_err(to_py)?;
        }
        Ok(())
    }

    /// Bytes on disk, by component.
    fn stored_bytes(&self, path: &str) -> PyResult<(u64, u64, u64)> {
        let dir = std::path::Path::new(path);
        let sz = |p: std::path::PathBuf| p.metadata().map(|m| m.len()).unwrap_or(0);
        let parts: u64 = store::partition_paths(dir).map_err(to_py)?.into_iter().map(sz).sum();
        Ok((parts, sz(dir.join(store::SCHEMA_FILE)), sz(dir.join(store::MANIFEST_FILE))))
    }

    /// Search every genome in `self` against every genome in `target`.
    ///
    /// Returns `(jaccard, shared, n_query, n_target)` — row-major flat buffers of
    /// `n_query * n_target`. Jaccard is the mean over shared accessions, NaN where
    /// nothing is shared.
    ///
    /// Both sides are read as inverted indexes, so no forward k-mer index is
    /// needed anywhere. `block` sets the query-blocking width: the accumulator is
    /// `block * n_target`, so it is the knob that keeps the two-dimensional
    /// accumulator inside a RAM budget. Small blocks win — 128 keeps it in L2.
    ///
    /// Self-comparison (`db.search(db, ...)`) computes the strict upper triangle
    /// and mirrors it, for a further ~1.23x.
    ///
    /// Threads are never defaulted to every logical core: scaling is memory-bound
    /// and measured *negative* past ~16 on consumer hardware.
    #[pyo3(signature = (target, block = 128, threads = 1, stdev = false))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        target: &Database,
        block: usize,
        threads: usize,
        stdev: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, usize,
                   Option<Bound<'py, PyBytes>>)> {
        if self.partitions.is_empty() || target.partitions.is_empty() {
            return Err(PyRuntimeError::new_err("both databases must be sealed"));
        }
        if self.schema_key() != target.schema_key() {
            return Err(PyValueError::new_err(
                "schema mismatch: query and target must share accession list, k and alphabet",
            ));
        }

        // Self-comparison: compute the strict upper triangle and mirror it. The
        // result is symmetric by construction (Jaccard, the shared-accession set
        // and the mean over it are all symmetric), so half the work is redundant.
        let selfcmp = std::ptr::eq(self as *const Database, target as *const Database);

        let (qoffs, nq) = kernel::partition_offsets(&self.partitions);
        let (toffs, nt) = kernel::partition_offsets(&target.partitions);
        let mut jac = vec![0.0f64; nq * nt];
        let mut sh = vec![0u32; nq * nt];
        // Only allocated when asked: at 2,943 genomes this is another 69 MB.
        let mut sq = if stdev { vec![0.0f64; nq * nt] } else { Vec::new() };

        py.detach(|| {
            for (qi, qp) in self.partitions.iter().enumerate() {
                for (ti, tp) in target.partitions.iter().enumerate() {
                    if selfcmp && ti < qi {
                        continue; // mirrored from the (ti, qi) block below
                    }
                    let sym = selfcmp && ti == qi;
                    let cells = qp.n_genomes * tp.n_genomes;
                    let mut j = vec![0.0f64; cells];
                    let mut s = vec![0u32; cells];
                    let mut q = if stdev { vec![0.0f64; cells] } else { Vec::new() };
                    kernel::join_threaded(
                        qp, tp, block, threads, sym, &mut j, &mut s,
                        if stdev { Some(&mut q) } else { None },
                    );
                    for r in 0..qp.n_genomes {
                        let dst = (qoffs[qi] + r) * nt + toffs[ti];
                        let src = r * tp.n_genomes..(r + 1) * tp.n_genomes;
                        jac[dst..dst + tp.n_genomes].copy_from_slice(&j[src.clone()]);
                        sh[dst..dst + tp.n_genomes].copy_from_slice(&s[src.clone()]);
                        if stdev {
                            sq[dst..dst + tp.n_genomes].copy_from_slice(&q[src]);
                        }
                    }
                }
            }
            if selfcmp {
                // Mirror the upper triangle and fill the diagonal, which is
                // Jaccard 1.0 over every accession the genome carries.
                for i in 0..nq {
                    for t in (i + 1)..nt {
                        jac[t * nt + i] = jac[i * nt + t];
                        sh[t * nt + i] = sh[i * nt + t];
                        if stdev {
                            sq[t * nt + i] = sq[i * nt + t];
                        }
                    }
                }
                let mut g = 0usize;
                for p in &self.partitions {
                    for local in 0..p.n_genomes {
                        let k = p.accs.iter().filter(|a| a.present[local]).count() as u32;
                        jac[g * nt + g] = k as f64;
                        sh[g * nt + g] = k;
                        if stdev {
                            sq[g * nt + g] = k as f64; // every accession Jaccard 1.0
                        }
                        g += 1;
                    }
                }
            }
            // Standard deviation first: it needs the unreduced sum.
            if stdev {
                for i in 0..jac.len() {
                    sq[i] = kernel::stdev_from(jac[i], sq[i], sh[i]);
                }
            }
            for i in 0..jac.len() {
                jac[i] = if sh[i] == 0 { f64::NAN } else { jac[i] / sh[i] as f64 };
            }
        });

        let jb = PyBytes::new(py, unsafe {
            std::slice::from_raw_parts(jac.as_ptr() as *const u8, jac.len() * 8)
        });
        let sb = PyBytes::new(py, unsafe {
            std::slice::from_raw_parts(sh.as_ptr() as *const u8, sh.len() * 4)
        });
        let qb = if stdev {
            Some(PyBytes::new(py, unsafe {
                std::slice::from_raw_parts(sq.as_ptr() as *const u8, sq.len() * 8)
            }))
        } else {
            None
        };
        Ok((jb, sb, nq, nt, qb))
    }
}

fn to_py(e: std::io::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Open a database written by `Database.save`.
///
/// Reads schema, manifest and every partition. Partitions are separate files so
/// lazy per-partition loading is a later change that needs no format revision —
/// which is what a GTDB-scale database will want, since only one partition needs
/// to be resident at a time.
#[pyfunction]
fn open_database(path: &str) -> PyResult<Database> {
    let dir = std::path::Path::new(path);
    let schema = store::read_schema(dir).map_err(to_py)?;
    let manifest = store::read_manifest(dir).map_err(to_py)?;
    let alpha = Alphabet::new(schema.alphabet.as_bytes(), schema.k)
        .map_err(PyValueError::new_err)?;
    let kspace = alpha.kspace as usize;

    let mut partitions = Vec::new();
    for p in store::partition_paths(dir).map_err(to_py)? {
        let part = store::read_partition(&p).map_err(to_py)?;
        if part.n_acc != schema.accessions.len() || part.kspace != kspace {
            return Err(PyRuntimeError::new_err(format!(
                "{}: partition disagrees with schema ({} accessions / kspace {} \
                 vs {} / {})",
                p.display(), part.n_acc, part.kspace, schema.accessions.len(), kspace
            )));
        }
        partitions.push(part);
    }

    let names = manifest.iter().map(|m| m.name.clone()).collect();
    Ok(Database {
        accessions: schema.accessions,
        filter_mode: schema.filter_mode,
        source: schema.source,
        manifest,
        alphabet_str: schema.alphabet,
        k: schema.k,
        kspace,
        kmerizer: Kmerizer::new(alpha),
        names,
        sets: Vec::new(), // forward sets are not stored and are never needed
        partitions,
    })
}

/// Merge databases into `output`.
///
/// Cheap by construction: local genome IDs reference nothing outside their own
/// partition, so merging copies partition files and rewrites the manifest.
/// **No posting list is read, renumbered or rewritten.** Cost scales with the
/// number of genomes, not with the size of the databases.
///
/// Genomes already present — same content hash *and* name — are skipped rather
/// than duplicated. Ordinals are reassigned across the merged set, which
/// invalidates any stored result matrix keyed to the old order.
///
/// Returns `(genomes_written, duplicates_skipped, partitions)`.
#[pyfunction]
fn merge_databases(output: &str, inputs: Vec<String>) -> PyResult<(usize, usize, usize)> {
    if inputs.is_empty() {
        return Err(PyValueError::new_err("no input databases"));
    }
    let out = std::path::Path::new(output);

    let first = store::read_schema(std::path::Path::new(&inputs[0])).map_err(to_py)?;
    let mut manifest: Vec<ManifestEntry> = Vec::new();
    let mut seen: std::collections::HashSet<(u64, String)> = std::collections::HashSet::new();
    let mut skipped = 0usize;
    let mut next_partition = 0usize;

    std::fs::create_dir_all(out).map_err(to_py)?;

    for path in &inputs {
        let dir = std::path::Path::new(path);
        let schema = store::read_schema(dir).map_err(to_py)?;
        first.compatible_with(&schema).map_err(|e| {
            PyValueError::new_err(format!("{path}: incompatible with {}: {e}", inputs[0]))
        })?;

        let entries = store::read_manifest(dir).map_err(to_py)?;
        let parts = store::partition_paths(dir).map_err(to_py)?;

        // Which of this donor's partitions survive, and where they land.
        let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut kept: Vec<ManifestEntry> = Vec::new();
        for e in entries {
            if !seen.insert((e.content_hash, e.name.clone())) {
                skipped += 1;
                continue;
            }
            kept.push(e);
        }
        for e in &kept {
            if !remap.contains_key(&e.partition) {
                let dest = next_partition;
                next_partition += 1;
                remap.insert(e.partition, dest as u32);
                let src = parts.get(e.partition as usize).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "{path}: manifest references partition {} but only {} exist",
                        e.partition, parts.len()
                    ))
                })?;
                std::fs::copy(src, out.join(store::partition_file(dest))).map_err(to_py)?;
            }
        }
        for mut e in kept {
            e.partition = remap[&e.partition];
            e.ordinal = manifest.len() as u32;
            manifest.push(e);
        }
    }

    if manifest.is_empty() {
        return Err(PyRuntimeError::new_err("merge produced no genomes"));
    }

    let mut schema = first;
    schema.source = format!("merge of {} database(s)", inputs.len());
    store::write_schema(out, &schema).map_err(to_py)?;
    store::write_manifest(out, &manifest).map_err(to_py)?;
    Ok((manifest.len(), skipped, next_partition))
}

/// Jaccard to AAI percentage. NaN in, NaN out; uncensored by design.
#[pyfunction]
fn jaccard_to_aai(kaai: f64) -> f64 {
    aai::kaai_to_aai(kaai)
}

/// Jaccard corresponding to a given AAI, by inverting the same fit.
#[pyfunction]
fn aai_to_jaccard(aai: f64) -> f64 {
    aai::aai_to_kaai(aai)
}

/// Compare two genomes directly, without building an index.
///
/// Each argument is a list of `(accession_index, protein_sequence)`. Returns
/// `(mean_jaccard, shared_accessions, aai)`; Jaccard and AAI are NaN when the two
/// genomes share no accession, which is not the same as similarity zero.
///
/// This is the honest way to do a handful of comparisons — building an index for
/// two genomes is silly. It is also the oracle the indexed path is validated
/// against, sharing none of its machinery.
#[pyfunction]
#[pyo3(signature = (query, target, n_accessions, k = kmer::DEFAULT_K, alphabet = kmer::DEFAULT_ALPHABET))]
fn compare_pair(
    query: Vec<(usize, Vec<u8>)>,
    target: Vec<(usize, Vec<u8>)>,
    n_accessions: usize,
    k: usize,
    alphabet: &str,
) -> PyResult<(f64, u32, f64)> {
    let alpha = Alphabet::new(alphabet.as_bytes(), k).map_err(PyValueError::new_err)?;
    let mut km = Kmerizer::new(alpha);

    let mut build = |scps: Vec<(usize, Vec<u8>)>| -> PyResult<Vec<Vec<u32>>> {
        let mut out = vec![Vec::new(); n_accessions];
        for (acc, seq) in scps {
            if acc >= n_accessions {
                return Err(PyValueError::new_err(format!(
                    "accession index {acc} out of range (have {n_accessions})"
                )));
            }
            out[acc] = km.kmers(&seq);
        }
        Ok(out)
    };

    let q = build(query)?;
    let t = build(target)?;
    let c = pairwise::compare(&q, &t);
    Ok((c.mean_jaccard(), c.shared, c.aai()))
}

/// Sorted unique k-mer IDs for one sequence — exposed for testing and inspection.
#[pyfunction]
#[pyo3(signature = (seq, k = kmer::DEFAULT_K, alphabet = kmer::DEFAULT_ALPHABET))]
fn kmerize(seq: &[u8], k: usize, alphabet: &str) -> PyResult<Vec<u32>> {
    let alpha = Alphabet::new(alphabet.as_bytes(), k).map_err(PyValueError::new_err)?;
    Ok(Kmerizer::new(alpha).kmers(seq))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_function(wrap_pyfunction!(jaccard_to_aai, m)?)?;
    m.add_function(wrap_pyfunction!(aai_to_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(kmerize, m)?)?;
    m.add_function(wrap_pyfunction!(compare_pair, m)?)?;
    m.add_function(wrap_pyfunction!(open_database, m)?)?;
    m.add_function(wrap_pyfunction!(merge_databases, m)?)?;
    m.add("DEFAULT_ALPHABET", kmer::DEFAULT_ALPHABET)?;
    m.add("DEFAULT_K", kmer::DEFAULT_K)?;
    m.add("MAX_PARTITION", index::MAX_PARTITION)?;
    Ok(())
}
