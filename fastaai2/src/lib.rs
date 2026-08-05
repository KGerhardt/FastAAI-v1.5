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

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use index::Partition;
use kmer::{Alphabet, Kmerizer};

/// A set of genomes k-merised against one accession list.
///
/// Build with `add_genome`, then `seal()` to construct the inverted index.
/// Query and target databases are the same type — the role is decided at call
/// time, so all-vs-all is simply `db.search(db)`.
#[pyclass]
pub struct Database {
    accessions: Vec<String>,
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
    #[pyo3(signature = (target, block = 128, threads = 1))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        target: &Database,
        block: usize,
        threads: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, usize)> {
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

        py.detach(|| {
            for (qi, qp) in self.partitions.iter().enumerate() {
                for (ti, tp) in target.partitions.iter().enumerate() {
                    if selfcmp && ti < qi {
                        continue; // mirrored from the (ti, qi) block below
                    }
                    let sym = selfcmp && ti == qi;
                    let mut j = vec![0.0f64; qp.n_genomes * tp.n_genomes];
                    let mut s = vec![0u32; qp.n_genomes * tp.n_genomes];
                    kernel::join_threaded(qp, tp, block, threads, sym, &mut j, &mut s);
                    for r in 0..qp.n_genomes {
                        let dst = (qoffs[qi] + r) * nt + toffs[ti];
                        jac[dst..dst + tp.n_genomes]
                            .copy_from_slice(&j[r * tp.n_genomes..(r + 1) * tp.n_genomes]);
                        sh[dst..dst + tp.n_genomes]
                            .copy_from_slice(&s[r * tp.n_genomes..(r + 1) * tp.n_genomes]);
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
                    }
                }
                let mut g = 0usize;
                for p in &self.partitions {
                    for local in 0..p.n_genomes {
                        let k = p.accs.iter().filter(|a| a.present[local]).count() as u32;
                        jac[g * nt + g] = k as f64;
                        sh[g * nt + g] = k;
                        g += 1;
                    }
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
        Ok((jb, sb, nq, nt))
    }
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
    m.add("DEFAULT_ALPHABET", kmer::DEFAULT_ALPHABET)?;
    m.add("DEFAULT_K", kmer::DEFAULT_K)?;
    m.add("MAX_PARTITION", index::MAX_PARTITION)?;
    Ok(())
}
