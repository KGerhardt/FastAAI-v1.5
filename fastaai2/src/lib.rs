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
pub mod report;
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
    /// Digest of the HMM set this was built against; empty when unknown.
    models: String,
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
    /// Resident partitions. Populated by `seal`; empty for a database opened
    /// from disk, which loads partitions on demand instead — see `part`.
    partitions: Vec<Partition>,
    /// Partition files, when this database is backed by a directory. A database
    /// far larger than RAM is searched by reloading these, never by holding
    /// them all resident.
    part_paths: Vec<std::path::PathBuf>,
    /// Genomes per partition, taken from the manifest so that partition offsets
    /// are known without reading a single partition file.
    part_genomes: Vec<usize>,
}

/// A partition either borrowed from memory or loaded for this use alone.
///
/// The distinction is invisible to callers, which is the point: the same search
/// code serves a database built in memory and one streamed from disk.
enum PartRef<'a> {
    Resident(&'a Partition),
    Loaded(Box<Partition>),
}

impl std::ops::Deref for PartRef<'_> {
    type Target = Partition;
    fn deref(&self) -> &Partition {
        match self {
            PartRef::Resident(p) => p,
            PartRef::Loaded(p) => p,
        }
    }
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
            models: self.models.clone(),
        }
    }

    /// Partition count, whichever way this database is backed.
    fn n_parts(&self) -> usize {
        if self.part_paths.is_empty() { self.partitions.len() } else { self.part_paths.len() }
    }

    fn sealed(&self) -> bool {
        !self.partitions.is_empty() || !self.part_paths.is_empty()
    }

    /// Borrow partition *i*, reading it from disk if this database is not
    /// resident. The caller drops it when done, so peak footprint is bounded by
    /// the partitions actually in hand rather than by the size of the database.
    fn part(&self, i: usize) -> std::io::Result<PartRef<'_>> {
        if self.part_paths.is_empty() {
            Ok(PartRef::Resident(&self.partitions[i]))
        } else {
            store::read_partition(&self.part_paths[i]).map(|p| PartRef::Loaded(Box::new(p)))
        }
    }

    /// Genomes in each partition, and the running offset of each into the full
    /// genome order. Reads no partition files.
    fn offsets(&self) -> (Vec<usize>, usize) {
        let counts: Vec<usize> = if self.part_paths.is_empty() {
            self.partitions.iter().map(|p| p.n_genomes).collect()
        } else {
            self.part_genomes.clone()
        };
        let mut offs = Vec::with_capacity(counts.len());
        let mut total = 0usize;
        for c in counts {
            offs.push(total);
            total += c;
        }
        (offs, total)
    }

    /// Fold over every partition, one resident at a time.
    fn fold_parts<T>(&self, init: T, mut f: impl FnMut(T, &Partition) -> T) -> std::io::Result<T> {
        let mut acc = init;
        for i in 0..self.n_parts() {
            let p = self.part(i)?;
            acc = f(acc, &p);
        }
        Ok(acc)
    }

    /// Compute one `q x t` block, reduced to final values.
    ///
    /// Holds two partitions and one block-sized result — never the database and
    /// never the full matrix. Shared by the buffer-returning and file-writing
    /// entry points so the two cannot drift.
    fn block_values(
        &self,
        target: &Database,
        qi: usize,
        ti: usize,
        block: usize,
        threads: usize,
        stdev: bool,
    ) -> std::io::Result<Block> {
        // Symmetric only when this is literally the same partition of the same
        // database. Blocks (qi, ti) and (ti, qi) of a self-search are transposes,
        // but each is written independently, so neither is skipped here.
        let selfblock =
            std::ptr::eq(self as *const Database, target as *const Database) && qi == ti;

        let qp = self.part(qi)?;
        let tp = if selfblock { None } else { Some(target.part(ti)?) };
        let tpr: &Partition = match &tp {
            Some(p) => p,
            None => &qp,
        };

        let (nq, nt) = (qp.n_genomes, tpr.n_genomes);
        let cells = nq * nt;
        let mut jac = vec![0.0f64; cells];
        let mut sh = vec![0u32; cells];
        let mut sq = if stdev { vec![0.0f64; cells] } else { Vec::new() };

        kernel::join_threaded(
            &qp, tpr, block, threads, selfblock, &mut jac, &mut sh,
            if stdev { Some(&mut sq) } else { None },
        );
        if selfblock {
            for i in 0..nq {
                for t in (i + 1)..nt {
                    jac[t * nt + i] = jac[i * nt + t];
                    sh[t * nt + i] = sh[i * nt + t];
                    if stdev {
                        sq[t * nt + i] = sq[i * nt + t];
                    }
                }
            }
            for local in 0..nq {
                let k = qp.accs.iter().filter(|a| a.present[local]).count() as u32;
                jac[local * nt + local] = k as f64;
                sh[local * nt + local] = k;
                if stdev {
                    sq[local * nt + local] = k as f64;
                }
            }
        }
        if stdev {
            for i in 0..cells {
                sq[i] = kernel::stdev_from(jac[i], sq[i], sh[i]);
            }
        }
        for i in 0..cells {
            jac[i] = if sh[i] == 0 { f64::NAN } else { jac[i] / sh[i] as f64 };
        }

        // Accessions carried by each genome. `poss_shared_SCPs` is the smaller
        // of the two: a pair cannot share more markers than the poorer genome
        // has.
        let count = |p: &Partition, g: usize| {
            p.accs.iter().filter(|a| a.present[g]).count() as u32
        };
        let qcounts = (0..nq).map(|g| count(&qp, g)).collect();
        let tcounts = (0..nt).map(|g| count(tpr, g)).collect();

        Ok(Block { jac, sh, sq, nq, nt, qcounts, tcounts, selfblock })
    }
}

/// One computed `q x t` block, reduced to final values.
struct Block {
    jac: Vec<f64>,
    sh: Vec<u32>,
    sq: Vec<f64>,
    nq: usize,
    nt: usize,
    /// Accessions per genome, for `poss_shared_SCPs`.
    qcounts: Vec<u32>,
    tcounts: Vec<u32>,
    /// True when this block is a partition against itself in the same database,
    /// which is the only place a genome meets itself. Its diagonal is identity
    /// by definition rather than by measurement.
    selfblock: bool,
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
            models: String::new(),
            manifest: Vec::new(),
            alphabet_str: alphabet.to_string(),
            k,
            kspace,
            kmerizer: Kmerizer::new(alpha),
            names: Vec::new(),
            sets: Vec::new(),
            partitions: Vec::new(),
            part_paths: Vec::new(),
            part_genomes: Vec::new(),
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
        self.n_parts()
    }

    /// True when partitions are read from disk per block rather than held
    /// resident — the mode a database larger than RAM must be searched in.
    #[getter]
    fn is_streamed(&self) -> bool {
        !self.part_paths.is_empty()
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
        self.sealed()
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
    fn scp_counts(&self) -> PyResult<Vec<usize>> {
        // Presence flags in the inverted index carry this directly.
        self.fold_parts(Vec::with_capacity(self.names.len()), |mut out, p| {
            for g in 0..p.n_genomes {
                out.push(p.accs.iter().filter(|a| a.present[g]).count());
            }
            out
        })
        .map_err(to_py)
    }

    /// Index footprint in bytes, or 0 if unsealed. For a streamed database this
    /// is what the index *would* occupy resident, not what it currently does.
    fn index_bytes(&self) -> PyResult<usize> {
        self.fold_parts(0usize, |a, p| a + p.index_bytes()).map_err(to_py)
    }

    /// What a dense k-mer index would have cost, for comparison.
    fn dense_index_bytes(&self) -> PyResult<usize> {
        self.fold_parts(0usize, |a, p| a + p.dense_index_bytes()).map_err(to_py)
    }

    /// Fraction of the k-mer space occupied, averaged over accessions and
    /// partitions. Below 0.5, sparse storage is the smaller layout.
    fn occupancy(&self) -> PyResult<f64> {
        let n = self.n_parts();
        if n == 0 {
            return Ok(0.0);
        }
        let sum = self.fold_parts(0.0f64, |a, p| a + p.occupancy()).map_err(to_py)?;
        Ok(sum / n as f64)
    }

    /// Resident bytes of the single largest partition — the figure that must fit
    /// a per-thread RAM budget, since a streamed search holds at most the query
    /// and target partitions of one block.
    fn largest_partition_bytes(&self) -> PyResult<usize> {
        self.fold_parts(0usize, |a, p| a.max(p.index_bytes())).map_err(to_py)
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

    /// Digest of the HMM set this database was built against, or empty when it
    /// was built without one. See `ModelSet.fingerprint`.
    #[getter]
    fn models(&self) -> String { self.models.clone() }

    #[setter]
    fn set_models(&mut self, v: &str) { self.models = v.to_string(); }

    /// Write the database to *path* as a directory.
    ///
    /// Only the inverted index is stored — the forward k-mer sets exist to build
    /// it and nothing downstream reads them. Partitions are written as separate
    /// files so appending genomes later touches no existing partition.
    fn save(&self, path: &str) -> PyResult<()> {
        if !self.sealed() {
            return Err(PyRuntimeError::new_err("database is not sealed"));
        }
        let dir = std::path::Path::new(path);
        std::fs::create_dir_all(dir).map_err(to_py)?;
        store::write_schema(dir, &self.schema_struct()).map_err(to_py)?;
        store::write_manifest(dir, &self.manifest).map_err(to_py)?;
        for i in 0..self.n_parts() {
            let dest = dir.join(store::partition_file(i));
            if self.part_paths.is_empty() {
                store::write_partition(&dest, &self.partitions[i]).map_err(to_py)?;
            } else if self.part_paths[i] != dest {
                // Streamed: the bytes on disk are already the ones we would
                // write, so copy rather than round-tripping through RAM.
                std::fs::copy(&self.part_paths[i], &dest).map_err(to_py)?;
            }
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
        if !self.sealed() || !target.sealed() {
            return Err(PyRuntimeError::new_err("both databases must be sealed"));
        }
        if self.schema_key() != target.schema_key() {
            return Err(PyValueError::new_err(
                "schema mismatch: query and target must share accession list, k and alphabet",
            ));
        }
        self.schema_struct()
            .compatible_with(&target.schema_struct())
            .map_err(PyValueError::new_err)?;

        // Self-comparison: compute the strict upper triangle and mirror it. The
        // result is symmetric by construction (Jaccard, the shared-accession set
        // and the mean over it are all symmetric), so half the work is redundant.
        let selfcmp = std::ptr::eq(self as *const Database, target as *const Database);

        let (qoffs, nq) = self.offsets();
        let (toffs, nt) = target.offsets();
        let mut jac = vec![0.0f64; nq * nt];
        let mut sh = vec![0u32; nq * nt];
        // Only allocated when asked: at 2,943 genomes this is another 69 MB.
        let mut sq = if stdev { vec![0.0f64; nq * nt] } else { Vec::new() };

        let outcome: std::io::Result<()> = py.detach(|| {
            for qi in 0..self.n_parts() {
                // One load per outer block; the inner side reloads beneath it.
                // Peak footprint is these two partitions, never the database.
                let qp = self.part(qi)?;
                for ti in 0..target.n_parts() {
                    if selfcmp && ti < qi {
                        continue; // mirrored from the (ti, qi) block below
                    }
                    let tp = if selfcmp && ti == qi {
                        None // same partition; reuse `qp` rather than load it twice
                    } else {
                        Some(target.part(ti)?)
                    };
                    let tp: &Partition = match &tp {
                        Some(p) => p,
                        None => &qp,
                    };
                    let sym = selfcmp && ti == qi;
                    let cells = qp.n_genomes * tp.n_genomes;
                    let mut j = vec![0.0f64; cells];
                    let mut s = vec![0u32; cells];
                    let mut q = if stdev { vec![0.0f64; cells] } else { Vec::new() };
                    kernel::join_threaded(
                        &qp, tp, block, threads, sym, &mut j, &mut s,
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
                for i in 0..self.n_parts() {
                    let p = self.part(i)?;
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
            Ok(())
        });
        outcome.map_err(to_py)?;

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

    /// Genomes in each partition, in partition order.
    ///
    /// Lets a caller slice `genome_names` per block without opening a partition.
    #[getter]
    fn partition_genomes(&self) -> Vec<usize> {
        if self.part_paths.is_empty() {
            self.partitions.iter().map(|p| p.n_genomes).collect()
        } else {
            self.part_genomes.clone()
        }
    }

    /// Search one query partition against one target partition.
    ///
    /// The unit a large search is actually made of. Each block is independent:
    /// it holds two partitions and a `q x t` result, both bounded by
    /// `PARTITION_SIZE` rather than by the size of either database, and it can
    /// be computed in any order or by any process. That is what makes an
    /// all-vs-all at GTDB scale expressible — the full matrix never exists.
    ///
    /// Results are final, not partial sums: the mean over shared accessions is
    /// already taken. A self-block (same database, `qi == ti`) computes its own
    /// upper triangle, mirrors it and fills its diagonal, so it needs nothing
    /// from any other block.
    #[pyo3(signature = (target, qi, ti, block = 128, threads = 1, stdev = false))]
    fn search_block<'py>(
        &self,
        py: Python<'py>,
        target: &Database,
        qi: usize,
        ti: usize,
        block: usize,
        threads: usize,
        stdev: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, usize,
                   Option<Bound<'py, PyBytes>>)> {
        if !self.sealed() || !target.sealed() {
            return Err(PyRuntimeError::new_err("both databases must be sealed"));
        }
        if self.schema_key() != target.schema_key() {
            return Err(PyValueError::new_err(
                "schema mismatch: query and target must share accession list, k and alphabet",
            ));
        }
        self.schema_struct()
            .compatible_with(&target.schema_struct())
            .map_err(PyValueError::new_err)?;
        if qi >= self.n_parts() || ti >= target.n_parts() {
            return Err(PyValueError::new_err(format!(
                "block ({qi}, {ti}) is outside {} x {} partitions",
                self.n_parts(), target.n_parts()
            )));
        }

        let Block { jac, sh, sq, nq, nt, .. } = py
            .detach(|| self.block_values(target, qi, ti, block, threads, stdev))
            .map_err(to_py)?;

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

    /// Compute one block and write it as TSV, without it passing through Python.
    ///
    /// A full self-block is 16,384^2 = 268M rows. Formatting that a row at a
    /// time from Python is not viable, so the worker that computes a block also
    /// writes it. This is the only writer: a whole search is one block when
    /// both sides fit a single partition, and many otherwise.
    ///
    /// *path* of `-` writes to stdout. Otherwise the block goes to a temporary
    /// name and is renamed, so a reader never sees a half-written file.
    ///
    /// Returns `(rows, compute_seconds)`. The split is reported because the two
    /// are different claims: throughput per thread describes the kernel, and
    /// folding formatting and disk into it would understate the engine while
    /// pretending to measure it.
    #[pyo3(signature = (target, qi, ti, path, block = 128, threads = 1,
                        stdev = false, emit = "both", style = "tsv"))]
    #[allow(clippy::too_many_arguments)]
    fn write_block(
        &self,
        py: Python<'_>,
        target: &Database,
        qi: usize,
        ti: usize,
        path: &str,
        block: usize,
        threads: usize,
        stdev: bool,
        emit: &str,
        style: &str,
    ) -> PyResult<(usize, f64)> {
        if !self.sealed() || !target.sealed() {
            return Err(PyRuntimeError::new_err("both databases must be sealed"));
        }
        if self.schema_key() != target.schema_key() {
            return Err(PyValueError::new_err(
                "schema mismatch: query and target must share accession list, k and alphabet",
            ));
        }
        self.schema_struct()
            .compatible_with(&target.schema_struct())
            .map_err(PyValueError::new_err)?;
        if qi >= self.n_parts() || ti >= target.n_parts() {
            return Err(PyValueError::new_err(format!(
                "block ({qi}, {ti}) is outside {} x {} partitions",
                self.n_parts(), target.n_parts()
            )));
        }
        let (want_jac, want_aai) = match emit {
            "both" => (true, true),
            "jaccard" => (true, false),
            "aai" => (false, true),
            other => {
                return Err(PyValueError::new_err(format!(
                    "emit must be one of aai, jaccard, both (got {other})"
                )))
            }
        };
        let matrix = match style {
            "tsv" => false,
            "matrix" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "style must be tsv or matrix (got {other})"
                )))
            }
        };

        let (qoffs, _) = self.offsets();
        let (toffs, _) = target.offsets();
        let (qstart, tstart) = (qoffs[qi], toffs[ti]);
        let dest = std::path::PathBuf::from(path);

        let written = py.detach(|| -> std::io::Result<(usize, f64)> {
            let t0 = std::time::Instant::now();
            let Block { jac, sh, sq, nq, nt, qcounts, tcounts, selfblock } =
                self.block_values(target, qi, ti, block, threads, stdev)?;
            let compute = t0.elapsed().as_secs_f64();

            // Temp name carries the pid. Two processes told to write the same
            // block would otherwise open the same file, interleave their rows
            // and rename the result into place — corruption that looks like a
            // complete block. Renaming is atomic, so if both do finish, one
            // simply replaces the other with identical bytes.
            let to_stdout = path == "-";
            let tmp = dest.with_extension(format!("part.{}", std::process::id()));
            let mut w: Box<dyn std::io::Write> = if to_stdout {
                Box::new(std::io::BufWriter::with_capacity(1 << 20, std::io::stdout()))
            } else {
                Box::new(std::io::BufWriter::with_capacity(
                    1 << 20,
                    std::fs::File::create(&tmp)?,
                ))
            };

            if matrix {
                // A Q x T grid for this partition pair, exactly as the TSV is a
                // Q x T listing for it. Nothing here needs the whole result, so
                // matrix output is not limited to searches that would fit one.
                let mut line = String::with_capacity(16 * nt);
                line.push_str("query_genome");
                for c in 0..nt {
                    line.push('\t');
                    line.push_str(&target.names[tstart + c]);
                }
                line.push('\n');
                std::io::Write::write_all(&mut w, line.as_bytes())?;

                for r in 0..nq {
                    line.clear();
                    line.push_str(&self.names[qstart + r]);
                    for c in 0..nt {
                        let idx = r * nt + c;
                        let j = jac[idx];
                        line.push('\t');
                        if selfblock && r == c {
                            // The genome against itself. Identity is given, not
                            // estimated, so the regression is not consulted.
                            report::fmt_py_round(&mut line, report::SELF_IDENTITY, 1);
                        } else {
                            report::aai_matrix_cell(&mut line, aai::kaai_to_aai(j),
                                                    sh[idx], j);
                        }
                    }
                    line.push('\n');
                    std::io::Write::write_all(&mut w, line.as_bytes())?;
                }
                std::io::Write::flush(&mut w)?;
                drop(w);
                if !to_stdout {
                    std::fs::rename(&tmp, &dest)?;
                }
                return Ok((nq * nt, compute));
            }

            // FastAAI 1's columns, names and order. `jacc_SD` is always present
            // and reads N/A when it was not asked for, exactly as v1 does, so a
            // parser written against v1 sees the shape it expects.
            let mut head = String::from("query\ttarget");
            if want_jac {
                head.push_str("\tavg_jacc_sim");
            }
            head.push_str("\tjacc_SD\tnum_shared_SCPs\tposs_shared_SCPs");
            if want_aai {
                head.push_str("\tAAI_estimate");
            }
            head.push('\n');
            std::io::Write::write_all(&mut w, head.as_bytes())?;

            // One reusable line buffer: 268M allocations per block is the thing
            // this function exists to avoid.
            let mut line = String::with_capacity(128);
            for r in 0..nq {
                let qname = &self.names[qstart + r];
                for c in 0..nt {
                    let idx = r * nt + c;
                    let (j, s) = (jac[idx], sh[idx]);
                    // v1 blanks every value column when a pair shares no
                    // accession: there is no measurement, not a measurement of
                    // zero.
                    let no_hit = s == 0;
                    line.clear();
                    line.push_str(qname);
                    line.push('\t');
                    line.push_str(&target.names[tstart + c]);
                    if want_jac {
                        line.push('\t');
                        if no_hit || j.is_nan() {
                            line.push_str(report::NO_HIT);
                        } else {
                            report::fmt_py_round(&mut line, j, 4);
                        }
                    }
                    line.push('\t');
                    if !stdev || no_hit || sq[idx].is_nan() {
                        line.push_str(report::NO_HIT);
                    } else {
                        report::fmt_py_round(&mut line, sq[idx], 4);
                    }
                    line.push('\t');
                    if no_hit {
                        line.push_str(report::NO_HIT);
                    } else {
                        let _ = std::fmt::Write::write_fmt(&mut line, format_args!("{s}"));
                    }
                    line.push('\t');
                    if no_hit {
                        line.push_str(report::NO_HIT);
                    } else {
                        let poss = qcounts[r].min(tcounts[c]);
                        let _ = std::fmt::Write::write_fmt(&mut line, format_args!("{poss}"));
                    }
                    if want_aai {
                        line.push('\t');
                        if selfblock && r == c {
                            // The genome against itself: identity is given, not
                            // estimated, so the regression is not consulted.
                            report::fmt_py_round(&mut line, report::SELF_IDENTITY, 2);
                        } else {
                            report::aai_label(&mut line, aai::kaai_to_aai(j), s, j);
                        }
                    }
                    line.push('\n');
                    std::io::Write::write_all(&mut w, line.as_bytes())?;
                }
            }
            std::io::Write::flush(&mut w)?;
            drop(w);
            if !to_stdout {
                std::fs::rename(&tmp, &dest)?;
            }
            Ok((nq * nt, compute))
        });
        written.map_err(to_py)
    }
}

fn to_py(e: std::io::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Open a database written by `Database.save`.
///
/// Reads schema and manifest only. Partitions stay on disk and are loaded per
/// block during a search, so opening a GTDB-scale database costs neither the
/// time nor the memory of reading it: peak footprint is the query and target
/// partitions of one block, not the whole index.
#[pyfunction]
fn open_database(path: &str) -> PyResult<Database> {
    let dir = std::path::Path::new(path);
    let schema = store::read_schema(dir).map_err(to_py)?;
    let manifest = store::read_manifest(dir).map_err(to_py)?;
    let alpha = Alphabet::new(schema.alphabet.as_bytes(), schema.k)
        .map_err(PyValueError::new_err)?;
    let kspace = alpha.kspace as usize;

    let part_paths = store::partition_paths(dir).map_err(to_py)?;

    // Genomes per partition, from the manifest — so a search knows its output
    // shape without opening a partition file.
    let mut part_genomes = vec![0usize; part_paths.len()];
    for m in &manifest {
        let p = m.partition as usize;
        if p >= part_genomes.len() {
            return Err(PyRuntimeError::new_err(format!(
                "manifest references partition {p} but only {} exist",
                part_paths.len()
            )));
        }
        part_genomes[p] += 1;
    }

    // Check the first partition against the schema now rather than mid-search.
    // A mismatched model set produces structurally valid, meaningless output, so
    // it must fail at open.
    if let Some(first) = part_paths.first() {
        let part = store::read_partition(first).map_err(to_py)?;
        if part.n_acc != schema.accessions.len() || part.kspace != kspace {
            return Err(PyRuntimeError::new_err(format!(
                "{}: partition disagrees with schema ({} accessions / kspace {} \
                 vs {} / {})",
                first.display(), part.n_acc, part.kspace, schema.accessions.len(), kspace
            )));
        }
    }

    let names = manifest.iter().map(|m| m.name.clone()).collect();
    Ok(Database {
        accessions: schema.accessions,
        filter_mode: schema.filter_mode,
        source: schema.source,
        models: schema.models,
        manifest,
        alphabet_str: schema.alphabet,
        k: schema.k,
        kspace,
        kmerizer: Kmerizer::new(alpha),
        names,
        sets: Vec::new(), // forward sets are not stored and are never needed
        partitions: Vec::new(), // streamed: see `part_paths`
        part_paths,
        part_genomes,
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
