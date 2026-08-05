//! On-disk database format.
//!
//! A database is a **directory**, not a single file:
//!
//! ```text
//! db/
//!   schema          k, alphabet, accession list, filter mode, source label
//!   manifest        one row per genome: ordinal, partition, local ID, hash, name
//!   part.00000      inverted index for partition 0
//!   part.00001      ...
//! ```
//!
//! A directory rather than one blob because **append is the operation that has
//! to stay cheap**. Adding genomes writes a new partition file and rewrites the
//! manifest; no existing partition is touched and no posting list is renumbered,
//! because local IDs reference nothing outside their own partition. FastAAI 1
//! grew a database with `INSERT ... ON CONFLICT DO UPDATE SET genomes = genomes
//! || (?)`, read-modify-writing every touched posting list, so the cost scaled
//! with the database rather than with the addition.
//!
//! Only the inverted index is stored. The forward k-mer sets exist solely to
//! build it, and keeping them would roughly triple a database — 95 GB instead of
//! 34 GB at GTDB scale — for something the k-mer join never reads.
//!
//! All integers little-endian. Every read is length-checked: a truncated or
//! corrupt file must produce an error, not a panic or silent misparse.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::index::{AccIndex, Encoding, Partition};

pub const SCHEMA_MAGIC: &[u8; 8] = b"FA2SCHM1";
pub const MANIFEST_MAGIC: &[u8; 8] = b"FA2MANI1";
pub const PARTITION_MAGIC: &[u8; 8] = b"FA2PART1";
pub const FORMAT_VERSION: u16 = 1;

pub const SCHEMA_FILE: &str = "schema";
pub const MANIFEST_FILE: &str = "manifest";

pub fn partition_file(i: usize) -> String {
    format!("part.{i:05}")
}

/// What two databases must agree on before they can be compared or merged.
///
/// Accession IDs are positions in this list, so comparing across different model
/// sets would produce structurally valid, biologically meaningless output. The
/// fields are stored plainly rather than hashed so a mismatch can say *what*
/// differs — "R220 vs R226" beats "fingerprint mismatch".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    pub k: usize,
    pub alphabet: String,
    pub accessions: Vec<String>,
    /// Best-hit resolution used to build this database; changes which protein is
    /// assigned to each accession, so it is part of comparability.
    pub filter_mode: String,
    /// Free-text provenance, e.g. "GTDB R232 bac120". Not part of equality.
    pub source: String,
}

impl Schema {
    /// Everything that must match, ignoring the provenance label.
    pub fn compatible_with(&self, other: &Schema) -> Result<(), String> {
        if self.k != other.k {
            return Err(format!("k differs: {} vs {}", self.k, other.k));
        }
        if self.alphabet != other.alphabet {
            return Err(format!(
                "alphabet differs: {:?} vs {:?}",
                self.alphabet, other.alphabet
            ));
        }
        if self.filter_mode != other.filter_mode {
            return Err(format!(
                "best-hit filter differs: {:?} vs {:?} — these assign different \
                 proteins to each accession",
                self.filter_mode, other.filter_mode
            ));
        }
        if self.accessions.len() != other.accessions.len() {
            return Err(format!(
                "accession count differs: {} vs {}",
                self.accessions.len(),
                other.accessions.len()
            ));
        }
        if let Some(i) = (0..self.accessions.len()).find(|&i| self.accessions[i] != other.accessions[i]) {
            return Err(format!(
                "accession {i} differs: {:?} vs {:?} — accession IDs are positions \
                 in this list, so the databases are not comparable",
                self.accessions[i], other.accessions[i]
            ));
        }
        Ok(())
    }
}

/// One genome's placement and identity.
#[derive(Clone, Debug)]
pub struct ManifestEntry {
    /// Canonical output order — the row/column index in a result matrix.
    /// Stored explicitly rather than derived, so it survives compaction,
    /// partition deletion and merge, none of which preserve partition or local ID.
    pub ordinal: u32,
    pub partition: u32,
    pub local: u16,
    pub scp_count: u16,
    /// Hash of the genome's k-mer sets, for duplicate detection across merges.
    pub content_hash: u64,
    pub name: String,
}

// ------------------------------------------------------------------ writing

struct W(Vec<u8>);

impl W {
    fn new(magic: &[u8; 8]) -> Self {
        let mut b = Vec::new();
        b.extend_from_slice(magic);
        b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        W(b)
    }
    fn u8(&mut self, v: u8) { self.0.push(v); }
    fn u16(&mut self, v: u16) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn u32(&mut self, v: u32) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn u64(&mut self, v: u64) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn u32s(&mut self, v: &[u32]) {
        self.u32(v.len() as u32);
        for &x in v { self.0.extend_from_slice(&x.to_le_bytes()); }
    }
    fn u16s(&mut self, v: &[u16]) {
        self.u32(v.len() as u32);
        for &x in v { self.0.extend_from_slice(&x.to_le_bytes()); }
    }
    /// Booleans as a bitmap: 122 accessions x 16,384 genomes is 2 MB as bytes,
    /// 250 KB as bits.
    fn bits(&mut self, v: &[bool]) {
        self.u32(v.len() as u32);
        for chunk in v.chunks(8) {
            let mut byte = 0u8;
            for (i, &b) in chunk.iter().enumerate() {
                if b { byte |= 1 << i; }
            }
            self.0.push(byte);
        }
    }
    fn finish(self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        std::fs::File::create(&tmp)?.write_all(&self.0)?;
        std::fs::rename(&tmp, path) // atomic: a reader never sees a half-written file
    }
}

// ------------------------------------------------------------------ reading

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> R<'a> {
    fn new(b: &'a [u8], magic: &[u8; 8], what: &str) -> io::Result<Self> {
        if b.len() < 10 || &b[..8] != magic {
            return Err(bad(format!("{what}: bad magic — not a FastAAI {what} file")));
        }
        let v = u16::from_le_bytes([b[8], b[9]]);
        if v != FORMAT_VERSION {
            return Err(bad(format!(
                "{what}: format version {v}, this build reads {FORMAT_VERSION}"
            )));
        }
        Ok(R { b, p: 10 })
    }
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            return Err(bad(format!(
                "truncated: wanted {n} bytes at {}, file has {}",
                self.p,
                self.b.len()
            )));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> io::Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> io::Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> io::Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> io::Result<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }
    fn str(&mut self) -> io::Result<String> {
        let n = self.u32()? as usize;
        let s = self.take(n)?;
        String::from_utf8(s.to_vec()).map_err(|e| bad(format!("invalid utf-8: {e}")))
    }
    fn u32s(&mut self) -> io::Result<Vec<u32>> {
        let n = self.u32()? as usize;
        let s = self.take(n * 4)?;
        Ok(s.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    fn u16s(&mut self) -> io::Result<Vec<u16>> {
        let n = self.u32()? as usize;
        let s = self.take(n * 2)?;
        Ok(s.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn bits(&mut self) -> io::Result<Vec<bool>> {
        let n = self.u32()? as usize;
        let s = self.take(n.div_ceil(8))?;
        Ok((0..n).map(|i| s[i / 8] & (1 << (i % 8)) != 0).collect())
    }
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

// ------------------------------------------------------------------- schema

pub fn write_schema(dir: &Path, s: &Schema) -> io::Result<()> {
    let mut w = W::new(SCHEMA_MAGIC);
    w.u8(s.k as u8);
    w.str(&s.alphabet);
    w.str(&s.filter_mode);
    w.str(&s.source);
    w.u32(s.accessions.len() as u32);
    for a in &s.accessions {
        w.str(a);
    }
    w.finish(&dir.join(SCHEMA_FILE))
}

pub fn read_schema(dir: &Path) -> io::Result<Schema> {
    let raw = std::fs::read(dir.join(SCHEMA_FILE))?;
    let mut r = R::new(&raw, SCHEMA_MAGIC, "schema")?;
    let k = r.u8()? as usize;
    let alphabet = r.str()?;
    let filter_mode = r.str()?;
    let source = r.str()?;
    let n = r.u32()? as usize;
    let mut accessions = Vec::with_capacity(n);
    for _ in 0..n {
        accessions.push(r.str()?);
    }
    Ok(Schema { k, alphabet, accessions, filter_mode, source })
}

// ----------------------------------------------------------------- manifest

pub fn write_manifest(dir: &Path, entries: &[ManifestEntry]) -> io::Result<()> {
    let mut w = W::new(MANIFEST_MAGIC);
    w.u32(entries.len() as u32);
    for e in entries {
        w.u32(e.ordinal);
        w.u32(e.partition);
        w.u16(e.local);
        w.u16(e.scp_count);
        w.u64(e.content_hash);
        w.str(&e.name);
    }
    w.finish(&dir.join(MANIFEST_FILE))
}

pub fn read_manifest(dir: &Path) -> io::Result<Vec<ManifestEntry>> {
    let raw = std::fs::read(dir.join(MANIFEST_FILE))?;
    let mut r = R::new(&raw, MANIFEST_MAGIC, "manifest")?;
    let n = r.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ManifestEntry {
            ordinal: r.u32()?,
            partition: r.u32()?,
            local: r.u16()?,
            scp_count: r.u16()?,
            content_hash: r.u64()?,
            name: r.str()?,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------- partition

pub fn write_partition(path: &Path, p: &Partition) -> io::Result<()> {
    let mut w = W::new(PARTITION_MAGIC);
    w.u32(p.n_genomes as u32);
    w.u32(p.n_acc as u32);
    w.u32(p.kspace as u32);
    for a in &p.accs {
        w.u8(match a.encoding {
            Encoding::Sparse => 0,
        });
        w.u32s(&a.kmers);
        w.u32s(&a.offsets);
        w.u16s(&a.postings);
        w.u32s(&a.kmer_counts);
        w.bits(&a.present);
    }
    w.finish(path)
}

pub fn read_partition(path: &Path) -> io::Result<Partition> {
    let raw = std::fs::read(path)?;
    let mut r = R::new(&raw, PARTITION_MAGIC, "partition")?;
    let n_genomes = r.u32()? as usize;
    let n_acc = r.u32()? as usize;
    let kspace = r.u32()? as usize;

    let mut accs = Vec::with_capacity(n_acc);
    for a in 0..n_acc {
        let encoding = match r.u8()? {
            0 => Encoding::Sparse,
            other => return Err(bad(format!("accession {a}: unknown encoding tag {other}"))),
        };
        let kmers = r.u32s()?;
        let offsets = r.u32s()?;
        let postings = r.u16s()?;
        let kmer_counts = r.u32s()?;
        let present = r.bits()?;

        // Structural checks. A corrupt file must fail here rather than produce
        // wrong numbers later, which is the failure mode that costs most.
        if offsets.len() != kmers.len() + 1 {
            return Err(bad(format!(
                "accession {a}: {} offsets for {} k-mers, expected {}",
                offsets.len(), kmers.len(), kmers.len() + 1
            )));
        }
        if offsets.first() != Some(&0) || offsets.last() != Some(&(postings.len() as u32)) {
            return Err(bad(format!("accession {a}: offsets do not span the postings")));
        }
        if !kmers.windows(2).all(|w| w[0] < w[1]) {
            return Err(bad(format!("accession {a}: k-mer IDs not strictly ascending")));
        }
        if kmer_counts.len() != n_genomes || present.len() != n_genomes {
            return Err(bad(format!("accession {a}: per-genome arrays wrong length")));
        }
        accs.push(AccIndex { encoding, kmers, offsets, postings, kmer_counts, present });
    }

    Ok(Partition { n_genomes, n_acc, kspace, accs })
}

/// FNV-1a over a genome's k-mer sets — identity for duplicate detection on merge,
/// not a security property.
pub fn content_hash(sets: &[Vec<u32>]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (a, kmers) in sets.iter().enumerate() {
        if kmers.is_empty() {
            continue;
        }
        for b in (a as u32).to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        for &k in kmers {
            for b in k.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

/// Partition files present in a database directory, in index order.
pub fn partition_paths(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for i in 0.. {
        let p = dir.join(partition_file(i));
        if !p.exists() {
            break;
        }
        out.push(p);
    }
    if out.is_empty() {
        return Err(bad(format!("{}: no partition files", dir.display())));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fastaai_store_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn sets() -> Vec<Vec<Vec<u32>>> {
        vec![
            vec![vec![1u32, 3, 5], vec![10, 11]],
            vec![vec![3u32, 5, 7], vec![]],
            vec![vec![], vec![10, 12]],
        ]
    }

    fn schema() -> Schema {
        Schema {
            k: 4,
            alphabet: "ACDEFGHIKLMNPQRSTVWY".into(),
            accessions: vec!["acc0".into(), "acc1".into()],
            filter_mode: "v1".into(),
            source: "test".into(),
        }
    }

    #[test]
    fn partition_round_trips_exactly() {
        let d = tmpdir("part");
        let p = Partition::build(&sets(), 2, 16).unwrap();
        let path = d.join(partition_file(0));
        write_partition(&path, &p).unwrap();
        let q = read_partition(&path).unwrap();

        assert_eq!(q.n_genomes, p.n_genomes);
        assert_eq!(q.n_acc, p.n_acc);
        assert_eq!(q.kspace, p.kspace);
        for (a, b) in q.accs.iter().zip(&p.accs) {
            assert_eq!(a.kmers, b.kmers);
            assert_eq!(a.offsets, b.offsets);
            assert_eq!(a.postings, b.postings);
            assert_eq!(a.kmer_counts, b.kmer_counts);
            assert_eq!(a.present, b.present);
            assert_eq!(a.encoding, b.encoding);
        }
    }

    #[test]
    fn round_tripped_partition_gives_identical_results() {
        let d = tmpdir("results");
        let s = sets();
        let n = s.len();
        let p = Partition::build(&s, 2, 16).unwrap();
        let path = d.join(partition_file(0));
        write_partition(&path, &p).unwrap();
        let q = read_partition(&path).unwrap();

        let (mut j1, mut s1) = (vec![0.0; n * n], vec![0u32; n * n]);
        let (mut j2, mut s2) = (vec![0.0; n * n], vec![0u32; n * n]);
        crate::kernel::join_into(&p, &p, 2, &mut j1, &mut s1);
        crate::kernel::join_into(&q, &q, 2, &mut j2, &mut s2);
        assert_eq!(s1, s2);
        assert_eq!(j1, j2, "a stored index must compute the same numbers");
    }

    #[test]
    fn schema_and_manifest_round_trip() {
        let d = tmpdir("schema");
        write_schema(&d, &schema()).unwrap();
        assert_eq!(read_schema(&d).unwrap(), schema());

        let entries = vec![
            ManifestEntry { ordinal: 0, partition: 0, local: 0, scp_count: 2,
                            content_hash: 42, name: "g0".into() },
            ManifestEntry { ordinal: 1, partition: 0, local: 1, scp_count: 1,
                            content_hash: 43, name: "g1".into() },
        ];
        write_manifest(&d, &entries).unwrap();
        let got = read_manifest(&d).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].name, "g1");
        assert_eq!(got[1].content_hash, 43);
        assert_eq!(got[0].scp_count, 2);
    }

    #[test]
    fn schema_compatibility_names_what_differs() {
        let a = schema();
        let mut b = a.clone();
        assert!(a.compatible_with(&b).is_ok());

        b.source = "different provenance".into();
        assert!(a.compatible_with(&b).is_ok(), "source is a label, not a constraint");

        let mut c = a.clone();
        c.k = 5;
        assert!(c.compatible_with(&a).unwrap_err().contains("k differs"));

        let mut e = a.clone();
        e.filter_mode = "rbh".into();
        assert!(a.compatible_with(&e).unwrap_err().contains("filter"));

        let mut f = a.clone();
        f.accessions[1] = "other".into();
        let msg = a.compatible_with(&f).unwrap_err();
        assert!(msg.contains("accession 1"), "must say which one: {msg}");
    }

    #[test]
    fn corrupt_files_error_rather_than_panic() {
        let d = tmpdir("corrupt");
        let p = Partition::build(&sets(), 2, 16).unwrap();
        let path = d.join(partition_file(0));
        write_partition(&path, &p).unwrap();
        let good = std::fs::read(&path).unwrap();

        // Wrong magic.
        let mut bad_magic = good.clone();
        bad_magic[..8].copy_from_slice(b"NOTAFAAI");
        std::fs::write(&path, &bad_magic).unwrap();
        assert!(read_partition(&path).is_err());

        // Wrong version.
        let mut bad_ver = good.clone();
        bad_ver[8..10].copy_from_slice(&99u16.to_le_bytes());
        std::fs::write(&path, &bad_ver).unwrap();
        match read_partition(&path) {
            Err(e) => assert!(e.to_string().contains("version"), "{e}"),
            Ok(_) => panic!("a future format version must be refused, not guessed at"),
        }

        // Truncated at every length — none may panic.
        for cut in (10..good.len()).step_by(7) {
            std::fs::write(&path, &good[..cut]).unwrap();
            assert!(read_partition(&path).is_err(), "truncation at {cut} must error");
        }
    }

    #[test]
    fn content_hash_distinguishes_and_ignores_absent_accessions() {
        let a = vec![vec![1u32, 2], vec![3]];
        let b = vec![vec![1u32, 2], vec![4]];
        assert_ne!(content_hash(&a), content_hash(&b));
        assert_eq!(content_hash(&a), content_hash(&a.clone()));
        // An accession present-but-empty is the same as absent.
        assert_eq!(content_hash(&vec![vec![1u32, 2], vec![]]),
                   content_hash(&vec![vec![1u32, 2]]));
    }

    #[test]
    fn hash_is_order_sensitive_across_accessions() {
        // Same k-mers, different accessions, must not collide.
        assert_ne!(content_hash(&vec![vec![1u32, 2], vec![]]),
                   content_hash(&vec![vec![], vec![1u32, 2]]));
    }
}
