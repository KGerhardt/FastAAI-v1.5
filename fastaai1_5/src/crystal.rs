//! Reading crystals — the resolved-SCP rank — straight into an index.
//!
//! A crystal is a FASTA file of the single-copy proteins one genome recovered,
//! one record per accession, with provenance in each description line:
//!
//! ```text
//! >PF00380.26 genome=GCF_000007085.1 protein=NC_004116.1_1523 models=45d1... filter=v1
//! MKVLAATT...
//! ```
//!
//! Parsing lives here rather than in Python because it is a prefix on
//! k-merisation: the sequence is read, encoded and dropped without ever
//! becoming a Python object. Python's reader handed one genome at a time across
//! the boundary, which was 5 s for 2,943 genomes — fine at that size, and the
//! wrong shape at 500k.
//!
//! Reading is streaming and one file at a time, so peak memory is one crystal
//! rather than the collection. `needletail` handles gzip transparently, so a
//! run written with `--gzip` reads without a second path.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use needletail::parse_fastx_file;

/// What a crystal set says about how it was made. Every record must agree;
/// disagreement means two marker sets or two filters are being mixed, which
/// produces structurally valid and biologically meaningless AAI.
#[derive(Default, Debug)]
pub struct Provenance {
    pub models: HashSet<String>,
    pub filter: HashSet<String>,
}

impl Provenance {
    pub fn models_one(&self) -> String {
        self.models.iter().next().cloned().unwrap_or_default()
    }

    pub fn filter_one(&self) -> String {
        self.filter.iter().next().cloned().unwrap_or_default()
    }

    fn observe(&mut self, fields: &Fields) -> Result<(), String> {
        if let Some(m) = &fields.models {
            self.models.insert(m.clone());
            if self.models.len() > 1 {
                let mut v: Vec<_> = self.models.iter().cloned().collect();
                v.sort();
                return Err(format!(
                    "crystals disagree on the model set: {}. \
                     They cannot build one database.",
                    v.join(", ")
                ));
            }
        }
        if let Some(f) = &fields.filter {
            self.filter.insert(f.clone());
            if self.filter.len() > 1 {
                let mut v: Vec<_> = self.filter.iter().cloned().collect();
                v.sort();
                return Err(format!(
                    "crystals disagree on the best-hit filter: {}. Different \
                     filters give different SCP sets and different AAI.",
                    v.join(", ")
                ));
            }
        }
        Ok(())
    }
}

struct Fields {
    accession: String,
    genome: Option<String>,
    models: Option<String>,
    filter: Option<String>,
}

/// Undo the percent-encoding the writer applies to header values.
///
/// Values are whitespace-delimited `key=value`, so a genome named `my genome`
/// would otherwise parse as `my` and that truncation would become the genome's
/// identity. Only `%XX` is decoded; nothing else is touched.
fn percent_decode(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hi = (b[i + 1] as char).to_digit(16);
            let lo = (b[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parse_header(head: &[u8]) -> Result<Fields, String> {
    let text = String::from_utf8_lossy(head);
    let mut parts = text.split_whitespace();
    let accession = match parts.next() {
        Some(a) if !a.is_empty() => percent_decode(a),
        _ => return Err("empty FASTA header in crystal".to_string()),
    };
    let (mut genome, mut models, mut filter) = (None, None, None);
    for token in parts {
        if let Some((k, v)) = token.split_once('=') {
            match k {
                "genome" => genome = Some(percent_decode(v)),
                "models" => models = Some(percent_decode(v)),
                "filter" => filter = Some(percent_decode(v)),
                _ => {}
            }
        }
    }
    Ok(Fields { accession, genome, models, filter })
}

/// One genome's worth of a crystal file: its name and its (accession, sequence)
/// records, in file order.
pub struct CrystalGenome {
    pub name: String,
    pub records: Vec<(usize, Vec<u8>)>,
}

/// Read one crystal file, grouping records by their `genome=` field.
///
/// *acc_index* maps accession name to position in the model list; an accession
/// absent from it is an error rather than a skip, because a silently dropped
/// marker changes AAI without changing anything visible.
pub fn read_file(
    path: &Path,
    acc_index: &HashMap<String, usize>,
    prov: &mut Provenance,
) -> Result<Vec<CrystalGenome>, String> {
    let mut reader = parse_fastx_file(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let mut order: Vec<String> = Vec::new();
    let mut by_genome: HashMap<String, Vec<(usize, Vec<u8>)>> = HashMap::new();

    while let Some(rec) = reader.next() {
        let rec = rec.map_err(|e| format!("{}: {e}", path.display()))?;
        let fields = parse_header(rec.id()).map_err(|e| format!("{}: {e}", path.display()))?;
        prov.observe(&fields).map_err(|e| format!("{}: {e}", path.display()))?;

        let genome = fields.genome.ok_or_else(|| {
            format!(
                "{}: record {:?} has no genome= field; this is not a FastAAI crystal",
                path.display(),
                fields.accession
            )
        })?;
        let &acc = acc_index.get(&fields.accession).ok_or_else(|| {
            format!(
                "{}: accession {:?} is absent from the model set",
                path.display(),
                fields.accession
            )
        })?;

        let seq = rec.seq().to_vec();
        let entry = by_genome.entry(genome.clone()).or_insert_with(|| {
            order.push(genome.clone());
            Vec::new()
        });
        entry.push((acc, seq));
    }

    // Sorted so a build is reproducible regardless of how the filesystem or the
    // writer ordered things.
    order.sort();
    Ok(order
        .into_iter()
        .map(|name| {
            let mut records = by_genome.remove(&name).unwrap_or_default();
            records.sort_by(|a, b| a.0.cmp(&b.0));
            CrystalGenome { name, records }
        })
        .collect())
}
