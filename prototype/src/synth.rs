//! Synthetic scale-up from the real example genomes.
//!
//! **Mutation happens at the residue level, then sequences are re-kmerized.**
//!
//! An earlier version resampled kmer *IDs* from the pool of tetramers observed in
//! the 10 real genomes. That pool holds only ~183 tetramers per accession, so no
//! synthetic genome could ever contain a tetramer absent from the seed set: a
//! 16,384-genome partition collapsed to 22,337 distinct (accession, kmer) pairs
//! with a mean posting-list length of 8,123 — half the partition per list. Real
//! databases have far more distinct tetramers and far shorter lists, so any codec
//! benchmark run on that data would have been measuring an artifact.
//!
//! Substituting residues and re-kmerizing draws replacements from the biologically
//! plausible distribution, produces realistic list lengths, and spans a real range
//! of Jaccard values including the low end the seed genomes never reach.

use crate::data::Dataset;
use crate::kmer::Kmerizer;

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub const CLUSTER: usize = 32;

/// Residue sampler weighted by observed marginal frequency.
struct Residues {
    table: Vec<u8>,
}

impl Residues {
    fn new(ds: &Dataset) -> Self {
        let mut counts = [0u64; 256];
        for g in &ds.genomes {
            for s in &g.scps {
                for &b in &s.seq {
                    if b != b'*' {
                        counts[b as usize] += 1;
                    }
                }
            }
        }
        let total: u64 = counts.iter().sum();
        let mut table = Vec::with_capacity(4096);
        for (b, &c) in counts.iter().enumerate() {
            let n = ((c as f64 / total as f64) * 4096.0).round() as usize;
            for _ in 0..n {
                table.push(b as u8);
            }
        }
        assert!(!table.is_empty());
        Residues { table }
    }

    #[inline]
    fn sample(&self, rng: &mut Rng) -> u8 {
        self.table[rng.below(self.table.len())]
    }
}

/// Substitute each non-terminal residue with probability `rate`.
fn mutate(seq: &[u8], rate: f64, res: &Residues, rng: &mut Rng) -> Vec<u8> {
    let mut out = seq.to_vec();
    for b in out.iter_mut() {
        if *b == b'*' {
            continue;
        }
        if rng.unit() < rate {
            *b = res.sample(rng);
        }
    }
    out
}

/// Build `n` synthetic genomes as `sets[genome][accession] = sorted unique kmers`.
///
/// Cluster founders are diverged from a real parent at a rate drawn across a wide
/// range so the database spans close relatives through distant ones; members are
/// then diverged slightly from their founder.
pub fn synthesize(ds: &Dataset, km: &mut Kmerizer, n: usize) -> Vec<Vec<Vec<u32>>> {
    synthesize_seeded(ds, km, n, 0x9E3779B97F4A7C15)
}

/// As `synthesize`, with an explicit seed so independent partitions differ.
pub fn synthesize_seeded(
    ds: &Dataset,
    km: &mut Kmerizer,
    n: usize,
    seed: u64,
) -> Vec<Vec<Vec<u32>>> {
    let res = Residues::new(ds);
    let mut rng = Rng(seed | 1);

    let mut sets: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n);
    // Founder sequences, kept as residues so members can be mutated from them.
    let mut founder: Vec<Option<Vec<u8>>> = Vec::new();

    for i in 0..n {
        if i % CLUSTER == 0 {
            let parent = &ds.genomes[rng.below(ds.genomes.len())];
            // Wide spread of divergence between clusters.
            let rate = 0.02 + rng.unit() * 0.55;
            founder = vec![None; ds.n_acc];
            for s in &parent.scps {
                founder[s.acc as usize] = Some(mutate(&s.seq, rate, &res, &mut rng));
            }
        }
        let mut per_acc = vec![Vec::new(); ds.n_acc];
        for (a, f) in founder.iter().enumerate() {
            if let Some(seq) = f {
                let m = mutate(seq, 0.02, &res, &mut rng);
                per_acc[a] = km.kmers(&m);
            }
        }
        sets.push(per_acc);
    }
    sets
}
