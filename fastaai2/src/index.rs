//! Partition-local inverted index.
//!
//! A partition holds at most 65,536 genomes so genome IDs fit `u16` and are local
//! to the partition. Per accession the structure is CSR, direct-addressed by
//! k-mer ID: `offsets[kmer] .. offsets[kmer + 1]` slices `postings`.
//!
//! Two invariants are load-bearing and measured, not assumed:
//!
//! * **Postings are sorted ascending by local genome ID.** This makes the
//!   counting loop's accumulator writes a monotone sweep rather than a random
//!   scatter. Destroying the order costs ~2x (prototype `untiled_shuffled`).
//! * **IDs are `u16`, not `u32`.** Halving the hottest, largest structure is
//!   worth 1.26-1.49x under threading.

/// Hard cap from the `u16` local genome ID. An upper bound, **not a target**.
pub const MAX_PARTITION: usize = 65_536;

/// Genomes per partition.
///
/// Chosen against a 2 GiB-per-thread budget — the prevailing HPC cost model,
/// where RAM and cores cannot be requested independently — assuming one
/// partition in flight per thread. Using today's measured density of 15,561
/// posting entries per genome:
///
/// | P | index | offsets share | free of 2 GiB |
/// |---|---|---|---|
/// | 8,192 | 0.31 G | 23.1% | 1.69 G |
/// | **16,384** | **0.56 G** | **13.1%** | **1.44 G** |
/// | 32,768 | 1.04 G | 7.0% | 0.96 G |
/// | 65,536 | 2.01 G | 3.6% | **-0.01 G** |
///
/// The `u16` cap of 65,536 is already *over budget* before any working space, so
/// it cannot be the operating size. 16,384 leaves 1.44 GiB for the accumulator,
/// query k-mer sets, output buffering and preprocessing slack.
///
/// The 13.1% offsets overhead is a **storage** cost, not a resident one: with
/// streaming, one partition's 78.1 MB is live at a time regardless of how many
/// partitions exist. Trading disk for RAM is the right direction — a 900k-genome
/// database is ~30 GB on disk and roughly half that compressed.
pub const PARTITION_SIZE: usize = 16_384;

pub struct AccIndex {
    /// len = kspace + 1
    pub offsets: Vec<u32>,
    /// genome IDs, sorted ascending within each k-mer's run
    pub postings: Vec<u16>,
    /// |T_a| per local genome; 0 where the genome lacks this accession
    pub kmer_counts: Vec<u32>,
    /// whether each local genome carries this accession at all
    pub present: Vec<bool>,
}

pub struct Partition {
    pub n_genomes: usize,
    pub n_acc: usize,
    pub kspace: usize,
    pub accs: Vec<AccIndex>,
}

impl Partition {
    /// `sets[genome][accession]` = sorted unique k-mer IDs; empty when absent.
    pub fn build(sets: &[Vec<Vec<u32>>], n_acc: usize, kspace: usize) -> Result<Self, String> {
        let n_genomes = sets.len();
        if n_genomes > MAX_PARTITION {
            return Err(format!(
                "{n_genomes} genomes exceeds the {MAX_PARTITION} partition cap"
            ));
        }

        let mut accs = Vec::with_capacity(n_acc);
        for a in 0..n_acc {
            let mut offsets = vec![0u32; kspace + 1];
            for g in sets {
                for &km in &g[a] {
                    offsets[km as usize + 1] += 1;
                }
            }
            for i in 1..offsets.len() {
                offsets[i] += offsets[i - 1];
            }

            let total = offsets[kspace] as usize;
            let mut cursor = offsets.clone();
            let mut postings = vec![0u16; total];
            // Iterating genomes in ascending order leaves every run sorted.
            for (g, set) in sets.iter().enumerate() {
                for &km in &set[a] {
                    let slot = &mut cursor[km as usize];
                    postings[*slot as usize] = g as u16;
                    *slot += 1;
                }
            }

            let mut kmer_counts = vec![0u32; n_genomes];
            let mut present = vec![false; n_genomes];
            for (g, set) in sets.iter().enumerate() {
                let n = set[a].len();
                if n > 0 {
                    kmer_counts[g] = n as u32;
                    present[g] = true;
                }
            }

            accs.push(AccIndex { offsets, postings, kmer_counts, present });
        }

        Ok(Partition { n_genomes, n_acc, kspace, accs })
    }

    pub fn posting_entries(&self) -> usize {
        self.accs.iter().map(|a| a.postings.len()).sum()
    }

    /// Bytes held by postings plus offsets — the resident index footprint.
    pub fn index_bytes(&self) -> usize {
        self.accs
            .iter()
            .map(|a| a.postings.len() * 2 + a.offsets.len() * 4)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 genomes, 1 accession, tiny k-mer space.
    fn fixture() -> Partition {
        let sets = vec![
            vec![vec![1u32, 3, 5]],
            vec![vec![3u32, 5, 7]],
            vec![vec![]],
        ];
        Partition::build(&sets, 1, 8).unwrap()
    }

    #[test]
    fn postings_are_sorted_within_each_run() {
        let p = fixture();
        let a = &p.accs[0];
        for k in 0..p.kspace {
            let (s, e) = (a.offsets[k] as usize, a.offsets[k + 1] as usize);
            let run = &a.postings[s..e];
            assert!(run.windows(2).all(|w| w[0] < w[1]),
                    "k-mer {k} run not strictly ascending: {run:?}");
        }
    }

    #[test]
    fn csr_layout_is_consistent() {
        let p = fixture();
        let a = &p.accs[0];
        assert_eq!(a.offsets[0], 0);
        assert_eq!(*a.offsets.last().unwrap() as usize, a.postings.len());
        assert!(a.offsets.windows(2).all(|w| w[0] <= w[1]), "offsets must be monotone");
        assert_eq!(a.postings.len(), 6, "3 + 3 + 0 k-mer instances");
    }

    #[test]
    fn shared_kmers_list_every_carrier() {
        let p = fixture();
        let a = &p.accs[0];
        // k-mer 3 is in genomes 0 and 1; k-mer 1 only in genome 0.
        let run = |k: usize| {
            a.postings[a.offsets[k] as usize..a.offsets[k + 1] as usize].to_vec()
        };
        assert_eq!(run(3), vec![0, 1]);
        assert_eq!(run(1), vec![0]);
        assert_eq!(run(7), vec![1]);
        assert!(run(0).is_empty());
    }

    #[test]
    fn absent_accession_is_recorded_not_faked() {
        let p = fixture();
        let a = &p.accs[0];
        assert_eq!(a.present, vec![true, true, false]);
        assert_eq!(a.kmer_counts, vec![3, 3, 0]);
    }


    #[test]
    fn rejects_oversized_partition() {
        // Cheap construction: one accession, empty sets, just over the cap.
        let sets = vec![vec![Vec::new()]; MAX_PARTITION + 1];
        assert!(Partition::build(&sets, 1, 2).is_err());
    }
}
