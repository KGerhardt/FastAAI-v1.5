//! Partition-local inverted index.
//!
//! One partition holds at most 65,536 genomes, so genome IDs are `u16` and local.
//! Per accession we store a CSR structure direct-addressed by kmer ID:
//! `offsets[kmer]..offsets[kmer+1]` slices into the posting array.
//!
//! Three payload encodings are built so the kernels can be compared on identical
//! data:
//!   - `post_u32`  : baseline width, what a global-ID design would need
//!   - `post_u16`  : partition-local IDs
//!   - `post_delta`: varint-coded deltas over the sorted `u16` IDs

pub const MAX_PARTITION: usize = 65_536;

pub struct AccIndex {
    /// len = kspace + 1, element offsets into the posting arrays
    pub offsets: Vec<u32>,
    pub post_u32: Vec<u32>,
    pub post_u16: Vec<u16>,
    /// Same values, order destroyed within each kmer's run. Only for measuring
    /// how much the sorted layout is worth; never a real storage option, since
    /// sortedness is required by delta coding and cursor sweeps.
    pub post_shuffled: Vec<u16>,
    /// varint deltas; `byte_offsets[k]` is where kmer k's run begins
    pub post_delta: Vec<u8>,
    pub byte_offsets: Vec<u32>,
    /// |T_a| per local genome; 0 when the genome lacks this accession
    pub kmer_counts: Vec<u16>,
    pub present: Vec<bool>,
}

pub struct Partition {
    pub n_genomes: usize,
    pub n_acc: usize,
    pub kspace: usize,
    pub accs: Vec<AccIndex>,
    pub names: Vec<String>,
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline(always)]
pub fn get_varint(buf: &[u8], p: &mut usize) -> u32 {
    let mut v: u32 = 0;
    let mut shift = 0;
    loop {
        let b = buf[*p];
        *p += 1;
        v |= ((b & 0x7F) as u32) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

impl Partition {
    /// Only `post_u16` (plus offsets/counts/presence). Used when many partitions
    /// must be resident at once and the comparison payloads are not needed.
    pub fn build_lean(
        sets: &[Vec<Vec<u32>>],
        n_acc: usize,
        kspace: usize,
        names: Vec<String>,
    ) -> Self {
        let n_genomes = sets.len();
        assert!(n_genomes <= MAX_PARTITION, "partition cap is {MAX_PARTITION}");
        let mut accs = Vec::with_capacity(n_acc);
        for a in 0..n_acc {
            let mut offsets = vec![0u32; kspace + 1];
            for g in 0..n_genomes {
                for &km in &sets[g][a] {
                    offsets[km as usize + 1] += 1;
                }
            }
            for i in 1..offsets.len() {
                offsets[i] += offsets[i - 1];
            }
            let total = offsets[kspace] as usize;
            let mut cursor = offsets.clone();
            let mut post_u16 = vec![0u16; total];
            for g in 0..n_genomes {
                for &km in &sets[g][a] {
                    let slot = &mut cursor[km as usize];
                    post_u16[*slot as usize] = g as u16;
                    *slot += 1;
                }
            }
            let mut kmer_counts = vec![0u16; n_genomes];
            let mut present = vec![false; n_genomes];
            for g in 0..n_genomes {
                let n = sets[g][a].len();
                if n > 0 {
                    kmer_counts[g] = n as u16;
                    present[g] = true;
                }
            }
            accs.push(AccIndex {
                offsets,
                post_u32: Vec::new(),
                post_u16,
                post_shuffled: Vec::new(),
                post_delta: Vec::new(),
                byte_offsets: Vec::new(),
                kmer_counts,
                present,
            });
        }
        Partition { n_genomes, n_acc, kspace, accs, names }
    }

    /// `sets[genome][acc]` = sorted unique kmer IDs, empty when the accession is absent.
    pub fn build(sets: &[Vec<Vec<u32>>], n_acc: usize, kspace: usize, names: Vec<String>) -> Self {
        let n_genomes = sets.len();
        assert!(n_genomes <= MAX_PARTITION, "partition cap is {MAX_PARTITION}");

        let mut accs = Vec::with_capacity(n_acc);
        for a in 0..n_acc {
            // Count per kmer.
            let mut counts = vec![0u32; kspace + 1];
            for g in 0..n_genomes {
                for &km in &sets[g][a] {
                    counts[km as usize + 1] += 1;
                }
            }
            // Prefix sum -> offsets.
            let mut offsets = counts;
            for i in 1..offsets.len() {
                offsets[i] += offsets[i - 1];
            }
            let total = offsets[kspace] as usize;

            // Fill postings. Iterating genomes in ascending order yields sorted runs.
            let mut cursor = offsets.clone();
            let mut post_u32 = vec![0u32; total];
            for g in 0..n_genomes {
                for &km in &sets[g][a] {
                    let slot = &mut cursor[km as usize];
                    post_u32[*slot as usize] = g as u32;
                    *slot += 1;
                }
            }
            let post_u16: Vec<u16> = post_u32.iter().map(|&x| x as u16).collect();

            // Shuffled copy: same multiset, order destroyed within each run.
            let mut post_shuffled = post_u16.clone();
            let mut rs: u64 = 0x2545F4914F6CDD1D ^ (a as u64).wrapping_mul(0x9E3779B97F4A7C15);
            for k in 0..kspace {
                let (s, e) = (offsets[k] as usize, offsets[k + 1] as usize);
                if e - s < 2 {
                    continue;
                }
                for i in (s + 1..e).rev() {
                    rs ^= rs << 13;
                    rs ^= rs >> 7;
                    rs ^= rs << 17;
                    let j = s + (rs % ((i - s + 1) as u64)) as usize;
                    post_shuffled.swap(i, j);
                }
            }

            // Delta + varint over each kmer's run.
            let mut post_delta = Vec::with_capacity(total);
            let mut byte_offsets = vec![0u32; kspace + 1];
            for k in 0..kspace {
                byte_offsets[k] = post_delta.len() as u32;
                let (s, e) = (offsets[k] as usize, offsets[k + 1] as usize);
                let mut prev = 0u32;
                for &g in &post_u32[s..e] {
                    put_varint(&mut post_delta, g - prev);
                    prev = g;
                }
            }
            byte_offsets[kspace] = post_delta.len() as u32;

            let mut kmer_counts = vec![0u16; n_genomes];
            let mut present = vec![false; n_genomes];
            for g in 0..n_genomes {
                let n = sets[g][a].len();
                if n > 0 {
                    kmer_counts[g] = n as u16;
                    present[g] = true;
                }
            }

            accs.push(AccIndex {
                offsets,
                post_u32,
                post_u16,
                post_shuffled,
                post_delta,
                byte_offsets,
                kmer_counts,
                present,
            });
        }

        Partition { n_genomes, n_acc, kspace, accs, names }
    }

    pub fn posting_entries(&self) -> usize {
        self.accs
            .iter()
            .map(|a| if a.post_u32.is_empty() { a.post_u16.len() } else { a.post_u32.len() })
            .sum()
    }

    pub fn delta_bytes(&self) -> usize {
        self.accs.iter().map(|a| a.post_delta.len()).sum()
    }

    pub fn offset_bytes(&self) -> usize {
        self.accs.iter().map(|a| a.offsets.len() * 4).sum()
    }
}
