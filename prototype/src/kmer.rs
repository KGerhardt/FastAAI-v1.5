//! Dense ordinal tetramer encoding.
//!
//! v1 encodes a tetramer as the decimal concatenation of ASCII codes:
//!   ord(a)*10^6 + ord(b)*10^4 + ord(c)*10^2 + ord(d)
//! which is injective but spreads ~194k real tetramers over ~9e7 sparse values,
//! forcing a hash or B-tree lookup.
//!
//! Here: `id = ((a*|A| + b)*|A| + c)*|A| + d`, dense over `[0, |A|^k)`. With the
//! alphabet observed in real prodigal output (20 residues + `*`), |A| = 21 and
//! K = 194,481 — small enough to direct-address.

pub const K_LEN: usize = 4;

#[derive(Clone)]
pub struct Alphabet {
    /// byte -> ordinal, 0xFF for symbols outside the alphabet
    pub lut: [u8; 256],
    pub symbols: Vec<u8>,
    pub base: u32,
    /// |A|^k
    pub kspace: u32,
}

impl Alphabet {
    pub fn new(symbols: &[u8]) -> Self {
        let mut lut = [0xFFu8; 256];
        for (i, &s) in symbols.iter().enumerate() {
            lut[s as usize] = i as u8;
        }
        let base = symbols.len() as u32;
        Alphabet {
            lut,
            symbols: symbols.to_vec(),
            base,
            kspace: base.pow(K_LEN as u32),
        }
    }
}

/// Reusable kmerizer. The bitset is cleared through a touched-list rather than a
/// full memset, so cost is O(len) regardless of how large `kspace` is.
pub struct Kmerizer {
    pub alpha: Alphabet,
    bits: Vec<u64>,
    touched: Vec<u32>,
}

impl Kmerizer {
    pub fn new(alpha: Alphabet) -> Self {
        let words = (alpha.kspace as usize + 63) / 64;
        Kmerizer { alpha, bits: vec![0u64; words], touched: Vec::with_capacity(1024) }
    }

    /// Sorted, deduplicated tetramer IDs for one protein sequence.
    ///
    /// Any residue outside the alphabet aborts the tetramer window, matching the
    /// intent that unknown symbols do not silently alias onto a valid code.
    pub fn kmers(&mut self, seq: &[u8]) -> Vec<u32> {
        self.touched.clear();
        if seq.len() < K_LEN {
            return Vec::new();
        }

        let base = self.alpha.base;
        let kspace = self.alpha.kspace;
        let mut code: u32 = 0;
        let mut run: usize = 0; // consecutive in-alphabet residues

        for &b in seq {
            let v = self.alpha.lut[b as usize];
            if v == 0xFF {
                run = 0;
                code = 0;
                continue;
            }
            code = code.wrapping_mul(base).wrapping_add(v as u32) % kspace;
            run += 1;
            if run >= K_LEN {
                let w = (code >> 6) as usize;
                let m = 1u64 << (code & 63);
                if self.bits[w] & m == 0 {
                    self.bits[w] |= m;
                    self.touched.push(code);
                }
            }
        }

        // Clear only what was set.
        for &c in &self.touched {
            self.bits[(c >> 6) as usize] &= !(1u64 << (c & 63));
        }

        let mut out = std::mem::take(&mut self.touched);
        out.sort_unstable();
        self.touched = Vec::with_capacity(out.len().max(1024));
        out
    }
}

/// v1's encoding, for cross-checking that the two schemes induce the same sets.
pub fn v1_code(w: &[u8]) -> i32 {
    w[0] as i32 * 1_000_000 + w[1] as i32 * 10_000 + w[2] as i32 * 100 + w[3] as i32
}
