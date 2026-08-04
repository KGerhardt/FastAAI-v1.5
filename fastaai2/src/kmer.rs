//! Dense ordinal k-mer encoding.
//!
//! FastAAI 1 encoded a tetramer as the decimal concatenation of ASCII codes
//! (`ord(a)*10^6 + ord(b)*10^4 + ord(c)*10^2 + ord(d)`), which is injective but
//! scatters ~194k real tetramers across ~9e7 sparse values, forcing a hash or
//! B-tree lookup. Here the code is dense over `[0, |A|^k)` so the inverted index
//! can be direct-addressed by k-mer ID.
//!
//! The alphabet is the 20 amino acids. `|A|` = 20 and `K` = 20^4 = 160,000.
//!
//! **The stop codon `*` is deliberately excluded.** Prodigal emits exactly one,
//! terminal, per protein, and it carries no amino-acid information — it should
//! never contribute to a Jaccard comparison. Excluding it from the alphabet means
//! the single window spanning it is not emitted (out-of-alphabet residues break
//! the window), and shrinks the k-mer space by 17.7%: `K` drops from 194,481 to
//! 160,000, and the direct-addressed offsets array from 94.9 MB to 78.1 MB per
//! partition at 122 accessions.
//!
//! This is a deliberate divergence from FastAAI 1, which encoded `*` as ASCII 42
//! and therefore *did* include the terminal tetramer. Expect a small AAI
//! difference; see the equivalence harness.

/// The 20 amino acids. Stop (`*`) and ambiguity codes (`X`, `B`, `Z`, `J`, `U`,
/// `O`) are excluded and break the k-mer window rather than aliasing onto a code.
pub const DEFAULT_ALPHABET: &str = "ACDEFGHIKLMNPQRSTVWY";
pub const DEFAULT_K: usize = 4;

#[derive(Clone)]
pub struct Alphabet {
    /// byte -> ordinal; 0xFF marks a residue outside the alphabet
    lut: [u8; 256],
    pub symbols: Vec<u8>,
    pub base: u32,
    pub k: usize,
    /// |A|^k
    pub kspace: u32,
}

impl Alphabet {
    pub fn new(symbols: &[u8], k: usize) -> Result<Self, String> {
        if symbols.is_empty() {
            return Err("alphabet is empty".into());
        }
        if k == 0 || k > 8 {
            return Err(format!("k must be in 1..=8, got {k}"));
        }
        let base = symbols.len() as u32;
        let kspace = (base as u64).pow(k as u32);
        if kspace > u32::MAX as u64 {
            return Err(format!("|A|^k = {kspace} exceeds u32; reduce k or alphabet"));
        }

        let mut lut = [0xFFu8; 256];
        for (i, &s) in symbols.iter().enumerate() {
            if lut[s as usize] != 0xFF {
                return Err(format!("duplicate symbol {:?} in alphabet", s as char));
            }
            lut[s as usize] = i as u8;
        }

        Ok(Alphabet { lut, symbols: symbols.to_vec(), base, k, kspace: kspace as u32 })
    }

    #[inline]
    pub fn ordinal(&self, b: u8) -> u8 {
        self.lut[b as usize]
    }
}

impl Default for Alphabet {
    fn default() -> Self {
        Alphabet::new(DEFAULT_ALPHABET.as_bytes(), DEFAULT_K).expect("default alphabet is valid")
    }
}

/// Reusable k-merizer. The dedup bitset is cleared through a touched-list rather
/// than a memset, so cost is O(sequence length) no matter how large `kspace` is.
pub struct Kmerizer {
    pub alpha: Alphabet,
    bits: Vec<u64>,
    touched: Vec<u32>,
}

impl Kmerizer {
    pub fn new(alpha: Alphabet) -> Self {
        let words = (alpha.kspace as usize).div_ceil(64);
        Kmerizer { alpha, bits: vec![0u64; words], touched: Vec::with_capacity(1024) }
    }

    /// Sorted, deduplicated k-mer IDs for one protein sequence.
    ///
    /// A residue outside the alphabet resets the window rather than aliasing onto
    /// a valid code, so unknown symbols (`X`, `B`, `Z`, …) cannot silently
    /// manufacture matches.
    pub fn kmers(&mut self, seq: &[u8]) -> Vec<u32> {
        self.touched.clear();
        if seq.len() < self.alpha.k {
            return Vec::new();
        }

        let (base, kspace, k) = (self.alpha.base, self.alpha.kspace, self.alpha.k);
        let mut code: u32 = 0;
        let mut run: usize = 0;

        for &b in seq {
            let v = self.alpha.ordinal(b);
            if v == 0xFF {
                run = 0;
                code = 0;
                continue;
            }
            code = (code.wrapping_mul(base).wrapping_add(v as u32)) % kspace;
            run += 1;
            if run >= k {
                let (w, m) = ((code >> 6) as usize, 1u64 << (code & 63));
                if self.bits[w] & m == 0 {
                    self.bits[w] |= m;
                    self.touched.push(code);
                }
            }
        }

        for &c in &self.touched {
            self.bits[(c >> 6) as usize] &= !(1u64 << (c & 63));
        }

        let mut out = std::mem::take(&mut self.touched);
        out.sort_unstable();
        self.touched = Vec::with_capacity(out.len().max(1024));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn km() -> Kmerizer {
        Kmerizer::new(Alphabet::default())
    }

    #[test]
    fn kspace_matches_real_alphabet() {
        let a = Alphabet::default();
        assert_eq!(a.base, 20, "20 amino acids, stop excluded");
        assert_eq!(a.kspace, 20u32.pow(4));
        assert_eq!(a.kspace, 160_000);
    }

    #[test]
    fn stop_codon_is_not_in_the_alphabet() {
        let a = Alphabet::default();
        assert_eq!(a.ordinal(b'*'), 0xFF, "stop must be out-of-alphabet");
        // A protein ending in `*` loses exactly the one window spanning it.
        let with_stop = km().kmers(b"MKVLAATTGGHH*");
        let without = km().kmers(b"MKVLAATTGGHH");
        assert_eq!(with_stop, without);
    }

    #[test]
    fn encoding_is_positional_and_dense() {
        // 'A' is ordinal 0, so "AAAA" encodes to 0.
        assert_eq!(km().kmers(b"AAAA"), vec![0]);
        // 'C' is ordinal 1: "AAAC" -> 1, "AACA" -> 20.
        assert_eq!(km().kmers(b"AAAC"), vec![1]);
        assert_eq!(km().kmers(b"AACA"), vec![20]);
    }

    #[test]
    fn short_sequence_yields_nothing() {
        assert!(km().kmers(b"AAA").is_empty());
        assert!(km().kmers(b"").is_empty());
    }

    #[test]
    fn output_is_sorted_and_deduplicated() {
        let out = km().kmers(b"AAAAAAAA");
        assert_eq!(out.len(), 1, "all windows identical");
        let out = km().kmers(b"MKVLAAMKVLA");
        let mut sorted = out.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(out, sorted);
    }

    #[test]
    fn unknown_residue_breaks_the_window() {
        // 'X' is outside the alphabet: no k-mer may span it.
        let with_x = km().kmers(b"AAAAXAAAA");
        let clean = km().kmers(b"AAAA");
        assert_eq!(with_x, clean, "windows spanning X must not be emitted");
    }

    #[test]
    fn reuse_does_not_leak_state_between_calls() {
        let mut k = km();
        let a = k.kmers(b"MKVLAAT");
        let b = k.kmers(b"MKVLAAT");
        assert_eq!(a, b, "touched-list clearing must fully reset the bitset");
    }

    #[test]
    fn rejects_bad_configuration() {
        assert!(Alphabet::new(b"AAB", 4).is_err(), "duplicate symbol");
        assert!(Alphabet::new(b"", 4).is_err(), "empty alphabet");
        assert!(Alphabet::new(b"ACGT", 0).is_err(), "k = 0");
        assert!(Alphabet::new(DEFAULT_ALPHABET.as_bytes(), 8).is_err(), "20^8 overflows u32");
    }
}
