//! Alternative posting-list payload encodings, benchmarked against raw `u16`.
//!
//! All of these keep the same CSR shape (direct-addressed by kmer ID) and the same
//! sorted-ascending value order; they differ only in how a list's bytes are stored
//! and decoded. The kernel work — `cnt[value] += 1` — is identical in every case,
//! so any difference is decode cost versus bytes saved.
//!
//! Candidates:
//!   SVB     stream-vbyte over deltas (SIMD, branchless control bytes)
//!   BP      bitpacking / simdcomp BitPacker4x over sorted 128-blocks
//!   HYB     hand-rolled: bitmap for dense lists, raw u16 for sparse

use bitpacking::{BitPacker, BitPacker4x};
use stream_vbyte::{decode::decode, encode::encode, scalar::Scalar};

use crate::index::Partition;
use crate::kernel::Query;
use crate::QueryResult;

const BP_BLOCK: usize = BitPacker4x::BLOCK_LEN; // 128

pub struct CodecAcc {
    pub buf: Vec<u8>,
    /// byte offset of each kmer's encoded run, len = kspace + 1
    pub offs: Vec<u32>,
}

pub struct CodecIndex {
    pub name: &'static str,
    pub accs: Vec<CodecAcc>,
}

impl CodecIndex {
    pub fn bytes(&self) -> usize {
        self.accs.iter().map(|a| a.buf.len()).sum()
    }
}

// ---------------------------------------------------------------- stream-vbyte

pub fn build_svb(p: &Partition) -> CodecIndex {
    let mut accs = Vec::with_capacity(p.n_acc);
    for ai in &p.accs {
        let mut buf = Vec::new();
        let mut offs = vec![0u32; p.kspace + 1];
        let mut deltas: Vec<u32> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();
        for k in 0..p.kspace {
            offs[k] = buf.len() as u32;
            let (s, e) = (ai.offsets[k] as usize, ai.offsets[k + 1] as usize);
            if s == e {
                continue;
            }
            deltas.clear();
            let mut prev = 0u32;
            for &g in &ai.post_u16[s..e] {
                deltas.push(g as u32 - prev);
                prev = g as u32;
            }
            scratch.clear();
            scratch.resize(deltas.len() * 5 + 16, 0);
            let n = encode::<Scalar>(&deltas, &mut scratch);
            buf.extend_from_slice(&scratch[..n]);
        }
        offs[p.kspace] = buf.len() as u32;
        accs.push(CodecAcc { buf, offs });
    }
    CodecIndex { name: "SVB svb-scalar", accs }
}

pub fn count_svb(p: &Partition, idx: &CodecIndex, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];
    let mut dec: Vec<u32> = vec![0; 70_000];

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        let ci = &idx.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);

        for &km in qk {
            let k = km as usize;
            let count = (ai.offsets[k + 1] - ai.offsets[k]) as usize;
            if count == 0 {
                continue;
            }
            let (bs, be) = (ci.offs[k] as usize, ci.offs[k + 1] as usize);
            decode::<Scalar>(&ci.buf[bs..be], count, &mut dec[..count]);
            let mut run = 0u32;
            for &d in &dec[..count] {
                run += d;
                cnt[run as usize] += 1;
            }
        }

        fold(&mut res, ai, &cnt, qk.len() as u32, n);
    }
    res
}

// ------------------------------------------------------------------ bitpacking

pub fn build_bp(p: &Partition) -> CodecIndex {
    let bp = BitPacker4x::new();
    let mut accs = Vec::with_capacity(p.n_acc);
    for ai in &p.accs {
        let mut buf = Vec::new();
        let mut offs = vec![0u32; p.kspace + 1];
        let mut block = [0u32; BP_BLOCK];
        let mut comp = vec![0u8; 4 * BP_BLOCK + 16];
        for k in 0..p.kspace {
            offs[k] = buf.len() as u32;
            let (s, e) = (ai.offsets[k] as usize, ai.offsets[k + 1] as usize);
            let len = e - s;
            if len == 0 {
                continue;
            }
            let vals = &ai.post_u16[s..e];
            let nblocks = len / BP_BLOCK;
            let mut initial = 0u32;
            for b in 0..nblocks {
                for i in 0..BP_BLOCK {
                    block[i] = vals[b * BP_BLOCK + i] as u32;
                }
                let nb = bp.num_bits_sorted(initial, &block);
                let clen = bp.compress_sorted(initial, &block, &mut comp, nb);
                buf.push(nb);
                buf.extend_from_slice(&comp[..clen]);
                initial = block[BP_BLOCK - 1];
            }
            // tail below one block: raw u16
            for &v in &vals[nblocks * BP_BLOCK..] {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        offs[p.kspace] = buf.len() as u32;
        accs.push(CodecAcc { buf, offs });
    }
    CodecIndex { name: "BP  bitpacking4x", accs }
}

pub fn count_bp(p: &Partition, idx: &CodecIndex, q: &Query) -> QueryResult {
    let bp = BitPacker4x::new();
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];
    let mut dec = [0u32; BP_BLOCK];

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        let ci = &idx.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);

        for &km in qk {
            let k = km as usize;
            let len = (ai.offsets[k + 1] - ai.offsets[k]) as usize;
            if len == 0 {
                continue;
            }
            let mut bpos = ci.offs[k] as usize;
            let nblocks = len / BP_BLOCK;
            let mut initial = 0u32;
            for _ in 0..nblocks {
                let nb = ci.buf[bpos];
                bpos += 1;
                let clen = (nb as usize * BP_BLOCK) / 8;
                bp.decompress_sorted(initial, &ci.buf[bpos..bpos + clen], &mut dec, nb);
                bpos += clen;
                for &v in dec.iter() {
                    cnt[v as usize] += 1;
                }
                initial = dec[BP_BLOCK - 1];
            }
            for _ in 0..(len % BP_BLOCK) {
                let v = u16::from_le_bytes([ci.buf[bpos], ci.buf[bpos + 1]]);
                bpos += 2;
                cnt[v as usize] += 1;
            }
        }

        fold(&mut res, ai, &cnt, qk.len() as u32, n);
    }
    res
}

// ------------------------------------------------------- bitmap / u16 hybrid

/// A list is stored as a bitmap once that is smaller than `2 * len` bytes.
#[inline]
pub fn hybrid_threshold(n: usize) -> usize {
    n / 16
}

pub fn build_hybrid(p: &Partition) -> CodecIndex {
    let n = p.n_genomes;
    let thr = hybrid_threshold(n);
    let bitmap_bytes = (n + 7) / 8;
    let mut accs = Vec::with_capacity(p.n_acc);
    for ai in &p.accs {
        let mut buf = Vec::new();
        let mut offs = vec![0u32; p.kspace + 1];
        for k in 0..p.kspace {
            offs[k] = buf.len() as u32;
            let (s, e) = (ai.offsets[k] as usize, ai.offsets[k + 1] as usize);
            let len = e - s;
            if len == 0 {
                continue;
            }
            if len > thr {
                let start = buf.len();
                buf.resize(start + bitmap_bytes, 0);
                for &v in &ai.post_u16[s..e] {
                    buf[start + (v as usize >> 3)] |= 1u8 << (v & 7);
                }
            } else {
                for &v in &ai.post_u16[s..e] {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        offs[p.kspace] = buf.len() as u32;
        accs.push(CodecAcc { buf, offs });
    }
    CodecIndex { name: "HYB bitmap+u16", accs }
}

pub fn count_hybrid(p: &Partition, idx: &CodecIndex, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let thr = hybrid_threshold(n);
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        let ci = &idx.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);

        for &km in qk {
            let k = km as usize;
            let len = (ai.offsets[k + 1] - ai.offsets[k]) as usize;
            if len == 0 {
                continue;
            }
            let bs = ci.offs[k] as usize;
            if len > thr {
                // Dense: walk set bits.
                let bytes = &ci.buf[bs..bs + (n + 7) / 8];
                let words = bytes.len() / 8;
                for w in 0..words {
                    let mut word = u64::from_le_bytes(
                        bytes[w * 8..w * 8 + 8].try_into().unwrap());
                    let base = w * 64;
                    while word != 0 {
                        let b = word.trailing_zeros() as usize;
                        cnt[base + b] += 1;
                        word &= word - 1;
                    }
                }
                for i in words * 8..bytes.len() {
                    let mut byte = bytes[i];
                    while byte != 0 {
                        let b = byte.trailing_zeros() as usize;
                        cnt[i * 8 + b] += 1;
                        byte &= byte - 1;
                    }
                }
            } else {
                for j in 0..len {
                    let v = u16::from_le_bytes([ci.buf[bs + j * 2], ci.buf[bs + j * 2 + 1]]);
                    cnt[v as usize] += 1;
                }
            }
        }

        fold(&mut res, ai, &cnt, qk.len() as u32, n);
    }
    res
}

// ---------------------------------------------------------------------- shared

#[inline]
fn fold(res: &mut QueryResult, ai: &crate::index::AccIndex, cnt: &[u32], qn: u32, n: usize) {
    for t in 0..n {
        let i = cnt[t];
        let denom = ai.kmer_counts[t] as u32 + qn - i;
        if denom > 0 {
            res.jaccard_sum[t] += i as f64 / denom as f64;
        }
        if ai.present[t] {
            res.shared[t] += 1;
        }
    }
}
