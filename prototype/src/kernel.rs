//! The three counting kernels. All three must produce bit-identical results;
//! they differ only in payload width and loop structure.

use crate::index::{get_varint, Partition};
use crate::QueryResult;

/// One query genome: (accession, sorted unique kmer IDs).
pub type Query = [(usize, Vec<u32>)];

/// (A) Baseline. Full-width accumulator, `u32` postings, no tiling.
/// This is the shape of v1's `np.bincount(..., minlength=num_tgts)` per SCP.
pub fn untiled_u32(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);

        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            for &g in &ai.post_u32[s..e] {
                cnt[g as usize] += 1;
            }
        }

        let qn = qk.len() as u32;
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
    res
}

/// (A16) Untiled but `u16` postings — isolates payload width from loop structure.
pub fn untiled_u16(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);

        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            for &g in &ai.post_u16[s..e] {
                cnt[g as usize] += 1;
            }
        }

        let qn = qk.len() as u32;
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
    res
}

/// (A-shuf) Identical to `untiled_u16` but reads the deliberately shuffled
/// posting copy. Sorted postings make the accumulator writes a monotone sweep;
/// this measures what the cost would be if they were a true random scatter.
pub fn untiled_shuffled(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);

        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            for &g in &ai.post_shuffled[s..e] {
                cnt[g as usize] += 1;
            }
        }

        let qn = qk.len() as u32;
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
    res
}

/// (B) `u16` postings, tiled with monotone forward cursors.
/// The scatter target is `tile` wide and stays L1-resident; `jaccard_sum` is
/// touched strictly sequentially as tiles advance.
pub fn tiled_u16(p: &Partition, q: &Query, tile: usize) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u16; tile];
    let mut cursors: Vec<u32> = Vec::new();

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cursors.clear();
        cursors.extend(qk.iter().map(|&km| ai.offsets[km as usize]));
        let qn = qk.len() as u32;

        let mut base = 0usize;
        while base < n {
            let end = (base + tile).min(n);
            let w = end - base;
            cnt[..w].iter_mut().for_each(|x| *x = 0);

            for (j, &km) in qk.iter().enumerate() {
                let stop = ai.offsets[km as usize + 1];
                let mut c = cursors[j];
                while c < stop {
                    let g = ai.post_u16[c as usize] as usize;
                    if g >= end {
                        break;
                    }
                    cnt[g - base] += 1;
                    c += 1;
                }
                cursors[j] = c;
            }

            for t in base..end {
                let i = cnt[t - base] as u32;
                let denom = ai.kmer_counts[t] as u32 + qn - i;
                if denom > 0 {
                    res.jaccard_sum[t] += i as f64 / denom as f64;
                }
                if ai.present[t] {
                    res.shared[t] += 1;
                }
            }
            base = end;
        }
    }
    res
}

/// (C) Delta+varint postings, tiled. Cursor carries (byte position, running
/// value, remaining count) plus one decoded lookahead, since varint decoding is
/// destructive and cannot peek. Sequential decode is exactly the access pattern
/// the tile sweep produces, so the two compose without extra work.
pub fn tiled_delta(p: &Partition, q: &Query, tile: usize) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u16; tile];

    let mut pos: Vec<u32> = Vec::new();
    let mut next: Vec<u32> = Vec::new();
    let mut left: Vec<u32> = Vec::new();

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        let buf = &ai.post_delta;

        pos.clear();
        next.clear();
        left.clear();
        for &km in qk {
            let k = km as usize;
            let count = ai.offsets[k + 1] - ai.offsets[k];
            let mut bp = ai.byte_offsets[k] as usize;
            if count > 0 {
                let v = get_varint(buf, &mut bp);
                next.push(v);
                left.push(count - 1);
            } else {
                next.push(u32::MAX);
                left.push(0);
            }
            pos.push(bp as u32);
        }

        let qn = qk.len() as u32;
        let mut base = 0usize;
        while base < n {
            let end = (base + tile).min(n);
            let w = end - base;
            cnt[..w].iter_mut().for_each(|x| *x = 0);

            for j in 0..qk.len() {
                let mut v = next[j];
                if v == u32::MAX {
                    continue;
                }
                let mut bp = pos[j] as usize;
                let mut rem = left[j];
                while (v as usize) < end {
                    cnt[v as usize - base] += 1;
                    if rem == 0 {
                        v = u32::MAX;
                        break;
                    }
                    v += get_varint(buf, &mut bp);
                    rem -= 1;
                }
                next[j] = v;
                pos[j] = bp as u32;
                left[j] = rem;
            }

            for t in base..end {
                let i = cnt[t - base] as u32;
                let denom = ai.kmer_counts[t] as u32 + qn - i;
                if denom > 0 {
                    res.jaccard_sum[t] += i as f64 / denom as f64;
                }
                if ai.present[t] {
                    res.shared[t] += 1;
                }
            }
            base = end;
        }
    }
    res
}

/// Total posting entries touched by a query — the kernel's fundamental work unit.
pub fn increments(p: &Partition, q: &Query) -> u64 {
    let mut n = 0u64;
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        for &km in qk {
            n += (ai.offsets[km as usize + 1] - ai.offsets[km as usize]) as u64;
        }
    }
    n
}

/// (M1) Materialize the target-ID stream, then reduce — v1's
/// `np.bincount(np.concatenate(...))` shape (fastaai.py:2733, 2800).
pub fn materialized_ids(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];
    let mut buf: Vec<u16> = Vec::with_capacity(1 << 20);

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);
        buf.clear();
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            buf.extend_from_slice(&ai.post_u16[s..e]);
        }
        for &g in &buf {
            cnt[g as usize] += 1;
        }
        fold_pub(&mut res, ai, &cnt, qk.len() as u32, n);
    }
    res
}

/// (M2) Materialize full (target, SCP, kmer) tuples, then reduce.
/// The proposed relational form. `acc` and `km` are loop-invariant here — they are
/// re-emitted per matching genome purely to make the record self-describing.
#[derive(Clone, Copy)]
pub struct Tup {
    pub g: u16,
    pub acc: u16,
    pub km: u32,
}

pub fn materialized_tuples(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];
    let mut buf: Vec<Tup> = Vec::with_capacity(1 << 20);

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);
        buf.clear();
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            for &g in &ai.post_u16[s..e] {
                buf.push(Tup { g, acc: *acc as u16, km });
            }
        }
        for t in &buf {
            cnt[t.g as usize] += 1;
        }
        fold_pub(&mut res, ai, &cnt, qk.len() as u32, n);
    }
    res
}

#[inline]
fn fold_pub(res: &mut QueryResult, ai: &crate::index::AccIndex, cnt: &[u32], qn: u32, n: usize) {
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

/// Query-batched kernel: the practical half of the "invert the search" idea.
///
/// The full inverted-vs-inverted form joins two kmer-indexed lists and needs an
/// Nq x Nt accumulator. This blocks only the *query* dimension (batch of <= 64)
/// and keeps the target dimension full, so the accumulator is `batch * n` and
/// stays in L2 — while each target posting list `T[k]` is read **once per batch**
/// instead of once per query genome.
///
/// Total increments are unchanged; index bytes read drop by up to the batch size.
pub fn batched(p: &Partition, qs: &[&Query], masks: &mut Vec<u64>, touched: &mut Vec<u32>)
    -> Vec<QueryResult>
{
    let n = p.n_genomes;
    let b = qs.len();
    assert!(b <= 64, "batch is a u64 mask");
    if masks.len() < p.kspace { masks.resize(p.kspace, 0); }

    let mut cnt = vec![0u32; b * n];
    let mut out: Vec<QueryResult> = (0..b).map(|_| QueryResult::new(n)).collect();

    // Per-accession query kmer counts, and which batch members carry the accession.
    for a in 0..p.n_acc {
        touched.clear();
        let mut qn = [0u32; 64];
        let mut any = false;
        for (qi, q) in qs.iter().enumerate() {
            if let Some((_, kk)) = q.iter().find(|(acc, _)| *acc == a) {
                qn[qi] = kk.len() as u32;
                any = true;
                for &k in kk.iter() {
                    if masks[k as usize] == 0 { touched.push(k); }
                    masks[k as usize] |= 1u64 << qi;
                }
            }
        }
        if !any { continue; }
        touched.sort_unstable();

        cnt.iter_mut().for_each(|x| *x = 0);
        let ai = &p.accs[a];
        for &k in touched.iter() {
            let m0 = masks[k as usize];
            let s = ai.offsets[k as usize] as usize;
            let e = ai.offsets[k as usize + 1] as usize;
            for &g in &ai.post_u16[s..e] {
                let mut m = m0;
                let row = g as usize * b;
                while m != 0 {
                    let qi = m.trailing_zeros() as usize;
                    cnt[row + qi] += 1;
                    m &= m - 1;
                }
            }
        }
        for &k in touched.iter() { masks[k as usize] = 0; }

        for qi in 0..b {
            if qn[qi] == 0 { continue; }
            for t in 0..n {
                let i = cnt[t * b + qi];
                let denom = ai.kmer_counts[t] as u32 + qn[qi] - i;
                if denom > 0 { out[qi].jaccard_sum[t] += i as f64 / denom as f64; }
                if ai.present[t] { out[qi].shared[t] += 1; }
            }
        }
    }
    out
}

// ===================== accumulator experiments =====================
// The counting loop is `cnt[g] += 1` over sorted u16 posting lists. Variants
// below attack the load-modify-store dependency chain and the accumulator
// footprint. Counter width is NEVER assumed: the true bound on a count is
// min(|Q_a|, |T_a|), itself bounded only by protein length and K, so any narrow
// counter must be gated on a measured maximum, not a presumed cap.

macro_rules! fold_into {
    ($res:expr, $ai:expr, $get:expr, $qn:expr, $n:expr) => {
        for t in 0..$n {
            let i: u32 = $get(t);
            let denom = $ai.kmer_counts[t] as u32 + $qn - i;
            if denom > 0 { $res.jaccard_sum[t] += i as f64 / denom as f64; }
            if $ai.present[t] { $res.shared[t] += 1; }
        }
    };
}

/// P2 — two independent partial accumulators, alternating by element.
/// Breaks the store-to-load forwarding chain when nearby increments collide.
pub fn acc_partial2(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut a0 = vec![0u32; n];
    let mut a1 = vec![0u32; n];
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        a0.iter_mut().for_each(|x| *x = 0);
        a1.iter_mut().for_each(|x| *x = 0);
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            let v = &ai.post_u16[s..e];
            let mut i = 0;
            while i + 2 <= v.len() {
                a0[v[i] as usize] += 1;
                a1[v[i + 1] as usize] += 1;
                i += 2;
            }
            while i < v.len() { a0[v[i] as usize] += 1; i += 1; }
        }
        let qn = qk.len() as u32;
        fold_into!(res, ai, |t: usize| a0[t] + a1[t], qn, n);
    }
    res
}

/// P4 — four independent partial accumulators.
pub fn acc_partial4(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut a = [vec![0u32; n], vec![0u32; n], vec![0u32; n], vec![0u32; n]];
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        for x in a.iter_mut() { x.iter_mut().for_each(|y| *y = 0); }
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            let v = &ai.post_u16[s..e];
            let mut i = 0;
            while i + 4 <= v.len() {
                a[0][v[i] as usize] += 1;
                a[1][v[i + 1] as usize] += 1;
                a[2][v[i + 2] as usize] += 1;
                a[3][v[i + 3] as usize] += 1;
                i += 4;
            }
            while i < v.len() { a[0][v[i] as usize] += 1; i += 1; }
        }
        let qn = qk.len() as u32;
        fold_into!(res, ai, |t: usize| a[0][t] + a[1][t] + a[2][t] + a[3][t], qn, n);
    }
    res
}

/// PF — software prefetch of the accumulator line `dist` elements ahead.
pub fn acc_prefetch(p: &Partition, q: &Query, dist: usize) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            let v = &ai.post_u16[s..e];
            for i in 0..v.len() {
                #[cfg(target_arch = "x86_64")]
                if i + dist < v.len() {
                    unsafe {
                        core::arch::x86_64::_mm_prefetch(
                            cnt.as_ptr().add(v[i + dist] as usize) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                cnt[v[i] as usize] += 1;
            }
        }
        let qn = qk.len() as u32;
        fold_into!(res, ai, |t: usize| cnt[t], qn, n);
    }
    res
}

/// U16 — narrow counters, halving the accumulator footprint.
/// SAFETY OF WIDTH: caller must have verified `max |Q_a| < u16::MAX`. `saturating`
/// is not acceptable (it would silently bias Jaccard); this is why the production
/// path needs a build-time check with a u32 fallback rather than an assumption.
pub fn acc_u16(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u16; n];
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            for &g in &ai.post_u16[s..e] { cnt[g as usize] += 1; }
        }
        let qn = qk.len() as u32;
        fold_into!(res, ai, |t: usize| cnt[t] as u32, qn, n);
    }
    res
}

/// U16P2 — narrow counters plus two partial accumulators.
pub fn acc_u16_partial2(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut a0 = vec![0u16; n];
    let mut a1 = vec![0u16; n];
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        a0.iter_mut().for_each(|x| *x = 0);
        a1.iter_mut().for_each(|x| *x = 0);
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            let v = &ai.post_u16[s..e];
            let mut i = 0;
            while i + 2 <= v.len() {
                a0[v[i] as usize] += 1;
                a1[v[i + 1] as usize] += 1;
                i += 2;
            }
            while i < v.len() { a0[v[i] as usize] += 1; i += 1; }
        }
        let qn = qk.len() as u32;
        fold_into!(res, ai, |t: usize| a0[t] as u32 + a1[t] as u32, qn, n);
    }
    res
}

/// COUNT-ONLY — accumulation without the Jaccard fold, to split the two costs.
/// Returns a checksum so the work cannot be optimised away.
pub fn count_only(p: &Partition, q: &Query) -> u64 {
    let n = p.n_genomes;
    let mut cnt = vec![0u32; n];
    let mut sink = 0u64;
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            for &g in &ai.post_u16[s..e] { cnt[g as usize] += 1; }
        }
        sink = sink.wrapping_add(cnt[0] as u64).wrapping_add(cnt[n - 1] as u64);
    }
    sink
}

/// FOLD-F32 — same result, but the per-target divide is done in f32 (8-wide)
/// while the running sum stays f64. Ratios of small integers are exact to ~6e-8
/// in f32, well inside the precision the stored Jaccard needs; accumulation
/// stays f64 so summing ~80 accessions does not drift.
pub fn fold_f32(p: &Partition, q: &Query) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    let mut cnt = vec![0u32; n];
    let mut ratio = vec![0f32; n];
    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        cnt.iter_mut().for_each(|x| *x = 0);
        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            for &g in &ai.post_u16[s..e] { cnt[g as usize] += 1; }
        }
        let qn = qk.len() as u32;
        // vectorisable: no branches, no f64 divide
        for t in 0..n {
            let i = cnt[t];
            let denom = ai.kmer_counts[t] as u32 + qn - i;
            ratio[t] = i as f32 / denom as f32;
        }
        for t in 0..n {
            res.jaccard_sum[t] += ratio[t] as f64;
            res.shared[t] += ai.present[t] as u32;
        }
    }
    res
}
