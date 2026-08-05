//! The counting kernel: a k-mer join.
//!
//! For each `(accession, k-mer)` the query and target posting lists are crossed,
//! `Q[k] x T[k]`, incrementing a counter per genome pair. The resulting count
//! *is* the intersection cardinality — no set is ever intersected pairwise, and
//! genome pairs sharing nothing cost nothing.
//!
//! Both sides are read as **inverted** indexes, so no forward k-mer index exists
//! anywhere and a targets-only database is closed under search.
//!
//! Measured on the real 2,943-genome Firmicutes index against the superseded
//! per-query kernel (see git history): 1.50x from the join, 1.85x with the
//! symmetric upper-triangle path — 5.41M vs 2.93M genome pairs/s, bit-identical.
//!
//! Variants measured and rejected, all in `prototype/`: L1 tiling (0.83x),
//! delta/varint payloads (0.43x), private accumulators (0.85x), software
//! prefetch (0.73x), narrow counters (noise). At ~0.53 ns per posting increment
//! the inner loop is at the scalar hardware limit — scatter increments cannot be
//! auto-vectorised and AVX2 has no conflict detection.

use crate::index::Partition;

/// Cumulative genome offset of each partition, plus the total width.
pub fn partition_offsets(parts: &[Partition]) -> (Vec<usize>, usize) {
    let mut offs = Vec::with_capacity(parts.len());
    let mut total = 0usize;
    for p in parts {
        offs.push(total);
        total += p.n_genomes;
    }
    (offs, total)
}

/// K-mer join: accumulate `Q[k] x T[k]` from two **inverted** indexes.
///
/// This needs no forward index on either side, which is what makes a
/// targets-only database closed under search — one stored format, no privileged
/// "query" database, no transpose.
///
/// The accumulator is two-dimensional (`block x n_target`), unlike the per-query
/// kernel's single row, so the query dimension must be blocked. `Q[k]` is sorted
/// by local ID, so restricting it to a block is a monotone cursor advance; the
/// cursor array persists across blocks and costs `n_acc * kspace * 4` bytes.
///
/// Zeroing and fold costs are identical to the per-query kernel — the block width
/// cancels — and total increments are the same sum reassociated. The only extra
/// work is the cursor sweep, `n_blocks * n_acc * kspace`.
pub fn join_into(
    qp: &Partition,
    tp: &Partition,
    block: usize,
    jaccard: &mut [f64],
    shared: &mut [u32],
) {
    join_range_into(qp, tp, 0, qp.n_genomes, block, false, jaccard, shared, None)
}

/// `join_into` restricted to query genomes `[qstart, qend)`.
///
/// `jaccard`/`shared` cover only that range. Cursors are seeded by binary search
/// so disjoint ranges run on separate threads without coordination.
///
/// **Accession is the outer loop.** Cursors then span `kspace` for the accession
/// in hand rather than `n_acc * kspace` for all of them at once — 640 KB instead
/// of 78 MB, which at 8 threads is 5 MB instead of 624 MB against a 2 GiB
/// partition budget. Jaccard accumulates straight into the caller's output, so
/// the separate `block * n_target` sum buffers disappear too.
pub fn join_range_into(
    qp: &Partition,
    tp: &Partition,
    qstart: usize,
    qend: usize,
    block: usize,
    symmetric: bool,
    jaccard: &mut [f64],
    shared: &mut [u32],
    mut sumsq: Option<&mut [f64]>,
) {
    let nt = tp.n_genomes;
    let nq = qend - qstart;
    assert_eq!(qp.n_acc, tp.n_acc);
    assert_eq!(jaccard.len(), nq * nt);
    assert_eq!(shared.len(), nq * nt);
    if nq == 0 {
        return;
    }

    let block = block.max(1).min(nq);
    jaccard.fill(0.0);
    shared.fill(0);
    if let Some(sq) = sumsq.as_deref_mut() {
        assert_eq!(sq.len(), nq * nt);
        sq.fill(0.0);
    }

    // Slot pairs for k-mers both sides carry, and a forward cursor per pair.
    // Sized by the intersection rather than by `kspace`, which is the second
    // payoff of sparse storage: at 15.8% occupancy the sweep is ~6x shorter and
    // the cursor array shrinks with it.
    let mut matches: Vec<(u32, u32)> = Vec::new();
    let mut cursors: Vec<u32> = Vec::new();
    let mut cnt = vec![0u32; block * nt];

    for a in 0..qp.n_acc {
        let qa = &qp.accs[a];
        let ta = &tp.accs[a];

        // Merge-join two sorted k-mer lists — no lookup, no binary search.
        matches.clear();
        let (mut i, mut j) = (0usize, 0usize);
        while i < qa.kmers.len() && j < ta.kmers.len() {
            match qa.kmers[i].cmp(&ta.kmers[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    matches.push((i as u32, j as u32));
                    i += 1;
                    j += 1;
                }
            }
        }

        // Seed each cursor at the first local ID >= qstart. Postings are sorted,
        // so this is a partition point; it is also what lets disjoint query
        // ranges run on separate threads without coordination.
        cursors.clear();
        cursors.reserve(matches.len());
        for &(qi, _) in &matches {
            let (s, e) = (qa.offsets[qi as usize] as usize, qa.offsets[qi as usize + 1] as usize);
            let skip = qa.postings[s..e].partition_point(|&g| (g as usize) < qstart);
            cursors.push((s + skip) as u32);
        }

        let mut qlo = qstart;
        while qlo < qend {
            let qhi = (qlo + block).min(qend);
            let w = qhi - qlo;
            cnt[..w * nt].fill(0);

            for (m, &(qi, ti)) in matches.iter().enumerate() {
                let stop = qa.offsets[qi as usize + 1];
                let mut c = cursors[m];
                if c >= stop {
                    continue;
                }
                let run_start = ta.offsets[ti as usize] as usize;
                let ts = &ta.postings[run_start..ta.offsets[ti as usize + 1] as usize];
                while c < stop {
                    let qg = qa.postings[c as usize] as usize;
                    if qg >= qhi {
                        break;
                    }
                    // Upper triangle only: entries after the query's own slot are
                    // exactly the targets with a larger local ID, the runs being
                    // sorted and identical when self-comparing.
                    let run = if symmetric {
                        &ts[(c as usize - run_start + 1).min(ts.len())..]
                    } else {
                        ts
                    };
                    if !run.is_empty() {
                        let row = (qg - qlo) * nt;
                        for &tg in run {
                            cnt[row + tg as usize] += 1;
                        }
                    }
                    c += 1;
                }
                cursors[m] = c;
            }

            // Runs for every query genome in the block, including those with no
            // matched k-mers — an accession both carry still counts as shared.
            for qi in 0..w {
                let qn = qa.kmer_counts[qlo + qi];
                if qn == 0 {
                    continue; // query genome lacks this accession
                }
                let src = qi * nt;
                let dst = (qlo + qi - qstart) * nt;
                let tlo = if symmetric { qlo + qi + 1 } else { 0 };
                // Two loop bodies rather than a per-element branch. A target
                // lacking this accession yields j = 0, contributing nothing to
                // either sum, and `shared` is not incremented — so the mean and
                // variance are over shared accessions without a mask.
                if let Some(sq) = sumsq.as_deref_mut() {
                    for t in tlo..nt {
                        let i = cnt[src + t];
                        let denom = ta.kmer_counts[t] + qn - i;
                        if denom > 0 {
                            let j = i as f64 / denom as f64;
                            jaccard[dst + t] += j;
                            sq[dst + t] += j * j;
                        }
                        if ta.present[t] {
                            shared[dst + t] += 1;
                        }
                    }
                } else {
                    for t in tlo..nt {
                        let i = cnt[src + t];
                        let denom = ta.kmer_counts[t] + qn - i;
                        if denom > 0 {
                            jaccard[dst + t] += i as f64 / denom as f64;
                        }
                        if ta.present[t] {
                            shared[dst + t] += 1;
                        }
                    }
                }
            }
            qlo = qhi;
        }
    }
}

/// Upper bound on cursor bytes one join worker holds: four per k-mer both sides
/// could share, which is at most the smaller occupied set — not `kspace`.
pub fn join_cursor_bytes(qp: &Partition, tp: &Partition) -> usize {
    qp.accs
        .iter()
        .zip(&tp.accs)
        .map(|(q, t)| q.occupied().min(t.occupied()) * 4)
        .max()
        .unwrap_or(0)
}

/// Threaded k-mer join over one partition pair: disjoint query ranges, no shared
/// state. Cursor seeding by binary search is what makes the split free.
pub fn join_threaded(
    qp: &Partition,
    tp: &Partition,
    block: usize,
    threads: usize,
    symmetric: bool,
    jaccard: &mut [f64],
    shared: &mut [u32],
    sumsq: Option<&mut [f64]>,
) {
    let (nq, nt) = (qp.n_genomes, tp.n_genomes);
    let threads = threads.max(1).min(nq.max(1));
    let per = nq.div_ceil(threads);

    let jchunks: Vec<&mut [f64]> = jaccard.chunks_mut(per * nt).collect();
    let schunks: Vec<&mut [u32]> = shared.chunks_mut(per * nt).collect();
    // Chunked the same way so each worker owns disjoint rows of every output.
    let mut qchunks: Vec<Option<&mut [f64]>> = match sumsq {
        Some(sq) => sq.chunks_mut(per * nt).map(Some).collect(),
        None => (0..jchunks.len()).map(|_| None).collect(),
    };
    while qchunks.len() < jchunks.len() {
        qchunks.push(None);
    }

    std::thread::scope(|sc| {
        for (i, ((jc, scn), qc)) in jchunks.into_iter().zip(schunks).zip(qchunks).enumerate() {
            let (qs, qe) = (i * per, ((i + 1) * per).min(nq));
            if qs >= qe {
                continue;
            }
            sc.spawn(move || {
                join_range_into(qp, tp, qs, qe, block, symmetric, jc, scn, qc)
            });
        }
    });
}

/// Standard deviation of Jaccard across shared accessions, from the running sum
/// and sum of squares.
///
/// `sqrt(E[j^2] - (E[j])^2)`. The usual objection to this form is catastrophic
/// cancellation, which does not bite with Jaccard bounded to [0, 1]: real values
/// run ~0.09 with spread ~0.005, costing about 2.5 significant digits of f64's
/// 16. Welford would avoid it at the price of a division per element, in a fold
/// that is ~10% of runtime.
///
/// **The variance is clamped at zero** — near-identical per-accession Jaccards
/// round negative, and `sqrt` of that is NaN.
#[inline]
pub fn stdev_from(sum: f64, sumsq: f64, shared: u32) -> f64 {
    if shared == 0 {
        return f64::NAN;
    }
    let n = shared as f64;
    let mean = sum / n;
    (sumsq / n - mean * mean).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Partition;
    use std::collections::HashSet;

    /// Brute-force reference: plain set arithmetic over the original k-mer sets,
    /// sharing no machinery with the inverted index. Mirrors FastAAI 1 — Jaccard
    /// is summed over every target (contributing 0 where the target lacks the
    /// accession) while `shared` counts only accessions both genomes carry.
    fn reference(sets: &[Vec<Vec<u32>>]) -> (Vec<f64>, Vec<u32>) {
        let n = sets.len();
        let (mut js, mut sh) = (vec![0.0f64; n * n], vec![0u32; n * n]);
        for qi in 0..n {
            for (a, qk) in sets[qi].iter().enumerate() {
                if qk.is_empty() {
                    continue;
                }
                let q: HashSet<u32> = qk.iter().copied().collect();
                for t in 0..n {
                    let tk = &sets[t][a];
                    if tk.is_empty() {
                        continue;
                    }
                    let ts: HashSet<u32> = tk.iter().copied().collect();
                    let inter = q.intersection(&ts).count();
                    js[qi * n + t] += inter as f64 / (q.len() + ts.len() - inter) as f64;
                    sh[qi * n + t] += 1;
                }
            }
        }
        (js, sh)
    }

    fn fixture() -> Vec<Vec<Vec<u32>>> {
        vec![
            vec![vec![1, 2, 3, 4], vec![10, 11]],
            vec![vec![3, 4, 5, 6], vec![10, 12]],
            vec![vec![1, 2, 3, 4], vec![]], // identical acc0, missing acc1
            vec![vec![], vec![]],            // shares nothing
        ]
    }

    /// Nine genomes with enough structure that splitting or blocking can go wrong.
    fn wide_fixture() -> Vec<Vec<Vec<u32>>> {
        (0..9)
            .map(|i| {
                let mut g = vec![
                    (0..4u32).map(|j| (i + j) % 13).collect::<Vec<_>>(),
                    if i % 3 == 0 { vec![] } else { vec![i % 7, (i + 2) % 7] },
                ];
                for v in g.iter_mut() {
                    v.sort_unstable();
                    v.dedup();
                }
                g
            })
            .collect()
    }

    #[test]
    fn matches_brute_force_reference() {
        let sets = fixture();
        let n = sets.len();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let (wj, ws) = reference(&sets);
        for block in [1usize, 2, 4, 16] {
            let mut j = vec![0.0; n * n];
            let mut s = vec![0u32; n * n];
            join_into(&p, &p, block, &mut j, &mut s);
            assert_eq!(s, ws, "block={block}: shared differs");
            for i in 0..j.len() {
                assert!((j[i] - wj[i]).abs() < 1e-12, "block={block} cell {i}");
            }
        }
    }

    #[test]
    fn identical_genomes_score_one() {
        let sets = fixture();
        let n = sets.len();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut j = vec![0.0; n * n];
        let mut s = vec![0u32; n * n];
        join_into(&p, &p, 2, &mut j, &mut s);
        assert!((j[0] / s[0] as f64 - 1.0).abs() < 1e-12);
        assert_eq!(s[2], 1);
        assert!((j[2] / s[2] as f64 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_shared_accession_yields_zero_shared_not_a_score() {
        let sets = fixture();
        let n = sets.len();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut j = vec![0.0; n * n];
        let mut s = vec![0u32; n * n];
        join_into(&p, &p, 2, &mut j, &mut s);
        assert_eq!(s[3], 0, "shared must be 0, distinguishing it from AAI 0");
        assert_eq!(j[3], 0.0);
    }

    #[test]
    fn threading_does_not_change_results() {
        let sets = wide_fixture();
        let n = sets.len();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut jb = vec![0.0; n * n];
        let mut sb = vec![0u32; n * n];
        join_into(&p, &p, 2, &mut jb, &mut sb);
        for threads in [1usize, 2, 4, 8] {
            let mut j = vec![0.0; n * n];
            let mut s = vec![0u32; n * n];
            join_threaded(&p, &p, 2, threads, false, &mut j, &mut s, None);
            assert_eq!(s, sb, "threads={threads}");
            for i in 0..j.len() {
                assert!((j[i] - jb[i]).abs() < 1e-12, "threads={threads} cell {i}");
            }
        }
    }

    #[test]
    fn partitioning_does_not_change_results() {
        let sets = wide_fixture();
        let n = sets.len();
        let (wj, ws) = reference(&sets);
        for chunk in [1usize, 2, 3, 4, 5, 8, 9] {
            let parts: Vec<Partition> =
                sets.chunks(chunk).map(|c| Partition::build(c, 2, 16).unwrap()).collect();
            let (offs, width) = partition_offsets(&parts);
            assert_eq!(width, n, "chunk={chunk}");
            let mut j = vec![0.0; n * n];
            let mut s = vec![0u32; n * n];
            for (qi, qp) in parts.iter().enumerate() {
                for (ti, tp) in parts.iter().enumerate() {
                    let (qn, tn) = (qp.n_genomes, tp.n_genomes);
                    let mut bj = vec![0.0; qn * tn];
                    let mut bs = vec![0u32; qn * tn];
                    join_into(qp, tp, 2, &mut bj, &mut bs);
                    for r in 0..qn {
                        let dst = (offs[qi] + r) * n + offs[ti];
                        j[dst..dst + tn].copy_from_slice(&bj[r * tn..(r + 1) * tn]);
                        s[dst..dst + tn].copy_from_slice(&bs[r * tn..(r + 1) * tn]);
                    }
                }
            }
            assert_eq!(s, ws, "chunk={chunk}: shared differs");
            for i in 0..j.len() {
                assert!((j[i] - wj[i]).abs() < 1e-12, "chunk={chunk} cell {i}");
            }
        }
    }

    #[test]
    fn asymmetric_partitions_agree_with_the_reference() {
        let sets = wide_fixture();
        let n = sets.len();
        let qp = Partition::build(&sets[..4], 2, 16).unwrap();
        let tp = Partition::build(&sets, 2, 16).unwrap();
        let (wj, ws) = reference(&sets);
        let mut j = vec![0.0; 4 * n];
        let mut s = vec![0u32; 4 * n];
        join_into(&qp, &tp, 2, &mut j, &mut s);
        for r in 0..4 {
            for t in 0..n {
                assert_eq!(s[r * n + t], ws[r * n + t]);
                assert!((j[r * n + t] - wj[r * n + t]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn symmetric_join_fills_the_upper_triangle_only() {
        let sets = wide_fixture();
        let n = sets.len();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut jf = vec![0.0; n * n];
        let mut sf = vec![0u32; n * n];
        join_into(&p, &p, 3, &mut jf, &mut sf);
        let mut ju = vec![0.0; n * n];
        let mut su = vec![0u32; n * n];
        join_range_into(&p, &p, 0, n, 3, true, &mut ju, &mut su, None);
        for i in 0..n {
            for t in 0..n {
                if t > i {
                    assert!((ju[i * n + t] - jf[i * n + t]).abs() < 1e-12, "upper ({i},{t})");
                    assert_eq!(su[i * n + t], sf[i * n + t], "upper shared ({i},{t})");
                } else {
                    assert_eq!(ju[i * n + t], 0.0, "lower/diagonal untouched");
                    assert_eq!(su[i * n + t], 0, "lower/diagonal untouched");
                }
            }
        }
    }

    #[test]
    fn global_index_is_partition_times_size_plus_local() {
        let sets = wide_fixture();
        let parts: Vec<Partition> =
            sets.chunks(4).map(|c| Partition::build(c, 2, 16).unwrap()).collect();
        let (offs, width) = partition_offsets(&parts);
        assert_eq!(offs, vec![0, 4, 8]);
        assert_eq!(width, 9);
        for (pi, &off) in offs.iter().enumerate() {
            if parts[pi].n_genomes == 4 {
                assert_eq!(off, pi * 4, "arithmetic mapping holds for full partitions");
            }
        }
        assert_eq!(parts[2].n_genomes, 1, "final partition is short");
    }

    #[test]
    fn cursor_array_is_bounded_by_occupancy_not_kspace() {
        // The point of the accession-outer loop: 0.64 MB per worker at
        // k=4/|A|=20, not the 78 MB an (accession, k-mer) cursor table would cost.
        let sets = fixture();
        let p = Partition::build(&sets, 2, 16).unwrap();
        // Bounded by the occupied intersection, not the k-mer space.
        assert!(join_cursor_bytes(&p, &p) <= p.kspace * 4);
        assert_eq!(join_cursor_bytes(&p, &p),
                   p.accs.iter().map(|a| a.occupied() * 4).max().unwrap());
    }
}
