//! The counting kernel.
//!
//! For each query k-mer, gather that k-mer's posting list and increment a counter
//! per target genome. The resulting count *is* the intersection cardinality — no
//! set is ever intersected pairwise, and targets sharing nothing cost nothing.
//!
//! This is the only kernel shipped. The prototype measured tiled variants (0.83x),
//! delta/varint payloads (0.43x), query batching (0.96x), private accumulators
//! (0.85x), software prefetch (0.73x) and narrow counters (noise). All lost. At
//! ~0.53 ns per increment this loop is at the scalar hardware limit: scatter
//! increments cannot be auto-vectorised and AVX2 has no conflict detection.

use crate::index::Partition;

/// One query genome: `(accession, sorted unique k-mer IDs)`.
pub type Query = [(usize, Vec<u32>)];

/// Jaccard sums and shared-accession counts against every target.
/// Kept unreduced so the caller controls the `sum / shared` division.
#[derive(Clone, Debug)]
pub struct QueryResult {
    pub jaccard_sum: Vec<f64>,
    pub shared: Vec<u32>,
}

impl QueryResult {
    pub fn new(n: usize) -> Self {
        QueryResult { jaccard_sum: vec![0.0; n], shared: vec![0; n] }
    }

    /// Mean Jaccard over shared accessions; NaN where nothing is shared.
    pub fn means(&self) -> Vec<f64> {
        self.jaccard_sum
            .iter()
            .zip(&self.shared)
            .map(|(s, &c)| if c == 0 { f64::NAN } else { s / c as f64 })
            .collect()
    }
}

/// Count one query genome against every genome in `p`.
pub fn count(p: &Partition, q: &Query, scratch: &mut Vec<u32>) -> QueryResult {
    let n = p.n_genomes;
    let mut res = QueryResult::new(n);
    scratch.clear();
    scratch.resize(n, 0);

    for (acc, qk) in q {
        let ai = &p.accs[*acc];
        scratch.iter_mut().for_each(|x| *x = 0);

        for &km in qk {
            let s = ai.offsets[km as usize] as usize;
            let e = ai.offsets[km as usize + 1] as usize;
            // Sorted postings make this a monotone sweep of `scratch`.
            for &g in &ai.postings[s..e] {
                scratch[g as usize] += 1;
            }
        }

        let qn = qk.len() as u32;
        for t in 0..n {
            let i = scratch[t];
            // |Q| + |T| - |intersection|; zero only when both sides are empty.
            let denom = ai.kmer_counts[t] + qn - i;
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

/// Posting entries a query will touch — the kernel's fundamental work unit.
pub fn increments(p: &Partition, q: &Query) -> u64 {
    q.iter()
        .flat_map(|(acc, qk)| {
            let ai = &p.accs[*acc];
            qk.iter().map(move |&km| {
                (ai.offsets[km as usize + 1] - ai.offsets[km as usize]) as u64
            })
        })
        .sum()
}

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

/// Run every query against every partition, writing row-major into `jaccard` and
/// `shared`. Row width is the total genome count across all partitions.
///
/// Genome *g* lives at global index `partition_offset + local`, so with a uniform
/// partition size P this is exactly `partition * P + local` — the arithmetic
/// mapping. Nothing here depends on that; offsets are computed from actual sizes
/// so a short final partition works unchanged.
pub fn search_partitions_into(
    parts: &[Partition],
    queries: &[Vec<(usize, Vec<u32>)>],
    threads: usize,
    jaccard: &mut [f64],
    shared: &mut [u32],
) {
    let (offs, width) = partition_offsets(parts);
    assert_eq!(jaccard.len(), queries.len() * width);
    assert_eq!(shared.len(), queries.len() * width);

    let threads = threads.max(1).min(queries.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);

    let jrows: Vec<_> = jaccard.chunks_mut(width).map(std::sync::Mutex::new).collect();
    let srows: Vec<_> = shared.chunks_mut(width).map(std::sync::Mutex::new).collect();

    std::thread::scope(|sc| {
        for _ in 0..threads {
            let (next, jrows, srows, offs) = (&next, &jrows, &srows, &offs);
            sc.spawn(move || {
                let mut scratch: Vec<u32> = Vec::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= queries.len() {
                        break;
                    }
                    let mut jrow = jrows[i].lock().unwrap();
                    let mut srow = srows[i].lock().unwrap();
                    for (p, &off) in parts.iter().zip(offs) {
                        let r = count(p, &queries[i], &mut scratch);
                        jrow[off..off + p.n_genomes].copy_from_slice(&r.jaccard_sum);
                        srow[off..off + p.n_genomes].copy_from_slice(&r.shared);
                    }
                }
            });
        }
    });
}

/// Run every query against `p`, writing row-major into `jaccard` and `shared`.
///
/// Work is stolen one query at a time. The prototype measured that letting
/// threads land on unrelated targets beats target-major ordering (7.93M vs 6.95M
/// pairs/s) — at partition sizes far above L3 there is no reuse to preserve, and
/// ordering only adds barriers.
///
/// Scaling is memory-bound: measured 6.17x at 16 threads and *negative* beyond
/// (20 threads was slower than 16 on a 6P+8E laptop). Callers should cap threads
/// rather than using every logical core.
pub fn search_into(
    p: &Partition,
    queries: &[Vec<(usize, Vec<u32>)>],
    threads: usize,
    jaccard: &mut [f64],
    shared: &mut [u32],
) {
    let n = p.n_genomes;
    assert_eq!(jaccard.len(), queries.len() * n);
    assert_eq!(shared.len(), queries.len() * n);

    let threads = threads.max(1).min(queries.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);

    // Each row is written by exactly one thread, so hand out disjoint row slices.
    let jrows: Vec<&mut [f64]> = jaccard.chunks_mut(n).collect();
    let srows: Vec<&mut [u32]> = shared.chunks_mut(n).collect();
    let jcell: Vec<_> = jrows.into_iter().map(std::sync::Mutex::new).collect();
    let scell: Vec<_> = srows.into_iter().map(std::sync::Mutex::new).collect();

    std::thread::scope(|sc| {
        for _ in 0..threads {
            let (next, jcell, scell) = (&next, &jcell, &scell);
            sc.spawn(move || {
                let mut scratch: Vec<u32> = Vec::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= queries.len() {
                        break;
                    }
                    let r = count(p, &queries[i], &mut scratch);
                    jcell[i].lock().unwrap().copy_from_slice(&r.jaccard_sum);
                    scell[i].lock().unwrap().copy_from_slice(&r.shared);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Partition;
    use std::collections::HashSet;

    /// Brute-force reference: plain set arithmetic, sharing no machinery with the
    /// inverted index. Mirrors FastAAI 1 — Jaccard is summed over every target
    /// (contributing 0 where the target lacks the accession) while `shared`
    /// counts only accessions both genomes carry.
    fn reference(sets: &[Vec<Vec<u32>>], qi: usize) -> (Vec<f64>, Vec<u32>) {
        let n = sets.len();
        let (mut js, mut sh) = (vec![0.0f64; n], vec![0u32; n]);
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
                let tset: HashSet<u32> = tk.iter().copied().collect();
                let inter = q.intersection(&tset).count();
                let union = q.len() + tset.len() - inter;
                js[t] += inter as f64 / union as f64;
                sh[t] += 1;
            }
        }
        (js, sh)
    }

    fn fixture() -> Vec<Vec<Vec<u32>>> {
        vec![
            vec![vec![1, 2, 3, 4], vec![10, 11]],
            vec![vec![3, 4, 5, 6], vec![10, 12]],
            vec![vec![1, 2, 3, 4], vec![]],       // identical acc0, missing acc1
            vec![vec![], vec![]],                  // shares nothing
        ]
    }

    #[test]
    fn matches_brute_force_reference() {
        let sets = fixture();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut scratch = Vec::new();
        for qi in 0..sets.len() {
            let q: Vec<(usize, Vec<u32>)> = sets[qi].iter().enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(a, v)| (a, v.clone())).collect();
            let got = count(&p, &q, &mut scratch);
            let (wj, ws) = reference(&sets, qi);
            for t in 0..sets.len() {
                assert_eq!(got.shared[t], ws[t], "shared q{qi} t{t}");
                assert!((got.jaccard_sum[t] - wj[t]).abs() < 1e-12,
                        "jaccard q{qi} t{t}: {} vs {}", got.jaccard_sum[t], wj[t]);
            }
        }
    }

    #[test]
    fn identical_genomes_score_one() {
        let sets = fixture();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut scratch = Vec::new();
        let q: Vec<(usize, Vec<u32>)> = vec![(0, sets[0][0].clone()), (1, sets[0][1].clone())];
        let r = count(&p, &q, &mut scratch);
        assert!((r.means()[0] - 1.0).abs() < 1e-12, "self-comparison must be 1.0");
        // Genome 2 matches on acc0 only, and carries no acc1.
        assert_eq!(r.shared[2], 1);
        assert!((r.means()[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_shared_accession_yields_nan_not_zero() {
        let sets = fixture();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut scratch = Vec::new();
        let q: Vec<(usize, Vec<u32>)> = vec![(0, sets[0][0].clone())];
        let r = count(&p, &q, &mut scratch);
        assert_eq!(r.shared[3], 0);
        assert!(r.means()[3].is_nan(), "unshared pairs must be NaN, not 0.0");
    }

    #[test]
    fn threaded_search_matches_serial() {
        let sets = fixture();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let queries: Vec<Vec<(usize, Vec<u32>)>> = sets.iter().map(|g| {
            g.iter().enumerate().filter(|(_, v)| !v.is_empty())
                .map(|(a, v)| (a, v.clone())).collect()
        }).collect();
        let n = sets.len();
        for threads in [1usize, 2, 8] {
            let mut j = vec![0.0; n * n];
            let mut s = vec![0u32; n * n];
            search_into(&p, &queries, threads, &mut j, &mut s);
            let mut scratch = Vec::new();
            for qi in 0..n {
                let r = count(&p, &queries[qi], &mut scratch);
                assert_eq!(&s[qi * n..(qi + 1) * n], &r.shared[..], "threads={threads}");
                for t in 0..n {
                    assert!((j[qi * n + t] - r.jaccard_sum[t]).abs() < 1e-12,
                            "threads={threads} q{qi} t{t}");
                }
            }
        }
    }

    /// Nine genomes, enough structure that splitting can go wrong.
    fn wide_fixture() -> Vec<Vec<Vec<u32>>> {
        (0..9)
            .map(|i| {
                vec![
                    (0..4u32).map(|j| (i + j) % 13).collect::<Vec<_>>(),
                    if i % 3 == 0 { vec![] } else { vec![i % 7, (i + 2) % 7] },
                ]
            })
            .map(|mut g| {
                for v in g.iter_mut() {
                    v.sort_unstable();
                    v.dedup();
                }
                g
            })
            .collect()
    }

    #[test]
    fn partitioning_does_not_change_results() {
        let sets = wide_fixture();
        let n = sets.len();
        let queries: Vec<Vec<(usize, Vec<u32>)>> = sets
            .iter()
            .map(|g| {
                g.iter().enumerate().filter(|(_, v)| !v.is_empty())
                    .map(|(a, v)| (a, v.clone())).collect()
            })
            .collect();

        let whole = vec![Partition::build(&sets, 2, 16).unwrap()];
        let mut jw = vec![0.0; n * n];
        let mut sw = vec![0u32; n * n];
        search_partitions_into(&whole, &queries, 1, &mut jw, &mut sw);

        // Every chunking, including one that leaves a short final partition.
        for chunk in [1usize, 2, 3, 4, 5, 8, 9] {
            let parts: Vec<Partition> = sets
                .chunks(chunk)
                .map(|c| Partition::build(c, 2, 16).unwrap())
                .collect();
            let (_, width) = partition_offsets(&parts);
            assert_eq!(width, n, "chunk={chunk}: widths must agree");

            for threads in [1usize, 4] {
                let mut j = vec![0.0; n * n];
                let mut s = vec![0u32; n * n];
                search_partitions_into(&parts, &queries, threads, &mut j, &mut s);
                assert_eq!(s, sw, "chunk={chunk} threads={threads}: shared differs");
                for i in 0..j.len() {
                    assert!((j[i] - jw[i]).abs() < 1e-12,
                            "chunk={chunk} threads={threads} cell {i}: {} vs {}", j[i], jw[i]);
                }
            }
        }
    }

    #[test]
    fn global_index_is_partition_times_size_plus_local() {
        // The arithmetic mapping holds while partitions are uniform; the offset
        // table is what keeps a short final partition correct.
        let sets = wide_fixture();
        let parts: Vec<Partition> = sets.chunks(4).map(|c| Partition::build(c, 2, 16).unwrap()).collect();
        let (offs, width) = partition_offsets(&parts);
        assert_eq!(offs, vec![0, 4, 8]);
        assert_eq!(width, 9);
        // Uniform partitions: global = partition * P + local.
        for (pi, &off) in offs.iter().enumerate() {
            if parts[pi].n_genomes == 4 {
                assert_eq!(off, pi * 4, "arithmetic mapping holds for full partitions");
            }
        }
        // The short tail is exactly why the offset table exists: 8 != 2 * 4 would
        // still hold here, but a short partition *before* the last one would break
        // the arithmetic, and the table would not.
        assert_eq!(parts[2].n_genomes, 1, "final partition is short");
    }

    #[test]
    fn join_matches_the_per_query_kernel() {
        let sets = wide_fixture();
        let n = sets.len();
        let queries: Vec<Vec<(usize, Vec<u32>)>> = sets.iter().map(|g| {
            g.iter().enumerate().filter(|(_, v)| !v.is_empty())
                .map(|(a, v)| (a, v.clone())).collect()
        }).collect();

        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut jw = vec![0.0; n * n];
        let mut sw = vec![0u32; n * n];
        search_partitions_into(std::slice::from_ref(&p), &queries, 1, &mut jw, &mut sw);

        for block in [1usize, 2, 3, 5, 9, 32] {
            let mut j = vec![0.0; n * n];
            let mut s = vec![0u32; n * n];
            join_into(&p, &p, block, &mut j, &mut s);
            assert_eq!(s, sw, "block={block}: shared differs");
            for i in 0..j.len() {
                assert!((j[i] - jw[i]).abs() < 1e-12,
                        "block={block} cell {i}: {} vs {}", j[i], jw[i]);
            }
        }
    }

    #[test]
    fn join_works_across_asymmetric_partitions() {
        // Query and target partitions need not be the same size.
        let sets = wide_fixture();
        let qp = Partition::build(&sets[..4], 2, 16).unwrap();
        let tp = Partition::build(&sets, 2, 16).unwrap();
        let queries: Vec<Vec<(usize, Vec<u32>)>> = sets[..4].iter().map(|g| {
            g.iter().enumerate().filter(|(_, v)| !v.is_empty())
                .map(|(a, v)| (a, v.clone())).collect()
        }).collect();

        let (nq, nt) = (4, sets.len());
        let mut jw = vec![0.0; nq * nt];
        let mut sw = vec![0u32; nq * nt];
        search_partitions_into(std::slice::from_ref(&tp), &queries, 1, &mut jw, &mut sw);

        let mut j = vec![0.0; nq * nt];
        let mut s = vec![0u32; nq * nt];
        join_into(&qp, &tp, 2, &mut j, &mut s);
        assert_eq!(s, sw);
        for i in 0..j.len() {
            assert!((j[i] - jw[i]).abs() < 1e-12, "cell {i}");
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
        join_range_into(&p, &p, 0, n, 3, true, &mut ju, &mut su);

        for i in 0..n {
            for t in 0..n {
                if t > i {
                    assert!((ju[i * n + t] - jf[i * n + t]).abs() < 1e-12,
                            "upper cell ({i},{t})");
                    assert_eq!(su[i * n + t], sf[i * n + t], "upper shared ({i},{t})");
                } else {
                    assert_eq!(ju[i * n + t], 0.0, "lower/diagonal must be untouched");
                    assert_eq!(su[i * n + t], 0, "lower/diagonal must be untouched");
                }
            }
        }
    }

    #[test]
    fn increments_counts_posting_entries_touched() {
        let sets = fixture();
        let p = Partition::build(&sets, 2, 16).unwrap();
        // acc0 k-mers {1,2,3,4}: genome0 and genome2 carry 1,2,3,4; genome1 carries 3,4.
        // runs: 1->[0,2] 2->[0,2] 3->[0,1,2] 4->[0,1,2] = 2+2+3+3 = 10
        let q: Vec<(usize, Vec<u32>)> = vec![(0, vec![1, 2, 3, 4])];
        assert_eq!(increments(&p, &q), 10);
    }
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
    join_range_into(qp, tp, 0, qp.n_genomes, block, false, jaccard, shared)
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
) {
    let (nt, ks) = (tp.n_genomes, qp.kspace);
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

    let mut cursors = vec![0u32; ks];
    let mut cnt = vec![0u32; block * nt];

    for a in 0..qp.n_acc {
        let qa = &qp.accs[a];
        let ta = &tp.accs[a];

        // Seed at the first local ID >= qstart; postings are sorted, so this is
        // a partition point rather than a scan.
        for k in 0..ks {
            let (s, e) = (qa.offsets[k] as usize, qa.offsets[k + 1] as usize);
            let skip = qa.postings[s..e].partition_point(|&g| (g as usize) < qstart);
            cursors[k] = (s + skip) as u32;
        }

        let mut qlo = qstart;
        while qlo < qend {
            let qhi = (qlo + block).min(qend);
            let w = qhi - qlo;
            cnt[..w * nt].fill(0);

            for k in 0..ks {
                let stop = qa.offsets[k + 1];
                let mut c = cursors[k];
                if c >= stop {
                    continue;
                }
                let run_start = ta.offsets[k] as usize;
                let ts = &ta.postings[run_start..ta.offsets[k + 1] as usize];
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
                cursors[k] = c;
            }

            for qi in 0..w {
                let qn = qa.kmer_counts[qlo + qi];
                if qn == 0 {
                    continue; // query genome lacks this accession
                }
                let src = qi * nt;
                let dst = (qlo + qi - qstart) * nt;
                let tlo = if symmetric { qlo + qi + 1 } else { 0 };
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
            qlo = qhi;
        }
    }
}

/// Cursor-array bytes one join worker holds — `kspace`, not `n_acc * kspace`.
pub fn join_cursor_bytes(qp: &Partition) -> usize {
    qp.kspace * 4
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
) {
    let (nq, nt) = (qp.n_genomes, tp.n_genomes);
    let threads = threads.max(1).min(nq.max(1));
    let per = nq.div_ceil(threads);

    let jchunks: Vec<&mut [f64]> = jaccard.chunks_mut(per * nt).collect();
    let schunks: Vec<&mut [u32]> = shared.chunks_mut(per * nt).collect();

    std::thread::scope(|sc| {
        for (i, (jc, scn)) in jchunks.into_iter().zip(schunks).enumerate() {
            let (qs, qe) = (i * per, ((i + 1) * per).min(nq));
            if qs >= qe {
                continue;
            }
            sc.spawn(move || join_range_into(qp, tp, qs, qe, block, symmetric, jc, scn));
        }
    });
}
