//! Scheduling study: does it matter *which* target partition concurrent threads
//! are working on?
//!
//! Schedule A "block-parallel"  — threads pull (query, target-partition) work items
//!   in interleaved order, so at any instant different threads are streaming
//!   *different* target indexes. This is the naive one-thread-per-block model.
//!
//! Schedule B "target-major"    — outer sequential loop over target partitions,
//!   inner parallel over queries. At any instant every thread is streaming the
//!   *same* target index.
//!
//! Both do identical total work. Any difference is memory-system behaviour.
//!
//! usage: sched <data.bin> <genomes_per_partition> <n_partitions> <n_queries>

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use fastaai_rs::data;
use fastaai_rs::index::Partition;
use fastaai_rs::kernel;
use fastaai_rs::kmer::{Alphabet, Kmerizer};
use fastaai_rs::synth;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().unwrap_or("data.bin".into());
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let np: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let nq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16);

    let ds = data::load(&path).expect("load");
    let alpha = Alphabet::new(&ds.alphabet);
    let ks = alpha.kspace as usize;
    let mut km = Kmerizer::new(alpha.clone());

    println!("building {np} partitions x {n} genomes ...");
    let mut parts = Vec::with_capacity(np);
    let mut queries: Vec<Vec<(usize, Vec<u32>)>> = Vec::new();
    for p in 0..np {
        let sets = synth::synthesize_seeded(&ds, &mut km, n, 0x9E3779B97F4A7C15 ^ (p as u64 * 0x1234567));
        if p == 0 {
            queries = (0..nq)
                .map(|i| {
                    let gi = (i * n / nq).min(n - 1);
                    sets[gi].iter().enumerate().filter(|(_, v)| !v.is_empty())
                        .map(|(x, v)| (x, v.clone())).collect()
                })
                .collect();
        }
        let names: Vec<String> = (0..n).map(|i| format!("p{p}g{i}")).collect();
        parts.push(Partition::build_lean(&sets, ds.n_acc, ks, names));
    }

    let idx_mb: f64 = parts.iter().map(|p| p.posting_entries() as f64 * 2.0).sum::<f64>() / 1e6;
    let offs_mb: f64 = parts.iter().map(|p| p.offset_bytes() as f64).sum::<f64>() / 1e6;
    println!("resident index: {idx_mb:.0} MB postings + {offs_mb:.0} MB offsets");

    // Work = every query against every partition.
    let items: Vec<(usize, usize)> =
        (0..np).flat_map(|t| (0..nq).map(move |q| (q, t))).collect();
    let total_inc: u64 = items.iter()
        .map(|&(q, t)| kernel::increments(&parts[t], &queries[q])).sum();
    let total_pairs: u64 = (nq * np * n) as u64;
    println!("work: {} items, {total_inc} increments, {total_pairs} genome pairs\n",
             items.len());

    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8);

    let report = |label: &str, secs: f64| {
        println!("{label:<28} {secs:>7.3} s  {:>8.1} M inc/s  {:>9.0} pairs/s  {:>8.0} pairs/s/thread",
                 total_inc as f64 / secs / 1e6,
                 total_pairs as f64 / secs,
                 total_pairs as f64 / secs / nt as f64);
    };

    // ---- single thread, for the per-thread baseline --------------------------
    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for &(q, t) in &items {
        acc += kernel::untiled_u16(&parts[t], &queries[q]).jaccard_sum[0];
    }
    std::hint::black_box(acc);
    let single = t0.elapsed().as_secs_f64();
    println!("{:<28} {single:>7.3} s  {:>8.1} M inc/s  {:>9.0} pairs/s  {:>8.0} pairs/s/thread",
             "1 thread (reference)",
             total_inc as f64 / single / 1e6,
             total_pairs as f64 / single,
             total_pairs as f64 / single);

    // ---- Schedule A: interleaved targets, work-stealing over blocks ----------
    // Interleave so adjacent items hit different target partitions.
    let mut inter: Vec<(usize, usize)> = Vec::with_capacity(items.len());
    for q in 0..nq {
        for t in 0..np {
            inter.push((q, t));
        }
    }
    let cursor = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            let (cursor, inter, parts, queries) = (&cursor, &inter, &parts, &queries);
            sc.spawn(move || {
                let mut a = 0.0f64;
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= inter.len() {
                        break;
                    }
                    let (q, t) = inter[i];
                    a += kernel::untiled_u16(&parts[t], &queries[q]).jaccard_sum[0];
                }
                std::hint::black_box(a);
            });
        }
    });
    report("A block-parallel (mixed t)", t0.elapsed().as_secs_f64());

    // ---- thread sweep on schedule A ------------------------------------------
    println!("\n{:<10} {:>9} {:>14} {:>16} {:>10}", "threads", "secs", "pairs/s", "pairs/s/thread", "scaling");
    for &t in &[1usize, 2, 4, 6, 8, 12, 16, 20] {
        if t > nt { continue; }
        let cursor = AtomicUsize::new(0);
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for _ in 0..t {
                let (cursor, inter, parts, queries) = (&cursor, &inter, &parts, &queries);
                sc.spawn(move || {
                    let mut a = 0.0f64;
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= inter.len() { break; }
                        let (q, tt) = inter[i];
                        a += kernel::untiled_u16(&parts[tt], &queries[q]).jaccard_sum[0];
                    }
                    std::hint::black_box(a);
                });
            }
        });
        let s = t0.elapsed().as_secs_f64();
        println!("{t:<10} {s:>9.3} {:>14.0} {:>16.0} {:>9.2}x",
                 total_pairs as f64 / s,
                 total_pairs as f64 / s / t as f64,
                 single / s);
    }

    // ---- Schedule B: target-major, all threads on one target at a time -------
    let t0 = Instant::now();
    for t in 0..np {
        let cursor = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..nt {
                let (cursor, parts, queries) = (&cursor, &parts, &queries);
                sc.spawn(move || {
                    let mut a = 0.0f64;
                    loop {
                        let q = cursor.fetch_add(1, Ordering::Relaxed);
                        if q >= queries.len() {
                            break;
                        }
                        a += kernel::untiled_u16(&parts[t], &queries[q]).jaccard_sum[0];
                    }
                    std::hint::black_box(a);
                });
            }
        });
    }
    report("B target-major (shared t)", t0.elapsed().as_secs_f64());
}
