//! Accumulator variants. Every one verified against the production kernel first.
use std::time::Instant;
use fastaai_rs::{data, kernel, synth};
use fastaai_rs::index::Partition;
use fastaai_rs::kmer::{Alphabet, Kmerizer};
use fastaai_rs::QueryResult;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().unwrap_or("data.bin".into());
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let nq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let ds = data::load(&path).expect("load");
    let alpha = Alphabet::new(&ds.alphabet);
    let mut km = Kmerizer::new(alpha.clone());
    let sets = synth::synthesize(&ds, &mut km, n);
    let names: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let part = Partition::build_lean(&sets, ds.n_acc, alpha.kspace as usize, names);
    let queries: Vec<Vec<(usize, Vec<u32>)>> = (0..nq).map(|i| {
        let gi = (i * n / nq).min(n - 1);
        sets[gi].iter().enumerate().filter(|(_, v)| !v.is_empty())
            .map(|(x, v)| (x, v.clone())).collect()
    }).collect();

    // Widest count actually reachable here, so the u16 variants are honest.
    let max_q = sets.iter().flat_map(|g| g.iter()).map(|v| v.len()).max().unwrap();
    drop(sets);
    let incs: u64 = queries.iter().map(|q| kernel::increments(&part, q)).sum();
    println!("{n} genomes, {incs} increments");
    println!("largest |Q_a| observed: {max_q}  (u16 counters valid only while < 65535)\n");

    let check = |l: &str, g: &QueryResult, w: &QueryResult| {
        for t in 0..part.n_genomes {
            assert_eq!(g.shared[t], w.shared[t], "{l} shared");
            assert!((g.jaccard_sum[t] - w.jaccard_sum[t]).abs() < 1e-9, "{l} jaccard");
        }
    };
    for q in &queries {
        let w = kernel::untiled_u16(&part, q);
        check("P2", &kernel::acc_partial2(&part, q), &w);
        check("P4", &kernel::acc_partial4(&part, q), &w);
        check("PF", &kernel::acc_prefetch(&part, q, 8), &w);
        check("U16", &kernel::acc_u16(&part, q), &w);
        check("U16P2", &kernel::acc_u16_partial2(&part, q), &w);
        let f32r = kernel::fold_f32(&part, q);
        for t in 0..part.n_genomes {
            assert_eq!(f32r.shared[t], w.shared[t], "F32 shared");
            let d = (f32r.jaccard_sum[t] - w.jaccard_sum[t]).abs();
            assert!(d < 1e-4, "F32 jaccard drift {d}");
        }
    }
    println!("all variants verified identical: OK\n");

    let mut base = 0.0f64;
    let mut run = |label: &str, f: &dyn Fn(&Partition, &[(usize, Vec<u32>)]) -> f64, is_base: bool| {
        // best of 3
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut acc = 0.0;
            for q in &queries { acc += f(&part, q); }
            std::hint::black_box(acc);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        if is_base { base = best; }
        println!("{label:<34} {best:>7.3} s  {:>6.3} ns/inc  {:>6.2}x  acc {:>6.0} KB",
                 best * 1e9 / incs as f64, base / best,
                 label_bytes(label, n));
    };
    fn label_bytes(l: &str, n: usize) -> f64 {
        let w = if l.contains("u16") { 2.0 } else { 4.0 };
        let m = if l.contains("x2") { 2.0 } else if l.contains("x4") { 4.0 } else { 1.0 };
        n as f64 * w * m / 1e3
    }

    run("A   u32 single (production)", &|p, q| kernel::untiled_u16(p, q).jaccard_sum[0], true);
    run("P2  u32 x2 partial", &|p, q| kernel::acc_partial2(p, q).jaccard_sum[0], false);
    run("P4  u32 x4 partial", &|p, q| kernel::acc_partial4(p, q).jaccard_sum[0], false);
    run("PF  u32 single + prefetch(8)", &|p, q| kernel::acc_prefetch(p, q, 8).jaccard_sum[0], false);
    run("PF  u32 single + prefetch(32)", &|p, q| kernel::acc_prefetch(p, q, 32).jaccard_sum[0], false);
    run("U   u16 single", &|p, q| kernel::acc_u16(p, q).jaccard_sum[0], false);
    run("UP2 u16 x2 partial", &|p, q| kernel::acc_u16_partial2(p, q).jaccard_sum[0], false);
    run("CNT count only, NO fold", &|p, q| kernel::count_only(p, q) as f64, false);
    run("F32 fold via f32 divide", &|p, q| kernel::fold_f32(p, q).jaccard_sum[0], false);
}
