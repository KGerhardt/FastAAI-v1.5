//! Does batching query genomes (partial search inversion) pay?
use std::time::Instant;
use fastaai_rs::{data, kernel, synth};
use fastaai_rs::index::Partition;
use fastaai_rs::kmer::{Alphabet, Kmerizer};

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().unwrap_or("data.bin".into());
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let nq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);

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
    drop(sets);
    let incs: u64 = queries.iter().map(|q| kernel::increments(&part, q)).sum();
    println!("{n} genomes, {nq} queries, {incs} increments\n");

    // correctness
    let refs: Vec<&[(usize, Vec<u32>)]> = queries.iter().map(|q| q.as_slice()).collect();
    let (mut m, mut t) = (Vec::new(), Vec::new());
    let got = kernel::batched(&part, &refs[..8], &mut m, &mut t);
    for i in 0..8 {
        let want = kernel::untiled_u16(&part, &queries[i]);
        for x in 0..part.n_genomes {
            assert_eq!(got[i].shared[x], want.shared[x], "shared q{i} t{x}");
            assert!((got[i].jaccard_sum[x] - want.jaccard_sum[x]).abs() < 1e-9,
                    "jaccard q{i} t{x}");
        }
    }
    println!("batched kernel verified against per-query baseline: OK\n");

    let t0 = Instant::now();
    let mut acc = 0.0;
    for q in &queries { acc += kernel::untiled_u16(&part, q).jaccard_sum[0]; }
    std::hint::black_box(acc);
    let base = t0.elapsed().as_secs_f64();
    println!("{:<28} {base:>7.3} s  {:>6.2} ns/inc  {:>6} 1.00x  acc {:>7.0} KB",
             "per-query (production)", base * 1e9 / incs as f64, "",
             part.n_genomes as f64 * 4.0 / 1e3);

    for b in [2usize, 4, 8, 16, 32, 64] {
        if b > nq { continue; }
        let (mut m, mut t) = (Vec::new(), Vec::new());
        let t0 = Instant::now();
        let mut acc = 0.0;
        for chunk in refs.chunks(b) {
            for r in kernel::batched(&part, chunk, &mut m, &mut t) { acc += r.jaccard_sum[0]; }
        }
        std::hint::black_box(acc);
        let s = t0.elapsed().as_secs_f64();
        println!("batch {b:<22} {s:>7.3} s  {:>6.2} ns/inc  {:>6.2}x       acc {:>7.0} KB",
                 s * 1e9 / incs as f64, base / s, (b * part.n_genomes) as f64 * 4.0 / 1e3);
    }

    // Under memory pressure: does the index-reuse saving finally pay?
    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(16);
    println!("\n=== {nt} threads (each thread its own query stream) ===");
    let total = incs * nt as u64;

    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            let (part, queries) = (&part, &queries);
            sc.spawn(move || {
                let mut a = 0.0;
                for q in queries.iter() { a += kernel::untiled_u16(part, q).jaccard_sum[0]; }
                std::hint::black_box(a);
            });
        }
    });
    let mb = t0.elapsed().as_secs_f64();
    println!("{:<28} {mb:>7.3} s  {:>6.2} ns/inc   1.00x",
             "per-query", mb * 1e9 / total as f64);

    for b in [4usize, 8, 16] {
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for _ in 0..nt {
                let (part, refs) = (&part, &refs);
                sc.spawn(move || {
                    let (mut m, mut t) = (Vec::new(), Vec::new());
                    let mut a = 0.0;
                    for chunk in refs.chunks(b) {
                        for r in kernel::batched(part, chunk, &mut m, &mut t) { a += r.jaccard_sum[0]; }
                    }
                    std::hint::black_box(a);
                });
            }
        });
        let s = t0.elapsed().as_secs_f64();
        println!("batch {b:<22} {s:>7.3} s  {:>6.2} ns/inc  {:>6.2}x",
                 s * 1e9 / total as f64, mb / s);
    }
}
