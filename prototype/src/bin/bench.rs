//! Kernel benchmark on a synthetic database scaled from the real example genomes.
//!
//! Synthesis: genomes are drawn in clusters (a founder diverged from a real
//! parent, then members diverged slightly from the founder) so that posting-list
//! length distributions resemble a real database of related organisms rather than
//! uniform noise. Reported statistics let the reader judge how representative it is.
//!
//! usage: bench <data.bin> <n_genomes> [n_queries]

use std::time::Instant;

use fastaai_rs::data;
use fastaai_rs::index::Partition;
use fastaai_rs::kernel;
use fastaai_rs::kmer::{Alphabet, Kmerizer};
use fastaai_rs::synth;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or("data.bin".into());
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let nq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let ds = data::load(&path).expect("load");
    let alpha = Alphabet::new(&ds.alphabet);
    let kspace = alpha.kspace as usize;
    let mut km = Kmerizer::new(alpha.clone());

    let sets = synth::synthesize(&ds, &mut km, n);
    let names: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let t0 = Instant::now();
    let part = Partition::build(&sets, ds.n_acc, kspace, names);
    let build_s = t0.elapsed().as_secs_f64();

    let entries = part.posting_entries();
    println!("=== synthetic database ===");
    println!("genomes            : {n}");
    println!("build time         : {build_s:.2} s");
    println!("posting entries    : {entries} ({:.1} per genome)", entries as f64 / n as f64);
    println!("  as u32           : {:.2} GB", entries as f64 * 4.0 / 1e9);
    println!("  as u16           : {:.2} GB", entries as f64 * 2.0 / 1e9);
    println!("  delta+varint     : {:.2} GB ({:.2} bytes/entry)",
             part.delta_bytes() as f64 / 1e9,
             part.delta_bytes() as f64 / entries as f64);
    println!("offsets (dense)    : {:.1} MB", part.offset_bytes() as f64 / 1e6);
    println!("accumulator @u32   : {:.0} KB   @u16: {:.0} KB",
             n as f64 * 4.0 / 1e3, n as f64 * 2.0 / 1e3);

    // Queries: sample across clusters so similarity is not all-or-nothing.
    let queries: Vec<Vec<(usize, Vec<u32>)>> = (0..nq)
        .map(|i| {
            let gi = (i * n / nq).min(n - 1);
            sets[gi]
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(a, v)| (a, v.clone()))
                .collect()
        })
        .collect();

    let incs: u64 = queries.iter().map(|q| kernel::increments(&part, q)).sum();
    println!("\nqueries            : {nq}");
    println!("increments total   : {incs} ({:.2e} per query)", incs as f64 / nq as f64);

    drop(sets);


    // Each variant: (label, closure over one query).
    type KFn = Box<dyn Fn(&Partition, &[(usize, Vec<u32>)]) -> f64 + Sync>;
    let tile = 8192usize.min(n);
    let variants: Vec<(String, KFn)> = vec![
        ("A   untiled  u32   (sorted)".into(),
            Box::new(|p: &Partition, q: &[(usize, Vec<u32>)]| kernel::untiled_u32(p, q).jaccard_sum[0]) as KFn),
        ("A16 untiled  u16   (sorted)".into(),
            Box::new(|p: &Partition, q: &[(usize, Vec<u32>)]| kernel::untiled_u16(p, q).jaccard_sum[0])),
        ("SH  untiled  u16   (SHUFFLED)".into(),
            Box::new(|p: &Partition, q: &[(usize, Vec<u32>)]| kernel::untiled_shuffled(p, q).jaccard_sum[0])),
        (format!("B   tiled    u16   tile={tile}"),
            Box::new(move |p: &Partition, q: &[(usize, Vec<u32>)]| kernel::tiled_u16(p, q, tile).jaccard_sum[0])),
        (format!("C   tiled    delta tile={tile}"),
            Box::new(move |p: &Partition, q: &[(usize, Vec<u32>)]| kernel::tiled_delta(p, q, tile).jaccard_sum[0])),
    ];

    let threads: Vec<usize> = vec![1, 4, std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8)];

    for &nt in &threads {
        println!("\n=== kernels, {nt} thread(s) ===");
        // Replicate the query list so every thread has comparable work.
        let work: Vec<&Vec<(usize, Vec<u32>)>> =
            (0..nt.max(1) * queries.len()).map(|i| &queries[i % queries.len()]).collect();
        let total_inc = incs * nt as u64;
        let mut base = 0.0f64;

        for (i, (label, f)) in variants.iter().enumerate() {
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                let chunk = (work.len() + nt - 1) / nt;
                for c in work.chunks(chunk) {
                    let part = &part;
                    sc.spawn(move || {
                        let mut acc = 0.0f64;
                        for q in c {
                            acc += f(part, q);
                        }
                        std::hint::black_box(acc);
                    });
                }
            });
            let secs = t0.elapsed().as_secs_f64();
            if i == 0 {
                base = secs;
            }
            let gbs = total_inc as f64 * 4.0 / secs / 1e9;
            println!("{label:<34} {secs:>7.3} s  {:>6.2} ns/inc  {:>7.1} M inc/s  {:>5.2}x  [{gbs:.1} GB/s @u32-equiv]",
                     secs * 1e9 / total_inc as f64,
                     total_inc as f64 / secs / 1e6,
                     base / secs);
        }
    }
}
