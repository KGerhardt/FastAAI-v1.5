//! Cost of materializing the match stream instead of consuming it in place.
use std::time::Instant;
use fastaai_rs::{data, kernel, synth};
use fastaai_rs::index::Partition;
use fastaai_rs::kmer::{Alphabet, Kmerizer};

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
    drop(sets);
    let incs: u64 = queries.iter().map(|q| kernel::increments(&part, q)).sum();
    println!("{n} genomes, {incs} increments (= tuples the relational form emits)\n");

    for q in &queries {
        let w = kernel::untiled_u16(&part, q);
        for (l, g) in [("M1", kernel::materialized_ids(&part, q)),
                       ("M2", kernel::materialized_tuples(&part, q))] {
            for t in 0..part.n_genomes {
                assert!((g.jaccard_sum[t] - w.jaccard_sum[t]).abs() < 1e-9, "{l}");
            }
        }
    }
    println!("materialized variants verified identical to direct: OK\n");

    let run = |label: &str, f: &dyn Fn(&Partition, &[(usize, Vec<u32>)]) -> f64, base: f64| -> f64 {
        let t0 = Instant::now();
        let mut acc = 0.0;
        for q in &queries { acc += f(&part, q); }
        std::hint::black_box(acc);
        let s = t0.elapsed().as_secs_f64();
        let rel = if base > 0.0 { format!("{:.2}x", base / s) } else { "1.00x".into() };
        println!("{label:<40} {s:>7.3} s  {:>6.2} ns/inc  {rel:>7}",
                 s * 1e9 / incs as f64);
        s
    };
    let b = run("direct  cnt[g]+=1 (production)", &|p, q| kernel::untiled_u16(p, q).jaccard_sum[0], 0.0);
    run("M1  materialize target IDs (u16)", &|p, q| kernel::materialized_ids(p, q).jaccard_sum[0], b);
    run("M2  materialize (G2,Pn,Kz) tuples", &|p, q| kernel::materialized_tuples(p, q).jaccard_sum[0], b);

    println!("\nbytes written to memory per increment:");
    println!("  direct : 0   (target ID consumed in a register)");
    println!("  M1     : 2   (u16 target ID)");
    println!("  M2     : 8   (u16 target + u16 SCP + u32 kmer, padded)");
    println!("\nfor this run M2 would materialize {:.1} GB", incs as f64 * 8.0 / 1e9);
}
