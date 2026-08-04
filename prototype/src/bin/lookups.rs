//! Is the inverted-vs-inverted (kmer-join) formulation worth it?
//!
//! Total increments are IDENTICAL in both formulations — both compute
//!     sum over (accession, kmer) of |Q[k]| * |T[k]|
//! just reassociated. So the only things that differ are (a) how many offset
//! lookups are needed and (b) how big the accumulator is. This measures both.

use fastaai_rs::{data, kernel, synth};
use fastaai_rs::index::Partition;
use fastaai_rs::kmer::{Alphabet, Kmerizer};

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().unwrap_or("data.bin".into());
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16384);

    let ds = data::load(&path).expect("load");
    let alpha = Alphabet::new(&ds.alphabet);
    let ks = alpha.kspace as usize;
    let mut km = Kmerizer::new(alpha.clone());
    let sets = synth::synthesize(&ds, &mut km, n);
    let names: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let part = Partition::build_lean(&sets, ds.n_acc, ks, names);

    // Current formulation, per query genome.
    let nq = 8usize;
    let mut lookups = 0u64;
    let mut incs = 0u64;
    for i in 0..nq {
        let gi = (i * n / nq).min(n - 1);
        let q: Vec<(usize, Vec<u32>)> = sets[gi].iter().enumerate()
            .filter(|(_, v)| !v.is_empty()).map(|(x, v)| (x, v.clone())).collect();
        lookups += q.iter().map(|(_, kk)| kk.len() as u64).sum::<u64>();
        incs += kernel::increments(&part, &q);
    }

    // Whole partition-pair, both formulations.
    let nonempty: u64 = part.accs.iter().map(|ai|
        (0..ks).filter(|&k| ai.offsets[k + 1] > ai.offsets[k]).count() as u64).sum();
    let cur_lookups = lookups / nq as u64 * n as u64;
    let total_inc = incs / nq as u64 * n as u64;

    println!("partition {n} x {n}\n");
    println!("per query genome:");
    println!("  offset lookups      : {:>16}", lookups / nq as u64);
    println!("  increments          : {:>16}", incs / nq as u64);
    println!("  increments / lookup : {:>16.0}   <- access-weighted mean list length",
             incs as f64 / lookups as f64);
    println!("\nwhole partition-pair ({n} queries x {n} targets):");
    println!("  increments (BOTH formulations, identical) : {total_inc:>16.3e}", total_inc = total_inc as f64);
    println!("  lookups, current  (per query, per kmer)   : {cur_lookups:>16.3e}", cur_lookups = cur_lookups as f64);
    println!("  lookups, inverted (per non-empty kmer)    : {:>16.3e}", nonempty as f64);
    println!("  lookup reduction                          : {:>15.1}x",
             cur_lookups as f64 / nonempty as f64);
    println!("\n  lookups as share of all operations, current  : {:>8.4}%",
             cur_lookups as f64 / (cur_lookups + total_inc) as f64 * 100.0);
    println!("  lookups as share of all operations, inverted : {:>8.4}%",
             nonempty as f64 / (nonempty + total_inc) as f64 * 100.0);

    println!("\naccumulator footprint:");
    println!("  current  1-D  n x u32           : {:>10.1} KB", n as f64 * 4.0 / 1e3);
    println!("  inverted 2-D  n x n x u32       : {:>10.1} GB", (n as f64).powi(2) * 4.0 / 1e9);
    for b in [128usize, 256, 512, 1024, 4096] {
        println!("    2-D blocked {b:>5} x {b:<5}      : {:>10.1} KB{}",
                 (b as f64).powi(2) * 4.0 / 1e3,
                 if (b*b*4) <= 32768 { "  (fits L1)" }
                 else if (b*b*4) <= 1_250_000 { "  (fits L2)" }
                 else if (b*b*4) <= 24_000_000 { "  (fits L3)" } else { "" });
    }
}
