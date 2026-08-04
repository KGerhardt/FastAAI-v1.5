//! Compare posting-list payload encodings: size and in-kernel decode throughput.
//! Every codec is verified against the raw-u16 baseline before being timed.

use std::time::Instant;

use fastaai_rs::codecs;
use fastaai_rs::data;
use fastaai_rs::index::Partition;
use fastaai_rs::kernel;
use fastaai_rs::kmer::{Alphabet, Kmerizer};
use fastaai_rs::synth;
use fastaai_rs::QueryResult;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or("data.bin".into());
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let nq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let ds = data::load(&path).expect("load");
    let alpha = Alphabet::new(&ds.alphabet);
    let mut km = Kmerizer::new(alpha.clone());
    let sets = synth::synthesize(&ds, &mut km, n);
    let names: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let part = Partition::build(&sets, ds.n_acc, alpha.kspace as usize, names);

    let queries: Vec<Vec<(usize, Vec<u32>)>> = (0..nq)
        .map(|i| {
            let gi = (i * n / nq).min(n - 1);
            sets[gi].iter().enumerate().filter(|(_, v)| !v.is_empty())
                .map(|(a, v)| (a, v.clone())).collect()
        })
        .collect();
    drop(sets);

    let entries = part.posting_entries();
    let incs: u64 = queries.iter().map(|q| kernel::increments(&part, q)).sum();
    println!("genomes {n}, entries {entries}, increments {incs}\n");

    let svb = codecs::build_svb(&part);
    let bp = codecs::build_bp(&part);
    let hyb = codecs::build_hybrid(&part);

    let raw_u16 = entries * 2;
    println!("{:<22} {:>10} {:>12} {:>10}", "encoding", "MB", "bytes/entry", "vs u16");
    let show = |name: &str, b: usize| {
        println!("{name:<22} {:>10.1} {:>12.2} {:>9.2}x",
                 b as f64 / 1e6, b as f64 / entries as f64, raw_u16 as f64 / b as f64);
    };
    show("raw u32", entries * 4);
    show("raw u16 (baseline)", raw_u16);
    show("delta+varint scalar", part.delta_bytes());
    show(svb.name, svb.bytes());
    show(bp.name, bp.bytes());
    show(hyb.name, hyb.bytes());

    // Correctness before speed.
    let check = |label: &str, got: &QueryResult, want: &QueryResult| {
        for t in 0..part.n_genomes {
            assert_eq!(got.shared[t], want.shared[t], "{label}: shared[{t}]");
            let d = (got.jaccard_sum[t] - want.jaccard_sum[t]).abs();
            assert!(d < 1e-9, "{label}: jaccard[{t}] off by {d}");
        }
    };
    for q in &queries {
        let want = kernel::untiled_u16(&part, q);
        check("SVB", &codecs::count_svb(&part, &svb, q), &want);
        check("BP", &codecs::count_bp(&part, &bp, q), &want);
        check("HYB", &codecs::count_hybrid(&part, &hyb, q), &want);
    }
    println!("\nall codecs verified against raw-u16 baseline: OK");

    let threads = [1usize, std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8)];
    type F<'a> = Box<dyn Fn(&Partition, &[(usize, Vec<u32>)]) -> f64 + Sync + 'a>;
    let variants: Vec<(String, F)> = vec![
        (
            "raw u16 (baseline)".into(),
            Box::new(|p: &Partition, q: &[(usize, Vec<u32>)]| {
                kernel::untiled_u16(p, q).jaccard_sum[0]
            }) as F,
        ),
        (
            svb.name.into(),
            Box::new(|p: &Partition, q: &[(usize, Vec<u32>)]| {
                codecs::count_svb(p, &svb, q).jaccard_sum[0]
            }),
        ),
        (
            bp.name.into(),
            Box::new(|p: &Partition, q: &[(usize, Vec<u32>)]| {
                codecs::count_bp(p, &bp, q).jaccard_sum[0]
            }),
        ),
        (
            hyb.name.into(),
            Box::new(|p: &Partition, q: &[(usize, Vec<u32>)]| {
                codecs::count_hybrid(p, &hyb, q).jaccard_sum[0]
            }),
        ),
    ];

    for &nt in &threads {
        println!("\n=== {nt} thread(s) ===");
        let work: Vec<&Vec<(usize, Vec<u32>)>> =
            (0..nt * queries.len()).map(|i| &queries[i % queries.len()]).collect();
        let total = incs * nt as u64;
        let mut base = 0.0;
        for (i, (label, f)) in variants.iter().enumerate() {
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                let chunk = (work.len() + nt - 1) / nt;
                for c in work.chunks(chunk) {
                    let part = &part;
                    sc.spawn(move || {
                        let mut a = 0.0;
                        for q in c { a += f(part, q); }
                        std::hint::black_box(a);
                    });
                }
            });
            let s = t0.elapsed().as_secs_f64();
            if i == 0 { base = s; }
            println!("{label:<22} {s:>7.3} s  {:>6.2} ns/inc  {:>8.1} M inc/s  {:>5.2}x",
                     s * 1e9 / total as f64, total as f64 / s / 1e6, base / s);
        }
    }
}
