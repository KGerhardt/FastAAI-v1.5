//! Correctness checks against FastAAI v1 on the 10 real example genomes.
//!
//!   1. dense encoding is a bijection with v1's decimal-ASCII encoding
//!   2. all three kernels agree bit-for-bit
//!   3. emits rust_results.tsv for comparison against the numpy reference

use std::collections::HashMap;
use std::io::Write;

use fastaai_rs::aai::kaai_to_aai;
use fastaai_rs::data;
use fastaai_rs::index::Partition;
use fastaai_rs::kernel;
use fastaai_rs::kmer::{v1_code, Alphabet, Kmerizer, K_LEN};

fn main() {
    let path = std::env::args().nth(1).unwrap_or("data.bin".into());
    let ds = data::load(&path).expect("load data.bin");

    let alpha = Alphabet::new(&ds.alphabet);
    println!(
        "alphabet: {} symbols ({}), kspace = {}^{} = {}",
        alpha.base,
        String::from_utf8_lossy(&alpha.symbols),
        alpha.base,
        K_LEN,
        alpha.kspace
    );
    println!("{} genomes, {} accessions\n", ds.genomes.len(), ds.n_acc);

    // ---- 1. encoding bijection -------------------------------------------------
    //
    // Both encodings are monotone in lexicographic tetramer order (v1's positional
    // weights 1e6/1e4/1e2/1 dominate because ord <= 89), so the two sorted arrays
    // must correspond element-wise. That makes this an exact check, not a sampled one.
    let mut km = Kmerizer::new(alpha.clone());
    let mut v1_to_dense: HashMap<i32, u32> = HashMap::new();
    let mut dense_to_v1: HashMap<u32, i32> = HashMap::new();
    let (mut n_scp, mut n_kmer) = (0usize, 0usize);

    for g in &ds.genomes {
        for s in &g.scps {
            let dense = km.kmers(&s.seq);
            assert_eq!(
                dense.len(),
                s.v1_codes.len(),
                "{} acc{}: cardinality {} != v1 {}",
                g.name,
                s.acc,
                dense.len(),
                s.v1_codes.len()
            );
            for (&d, &v) in dense.iter().zip(&s.v1_codes) {
                if let Some(&prev) = v1_to_dense.get(&v) {
                    assert_eq!(prev, d, "v1 code {v} maps to both {prev} and {d}");
                }
                if let Some(&prev) = dense_to_v1.get(&d) {
                    assert_eq!(prev, v, "dense code {d} maps to both {prev} and {v}");
                }
                v1_to_dense.insert(v, d);
                dense_to_v1.insert(d, v);
            }
            n_scp += 1;
            n_kmer += dense.len();
        }
    }
    println!("[1] encoding bijection: OK");
    println!("    {n_scp} SCPs, {n_kmer} kmers, {} distinct tetramers", v1_to_dense.len());

    // Independent spot-check: re-derive codes straight from the residues and
    // confirm every sliding window of the first SCP is accounted for in both
    // encodings.
    let s0 = &ds.genomes[0].scps[0];
    let dense0 = km.kmers(&s0.seq);
    for w in s0.seq.windows(K_LEN) {
        let v = v1_code(w);
        assert!(s0.v1_codes.binary_search(&v).is_ok(), "v1 code {v} missing");
        let d = v1_to_dense[&v];
        assert!(dense0.binary_search(&d).is_ok(), "dense code {d} missing");
    }
    let w = &s0.seq[0..K_LEN];
    println!(
        "    spot-check: all {} windows of SCP 0 present in both encodings",
        s0.seq.len() - K_LEN + 1
    );
    println!(
        "    e.g. {:?} -> v1 {} -> dense {}",
        String::from_utf8_lossy(w),
        v1_code(w),
        v1_to_dense[&v1_code(w)]
    );

    // ---- build the partition ---------------------------------------------------
    let mut sets: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut names = Vec::new();
    for g in &ds.genomes {
        let mut per_acc = vec![Vec::new(); ds.n_acc];
        for s in &g.scps {
            per_acc[s.acc as usize] = km.kmers(&s.seq);
        }
        sets.push(per_acc);
        names.push(g.name.clone());
    }
    let part = Partition::build(&sets, ds.n_acc, alpha.kspace as usize, names);
    println!(
        "\npartition: {} genomes, {} posting entries, offsets {:.1} MB, u16 {:.1} MB, delta {:.1} MB",
        part.n_genomes,
        part.posting_entries(),
        part.offset_bytes() as f64 / 1e6,
        part.posting_entries() as f64 * 2.0 / 1e6,
        part.delta_bytes() as f64 / 1e6
    );

    // ---- 2. kernels agree ------------------------------------------------------
    let queries: Vec<Vec<(usize, Vec<u32>)>> = sets
        .iter()
        .map(|per_acc| {
            per_acc
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(a, v)| (a, v.clone()))
                .collect()
        })
        .collect();

    for (qi, q) in queries.iter().enumerate() {
        let a = kernel::untiled_u32(&part, q);
        let b = kernel::tiled_u16(&part, q, 8192);
        let c = kernel::tiled_delta(&part, q, 8192);
        for t in 0..part.n_genomes {
            assert_eq!(a.shared[t], b.shared[t], "shared mismatch A/B q{qi} t{t}");
            assert_eq!(a.shared[t], c.shared[t], "shared mismatch A/C q{qi} t{t}");
            let (x, y, z) = (a.jaccard_sum[t], b.jaccard_sum[t], c.jaccard_sum[t]);
            assert!((x - y).abs() < 1e-12, "sum A/B q{qi} t{t}: {x} vs {y}");
            assert!((x - z).abs() < 1e-12, "sum A/C q{qi} t{t}: {x} vs {z}");
        }
    }
    println!("[2] all three kernels agree across {} queries: OK", queries.len());

    // ---- 3. emit results -------------------------------------------------------
    let mut out = std::fs::File::create("rust_results.tsv").unwrap();
    writeln!(out, "query\ttarget\tavg_jacc\tshared_scps\taai").unwrap();
    for (qi, q) in queries.iter().enumerate() {
        let r = kernel::tiled_u16(&part, q, 8192);
        let means = r.means();
        for t in 0..part.n_genomes {
            writeln!(
                out,
                "{}\t{}\t{:.17e}\t{}\t{:.17e}",
                part.names[qi], part.names[t], means[t], r.shared[t], kaai_to_aai(means[t])
            )
            .unwrap();
        }
    }
    println!("[3] wrote rust_results.tsv");

    let q0 = &queries[0];
    println!("\nincrements for query 0: {}", kernel::increments(&part, q0));
}
