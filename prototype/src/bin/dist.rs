//! Posting-list length distribution. Determines which encodings can help:
//! block codecs (bitpacking, stream-vbyte) need long runs to amortise, and a
//! bitmap container only beats a sorted u16 list once a list is dense.

use fastaai_rs::data;
use fastaai_rs::index::Partition;
use fastaai_rs::kmer::{Alphabet, Kmerizer};
use fastaai_rs::synth;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or("data.bin".into());
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16384);

    let ds = data::load(&path).expect("load");
    let alpha = Alphabet::new(&ds.alphabet);
    let kspace = alpha.kspace as usize;
    let mut km = Kmerizer::new(alpha.clone());
    let sets = synth::synthesize(&ds, &mut km, n);
    let names: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let part = Partition::build(&sets, ds.n_acc, kspace, names);

    // Bucket lists by length; report share of entries and of lists.
    let edges = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                 8192, 16384, 32768, usize::MAX];
    let mut list_ct = vec![0u64; edges.len()];
    let mut entry_ct = vec![0u64; edges.len()];
    let mut nonempty = 0u64;
    let mut total_entries = 0u64;

    for a in &part.accs {
        for k in 0..kspace {
            let len = (a.offsets[k + 1] - a.offsets[k]) as usize;
            if len == 0 {
                continue;
            }
            nonempty += 1;
            total_entries += len as u64;
            let b = edges.iter().position(|&e| len <= e).unwrap();
            list_ct[b] += 1;
            entry_ct[b] += len as u64;
        }
    }

    println!("partition: {n} genomes, {} accessions, kspace {kspace}", part.n_acc);
    println!("non-empty posting lists : {nonempty}");
    println!("total entries           : {total_entries}");
    println!("mean list length        : {:.1}", total_entries as f64 / nonempty as f64);
    println!("\n{:>10}  {:>12} {:>7}  {:>14} {:>7}  {:>8}",
             "len <=", "lists", "%", "entries", "%", "cum entry%");

    let mut cum = 0.0;
    for (i, &e) in edges.iter().enumerate() {
        if list_ct[i] == 0 {
            continue;
        }
        let ep = entry_ct[i] as f64 / total_entries as f64 * 100.0;
        cum += ep;
        let label = if e == usize::MAX { "inf".to_string() } else { e.to_string() };
        println!("{label:>10}  {:>12} {:>6.2}%  {:>14} {:>6.2}%  {:>7.2}%",
                 list_ct[i], list_ct[i] as f64 / nonempty as f64 * 100.0,
                 entry_ct[i], ep, cum);
    }

    // Bitmap crossover: a dense bitmap costs n/8 bytes; a sorted u16 list 2*len.
    let cross = n / 16;
    let (mut big_lists, mut big_entries) = (0u64, 0u64);
    for a in &part.accs {
        for k in 0..kspace {
            let len = (a.offsets[k + 1] - a.offsets[k]) as usize;
            if len > cross {
                big_lists += 1;
                big_entries += len as u64;
            }
        }
    }
    println!("\nbitmap crossover (len > n/16 = {cross}):");
    println!("  lists above   : {big_lists} ({:.2}% of non-empty)",
             big_lists as f64 / nonempty as f64 * 100.0);
    println!("  entries above : {big_entries} ({:.2}% of all entries)",
             big_entries as f64 / total_entries as f64 * 100.0);
    println!("  u16 bytes for those : {:.1} MB", big_entries as f64 * 2.0 / 1e6);
    println!("  bitmap bytes instead: {:.1} MB", big_lists as f64 * (n as f64 / 8.0) / 1e6);

    println!("\nper-list container overhead if one object per list:");
    println!("  {nonempty} objects x 24 B (Vec header) = {:.2} GB",
             nonempty as f64 * 24.0 / 1e9);
}
