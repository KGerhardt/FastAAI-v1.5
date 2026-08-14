//! FastAAI 1.5 engine prototype — dense kmer encoding, partition-local inverted
//! index, and three counting kernels for comparison.
//!
//! Targets FastAAI v1's fixed 122-accession SCP set so results can be checked
//! against the reference implementation.

pub mod kmer;
pub mod data;
pub mod index;
pub mod kernel;
pub mod aai;
pub mod synth;
pub mod codecs;

/// Jaccard sums and shared-SCP counts for one query against every target.
/// Kept separate so callers can reproduce v1's `sum / shared_count` exactly.
#[derive(Clone, Debug)]
pub struct QueryResult {
    pub jaccard_sum: Vec<f64>,
    pub shared: Vec<u32>,
}

impl QueryResult {
    pub fn new(n: usize) -> Self {
        QueryResult { jaccard_sum: vec![0.0; n], shared: vec![0; n] }
    }

    /// Mean Jaccard over shared SCPs; NaN where nothing is shared (v1's "N/A").
    pub fn means(&self) -> Vec<f64> {
        self.jaccard_sum
            .iter()
            .zip(&self.shared)
            .map(|(s, &c)| if c == 0 { f64::NAN } else { s / c as f64 })
            .collect()
    }
}
