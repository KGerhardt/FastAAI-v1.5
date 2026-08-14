//! Direct comparison of two genomes, without an index.
//!
//! Plain set arithmetic: intersect the accessions both genomes carry, take
//! Jaccard per accession, average, transform to AAI. This is what the inverted
//! index computes the fast way — the same sum, differently associated.
//!
//! It exists for two reasons, and the second is the important one:
//!
//! 1. A single comparison is a reasonable thing to ask for, and building an
//!    index for two genomes is silly.
//! 2. **It is the oracle the index path is validated against.** Sharing no
//!    machinery with `kernel` or `index` means agreement between them is
//!    evidence rather than tautology — this is what established the 2-ULP match
//!    against FastAAI 1. A public function cannot quietly rot the way a test
//!    helper can.
//!
//! Cost is O(|Q| + |T|) per accession, so it is only sensible for a handful of
//! comparisons. Anything larger should go through the engine, where a query is
//! amortised across every target at once.

/// Sorted, deduplicated k-mer IDs per accession; empty where absent.
pub type ScpSets = [Vec<u32>];

/// Jaccard summed over shared accessions, and the count of them.
///
/// Kept unreduced for the same reason the kernel does: the caller decides how to
/// treat "nothing shared", which is not the same as "similarity zero".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub jaccard_sum: f64,
    pub shared: u32,
}

impl Comparison {
    /// Mean Jaccard over shared accessions; NaN when nothing is shared.
    pub fn mean_jaccard(&self) -> f64 {
        if self.shared == 0 {
            f64::NAN
        } else {
            self.jaccard_sum / self.shared as f64
        }
    }

    /// AAI percentage, uncensored. NaN when nothing is shared.
    pub fn aai(&self) -> f64 {
        crate::aai::kaai_to_aai(self.mean_jaccard())
    }
}

/// Intersection size of two sorted, deduplicated slices, by merge.
fn intersection_len(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut n) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// Compare two genomes given their per-accession k-mer sets.
///
/// Both slices must be indexed by the same accession ordering — the caller is
/// responsible for that, exactly as a shared database schema guarantees it for
/// the index path.
pub fn compare(query: &ScpSets, target: &ScpSets) -> Comparison {
    let n = query.len().min(target.len());
    let mut jaccard_sum = 0.0f64;
    let mut shared = 0u32;

    for a in 0..n {
        let (q, t) = (&query[a], &target[a]);
        if q.is_empty() || t.is_empty() {
            continue; // an accession only one genome carries is not shared
        }
        let inter = intersection_len(q, t);
        let union = q.len() + t.len() - inter;
        if union > 0 {
            jaccard_sum += inter as f64 / union as f64;
        }
        shared += 1;
    }

    Comparison { jaccard_sum, shared }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_genomes_score_one() {
        let g = vec![vec![1u32, 2, 3], vec![10, 11]];
        let c = compare(&g, &g);
        assert_eq!(c.shared, 2);
        assert!((c.mean_jaccard() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn disjoint_kmers_score_zero_but_still_share_the_accession() {
        let a = vec![vec![1u32, 2, 3]];
        let b = vec![vec![7u32, 8, 9]];
        let c = compare(&a, &b);
        assert_eq!(c.shared, 1, "the accession is present in both");
        assert_eq!(c.mean_jaccard(), 0.0);
    }

    #[test]
    fn no_shared_accession_is_nan_not_zero() {
        let a = vec![vec![1u32, 2], vec![]];
        let b = vec![vec![], vec![5u32, 6]];
        let c = compare(&a, &b);
        assert_eq!(c.shared, 0);
        assert!(c.mean_jaccard().is_nan(), "must be distinguishable from AAI 0");
        assert!(c.aai().is_nan());
    }

    #[test]
    fn half_overlap() {
        // |∩| = 2, |∪| = 6  ->  1/3
        let a = vec![vec![1u32, 2, 3, 4]];
        let b = vec![vec![3u32, 4, 5, 6]];
        assert!((compare(&a, &b).mean_jaccard() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn averages_over_shared_accessions_only() {
        // acc0 matches fully, acc1 not at all, acc2 only in the query.
        let a = vec![vec![1u32, 2], vec![3, 4], vec![9]];
        let b = vec![vec![1u32, 2], vec![7, 8], vec![]];
        let c = compare(&a, &b);
        assert_eq!(c.shared, 2);
        assert!((c.mean_jaccard() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn is_symmetric() {
        let a = vec![vec![1u32, 2, 3, 4], vec![10, 11]];
        let b = vec![vec![3u32, 4, 5], vec![10, 12, 13]];
        assert_eq!(compare(&a, &b), compare(&b, &a));
    }

    #[test]
    fn intersection_merge_is_correct() {
        assert_eq!(intersection_len(&[], &[1, 2]), 0);
        assert_eq!(intersection_len(&[1, 2, 3], &[]), 0);
        assert_eq!(intersection_len(&[1, 3, 5], &[2, 4, 6]), 0);
        assert_eq!(intersection_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(intersection_len(&[1, 5, 9], &[5, 9, 11]), 2);
    }

    #[test]
    fn ragged_inputs_use_the_shorter_accession_list() {
        let a = vec![vec![1u32, 2]];
        let b = vec![vec![1u32, 2], vec![3, 4]];
        assert_eq!(compare(&a, &b).shared, 1);
    }

    /// The property that makes this useful as an oracle: it must agree with the
    /// indexed path. `kernel` tests the converse direction on larger fixtures.
    #[test]
    fn agrees_with_the_indexed_kernel() {
        use crate::index::Partition;
        use crate::kernel::join_into;

        let sets = vec![
            vec![vec![1u32, 2, 3, 4], vec![10, 11]],
            vec![vec![3u32, 4, 5, 6], vec![10, 12]],
            vec![vec![1u32, 2, 3, 4], vec![]],
            vec![vec![], vec![]],
        ];
        let n = sets.len();
        let p = Partition::build(&sets, 2, 16).unwrap();
        let mut j = vec![0.0; n * n];
        let mut s = vec![0u32; n * n];
        join_into(&p, &p, 2, &mut j, &mut s);

        for q in 0..n {
            for t in 0..n {
                let c = compare(&sets[q], &sets[t]);
                assert_eq!(c.shared, s[q * n + t], "shared ({q},{t})");
                assert!((c.jaccard_sum - j[q * n + t]).abs() < 1e-12, "sum ({q},{t})");
            }
        }
    }
}
