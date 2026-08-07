//! Jaccard to AAI.
//!
//! Matches FastAAI 1 (`fastaai.py:2302`) exactly:
//!
//! ```text
//! aai = (-0.3087057 + 1.810741 * exp(-(-0.2607023 * ln(J))^(1/3.435))) * 100
//! ```
//!
//! v1 censored the result to the literal strings `"<30%"` and `">90%"`. That
//! censoring is **not** applied here. It is a display decision, and baking it into
//! stored values discards precision in exactly the regime FastAAI exists to serve:
//! the usable band runs from J ~ 0.006 (AAI 30%) to J ~ 0.843 (AAI 90%), so at
//! extreme distance the interesting values sit near the bottom of the range.
//!
//! Callers should store raw Jaccard and apply the transform at presentation.

const A: f64 = -0.3087057;
const B: f64 = 1.810741;
const C: f64 = -0.2607023;
const D: f64 = 3.435;

/// Regression estimate of AAI as a percentage. NaN for non-positive Jaccard.
///
/// Note the regression is unbounded above: identical genomes (J = 1) extrapolate
/// past 100%. That is a property of the fit, not an error — clamp at display time.
pub fn kaai_to_aai(kaai: f64) -> f64 {
    if !(kaai > 0.0) {
        return f64::NAN;
    }
    let x = (C * kaai.ln()).powf(1.0 / D);
    (B * (-x).exp() + A) * 100.0
}

/// Jaccard corresponding to a given AAI, by inverting the same fit. Useful for
/// stating the usable dynamic range.
pub fn aai_to_kaai(aai: f64) -> f64 {
    // aai/100 = B*exp(-x) + A       with x = (C * ln J)^(1/D)
    //   =>  x = -ln((aai/100 - A) / B)
    //   =>  ln J = x^D / C          (C is negative, so no further sign flip)
    let t = (aai / 100.0 - A) / B;
    if t <= 0.0 {
        return f64::NAN;
    }
    let x = -t.ln();
    if x <= 0.0 {
        return f64::NAN; // above the fit's ceiling
    }
    (x.powf(D) / C).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_the_inverse() {
        for aai in [35.0, 50.0, 65.0, 80.0, 90.0] {
            let j = aai_to_kaai(aai);
            assert!((kaai_to_aai(j) - aai).abs() < 1e-6, "aai {aai} -> J {j} -> back");
        }
    }

    #[test]
    fn reproduces_the_documented_band() {
        // The values quoted throughout the design notes.
        assert!((aai_to_kaai(30.0) - 0.0057).abs() < 5e-4);
        assert!((aai_to_kaai(65.0) - 0.4447).abs() < 5e-4);
        assert!((aai_to_kaai(90.0) - 0.8430).abs() < 5e-4);
    }

    /// Monotone far below anything reachable, not only across the usable band.
    /// A non-monotone patch would make AAI non-comparable in exactly the regime
    /// this tool exists to serve — the bottom of the range.
    #[test]
    fn is_monotone_increasing_over_the_whole_domain() {
        let mut prev = f64::NEG_INFINITY;
        // Log-spaced from 1e-300, then linear across the band.
        for i in 0..=600 {
            let j = 10f64.powf(-300.0 + 297.0 * (i as f64 / 600.0));
            let v = kaai_to_aai(j);
            assert!(v > prev, "not monotone at J = {j:e}");
            prev = v;
        }
        // Continue upward from where the log sweep stopped. Restarting below it
        // would compare against a larger J and fail on the sampling, not on the
        // function.
        for i in 1..=2000 {
            let j = 1e-3 + (1.0 - 1e-3) * (i as f64 / 2000.0);
            let v = kaai_to_aai(j);
            assert!(v > prev, "not monotone at J = {j}");
            prev = v;
        }
    }

    /// The fit's asymptote is A*100 = -30.87%, so AAI is negative for tiny
    /// Jaccard. Unreachable in practice: a per-SCP Jaccard is at least
    /// ~1/660 and the mean over ~122 accessions bottoms out near 1e-5, seven
    /// orders of magnitude above the crossing.
    #[test]
    fn the_negative_tail_exists_but_is_out_of_reach() {
        assert!(kaai_to_aai(1e-13) < 0.0);
        assert!(kaai_to_aai(1e-11) > 0.0);
        assert!(kaai_to_aai(1.0 / 660.0 / 122.0) > 0.0);
        // ...and it never dives below the asymptote.
        assert!(kaai_to_aai(f64::MIN_POSITIVE) > A * 100.0);
    }

    #[test]
    fn non_positive_jaccard_is_nan_not_a_number() {
        assert!(kaai_to_aai(0.0).is_nan());
        assert!(kaai_to_aai(-0.1).is_nan());
        assert!(kaai_to_aai(f64::NAN).is_nan());
    }

    #[test]
    fn identical_genomes_extrapolate_above_100() {
        // Documents the known behaviour rather than pretending it is clamped.
        assert!(kaai_to_aai(1.0) > 100.0, "regression is unbounded above");
    }
}
