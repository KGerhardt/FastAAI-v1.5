//! Formatting results for output.
//!
//! Lives in Rust because a worker writes its own `q x t` block directly: a full
//! self-block is 16,384^2 = 268M rows, which is not a thing to format one row at
//! a time from Python.
//!
//! The formatting must match `fastaai.cli` digit for digit, because the two
//! paths write the same numbers and a user comparing a block against a
//! single-file run must not see them disagree. `tests/test_streaming.py`
//! compares the two outputs and is what holds them together.

use std::fmt::Write as _;

/// The band the Jaccard->AAI regression has sensitivity across. Outside it the
/// estimate cannot support a specific figure, so the output is categorical.
pub const AAI_FLOOR: f64 = 30.0;
pub const AAI_CEILING: f64 = 90.0;
pub const LABEL_BELOW: &str = "<30%";
pub const LABEL_ABOVE: &str = ">90%";

/// One AAI estimate, categorical where the regression cannot resolve.
///
/// Precedence matters: no measurement outranks the floor, and the floor
/// outranks the ceiling so that zero Jaccard — which `log(0)` places *above*
/// the ceiling — reports as `<30%` rather than as its inverse.
pub fn aai_label(out: &mut String, aai: f64, shared: u32, jac: f64) {
    if shared == 0 || jac.is_nan() {
        out.push_str("NA");
    } else if jac == 0.0 || aai.is_nan() || aai < AAI_FLOOR {
        out.push_str(LABEL_BELOW);
    } else if aai > AAI_CEILING {
        out.push_str(LABEL_ABOVE);
    } else {
        let _ = write!(out, "{aai:.2}");
    }
}

/// Python's `%.{sig}g`, which is what the single-file writer uses.
///
/// Reimplemented rather than approximated with `{:.n}`: `%g` switches to
/// exponent form outside a range and strips trailing zeros, and Jaccard at the
/// distances this tool exists to serve is small enough to reach that switch.
pub fn fmt_g(out: &mut String, v: f64, sig: usize) {
    if v.is_nan() {
        out.push_str("nan");
        return;
    }
    if v == 0.0 {
        out.push('0');
        return;
    }
    if v.is_infinite() {
        out.push_str(if v > 0.0 { "inf" } else { "-inf" });
        return;
    }

    let exp = v.abs().log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        let s = format!("{:.*e}", sig.saturating_sub(1), v);
        let (mantissa, e) = s.split_once('e').expect("rust exponent form");
        out.push_str(trim_zeros(mantissa));
        let ev: i32 = e.parse().expect("exponent digits");
        let _ = write!(out, "e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs());
    } else {
        let dec = (sig as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", dec, v);
        out.push_str(trim_zeros(&s));
    }
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(v: f64, sig: usize) -> String {
        let mut s = String::new();
        fmt_g(&mut s, v, sig);
        s
    }

    fn lab(aai: f64, shared: u32, jac: f64) -> String {
        let mut s = String::new();
        aai_label(&mut s, aai, shared, jac);
        s
    }

    /// Expected values are what Python's f"{v:.10g}" produces.
    #[test]
    fn matches_python_percent_g() {
        assert_eq!(g(1.0, 10), "1");
        assert_eq!(g(0.875, 10), "0.875");
        assert_eq!(g(0.8035714286, 10), "0.8035714286");
        assert_eq!(g(0.9333333333333333, 10), "0.9333333333");
        assert_eq!(g(0.0, 10), "0");
        assert_eq!(g(0.5, 6), "0.5");
    }

    /// Small Jaccard is the normal case at the distances this tool serves, so
    /// the exponent switch is on the hot path, not an edge case.
    #[test]
    fn small_values_take_exponent_form_as_python_does() {
        assert_eq!(g(1.2345e-5, 10), "1.2345e-05");
        assert_eq!(g(9.0e-7, 10), "9e-07");
        assert_eq!(g(1e-4, 10), "0.0001"); // exp == -4 stays positional
    }

    #[test]
    fn zero_jaccard_is_below_the_floor_not_above_the_ceiling() {
        assert_eq!(lab(f64::NAN, 50, 0.0), "<30%");
    }

    #[test]
    fn no_shared_accessions_is_not_a_low_score() {
        assert_eq!(lab(f64::NAN, 0, f64::NAN), "NA");
    }

    #[test]
    fn band_edges_stay_numeric() {
        assert_eq!(lab(30.0, 50, 0.006), "30.00");
        assert_eq!(lab(90.0, 50, 0.843), "90.00");
        assert_eq!(lab(29.99, 50, 0.006), "<30%");
        assert_eq!(lab(90.01, 50, 0.9), ">90%");
    }
}
