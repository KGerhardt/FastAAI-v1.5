//! Jaccard -> AAI, matching fastaai.py:2302 exactly.
//!
//!   aai_hat = (-0.3087057 + 1.810741 * exp(-(-0.2607023 * ln(kaai))^(1/3.435))) * 100
//!
//! v1 censors the result to the strings "<30%" and ">90%". That censoring is a
//! display decision and is deliberately not applied here — the raw value is kept
//! so precision survives at the low-Jaccard end.

pub fn kaai_to_aai(kaai: f64) -> f64 {
    if !(kaai > 0.0) {
        return f64::NAN;
    }
    let l = kaai.ln();
    let x = (-0.2607023 * l).powf(1.0 / 3.435);
    (1.810741 * (-x).exp() - 0.3087057) * 100.0
}

/// Jaccard corresponding to a given AAI, by numeric inversion. Used to state the
/// usable dynamic range: AAI 30% ~ J 0.0057, 65% ~ 0.445, 90% ~ 0.843.
pub fn aai_to_kaai(aai: f64) -> f64 {
    let t = (aai / 100.0 + 0.3087057) / 1.810741;
    if t <= 0.0 {
        return f64::NAN;
    }
    let x = -t.ln();
    (-(x.powf(3.435)) / 0.2607023).exp()
}
