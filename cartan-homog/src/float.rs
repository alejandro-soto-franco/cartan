//! Float transcendentals routed through libm under no_std, matching the pattern
//! in `cartan-core` (see `cartan-core/src/rotor.rs`). Under std they use the
//! inherent `f64` methods.
//!
//! `abs` and `max` are not shimmed: both are available in `core` and compile
//! bare-metal unchanged.

#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    { x.sqrt() }
    #[cfg(not(feature = "std"))]
    { libm::sqrt(x) }
}

#[inline]
pub(crate) fn powi(x: f64, n: i32) -> f64 {
    #[cfg(feature = "std")]
    { x.powi(n) }
    #[cfg(not(feature = "std"))]
    { libm::pow(x, n as f64) }
}

#[inline]
pub(crate) fn powf(x: f64, n: f64) -> f64 {
    #[cfg(feature = "std")]
    { x.powf(n) }
    #[cfg(not(feature = "std"))]
    { libm::pow(x, n) }
}

#[inline]
pub(crate) fn ln(x: f64) -> f64 {
    #[cfg(feature = "std")]
    { x.ln() }
    #[cfg(not(feature = "std"))]
    { libm::log(x) }
}

#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    #[cfg(feature = "std")]
    { x.sin() }
    #[cfg(not(feature = "std"))]
    { libm::sin(x) }
}

#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    #[cfg(feature = "std")]
    { x.cos() }
    #[cfg(not(feature = "std"))]
    { libm::cos(x) }
}

#[inline]
pub(crate) fn acos(x: f64) -> f64 {
    #[cfg(feature = "std")]
    { x.acos() }
    #[cfg(not(feature = "std"))]
    { libm::acos(x) }
}

#[inline]
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    #[cfg(feature = "std")]
    { y.atan2(x) }
    #[cfg(not(feature = "std"))]
    { libm::atan2(y, x) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Guards the no_std branch against a wrong libm mapping: each shim must
    // agree with the std inherent method it replaces.
    #[test]
    fn shims_match_inherent_methods() {
        assert!((sqrt(2.0) - 2.0_f64.sqrt()).abs() < 1e-15);
        assert!((powi(1.5, 3) - 1.5_f64.powi(3)).abs() < 1e-15);
        assert!((powf(1.5, 0.5) - 1.5_f64.powf(0.5)).abs() < 1e-15);
        assert!((ln(3.0) - 3.0_f64.ln()).abs() < 1e-15);
        assert!((sin(0.7) - 0.7_f64.sin()).abs() < 1e-15);
        assert!((cos(0.7) - 0.7_f64.cos()).abs() < 1e-15);
        assert!((acos(0.3) - 0.3_f64.acos()).abs() < 1e-15);
        assert!((atan2(0.4, 0.9) - 0.4_f64.atan2(0.9)).abs() < 1e-15);
    }
}
