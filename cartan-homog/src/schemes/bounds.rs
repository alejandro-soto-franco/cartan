//! Voigt (arithmetic mean) and Reuss (harmonic mean) bounds.

use crate::{error::HomogError, rve::Rve,
            schemes::{Effective, Scheme, SchemeOpts},
            tensor::TensorOrder};

#[derive(Clone, Debug, Default)]
pub struct VoigtBound;

impl<O: TensorOrder> Scheme<O> for VoigtBound {
    fn homogenize(&self, rve: &Rve<O>, _opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let mut acc = O::zero();
        for ph in &rve.phases {
            acc = O::add(&acc, &O::scale(&ph.property, ph.fraction));
        }
        Ok(Effective { tensor: acc, concentration: None, iterations: None, residual: None })
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReussBound;

impl<O: TensorOrder> Scheme<O> for ReussBound {
    fn homogenize(&self, rve: &Rve<O>, _opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let mut acc = O::zero();
        for ph in &rve.phases {
            let inv = O::inverse(&ph.property)?;
            acc = O::add(&acc, &O::scale(&inv, ph.fraction));
        }
        let eff = O::inverse(&acc)?;
        Ok(Effective { tensor: eff, concentration: None, iterations: None, residual: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rve::Phase, shapes::Sphere, tensor::Order2};
    use alloc::{string::String, sync::Arc};
    use approx::assert_relative_eq;

    fn two_phase(k0: f64, k1: f64, f1: f64) -> Rve<Order2> {
        let mut r = Rve::<Order2>::new();
        r.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(k0), fraction: 1.0 - f1 });
        r.add_phase(Phase { name: String::from("I"), shape: Arc::new(Sphere),
            property: Order2::scalar(k1), fraction: f1 });
        r.set_matrix("M");
        r
    }

    #[test]
    fn voigt_arithmetic_mean() {
        let rve = two_phase(1.0, 10.0, 0.3);
        let e = VoigtBound.homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert_relative_eq!(e.tensor[(0, 0)], 0.7 * 1.0 + 0.3 * 10.0, epsilon = 1e-12);
    }

    #[test]
    fn reuss_harmonic_mean() {
        let rve = two_phase(1.0, 10.0, 0.3);
        let e = ReussBound.homogenize(&rve, &SchemeOpts::default()).unwrap();
        let expected = 1.0 / (0.7 / 1.0 + 0.3 / 10.0);
        assert_relative_eq!(e.tensor[(0, 0)], expected, epsilon = 1e-12);
    }

    #[test]
    fn voigt_above_reuss() {
        let rve = two_phase(1.0, 10.0, 0.3);
        let v = VoigtBound.homogenize(&rve, &SchemeOpts::default()).unwrap();
        let r = ReussBound.homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert!(v.tensor[(0, 0)] > r.tensor[(0, 0)]);
    }
}
