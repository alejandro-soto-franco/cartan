//! Dilute (strain) and dilute-stress schemes.

use crate::{error::HomogError, rve::Rve,
            schemes::{Effective, Scheme, SchemeOpts},
            tensor::TensorOrder};
use alloc::vec::Vec;

#[derive(Clone, Debug, Default)]
pub struct Dilute;

impl<O: TensorOrder> Scheme<O> for Dilute {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let c_ref = rve.reference_property(None)?;
        let mut c_eff = c_ref.clone();
        let mut concs: Option<Vec<O::KmMatrix>> = opts.store_concentration.then(Vec::new);
        for ph in &rve.phases {
            if rve.is_matrix_phase(&ph.name) { continue; }
            let a = ph.shape.concentration_dilute(&c_ref, &ph.property, &opts.integration)?;
            let dc = O::sub(&ph.property, &c_ref);
            let dc_a = O::mat_mul(&dc, &a);
            c_eff = O::add(&c_eff, &O::scale(&dc_a, ph.fraction));
            if let Some(v) = concs.as_mut() { v.push(a); }
        }
        Ok(Effective { tensor: c_eff, concentration: concs, iterations: None, residual: None })
    }
}

#[derive(Clone, Debug, Default)]
pub struct DiluteStress;

impl<O: TensorOrder> Scheme<O> for DiluteStress {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let c_ref = rve.reference_property(None)?;
        let s_ref = O::inverse(&c_ref)?;
        let mut s_eff = s_ref.clone();
        for ph in &rve.phases {
            if rve.is_matrix_phase(&ph.name) { continue; }
            let s_phase = O::inverse(&ph.property)?;
            let a = ph.shape.concentration_dilute(&c_ref, &ph.property, &opts.integration)?;
            let b = O::mat_mul(&O::mat_mul(&ph.property, &a), &s_ref);
            let ds = O::sub(&s_phase, &s_ref);
            let ds_b = O::mat_mul(&ds, &b);
            s_eff = O::add(&s_eff, &O::scale(&ds_b, ph.fraction));
        }
        Ok(Effective { tensor: O::inverse(&s_eff)?, concentration: None, iterations: None, residual: None })
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
    fn dilute_equals_matrix_at_zero_fraction() {
        let rve = two_phase(1.0, 10.0, 0.0);
        let e = Dilute.homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert_relative_eq!(e.tensor[(0, 0)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn dilute_between_bounds() {
        let rve = two_phase(1.0, 5.0, 0.05);
        let e = Dilute.homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert!(e.tensor[(0, 0)] > 1.0);
    }
}
