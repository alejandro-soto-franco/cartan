//! Self-consistent scheme with SPD-geodesic fixed-point iteration.
//! Each phase embedded in the effective medium (no privileged matrix).

use crate::{error::HomogError, rve::Rve,
            schemes::{Effective, Scheme, SchemeOpts, VoigtBound},
            tensor::TensorOrder};
use alloc::vec::Vec;

#[derive(Clone, Debug, Default)]
pub struct SelfConsistent;

impl<O: TensorOrder> Scheme<O> for SelfConsistent {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        // Initial guess = Voigt bound (always SPD for SPD phases).
        let mut c_hom = VoigtBound.homogenize(rve, opts)?.tensor;
        let mut last_res = f64::INFINITY;

        for iter in 0..opts.max_iter {
            let mut a_dils: Vec<O::KmMatrix> = Vec::with_capacity(rve.phases.len());
            for ph in &rve.phases {
                a_dils.push(ph.shape.concentration_dilute(&c_hom, &ph.property, &opts.integration)?);
            }
            let mut sum_fa  = O::zero();
            let mut sum_fca = O::zero();
            for (ph, a) in rve.phases.iter().zip(a_dils.iter()) {
                sum_fa  = O::add(&sum_fa,  &O::scale(a, ph.fraction));
                let ca  = O::mat_mul(&ph.property, a);
                sum_fca = O::add(&sum_fca, &O::scale(&ca, ph.fraction));
            }
            let c_tilde = O::mat_mul(&sum_fca, &O::inverse(&sum_fa)?);

            let c_next = if opts.spd_iteration {
                O::spd_geodesic_step(&c_hom, &c_tilde, opts.damping)?
            } else {
                O::add(&O::scale(&c_hom, 1.0 - opts.damping),
                       &O::scale(&c_tilde, opts.damping))
            };

            let res = O::frobenius_norm(&O::sub(&c_next, &c_hom))
                      / O::frobenius_norm(&c_hom).max(1e-300);
            c_hom = c_next;

            if res < opts.rel_tol {
                let concs = opts.store_concentration.then_some(a_dils);
                return Ok(Effective { tensor: c_hom, concentration: concs,
                                       iterations: Some(iter + 1), residual: Some(res) });
            }
            last_res = res;
        }
        Err(HomogError::DidNotConverge { iters: opts.max_iter, residual: last_res })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rve::Phase, shapes::Sphere, tensor::Order2};
    use alloc::{string::String, sync::Arc};

    #[test]
    fn sc_converges_on_two_phase_iso() {
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("A"), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0), fraction: 0.5 });
        rve.add_phase(Phase { name: String::from("B"), shape: Arc::new(Sphere),
            property: Order2::scalar(3.0), fraction: 0.5 });
        rve.set_matrix("A");
        let e = SelfConsistent.homogenize(&rve, &SchemeOpts::default()).unwrap();
        // SC for 50/50 two-phase spheres iso: between 1 and 3, close to geometric mean ~1.7.
        assert!(e.tensor[(0, 0)] > 1.4 && e.tensor[(0, 0)] < 2.2,
                "expected 1.4..2.2, got {}", e.tensor[(0, 0)]);
    }
}
