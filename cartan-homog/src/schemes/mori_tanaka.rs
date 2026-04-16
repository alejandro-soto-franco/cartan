//! Mori-Tanaka scheme: each inclusion embedded in the matrix reference medium,
//! effective property from the strain-concentration sum constraint.

use crate::{error::HomogError, rve::Rve,
            schemes::{Effective, Scheme, SchemeOpts},
            tensor::TensorOrder};
use alloc::vec::Vec;

#[derive(Clone, Debug, Default)]
pub struct MoriTanaka;

impl<O: TensorOrder> Scheme<O> for MoriTanaka {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let c_ref = rve.reference_property(None)?;
        let mut a_dils: Vec<O::KmMatrix> = Vec::with_capacity(rve.phases.len());
        for ph in &rve.phases {
            if rve.is_matrix_phase(&ph.name) {
                a_dils.push(O::identity());
            } else {
                a_dils.push(ph.shape.concentration_dilute(&c_ref, &ph.property, &opts.integration)?);
            }
        }
        let mut sum_fa = O::zero();
        let mut sum_fca = O::zero();
        for (ph, a) in rve.phases.iter().zip(a_dils.iter()) {
            sum_fa  = O::add(&sum_fa,  &O::scale(a, ph.fraction));
            let ca  = O::mat_mul(&ph.property, a);
            sum_fca = O::add(&sum_fca, &O::scale(&ca, ph.fraction));
        }
        let sum_fa_inv = O::inverse(&sum_fa)?;
        let c_eff = O::mat_mul(&sum_fca, &sum_fa_inv);
        let concs = opts.store_concentration.then(|| a_dils);
        Ok(Effective { tensor: c_eff, concentration: concs, iterations: None, residual: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rve::Phase, shapes::Sphere, tensor::Order2};
    use alloc::{string::String, sync::Arc};
    use approx::assert_relative_eq;

    #[test]
    fn mt_matches_analytic_for_spheres_iso() {
        let k0 = 1.0;
        let k1 = 5.0;
        let f = 0.3;
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(k0), fraction: 1.0 - f });
        rve.add_phase(Phase { name: String::from("I"), shape: Arc::new(Sphere),
            property: Order2::scalar(k1), fraction: f });
        rve.set_matrix("M");
        let e = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
        // k_eff / k_0 = 1 + 3f(k1-k0) / (3k0 + (1-f)(k1-k0))
        let dk = k1 - k0;
        let expected = k0 * (1.0 + 3.0 * f * dk / (3.0 * k0 + (1.0 - f) * dk));
        assert_relative_eq!(e.tensor[(0, 0)], expected, epsilon = 1e-10);
    }

    #[test]
    fn mt_equals_matrix_at_zero_inclusion() {
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(3.0), fraction: 1.0 });
        rve.set_matrix("M");
        let e = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert_relative_eq!(e.tensor[(0, 0)], 3.0, epsilon = 1e-12);
    }
}
