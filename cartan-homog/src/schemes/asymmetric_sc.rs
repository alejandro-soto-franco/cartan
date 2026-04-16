//! Asymmetric self-consistent: matrix contributes directly as C_0; inclusions
//! embedded in the evolving effective medium.

use crate::{error::HomogError, rve::Rve,
            schemes::{Effective, Scheme, SchemeOpts},
            tensor::TensorOrder};

#[derive(Clone, Debug, Default)]
pub struct AsymmetricSc;

impl<O: TensorOrder> Scheme<O> for AsymmetricSc {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let c0 = rve.matrix_property()?.clone();
        let mut c_hom = c0.clone();
        let mut last_res = f64::INFINITY;

        for iter in 0..opts.max_iter {
            let mut acc = c0.clone();
            for ph in &rve.phases {
                if rve.is_matrix_phase(&ph.name) { continue; }
                // A_r = (I + P(C^ASC) : (C_r - C^ASC))^{-1}  — concentration tensor
                // contribution = (C_r - C_0) : A_r           — contrast is matrix-relative
                let dc_hom = O::sub(&ph.property, &c_hom);
                let dc_matrix = O::sub(&ph.property, &c0);
                let p = ph.shape.hill(&c_hom, &opts.integration)?;
                let arg = O::add(&O::identity(), &O::mat_mul(&p, &dc_hom));
                let a_r = O::inverse(&arg)?;
                let contrib = O::scale(&O::mat_mul(&dc_matrix, &a_r), ph.fraction);
                acc = O::add(&acc, &contrib);
            }
            let c_next = if opts.spd_iteration {
                O::spd_geodesic_step(&c_hom, &acc, opts.damping)?
            } else {
                O::add(&O::scale(&c_hom, 1.0 - opts.damping),
                       &O::scale(&acc, opts.damping))
            };
            let res = O::frobenius_norm(&O::sub(&c_next, &c_hom))
                      / O::frobenius_norm(&c_hom).max(1e-300);
            c_hom = c_next;
            if res < opts.rel_tol {
                return Ok(Effective { tensor: c_hom, concentration: None,
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
    fn asc_reduces_to_matrix_at_zero_inclusion() {
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(2.0), fraction: 1.0 });
        rve.set_matrix("M");
        let e = AsymmetricSc.homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert!((e.tensor[(0, 0)] - 2.0).abs() < 1e-6);
    }
}
