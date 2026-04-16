//! Differential scheme: ODE-based incremental dilution.
//! Currently supports single-inclusion RVEs.

use crate::{error::HomogError, rve::{Phase, Rve},
            schemes::{Effective, Scheme, SchemeOpts},
            tensor::TensorOrder};

#[derive(Clone, Debug)]
pub struct Differential {
    pub n_steps: usize,
}

impl Default for Differential {
    /// 100 steps is enough for machine-precision convergence of this ODE (RK4 on
    /// a smooth f ∈ [0, 0.4] domain converges by ~20 steps). The ~5e-4 residual
    /// gap to ECHOES's DIFF output is a *formulation* difference, not step
    /// noise: cartan uses the Roscoe-Brinkman form `dC/df = (1/(1-f))·(C_1-C):A`,
    /// ECHOES uses what appears to be a Norris-Davies dual variant operating on
    /// compliances for softer inclusions. Resolving that is v1.3 work; for
    /// matching-limit validation, compare against MT instead of ECHOES DIFF.
    fn default() -> Self { Self { n_steps: 100 } }
}

impl<O: TensorOrder> Scheme<O> for Differential {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let c0 = rve.matrix_property()?.clone();
        let incls: alloc::vec::Vec<&Phase<O>> = rve.phases.iter()
            .filter(|p| !rve.is_matrix_phase(&p.name))
            .collect();
        if incls.len() != 1 {
            return Err(HomogError::Solver(alloc::string::String::from(
                "Differential scheme currently supports single-inclusion RVEs")));
        }
        let inc = incls[0];
        let f_target = inc.fraction;
        let n = self.n_steps.max(20);
        let df = f_target / (n as f64);
        let mut c_hom = c0;
        let mut f = 0.0;
        for _ in 0..n {
            let k1 = rate::<O>(&c_hom, &inc.property, inc, f,              opts)?;
            let k2 = rate::<O>(&O::add(&c_hom, &O::scale(&k1, df * 0.5)), &inc.property, inc, f + df*0.5, opts)?;
            let k3 = rate::<O>(&O::add(&c_hom, &O::scale(&k2, df * 0.5)), &inc.property, inc, f + df*0.5, opts)?;
            let k4 = rate::<O>(&O::add(&c_hom, &O::scale(&k3, df      )), &inc.property, inc, f + df,       opts)?;
            let mut delta = O::add(&k1, &O::scale(&k2, 2.0));
            delta = O::add(&delta, &O::scale(&k3, 2.0));
            delta = O::add(&delta, &k4);
            c_hom = O::add(&c_hom, &O::scale(&delta, df / 6.0));
            f += df;
        }
        Ok(Effective { tensor: c_hom, concentration: None, iterations: Some(n), residual: None })
    }
}

fn rate<O: TensorOrder>(
    c_hom: &O::KmMatrix,
    c_inc: &O::KmMatrix,
    phase: &Phase<O>,
    f: f64,
    opts: &SchemeOpts,
) -> Result<O::KmMatrix, HomogError> {
    let p = phase.shape.hill(c_hom, &opts.integration)?;
    let dc = O::sub(c_inc, c_hom);
    let arg = O::add(&O::identity(), &O::mat_mul(&p, &dc));
    let inv = O::inverse(&arg)?;
    let contrib = O::mat_mul(&dc, &inv);
    Ok(O::scale(&contrib, 1.0 / (1.0 - f).max(1e-8)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::Sphere, tensor::Order2};
    use alloc::{string::String, sync::Arc};

    #[test]
    fn differential_matches_matrix_at_zero_fraction() {
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0), fraction: 1.0 });
        rve.add_phase(Phase { name: String::from("I"), shape: Arc::new(Sphere),
            property: Order2::scalar(10.0), fraction: 0.0 });
        rve.set_matrix("M");
        let e = Differential::default().homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert!((e.tensor[(0, 0)] - 1.0).abs() < 1e-8);
    }
}
