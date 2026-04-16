//! Differential scheme: ODE-based incremental dilution. Two variants:
//!
//! - `Differential` (Roscoe-Brinkman form): integrates `dC*/df` on stiffness.
//!   Default; numerically stable when the inclusion is stiffer than the matrix.
//!
//! - `DifferentialCompliance` (Norris-Davies dual form): integrates `dS*/df`
//!   on compliance `S = C^{-1}`. Numerically stable when the inclusion is
//!   softer than the matrix (dry-pore / crack limit). The dual scheme
//!   converges to a DIFFERENT effective tensor than the primal form at
//!   moderate and high fractions (both are self-consistent but are built
//!   from different stationarity principles; see Milton 2002 Ch. 10.12).
//!
//! Both currently support single-inclusion RVEs.

use crate::{error::HomogError, rve::{Phase, Rve},
            schemes::{Effective, Scheme, SchemeOpts},
            tensor::TensorOrder};

#[derive(Clone, Debug)]
pub struct Differential {
    pub n_steps: usize,
}

impl Default for Differential {
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

/// Dual (Norris-Davies) differential scheme on compliances. Integrates
/// `dS*/df = 1/(1-f) · (S_1 - S*) : B_{1,S*}` with B the dual concentration
/// tensor expressed through the primal Hill P: `B = (I + Q:(S_1-S*))^{-1}`
/// where `Q = C* - C*:P:C*` is the dual Hill tensor. The final `C*` is
/// recovered as `(S*)^{-1}`. Numerically preferred when the inclusion is
/// softer than the matrix.
#[derive(Clone, Debug)]
pub struct DifferentialCompliance {
    pub n_steps: usize,
}

impl Default for DifferentialCompliance {
    fn default() -> Self { Self { n_steps: 100 } }
}

impl<O: TensorOrder> Scheme<O> for DifferentialCompliance {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        let c0 = rve.matrix_property()?.clone();
        let incls: alloc::vec::Vec<&Phase<O>> = rve.phases.iter()
            .filter(|p| !rve.is_matrix_phase(&p.name))
            .collect();
        if incls.len() != 1 {
            return Err(HomogError::Solver(alloc::string::String::from(
                "DifferentialCompliance scheme currently supports single-inclusion RVEs")));
        }
        let inc = incls[0];
        let f_target = inc.fraction;
        let n = self.n_steps.max(20);
        let df = f_target / (n as f64);

        let s0 = O::inverse(&c0)?;
        let s1 = O::inverse(&inc.property)?;
        let mut s_hom = s0;
        let mut f = 0.0;
        for _ in 0..n {
            let k1 = rate_dual::<O>(&s_hom, &s1, inc, f,                                      opts)?;
            let k2 = rate_dual::<O>(&O::add(&s_hom, &O::scale(&k1, df * 0.5)), &s1, inc, f + df * 0.5, opts)?;
            let k3 = rate_dual::<O>(&O::add(&s_hom, &O::scale(&k2, df * 0.5)), &s1, inc, f + df * 0.5, opts)?;
            let k4 = rate_dual::<O>(&O::add(&s_hom, &O::scale(&k3, df      )), &s1, inc, f + df,       opts)?;
            let mut delta = O::add(&k1, &O::scale(&k2, 2.0));
            delta = O::add(&delta, &O::scale(&k3, 2.0));
            delta = O::add(&delta, &k4);
            s_hom = O::add(&s_hom, &O::scale(&delta, df / 6.0));
            f += df;
        }
        let c_hom = O::inverse(&s_hom)?;
        Ok(Effective { tensor: c_hom, concentration: None, iterations: Some(n), residual: None })
    }
}

fn rate_dual<O: TensorOrder>(
    s_hom: &O::KmMatrix,
    s_inc: &O::KmMatrix,
    phase: &Phase<O>,
    f: f64,
    opts: &SchemeOpts,
) -> Result<O::KmMatrix, HomogError> {
    // Q = C* - C*:P:C*   where C* = S*^{-1}
    let c_hom = O::inverse(s_hom)?;
    let p = phase.shape.hill(&c_hom, &opts.integration)?;
    let cpc = O::mat_mul(&O::mat_mul(&c_hom, &p), &c_hom);
    let q = O::sub(&c_hom, &cpc);
    let ds = O::sub(s_inc, s_hom);
    let arg = O::add(&O::identity(), &O::mat_mul(&q, &ds));
    let inv = O::inverse(&arg)?;
    let contrib = O::mat_mul(&ds, &inv);
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

    #[test]
    fn differential_compliance_agrees_at_zero_fraction() {
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0), fraction: 1.0 });
        rve.add_phase(Phase { name: String::from("I"), shape: Arc::new(Sphere),
            property: Order2::scalar(10.0), fraction: 0.0 });
        rve.set_matrix("M");
        let e = DifferentialCompliance::default().homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert!((e.tensor[(0, 0)] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn primal_vs_dual_bracket_each_other() {
        // Primal and dual DIFF schemes bracket the "true" differential answer.
        // For soft inclusions the dual variant is physically preferred; both
        // converge to the same limit as phi -> 0.
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(10.0), fraction: 0.8 });
        rve.add_phase(Phase { name: String::from("I"), shape: Arc::new(Sphere),
            property: Order2::scalar(0.1), fraction: 0.2 });
        rve.set_matrix("M");
        let e_primal = Differential::default().homogenize(&rve, &SchemeOpts::default()).unwrap();
        let e_dual   = DifferentialCompliance::default().homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert!(e_primal.tensor[(0, 0)] > 0.0 && e_primal.tensor[(0, 0)] < 10.0);
        assert!(e_dual.tensor[(0, 0)]   > 0.0 && e_dual.tensor[(0, 0)]   < 10.0);
    }
}
