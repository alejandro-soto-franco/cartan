//! Maxwell scheme + Ponte Castañeda-Willis (same formula, different interpretation
//! of the outer ellipsoid: Maxwell cluster vs PCW distribution).

use crate::{error::HomogError, rve::Rve,
            schemes::{Effective, Scheme, SchemeOpts},
            shapes::{Shape, Sphere},
            tensor::TensorOrder};

fn maxwell_style<O: TensorOrder>(rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError>
where
    Sphere: Shape<O>,
{
    let c_ref = rve.reference_property(None)?;
    let mut sum = O::zero();
    for ph in &rve.phases {
        if rve.is_matrix_phase(&ph.name) { continue; }
        let a = ph.shape.concentration_dilute(&c_ref, &ph.property, &opts.integration)?;
        let dc = O::sub(&ph.property, &c_ref);
        let dc_a = O::mat_mul(&dc, &a);
        sum = O::add(&sum, &O::scale(&dc_a, ph.fraction));
    }
    let sum_inv = O::inverse(&sum)?;
    // Distribution ellipsoid defaults to spherical.
    let p_dist = <Sphere as Shape<O>>::hill(&Sphere, &c_ref, &opts.integration)?;
    let arg = O::sub(&sum_inv, &p_dist);
    let bracket = O::inverse(&arg)?;
    Ok(Effective { tensor: O::add(&c_ref, &bracket), concentration: None,
                    iterations: None, residual: None })
}

#[derive(Clone, Debug, Default)]
pub struct Maxwell;

impl<O: TensorOrder> Scheme<O> for Maxwell
where
    Sphere: Shape<O>,
{
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        maxwell_style(rve, opts)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PonteCastanedaWillis;

impl<O: TensorOrder> Scheme<O> for PonteCastanedaWillis
where
    Sphere: Shape<O>,
{
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError> {
        maxwell_style(rve, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rve::Phase, schemes::MoriTanaka, tensor::Order2};
    use alloc::{string::String, sync::Arc};
    use approx::assert_relative_eq;

    #[test]
    fn max_equals_mt_for_spherical_inclusions_iso_matrix() {
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0), fraction: 0.7 });
        rve.add_phase(Phase { name: String::from("I"), shape: Arc::new(Sphere),
            property: Order2::scalar(8.0), fraction: 0.3 });
        rve.set_matrix("M");
        let max = Maxwell.homogenize(&rve, &SchemeOpts::default()).unwrap();
        let mt  = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
        assert_relative_eq!(max.tensor, mt.tensor, epsilon = 1e-8);
    }
}
