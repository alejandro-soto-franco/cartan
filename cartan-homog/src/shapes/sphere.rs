//! Sphere shape: closed-form Hill tensor for isotropic reference.

use crate::{error::HomogError, kelvin_mandel::iso_detect_order2,
            shapes::{IntegrationOpts, Shape},
            tensor::{Km3, Km6, Order2, Order4}};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sphere;

impl Shape<Order2> for Sphere {
    fn hill(&self, c_ref: &Km3, opts: &IntegrationOpts) -> Result<Km3, HomogError> {
        let (avg, aniso) = iso_detect_order2(c_ref);
        if aniso > 1e-8 {
            // Anisotropic-reference branch (v1.3): Lebedev quadrature.
            return crate::shapes::lebedev::hill_order2_anisotropic(c_ref, opts.lebedev_degree);
        }
        if avg <= 0.0 { return Err(HomogError::NotPositiveDefinite); }
        Ok(Km3::identity() * (1.0 / (3.0 * avg)))
    }
}

/// Extract bulk and shear moduli from a Kelvin-Mandel stiffness tensor assuming isotropy.
pub(crate) fn extract_k_mu_iso(c: &Km6) -> Result<(f64, f64), HomogError> {
    let (j, k_proj) = Order4::iso_projectors();
    let trace_j = j.trace();
    let trace_k = k_proj.trace();
    if trace_j.abs() < 1e-14 || trace_k.abs() < 1e-14 {
        return Err(HomogError::NotPositiveDefinite);
    }
    let three_k = (j * c).trace() / trace_j;
    let two_mu  = (k_proj * c).trace() / trace_k;
    Ok((three_k / 3.0, two_mu / 2.0))
}

impl Shape<Order4> for Sphere {
    fn hill(&self, c_ref: &Km6, _opts: &IntegrationOpts) -> Result<Km6, HomogError> {
        let (k0, mu0) = extract_k_mu_iso(c_ref)?;
        let c_iso = Order4::iso_stiff(k0, mu0);
        let diff = (c_ref - c_iso).norm() / c_ref.norm().max(1e-300);
        if diff > 1e-8 {
            return Err(HomogError::Geometry(
                alloc::string::String::from(
                    "Sphere::hill (Order4) closed form requires isotropic reference")));
        }
        if k0 <= 0.0 || mu0 <= 0.0 { return Err(HomogError::NotPositiveDefinite); }
        let (j, k_proj) = Order4::iso_projectors();
        let alpha = 1.0 / (3.0 * k0 + 4.0 * mu0);
        let beta  = 3.0 * (k0 + 2.0 * mu0) / (5.0 * mu0 * (3.0 * k0 + 4.0 * mu0));
        Ok(alpha * j + beta * k_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorOrder;
    use approx::assert_relative_eq;

    #[test]
    fn hill_sphere_iso_order2_matches_1_over_3k() {
        let c_ref = Order2::scalar(4.0);
        let p = <Sphere as Shape<Order2>>::hill(&Sphere, &c_ref, &IntegrationOpts::default()).unwrap();
        let expected = Km3::identity() * (1.0 / 12.0);
        assert_relative_eq!(p, expected, epsilon = 1e-12);
    }

    #[test]
    fn hill_sphere_iso_order2_anisotropic_ref_uses_lebedev() {
        // v1.3: anisotropic reference no longer rejected; falls through to
        // Lebedev quadrature and produces an SPD Hill tensor.
        let c_ref = crate::kelvin_mandel::ti_order2(10.0, 1.0);
        let p = <Sphere as Shape<Order2>>::hill(&Sphere, &c_ref, &IntegrationOpts::default()).unwrap();
        let eig = p.symmetric_eigen();
        assert!(eig.eigenvalues.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn dilute_concentration_sphere_5x_contrast() {
        let c_ref = Order2::scalar(1.0);
        let c_phase = Order2::scalar(5.0);
        let a = <Sphere as Shape<Order2>>::concentration_dilute(
            &Sphere, &c_ref, &c_phase, &IntegrationOpts::default()).unwrap();
        let expected = Km3::identity() * (3.0 / 7.0);
        assert_relative_eq!(a, expected, epsilon = 1e-12);
    }

    #[test]
    fn hill_sphere_iso_order4_hydrostatic_component() {
        let c_ref = Order4::iso_stiff(72.0, 32.0);
        let p: Km6 = <Sphere as Shape<Order4>>::hill(&Sphere, &c_ref, &IntegrationOpts::default()).unwrap();
        let (j, _k) = Order4::iso_projectors();
        let pj = (p * j).trace() / j.trace();
        let expected_alpha = 1.0 / (3.0 * 72.0 + 4.0 * 32.0);
        assert_relative_eq!(pj, expected_alpha, epsilon = 1e-12);
    }
}
