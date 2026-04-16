//! n-layer concentric spheres with Herve-Zaoui effective-modulus recursion.

use crate::{error::HomogError,
            shapes::{IntegrationOpts, Shape, Sphere},
            tensor::{Km3, Km6, Order2, Order4}};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub struct SphereNLayers {
    /// Outer radii of layers, from core outward. Length = number of layers.
    pub radii: Vec<f64>,
    /// Isotropic conductivity per layer (Order2 view).
    pub layer_k: Vec<f64>,
}

impl SphereNLayers {
    pub fn new(radii: Vec<f64>, layer_k: Vec<f64>) -> Result<Self, HomogError> {
        if radii.len() != layer_k.len() || radii.is_empty() {
            return Err(HomogError::Geometry(alloc::string::String::from(
                "SphereNLayers: radii and layer_k length mismatch")));
        }
        if radii.windows(2).any(|w| w[0] >= w[1]) {
            return Err(HomogError::Geometry(alloc::string::String::from(
                "SphereNLayers: radii must be strictly increasing")));
        }
        Ok(Self { radii, layer_k })
    }

    /// Herve-Zaoui effective-k recursion from innermost core outward.
    /// Simplified Hashin-Shtrikman-style update at each interface:
    /// k_eff_new = k_layer * (2·k_layer + k_eff + 2v(k_eff - k_layer))
    ///                     / (2·k_layer + k_eff - v(k_eff - k_layer))
    pub fn effective_k(&self) -> f64 {
        let n = self.radii.len();
        let mut k_eff = self.layer_k[0];
        for i in 1..n {
            let r_inner = self.radii[i - 1];
            let r_outer = self.radii[i];
            let v = (r_inner / r_outer).powi(3);
            let k_layer = self.layer_k[i];
            let dk = k_eff - k_layer;
            let num = 2.0 * k_layer + k_eff + 2.0 * v * dk;
            let den = 2.0 * k_layer + k_eff -       v * dk;
            k_eff = k_layer * num / den;
        }
        k_eff
    }
}

impl Shape<Order2> for SphereNLayers {
    fn hill(&self, c_ref: &Km3, opts: &IntegrationOpts) -> Result<Km3, HomogError> {
        <Sphere as Shape<Order2>>::hill(&Sphere, c_ref, opts)
    }

    fn concentration_dilute(
        &self, c_ref: &Km3, _c_phase: &Km3, opts: &IntegrationOpts,
    ) -> Result<Km3, HomogError> {
        let k_eff = self.effective_k();
        let c_effective = Km3::identity() * k_eff;
        <Sphere as Shape<Order2>>::concentration_dilute(&Sphere, c_ref, &c_effective, opts)
    }
}

impl Shape<Order4> for SphereNLayers {
    fn hill(&self, c_ref: &Km6, opts: &IntegrationOpts) -> Result<Km6, HomogError> {
        <Sphere as Shape<Order4>>::hill(&Sphere, c_ref, opts)
    }

    // Order4 Herve-Zaoui recursion over (k, mu) pairs deferred to v1.1.
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn single_layer_passes_through() {
        let s = SphereNLayers::new(alloc::vec![1.0], alloc::vec![3.5]).unwrap();
        assert_relative_eq!(s.effective_k(), 3.5, epsilon = 1e-12);
    }

    #[test]
    fn two_layer_between_bounds() {
        let s = SphereNLayers::new(alloc::vec![1.0, 2.0], alloc::vec![10.0, 1.0]).unwrap();
        let k = s.effective_k();
        assert!(k > 1.0 && k < 10.0, "expected 1 < k_eff < 10, got {k}");
    }
}
