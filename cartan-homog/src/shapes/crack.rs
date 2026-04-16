//! Penny-shaped crack: vanishingly oblate spheroid + Budiansky-O'Connell density ε = n·a³.

use crate::{error::HomogError,
            shapes::{IntegrationOpts, Shape, Spheroid},
            tensor::{Km3, Km6, Order2, Order4}};
use nalgebra::{Unit, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct PennyCrack {
    pub normal: Unit<Vector3<f64>>,
    pub density: f64,
    pub tiny_aspect: f64,
}

impl PennyCrack {
    pub fn new(normal: Unit<Vector3<f64>>, density: f64) -> Self {
        // tiny_aspect: default 1e-3 chosen to match ECHOES's penny-crack convention
        // (spheroidal(1e-3)). Smaller values reproduce the Kachanov/Budiansky-O'Connell
        // limit more faithfully but cause conditioning issues in I + P:(δC)^{-1}
        // near the void limit. 1e-3 is a pragmatic compromise.
        Self { normal, density, tiny_aspect: 1e-3 }
    }
}

impl Shape<Order2> for PennyCrack {
    fn hill(&self, c_ref: &Km3, opts: &IntegrationOpts) -> Result<Km3, HomogError> {
        let sph = Spheroid::new(self.normal, self.tiny_aspect);
        <Spheroid as Shape<Order2>>::hill(&sph, c_ref, opts)
    }
}

impl Shape<Order4> for PennyCrack {
    fn hill(&self, c_ref: &Km6, opts: &IntegrationOpts) -> Result<Km6, HomogError> {
        let sph = Spheroid::new(self.normal, self.tiny_aspect);
        <Spheroid as Shape<Order4>>::hill(&sph, c_ref, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorOrder;

    #[test]
    fn penny_crack_order2_normal_component_dominates() {
        let c = PennyCrack::new(Unit::new_normalize(Vector3::z()), 0.3);
        let c_ref = Order2::scalar(1.0);
        let p = <PennyCrack as Shape<Order2>>::hill(&c, &c_ref, &IntegrationOpts::default()).unwrap();
        assert!(p[(2, 2)] > 0.99, "expected P_zz close to 1/k_ref, got {}", p[(2, 2)]);
        assert!(p[(0, 0)].abs() < 2e-3);   // N_1 ~ 7.85e-4 at aspect=1e-3
    }

    #[test]
    fn penny_crack_rotates_with_normal() {
        let c = PennyCrack::new(Unit::new_normalize(Vector3::x()), 0.3);
        let c_ref = Order2::scalar(1.0);
        let p = <PennyCrack as Shape<Order2>>::hill(&c, &c_ref, &IntegrationOpts::default()).unwrap();
        assert!(p[(0, 0)] > 0.99);
        assert!(p[(2, 2)].abs() < 2e-3);
    }
}
