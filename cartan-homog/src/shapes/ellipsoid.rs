//! Generic triaxial ellipsoid with Carlson RD depolarising factors.

use crate::{error::HomogError, kelvin_mandel::iso_detect_order2,
            shapes::{IntegrationOpts, Shape},
            tensor::{Km3, Km6, Order2, Order4}};
use nalgebra::{Matrix3, Rotation3, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct Ellipsoid {
    pub semi_axes: Vector3<f64>,
    pub rotation: Rotation3<f64>,
}

impl Ellipsoid {
    pub fn new(semi_axes: Vector3<f64>, rotation: Rotation3<f64>) -> Self {
        debug_assert!(semi_axes.iter().all(|x| *x > 0.0));
        Self { semi_axes, rotation }
    }

    pub fn unit_sphere() -> Self {
        Self::new(Vector3::new(1.0, 1.0, 1.0), Rotation3::identity())
    }

    pub fn depolarising(&self) -> [f64; 3] {
        let a = self.semi_axes;
        let a1 = a.x * a.x;
        let a2 = a.y * a.y;
        let a3 = a.z * a.z;
        let prod = a.x * a.y * a.z;
        let n1 = prod * carlson_rd(a2, a3, a1) / 3.0;
        let n2 = prod * carlson_rd(a3, a1, a2) / 3.0;
        let n3 = prod * carlson_rd(a1, a2, a3) / 3.0;
        [n1, n2, n3]
    }
}

/// Carlson's symmetric integral R_D(x, y, z). Numerical Recipes §6.11.
pub fn carlson_rd(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    const ERRTOL: f64 = 1e-10;
    let mut sum = 0.0;
    let mut fac = 1.0;
    for _ in 0..64 {
        let sqrtx = x.sqrt();
        let sqrty = y.sqrt();
        let sqrtz = z.sqrt();
        let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        sum += fac / (sqrtz * (z + alamb));
        fac *= 0.25;
        x = 0.25 * (x + alamb);
        y = 0.25 * (y + alamb);
        z = 0.25 * (z + alamb);
        let ave = 0.2 * (x + y + 3.0 * z);
        let delx = (ave - x) / ave;
        let dely = (ave - y) / ave;
        let delz = (ave - z) / ave;
        if delx.abs() < ERRTOL && dely.abs() < ERRTOL && delz.abs() < ERRTOL {
            let ea = delx * dely;
            let eb = delz * delz;
            let ec = ea - eb;
            let ed = ea - 6.0 * eb;
            let ee = ed + ec + ec;
            return 3.0 * sum
                + fac * (1.0 + ed * (-3.0 / 14.0 + 0.25 * ed - 9.0 / 22.0 * delz * ee)
                              + delz * (ec / 6.0 + delz * (-9.0 / 22.0 * ec + 0.75 * delz * ea)))
                      / (ave * ave.sqrt());
        }
    }
    f64::NAN
}

impl Shape<Order2> for Ellipsoid {
    fn hill(&self, c_ref: &Km3, _opts: &IntegrationOpts) -> Result<Km3, HomogError> {
        let (avg, aniso) = iso_detect_order2(c_ref);
        if aniso > 1e-8 {
            return Err(HomogError::Geometry(alloc::string::String::from(
                "Ellipsoid::hill (Order2) closed form requires isotropic reference")));
        }
        if avg <= 0.0 { return Err(HomogError::NotPositiveDefinite); }

        let [n1, n2, n3] = self.depolarising();
        let principal = Matrix3::from_diagonal(&Vector3::new(n1, n2, n3)) / avg;
        let r = self.rotation.matrix();
        Ok(r * principal * r.transpose())
    }
}

impl Shape<Order4> for Ellipsoid {
    fn hill(&self, _c_ref: &Km6, _opts: &IntegrationOpts) -> Result<Km6, HomogError> {
        Err(HomogError::Geometry(alloc::string::String::from(
            "Ellipsoid::hill (Order4) pending Lebedev fallback in Task 28")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sphere_depolarising_is_1_3_each() {
        let e = Ellipsoid::unit_sphere();
        let n: [f64; 3] = e.depolarising();
        assert_relative_eq!(n[0] + n[1] + n[2], 1.0, epsilon = 1e-8);
        for v in n { assert_relative_eq!(v, 1.0/3.0, epsilon = 1e-8); }
    }

    #[test]
    fn elongated_has_small_axial_depolarising() {
        let e = Ellipsoid::new(Vector3::new(10.0, 1.0, 1.0), Rotation3::identity());
        let n = e.depolarising();
        assert!(n[0] < 0.05);
        assert!(n[1] > 0.4 && n[1] < 0.6);
    }
}
