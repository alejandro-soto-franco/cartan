//! Spheroid shape: aligned axisymmetric ellipsoid (axes a, a, c) with aspect c/a.
//! Closed-form depolarising factor N3(ω) for isotropic reference, Order2.
//! Order4 branch for Mura 1987 spheroid tensor.

use crate::{error::HomogError, kelvin_mandel::iso_detect_order2,
            shapes::{IntegrationOpts, Shape, Sphere},
            tensor::{Km3, Km6, Order2, Order4}};
use nalgebra::{Matrix3, Unit, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct Spheroid {
    pub axis: Unit<Vector3<f64>>,
    pub aspect: f64,   // c / a
}

impl Spheroid {
    pub fn new(axis: Unit<Vector3<f64>>, aspect: f64) -> Self { Self { axis, aspect } }

    pub fn depolarising_n3(&self) -> f64 {
        let w = self.aspect;
        if (w - 1.0).abs() < 1e-6 { return 1.0 / 3.0; }
        if w < 1.0 {
            // Oblate: N_z = 1/(1-w²) - w/(1-w²)^{3/2} · arccos(w).
            // Limits: w->0 gives 1 (strong depolarisation across thin disk),
            //         w->1 gives 1/3 (sphere).
            let one_minus_w2 = 1.0 - w * w;
            1.0 / one_minus_w2 - w / one_minus_w2.powf(1.5) * w.acos()
        } else {
            // Prolate: N_z = (1/(w²-1)) · (w/√(w²-1) · ln(w + √(w²-1)) - 1).
            // Limits: w->∞ gives 0 (weak depolarisation along needle axis),
            //         w->1 gives 1/3.
            let w2_minus_1 = w * w - 1.0;
            let s = w2_minus_1.sqrt();
            (w / s * (w + s).ln() - 1.0) / w2_minus_1
        }
    }
}

fn rotation_axis_angle(axis: &Vector3<f64>, angle: f64) -> Matrix3<f64> {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    Matrix3::new(
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c,
    )
}

fn rotation_from_z_to(to: &Vector3<f64>) -> Matrix3<f64> {
    let from = Vector3::z();
    let c = from.dot(to);
    if c > 1.0 - 1e-14 { return Matrix3::identity(); }
    if c < -1.0 + 1e-14 {
        let other = if from.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
        let ax = from.cross(&other).normalize();
        return rotation_axis_angle(&ax, core::f64::consts::PI);
    }
    let cross = from.cross(to);
    let s = cross.norm();
    let axis = cross / s;
    rotation_axis_angle(&axis, s.atan2(c))
}

impl Shape<Order2> for Spheroid {
    fn hill(&self, c_ref: &Km3, _opts: &IntegrationOpts) -> Result<Km3, HomogError> {
        let (avg, aniso) = iso_detect_order2(c_ref);
        if aniso > 1e-8 {
            return Err(HomogError::Geometry(alloc::string::String::from(
                "Spheroid::hill (Order2) closed form requires isotropic reference")));
        }
        if avg <= 0.0 { return Err(HomogError::NotPositiveDefinite); }

        let n3 = self.depolarising_n3();
        let n1 = (1.0 - n3) / 2.0;
        let principal = Matrix3::from_diagonal(&Vector3::new(n1, n1, n3)) / avg;
        let r = rotation_from_z_to(self.axis.as_ref());
        Ok(r * principal * r.transpose())
    }
}

// Order4 branch: for ω = 1, delegate to Sphere. For ω != 1, the closed-form Mura 1987
// transversely-isotropic Hill tensor is significant code (~150 LOC). v1 defers this
// to Lebedev quadrature (Task 28); for ω == 1 the exact branch works.
impl Shape<Order4> for Spheroid {
    fn hill(&self, c_ref: &Km6, opts: &IntegrationOpts) -> Result<Km6, HomogError> {
        if (self.aspect - 1.0).abs() < 1e-10 {
            return <Sphere as Shape<Order4>>::hill(&Sphere, c_ref, opts);
        }
        Err(HomogError::Geometry(alloc::string::String::from(
            "Spheroid::hill (Order4) for ω != 1 pending Mura 1987 closed form / Lebedev fallback")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorOrder;
    use approx::assert_relative_eq;

    #[test]
    fn omega_one_gives_one_third() {
        let s = Spheroid::new(Unit::new_normalize(Vector3::z()), 1.0);
        assert_relative_eq!(s.depolarising_n3(), 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn very_oblate_depolarises_to_one() {
        let s = Spheroid::new(Unit::new_normalize(Vector3::z()), 1e-4);
        assert!(s.depolarising_n3() > 0.999);
    }

    #[test]
    fn very_prolate_depolarises_to_zero() {
        let s = Spheroid::new(Unit::new_normalize(Vector3::z()), 1e4);
        assert!(s.depolarising_n3() < 1e-3);
    }

    #[test]
    fn sphere_limit_matches_sphere_hill() {
        let s = Spheroid::new(Unit::new_normalize(Vector3::z()), 1.0);
        let c_ref = Order2::scalar(2.0);
        let p = <Spheroid as Shape<Order2>>::hill(&s, &c_ref, &IntegrationOpts::default()).unwrap();
        let expected = Km3::identity() * (1.0 / 6.0);
        assert_relative_eq!(p, expected, epsilon = 1e-12);
    }
}
