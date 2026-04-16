//! TensorOrder trait: generic abstraction for 2nd-order (conductivity) and
//! 4th-order (elasticity) tensors in Kelvin-Mandel storage.

use crate::error::HomogError;
use nalgebra::{Matrix3, SMatrix};

pub type Km3 = Matrix3<f64>;
pub type Km6 = SMatrix<f64, 6, 6>;

pub trait TensorOrder: 'static + Sized + Clone + core::fmt::Debug {
    const KM_DIM: usize;
    type KmMatrix: Clone + core::fmt::Debug + Send + Sync + PartialEq;

    fn identity() -> Self::KmMatrix;
    fn zero() -> Self::KmMatrix;
    fn scalar(s: f64) -> Self::KmMatrix;

    fn add(a: &Self::KmMatrix, b: &Self::KmMatrix) -> Self::KmMatrix;
    fn sub(a: &Self::KmMatrix, b: &Self::KmMatrix) -> Self::KmMatrix;
    fn scale(a: &Self::KmMatrix, s: f64) -> Self::KmMatrix;
    fn mat_mul(a: &Self::KmMatrix, b: &Self::KmMatrix) -> Self::KmMatrix;

    fn inverse(a: &Self::KmMatrix) -> Result<Self::KmMatrix, HomogError>;
    fn frobenius_norm(a: &Self::KmMatrix) -> f64;

    /// Fallible view as an SPD point in the manifold of SPD(KM_DIM). Returns the
    /// symmetrised matrix if positive definite, Err(PercolationThreshold) otherwise.
    /// Use for affine-invariant distance / Karcher mean / geodesic interpolation.
    fn check_spd(a: &Self::KmMatrix, tol: f64) -> Result<Self::KmMatrix, HomogError>;

    /// One step of SPD-geodesic iteration: move from `from` toward `to` on the
    /// affine-invariant SPD(KM_DIM) manifold, with damping coefficient `damping`.
    /// Falls back to Euclidean damping when either operand is not SPD.
    fn spd_geodesic_step(
        from: &Self::KmMatrix, to: &Self::KmMatrix, damping: f64,
    ) -> Result<Self::KmMatrix, HomogError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Order2;

impl TensorOrder for Order2 {
    const KM_DIM: usize = 3;
    type KmMatrix = Km3;

    fn identity() -> Km3     { Km3::identity() }
    fn zero() -> Km3         { Km3::zeros() }
    fn scalar(s: f64) -> Km3 { Km3::identity() * s }

    fn add(a: &Km3, b: &Km3) -> Km3        { a + b }
    fn sub(a: &Km3, b: &Km3) -> Km3        { a - b }
    fn scale(a: &Km3, s: f64) -> Km3       { a * s }
    fn mat_mul(a: &Km3, b: &Km3) -> Km3    { a * b }

    fn inverse(a: &Km3) -> Result<Km3, HomogError> {
        a.try_inverse().ok_or(HomogError::SingularMatrix)
    }
    fn frobenius_norm(a: &Km3) -> f64 { a.norm() }

    fn check_spd(a: &Km3, tol: f64) -> Result<Km3, HomogError> {
        let sym = (a + a.transpose()) * 0.5;
        let eig = sym.symmetric_eigen();
        if eig.eigenvalues.iter().any(|v| *v <= tol) {
            return Err(HomogError::PercolationThreshold);
        }
        Ok(sym)
    }

    fn spd_geodesic_step(from: &Km3, to: &Km3, damping: f64) -> Result<Km3, HomogError> {
        use cartan_core::Manifold;
        use cartan_manifolds::Spd;
        let spd = Spd::<3>;
        match (Self::check_spd(from, 1e-14), Self::check_spd(to, 1e-14)) {
            (Ok(p), Ok(q)) => {
                let v = spd.log(&p, &q).map_err(|e| HomogError::Solver(alloc::format!("{e}")))?;
                Ok(spd.exp(&p, &(v * damping)))
            }
            _ => Ok(Self::add(&Self::scale(from, 1.0 - damping), &Self::scale(to, damping))),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Order4;

impl Order4 {
    /// Kelvin-Mandel hydrostatic and deviatoric projectors for isotropic algebra.
    pub fn iso_projectors() -> (Km6, Km6) {
        let mut j = Km6::zeros();
        for i in 0..3 { for jj in 0..3 { j[(i, jj)] = 1.0 / 3.0; } }
        let k = Km6::identity() - j;
        (j, k)
    }

    /// Build an isotropic stiffness tensor from bulk `k` and shear `mu`.
    /// C = 3k J + 2mu K in Kelvin-Mandel basis.
    pub fn iso_stiff(k: f64, mu: f64) -> Km6 {
        let (j, kp) = Self::iso_projectors();
        3.0 * k * j + 2.0 * mu * kp
    }
}

impl TensorOrder for Order4 {
    const KM_DIM: usize = 6;
    type KmMatrix = Km6;

    fn identity() -> Km6     { Km6::identity() }
    fn zero() -> Km6         { Km6::zeros() }
    fn scalar(s: f64) -> Km6 { Km6::identity() * s }

    fn add(a: &Km6, b: &Km6) -> Km6     { a + b }
    fn sub(a: &Km6, b: &Km6) -> Km6     { a - b }
    fn scale(a: &Km6, s: f64) -> Km6    { a * s }
    fn mat_mul(a: &Km6, b: &Km6) -> Km6 { a * b }

    fn inverse(a: &Km6) -> Result<Km6, HomogError> {
        a.try_inverse().ok_or(HomogError::SingularMatrix)
    }
    fn frobenius_norm(a: &Km6) -> f64 { a.norm() }

    fn check_spd(a: &Km6, tol: f64) -> Result<Km6, HomogError> {
        let sym = (a + a.transpose()) * 0.5;
        let eig = sym.symmetric_eigen();
        if eig.eigenvalues.iter().any(|v| *v <= tol) {
            return Err(HomogError::PercolationThreshold);
        }
        Ok(sym)
    }

    fn spd_geodesic_step(from: &Km6, to: &Km6, damping: f64) -> Result<Km6, HomogError> {
        use cartan_core::Manifold;
        use cartan_manifolds::Spd;
        let spd = Spd::<6>;
        match (Self::check_spd(from, 1e-14), Self::check_spd(to, 1e-14)) {
            (Ok(p), Ok(q)) => {
                let v = spd.log(&p, &q).map_err(|e| HomogError::Solver(alloc::format!("{e}")))?;
                Ok(spd.exp(&p, &(v * damping)))
            }
            _ => Ok(Self::add(&Self::scale(from, 1.0 - damping), &Self::scale(to, damping))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn order2_identity_is_identity() {
        let i = Order2::identity();
        let product = Order2::mat_mul(&i, &i);
        assert_relative_eq!(product, i, epsilon = 1e-12);
    }

    #[test]
    fn order2_inverse_of_scalar_k() {
        let k = Order2::scalar(5.0);
        let k_inv = Order2::inverse(&k).unwrap();
        let prod = Order2::mat_mul(&k, &k_inv);
        assert_relative_eq!(prod, Order2::identity(), epsilon = 1e-12);
    }

    #[test]
    fn order2_singular_errors() {
        let zero = Order2::zero();
        assert!(matches!(Order2::inverse(&zero), Err(HomogError::SingularMatrix)));
    }

    #[test]
    fn order2_check_spd_rejects_zero_eigenvalue() {
        let mut k = Order2::scalar(1.0);
        k[(2, 2)] = 0.0;
        assert!(matches!(Order2::check_spd(&k, 1e-12), Err(HomogError::PercolationThreshold)));
    }

    #[test]
    fn order4_iso_stiff_invertible() {
        let c = Order4::iso_stiff(72.0, 32.0);
        let inv = Order4::inverse(&c).unwrap();
        let prod = Order4::mat_mul(&c, &inv);
        assert_relative_eq!(prod, Order4::identity(), epsilon = 1e-10);
    }

    #[test]
    fn order4_iso_stiff_bulk_and_shear_eigenvalues() {
        let c = Order4::iso_stiff(72.0, 32.0);
        let (j, k_proj) = Order4::iso_projectors();
        // C = 3k·J + 2mu·K with J, K complementary projectors:
        //   trace(C·J)/trace(J) = 3k  and  trace(C·K)/trace(K) = 2mu.
        let eig_bulk  = (c * j).trace()      / j.trace();
        let eig_shear = (c * k_proj).trace() / k_proj.trace();
        assert_relative_eq!(eig_bulk,  3.0 * 72.0, epsilon = 1e-10);
        assert_relative_eq!(eig_shear, 2.0 * 32.0, epsilon = 1e-10);
    }
}
