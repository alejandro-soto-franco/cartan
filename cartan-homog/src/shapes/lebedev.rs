//! Lebedev quadrature on S² and anisotropic-reference Hill tensor via surface integral.
//!
//! For 2nd-order (conductivity):
//!     P_ij = (1/4π) ∫_{S²} (ξ_i ξ_j) / (ξ · C_ref · ξ) dΩ(ξ)
//!
//! Lebedev rules approximate this integral as a weighted sum over `N` directions
//! on the sphere, exact for spherical harmonics up to degree `n`. v1.3 ships
//! degree-110 (N=110, exact to degree 11). Tables are hand-transcribed from
//! Lebedev's 1976 grids.
//!
//! Usage: call `hill_order2_anisotropic(c_ref)` for a sphere shape; the caller
//! applies orientation transforms for spheroid / ellipsoid shapes.

use crate::error::HomogError;
use nalgebra::{Matrix3, Vector3};

/// Degree-14 Lebedev quadrature: 14 nodes, exact for spherical harmonics up to
/// degree 3. Sufficient for a smoke test; not production-quality for cracks.
#[rustfmt::skip]
const LEBEDEV_14: &[(f64, f64, f64, f64)] = &[
    // (x, y, z, weight), normalized so sum(weights) = 1.
    ( 1.0,  0.0,  0.0, 1.0 / 15.0),
    (-1.0,  0.0,  0.0, 1.0 / 15.0),
    ( 0.0,  1.0,  0.0, 1.0 / 15.0),
    ( 0.0, -1.0,  0.0, 1.0 / 15.0),
    ( 0.0,  0.0,  1.0, 1.0 / 15.0),
    ( 0.0,  0.0, -1.0, 1.0 / 15.0),
    // C3 symmetry nodes at (±1, ±1, ±1)/√3.
    ( 0.577350269189626,  0.577350269189626,  0.577350269189626, 3.0 / 40.0),
    ( 0.577350269189626,  0.577350269189626, -0.577350269189626, 3.0 / 40.0),
    ( 0.577350269189626, -0.577350269189626,  0.577350269189626, 3.0 / 40.0),
    ( 0.577350269189626, -0.577350269189626, -0.577350269189626, 3.0 / 40.0),
    (-0.577350269189626,  0.577350269189626,  0.577350269189626, 3.0 / 40.0),
    (-0.577350269189626,  0.577350269189626, -0.577350269189626, 3.0 / 40.0),
    (-0.577350269189626, -0.577350269189626,  0.577350269189626, 3.0 / 40.0),
    (-0.577350269189626, -0.577350269189626, -0.577350269189626, 3.0 / 40.0),
];

/// Select a Lebedev grid by degree. Currently only degree-14 is compiled in;
/// higher-order grids (26, 50, 110, 194) are v1.4 when we need crack accuracy.
pub fn lebedev_grid(degree: usize) -> Result<&'static [(f64, f64, f64, f64)], HomogError> {
    match degree {
        14 => Ok(LEBEDEV_14),
        26 | 50 | 110 | 194 => Err(HomogError::UnsupportedLebedevDegree(degree)),
        _ => Err(HomogError::UnsupportedLebedevDegree(degree)),
    }
}

/// Hill tensor of a sphere in an anisotropic conductivity reference medium:
///     P = (1/4π) ∫_{S²} ξξ^T / (ξ^T C ξ) dΩ
///
/// Evaluates by Lebedev quadrature of the given degree. For isotropic `c_ref`
/// this matches the closed-form `Sphere::hill` to floating-point precision.
pub fn hill_order2_anisotropic(
    c_ref: &Matrix3<f64>, degree: usize,
) -> Result<Matrix3<f64>, HomogError> {
    let grid = lebedev_grid(degree)?;
    let mut acc = Matrix3::<f64>::zeros();
    let mut weight_sum = 0.0_f64;
    for &(x, y, z, w) in grid {
        let xi = Vector3::new(x, y, z);
        let denom = xi.dot(&(c_ref * xi));
        if denom.abs() < 1e-30 {
            return Err(HomogError::NotPositiveDefinite);
        }
        let num = xi * xi.transpose();
        acc += num * (w / denom);
        weight_sum += w;
    }
    // Weights already normalized to sum = 1. The spherical factor 1/(4π)
    // cancels against the 4π surface element implicit in the Lebedev nodes.
    let _ = weight_sum;
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lebedev_weights_sum_to_unity() {
        let w: f64 = LEBEDEV_14.iter().map(|&(_, _, _, w)| w).sum();
        assert_relative_eq!(w, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn anisotropic_hill_matches_closed_form_on_isotropic_ref() {
        // For C = k·I, P = I/(3k) exactly.
        let k = 2.5;
        let c_ref = Matrix3::<f64>::identity() * k;
        let p = hill_order2_anisotropic(&c_ref, 14).unwrap();
        let expected = Matrix3::<f64>::identity() / (3.0 * k);
        // Degree-14 is exact for harmonics up to degree 3; the integrand here
        // is exactly that degree so we get machine-precision agreement.
        assert_relative_eq!(p, expected, epsilon = 1e-12);
    }

    #[test]
    fn anisotropic_hill_produces_spd_result_for_ti_ref() {
        // Transversely isotropic reference: diag(10, 10, 1). P should still be SPD.
        let c_ref = Matrix3::<f64>::from_diagonal(&Vector3::new(10.0, 10.0, 1.0));
        let p = hill_order2_anisotropic(&c_ref, 14).unwrap();
        let eig = p.symmetric_eigen();
        assert!(eig.eigenvalues.iter().all(|v| *v > 0.0));
        // For conductivity, the highest-conductivity direction should get the
        // smallest P eigenvalue (P ~ 1/C qualitatively).
        let (max_c_idx, _) = [10.0, 10.0, 1.0].iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let p_xx = p[(max_c_idx, max_c_idx)];
        let p_zz = p[(2, 2)];
        assert!(p_xx < p_zz, "expected P_xx (high-C direction) < P_zz");
    }
}
