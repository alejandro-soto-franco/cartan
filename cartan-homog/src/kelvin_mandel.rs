//! Kelvin-Mandel helpers: isotropy / transversely-isotropic constructors,
//! spectral queries, and SPD positivity checks. Used by shapes and schemes.

use nalgebra::{Matrix3, SymmetricEigen};

pub fn iso_order2(k: f64) -> Matrix3<f64> {
    Matrix3::<f64>::identity() * k
}

/// Transversely isotropic 2nd-order tensor with symmetry axis = z.
pub fn ti_order2(k_h: f64, k_v: f64) -> Matrix3<f64> {
    let mut m = Matrix3::<f64>::zeros();
    m[(0, 0)] = k_h;
    m[(1, 1)] = k_h;
    m[(2, 2)] = k_v;
    m
}

pub fn eigenvalues_order2(m: &Matrix3<f64>) -> [f64; 3] {
    let sym = (m + m.transpose()) * 0.5;
    let eig = SymmetricEigen::new(sym);
    let mut v = [eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]];
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

pub fn is_spd_order2(m: &Matrix3<f64>, tol: f64) -> bool {
    eigenvalues_order2(m).iter().all(|&e| e > tol)
}

/// Isotropy detector: returns (k_avg, anisotropy_relative) where anisotropy is
/// normalised by average diagonal. Used by closed-form Hill tensors to gate
/// the iso branch vs the Lebedev fallback.
pub fn iso_detect_order2(c: &Matrix3<f64>) -> (f64, f64) {
    let d = c.diagonal();
    let off = (c[(0,1)].powi(2) + c[(0,2)].powi(2) + c[(1,2)].powi(2)).sqrt();
    let avg = (d[0] + d[1] + d[2]) / 3.0;
    let diag_aniso = (d[0] - avg).abs() + (d[1] - avg).abs() + (d[2] - avg).abs();
    let relative = (diag_aniso + off) / avg.abs().max(1e-300);
    (avg, relative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn iso_diagonal() {
        let k = iso_order2(7.5);
        assert_relative_eq!(k[(0, 0)], 7.5);
        assert_relative_eq!(k[(1, 1)], 7.5);
        assert_relative_eq!(k[(2, 2)], 7.5);
        assert_relative_eq!(k[(0, 1)], 0.0);
    }

    #[test]
    fn ti_axis_z() {
        let k = ti_order2(10.0, 1.0);
        let ev = eigenvalues_order2(&k);
        assert_relative_eq!(ev[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(ev[1], 10.0, epsilon = 1e-12);
        assert_relative_eq!(ev[2], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn spd_check_detects_zero_diagonal() {
        let mut m = iso_order2(1.0);
        m[(2, 2)] = 0.0;
        assert!(!is_spd_order2(&m, 1e-10));
    }

    #[test]
    fn iso_detect_flags_ti_as_anisotropic() {
        let k_iso = iso_order2(1.0);
        let (_, aniso_iso) = iso_detect_order2(&k_iso);
        assert!(aniso_iso < 1e-12);

        let k_ti = ti_order2(10.0, 1.0);
        let (_, aniso_ti) = iso_detect_order2(&k_ti);
        assert!(aniso_ti > 0.1);
    }
}
