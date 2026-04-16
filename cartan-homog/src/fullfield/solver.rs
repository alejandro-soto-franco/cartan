//! Sparse linear solver for the cell problem.
//!
//! v1: conjugate gradient with Jacobi (diagonal) preconditioner. Adequate for
//! the smooth piecewise-constant conductivity case; degrades near high contrast
//! without AMG. v1.2: swap in algebraic multigrid.

use crate::error::HomogError;
use nalgebra::DVector;
use sprs::CsMat;

/// Preconditioned conjugate-gradient solve for `A x = b`, A symmetric positive definite.
/// Returns (x, iterations, final_residual).
pub fn pcg_jacobi(
    a: &CsMat<f64>, b: &DVector<f64>, tol: f64, max_iter: usize,
) -> Result<(DVector<f64>, usize, f64), HomogError> {
    let n = b.len();
    let diag = {
        let mut d = DVector::<f64>::zeros(n);
        for (&val, (i, j)) in a.iter() {
            if i == j { d[i] = val; }
        }
        d
    };
    // Jacobi preconditioner M^{-1} = diag(1/A_ii).
    let m_inv = |r: &DVector<f64>| -> DVector<f64> {
        let mut z = DVector::<f64>::zeros(r.len());
        for i in 0..r.len() {
            if diag[i].abs() > 1e-30 { z[i] = r[i] / diag[i]; }
        }
        z
    };
    let apply_a = |v: &DVector<f64>| -> DVector<f64> {
        let mut out = DVector::<f64>::zeros(v.len());
        for (&val, (i, j)) in a.iter() {
            out[i] += val * v[j];
        }
        out
    };
    let mut x = DVector::<f64>::zeros(n);
    let mut r = b - apply_a(&x);
    let b_norm = b.norm().max(1e-30);
    let mut z = m_inv(&r);
    let mut p = z.clone();
    let mut rz_old = r.dot(&z);
    for iter in 0..max_iter {
        let ap = apply_a(&p);
        let pap = p.dot(&ap);
        if pap.abs() < 1e-30 { break; }
        let alpha = rz_old / pap;
        x += alpha * &p;
        r -= alpha * &ap;
        let rel = r.norm() / b_norm;
        if rel < tol { return Ok((x, iter + 1, rel)); }
        z = m_inv(&r);
        let rz_new = r.dot(&z);
        let beta = rz_new / rz_old;
        p = &z + beta * &p;
        rz_old = rz_new;
    }
    let final_residual = (b - apply_a(&x)).norm() / b_norm;
    Err(HomogError::DidNotConverge { iters: max_iter, residual: final_residual })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn pcg_solves_diagonal_system() {
        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(1, 1, 3.0);
        tri.add_triplet(2, 2, 4.0);
        let a = tri.to_csc();
        let b = DVector::from_vec(alloc::vec![2.0, 6.0, 12.0]);
        let (x, iters, res) = pcg_jacobi(&a, &b, 1e-12, 100).unwrap();
        assert!(res < 1e-10);
        assert!(iters <= 3);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }
}
