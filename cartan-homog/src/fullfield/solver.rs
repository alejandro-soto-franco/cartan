//! Sparse linear solver for the cell problem.
//!
//! Three preconditioners, escalating cost/power:
//!   - Jacobi (diagonal): cheapest, good for smooth problems.
//!   - ILU(0) (incomplete LU with zero fill): 3-10x fewer CG iterations for
//!     piecewise-constant conductivity problems. Matches the sparsity of A.
//!   - Dense LU fallback: O(n^3) direct solve when iterative methods stall
//!     (used for strongly anisotropic / near-singular periodic systems).

use crate::error::HomogError;
use nalgebra::{DMatrix, DVector};
use sprs::CsMat;
use alloc::vec::Vec;

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

/// ILU(0) incomplete LU factorisation with zero fill-in. Stores L and U as
/// dense row arrays for simplicity (the factorisation preserves A's sparsity
/// but we don't bother compressing). Call `apply_ilu0` for the back-solve.
pub struct Ilu0 {
    /// L + U combined: L below diagonal (L's diag is 1 implicitly),
    /// diagonal is `U[i,i]`, above diagonal is `U[i,j]`.
    pub data: Vec<Vec<(usize, f64)>>,  // row i -> sorted (col, val) pairs
    pub n: usize,
}

impl Ilu0 {
    pub fn factor(a: &CsMat<f64>) -> Self {
        let n = a.rows();
        // Build per-row sparse representation from A.
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (&val, (i, j)) in a.iter() {
            rows[i].push((j, val));
        }
        for row in rows.iter_mut() {
            row.sort_by_key(|&(j, _)| j);
            // Collapse duplicate (i, j) entries if any (CSC may hold multiples from
            // assembly); add values to coalesce.
            let mut collapsed: Vec<(usize, f64)> = Vec::with_capacity(row.len());
            for &(j, v) in row.iter() {
                if let Some(last) = collapsed.last_mut() {
                    if last.0 == j { last.1 += v; continue; }
                }
                collapsed.push((j, v));
            }
            *row = collapsed;
        }

        // ILU(0): for i = 0..n, for k = 0..i with (i, k) != 0:
        //   A[i, k] /= A[k, k]
        //   for j = k+1..n with (i, j) != 0 and (k, j) != 0:
        //     A[i, j] -= A[i, k] * A[k, j]
        for i in 0..n {
            // Sparse access to row i; we modify in place.
            let mut i_row = core::mem::take(&mut rows[i]);
            for idx_k in 0..i_row.len() {
                let (k, _) = i_row[idx_k];
                if k >= i { break; }  // only pre-diagonal entries participate in the elimination loop

                // Find A[k, k] for scaling.
                let a_kk = rows[k].iter().find(|&&(j, _)| j == k)
                    .map(|&(_, v)| v)
                    .filter(|v| v.abs() > 1e-30);
                let a_kk = match a_kk {
                    Some(v) => v,
                    None => continue,   // null pivot: skip, ILU(0) accepts regularisation upstream
                };
                i_row[idx_k].1 /= a_kk;
                let a_ik = i_row[idx_k].1;

                // Update A[i, j] -= A[i, k] * A[k, j] for j > k where (i, j) already nonzero.
                for (j, val) in i_row.iter_mut() {
                    if *j <= k { continue; }
                    if let Some(&(_, a_kj)) = rows[k].iter().find(|&&(col, _)| col == *j) {
                        *val -= a_ik * a_kj;
                    }
                }
            }
            rows[i] = i_row;
        }

        Ilu0 { data: rows, n }
    }

    /// Apply M^{-1} r = (LU)^{-1} r via forward-sub then back-sub.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = self.n;
        // Forward substitution: L y = r (L has unit diagonal).
        let mut y = DVector::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = r[i];
            for &(j, v) in &self.data[i] {
                if j < i { sum -= v * y[j]; }
                else { break; }
            }
            y[i] = sum;
        }
        // Back substitution: U x = y.
        let mut x = DVector::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y[i];
            let mut diag = 0.0;
            for &(j, v) in &self.data[i] {
                if j < i { continue; }
                if j == i { diag = v; continue; }
                sum -= v * x[j];
            }
            if diag.abs() > 1e-30 { x[i] = sum / diag; } else { x[i] = 0.0; }
        }
        x
    }
}

/// ILU(0)-preconditioned conjugate-gradient solve.
pub fn pcg_ilu0(
    a: &CsMat<f64>, b: &DVector<f64>, tol: f64, max_iter: usize,
) -> Result<(DVector<f64>, usize, f64), HomogError> {
    let ilu = Ilu0::factor(a);
    let apply_a = |v: &DVector<f64>| -> DVector<f64> {
        let mut out = DVector::<f64>::zeros(v.len());
        for (&val, (i, j)) in a.iter() { out[i] += val * v[j]; }
        out
    };
    let n = b.len();
    let mut x = DVector::<f64>::zeros(n);
    let mut r = b - apply_a(&x);
    let b_norm = b.norm().max(1e-30);
    let mut z = ilu.apply(&r);
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
        z = ilu.apply(&r);
        let rz_new = r.dot(&z);
        let beta = rz_new / rz_old;
        p = &z + beta * &p;
        rz_old = rz_new;
    }
    let final_residual = (b - apply_a(&x)).norm() / b_norm;
    Err(HomogError::DidNotConverge { iters: max_iter, residual: final_residual })
}

/// Dense LU solve for `A x = b`. Pays a O(n^3) cost; use for small n or when
/// PCG fails to converge due to high contrast / periodic conditioning.
pub fn solve_dense_lu(a: &CsMat<f64>, b: &DVector<f64>) -> Result<DVector<f64>, HomogError> {
    let n = b.len();
    let mut dense = DMatrix::<f64>::zeros(n, n);
    for (&val, (i, j)) in a.iter() {
        dense[(i, j)] += val;
    }
    let lu = dense.lu();
    lu.solve(b).ok_or_else(|| HomogError::Solver(alloc::string::String::from(
        "dense LU solve failed: matrix is singular or near-singular")))
}

/// Solve ladder: Jacobi-PCG -> ILU(0)-PCG -> dense LU.
/// Each step is tried in turn with the given `tol`/`max_iter` for the CG passes.
pub fn solve_with_fallback(
    a: &CsMat<f64>, b: &DVector<f64>, tol: f64, max_iter: usize,
) -> Result<(DVector<f64>, usize, f64), HomogError> {
    match pcg_jacobi(a, b, tol, max_iter) {
        Ok(t) => return Ok(t),
        Err(HomogError::DidNotConverge { .. }) => {}
        Err(e) => return Err(e),
    }
    match pcg_ilu0(a, b, tol, max_iter) {
        Ok(t) => Ok(t),
        Err(HomogError::DidNotConverge { iters, residual }) => {
            let x = solve_dense_lu(a, b)?;
            Ok((x, iters, residual))
        }
        Err(e) => Err(e),
    }
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

    #[test]
    fn ilu0_pcg_solves_small_tridiagonal() {
        // Tridiag [2, -1, -1, 2, -1, ..., -1, 2] is SPD; ILU(0) is exact here.
        let n = 10;
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, 2.0);
            if i > 0 { tri.add_triplet(i, i - 1, -1.0); tri.add_triplet(i - 1, i, -1.0); }
        }
        let a = tri.to_csc();
        let b = DVector::from_vec(alloc::vec![1.0; n]);
        let (x, iters, res) = pcg_ilu0(&a, &b, 1e-12, 50).unwrap();
        assert!(res < 1e-10, "ILU-PCG residual {res}");
        assert!(iters < n, "ILU-PCG should converge in <n iters, got {iters}");
        // Reconstruct A*x and compare with b.
        let mut ax = DVector::<f64>::zeros(n);
        for (&val, (i, j)) in a.iter() { ax[i] += val * x[j]; }
        for i in 0..n { assert!((ax[i] - b[i]).abs() < 1e-8); }
    }
}
