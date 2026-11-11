//! Linear solves for the Hodge decomposition.

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};

use crate::Complex;
use crate::complex::SignedCsr;

/// Solves `L0 phi = B1 f` with `L0 = B1 B1^T`, by sparse Cholesky on `L0`
/// with the lowest vertex of every component grounded to zero, then shifts
/// `phi` to mean zero on each component. Grounding removes exactly the kernel
/// of `L0`, which is spanned by the component indicators, so the reduced
/// matrix is positive definite.
pub(crate) fn gradient_potential(complex: &Complex, f: &[f64]) -> Vec<f64> {
    let n = complex.n_vertices();
    let comp = complex.components();
    let n_comp = comp.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut grounded = vec![false; n];
    let mut seen = vec![false; n_comp];
    for v in 0..n {
        let c = comp[v] as usize;
        if !seen[c] {
            seen[c] = true;
            grounded[v] = true;
        }
    }
    let mut reduced = vec![usize::MAX; n];
    let mut r = 0;
    for v in 0..n {
        if !grounded[v] {
            reduced[v] = r;
            r += 1;
        }
    }
    let mut phi = vec![0.0; n];
    if r > 0 {
        let mut rhs_full = vec![0.0; n];
        complex.b1(f, &mut rhs_full);
        let mut degree = vec![0.0; r];
        let mut trips = Vec::with_capacity(2 * complex.n_edges() + r);
        for e in complex.edges() {
            let (a, b) = (reduced[e[0] as usize], reduced[e[1] as usize]);
            if a != usize::MAX {
                degree[a] += 1.0;
            }
            if b != usize::MAX {
                degree[b] += 1.0;
            }
            if a != usize::MAX && b != usize::MAX {
                trips.push(Triplet {
                    row: a,
                    col: b,
                    val: -1.0,
                });
                trips.push(Triplet {
                    row: b,
                    col: a,
                    val: -1.0,
                });
            }
        }
        for (i, &d) in degree.iter().enumerate() {
            trips.push(Triplet {
                row: i,
                col: i,
                val: d,
            });
        }
        let l0 = SparseColMat::<usize, f64>::try_new_from_triplets(r, r, &trips)
            .expect("reduced Laplacian assembles");
        let llt = l0
            .sp_cholesky(Side::Lower)
            .expect("grounded Laplacian is positive definite");
        let mut rhs = Mat::<f64>::from_fn(r, 1, |i, _| 0.0 * i as f64);
        for v in 0..n {
            if reduced[v] != usize::MAX {
                rhs[(reduced[v], 0)] = rhs_full[v];
            }
        }
        llt.solve_in_place(rhs.as_mut());
        for v in 0..n {
            if reduced[v] != usize::MAX {
                phi[v] = rhs[(reduced[v], 0)];
            }
        }
    }
    let mut sum = vec![0.0; n_comp];
    let mut count = vec![0usize; n_comp];
    for v in 0..n {
        sum[comp[v] as usize] += phi[v];
        count[comp[v] as usize] += 1;
    }
    for v in 0..n {
        let c = comp[v] as usize;
        phi[v] -= sum[c] / count[c] as f64;
    }
    phi
}

/// Outcome of a conjugate gradient solve.
pub(crate) struct CgReport {
    pub(crate) iterations: usize,
    /// `|b - A y| / |b|`, recomputed from `y` at the end.
    pub(crate) relative_residual: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out = B2^T B2 y`, with `scratch` sized to the edges.
fn normal_apply(b2: &SignedCsr, b2t: &SignedCsr, y: &[f64], scratch: &mut [f64], out: &mut [f64]) {
    b2.apply(y, scratch);
    b2t.apply(scratch, out);
}

/// Solves `B2^T B2 y = b` by conjugate gradient from `y = 0`. The system is
/// positive semidefinite and consistent, since `b` lies in the range of `B2^T`,
/// so the iteration converges to a solution and `B2 y` is the projection the
/// caller wants. The recursive residual drifts from the true one, so on
/// reaching the tolerance the true residual is recomputed and the iteration
/// restarts from the current `y` if it is still above it, at most four times.
pub(crate) fn curl_cg(
    complex: &Complex,
    b: &[f64],
    tol: f64,
    max_iter: usize,
    y: &mut [f64],
) -> CgReport {
    let (b2, b2t) = (&complex.b2, &complex.b2t);
    let t = b.len();
    let bnorm = dot(b, b).sqrt();
    if bnorm == 0.0 {
        y.iter_mut().for_each(|v| *v = 0.0);
        return CgReport {
            iterations: 0,
            relative_residual: 0.0,
        };
    }
    let mut scratch = vec![0.0; complex.n_edges()];
    let mut ay = vec![0.0; t];
    let mut iterations = 0;
    let mut best = f64::INFINITY;
    for _restart in 0..5 {
        normal_apply(b2, b2t, y, &mut scratch, &mut ay);
        let mut r: Vec<f64> = b.iter().zip(&ay).map(|(bi, ai)| bi - ai).collect();
        let true_res = dot(&r, &r).sqrt() / bnorm;
        if true_res <= tol || true_res >= best {
            best = best.min(true_res);
            break;
        }
        best = true_res;
        let mut p = r.clone();
        let mut ap = vec![0.0; t];
        let mut rr = dot(&r, &r);
        while iterations < max_iter && rr.sqrt() > tol * bnorm {
            normal_apply(b2, b2t, &p, &mut scratch, &mut ap);
            let alpha = rr / dot(&p, &ap);
            for i in 0..t {
                y[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rr_new = dot(&r, &r);
            let beta = rr_new / rr;
            for i in 0..t {
                p[i] = r[i] + beta * p[i];
            }
            rr = rr_new;
            iterations += 1;
        }
        if iterations >= max_iter {
            normal_apply(b2, b2t, y, &mut scratch, &mut ay);
            let res: f64 = b
                .iter()
                .zip(&ay)
                .map(|(bi, ai)| (bi - ai).powi(2))
                .sum::<f64>()
                .sqrt();
            best = best.min(res / bnorm);
            break;
        }
    }
    CgReport {
        iterations,
        relative_residual: best,
    }
}
