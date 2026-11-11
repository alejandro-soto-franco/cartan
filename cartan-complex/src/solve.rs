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
    /// `|b - A y| / |b|`, recomputed from the final `y`.
    pub(crate) relative_residual: f64,
    /// Whether that residual is within the tolerance.
    pub(crate) converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out = B2^T B2 y`, with `scratch` sized to the edges.
fn normal_apply(b2: &SignedCsr, b2t: &SignedCsr, y: &[f64], scratch: &mut [f64], out: &mut [f64]) {
    b2.apply(y, scratch);
    b2t.apply(scratch, out);
}

/// Iterations without a new minimum of the recursive residual after which a
/// pass is taken to have stagnated.
const STAGNATION_WINDOW: usize = 50;

/// Relative residual below which no pass iterates further. Steps taken at the
/// rounding floor of a semidefinite system amplify rounding error along its
/// kernel instead of reducing the residual.
const RESIDUAL_FLOOR: f64 = 1e-16;

/// Solves `B2^T B2 y = b` by conjugate gradient from `y = 0`. The system is
/// positive semidefinite and consistent, since `b` lies in the range of `B2^T`,
/// so the iteration converges to a solution and `B2 y` is the projection the
/// caller wants.
///
/// The recursive residual drifts from the true one, so each pass ends by
/// recomputing the true residual and, if it is still above the tolerance and
/// still falling, restarts from the current `y`, at most five passes. A pass
/// also ends when the recursive residual has not reached a new minimum for
/// `STAGNATION_WINDOW` iterations, which happens once it sits at the rounding
/// floor. The reported residual is always the true one of the returned `y`.
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
            converged: true,
        };
    }
    let mut scratch = vec![0.0; complex.n_edges()];
    let mut ay = vec![0.0; t];
    let true_residual = |y: &[f64], scratch: &mut [f64], ay: &mut [f64]| -> (Vec<f64>, f64) {
        normal_apply(b2, b2t, y, scratch, ay);
        let r: Vec<f64> = b.iter().zip(ay.iter()).map(|(bi, ai)| bi - ai).collect();
        let n = dot(&r, &r).sqrt() / bnorm;
        (r, n)
    };
    let mut iterations = 0;
    let mut previous = f64::INFINITY;
    for _pass in 0..5 {
        let (mut r, res) = true_residual(y, &mut scratch, &mut ay);
        if res <= tol || res >= previous || iterations >= max_iter {
            break;
        }
        previous = res;
        let mut p = r.clone();
        let mut ap = vec![0.0; t];
        let mut rr = dot(&r, &r);
        let (mut best_rr, mut since_best) = (rr, 0);
        let stop = tol.max(RESIDUAL_FLOOR) * bnorm;
        while iterations < max_iter && rr.sqrt() > stop && since_best < STAGNATION_WINDOW {
            normal_apply(b2, b2t, &p, &mut scratch, &mut ap);
            let pap = dot(&p, &ap);
            if !(pap.is_finite() && pap > 0.0) {
                break;
            }
            let alpha = rr / pap;
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
            if rr < best_rr {
                best_rr = rr;
                since_best = 0;
            } else {
                since_best += 1;
            }
        }
    }
    let (_, relative_residual) = true_residual(y, &mut scratch, &mut ay);
    CgReport {
        iterations,
        relative_residual,
        converged: relative_residual <= tol,
    }
}
