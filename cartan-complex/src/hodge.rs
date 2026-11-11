//! The Hodge decomposition of an edge flow.
//!
//! An edge flow `f` splits orthogonally as `f = grad + curl + harm`, with
//! `grad = B1^T phi` in the image of `B1^T`, `curl = B2 y` in the image of
//! `B2`, and `harm` in the kernel of both `B1` and `B2^T`. All three inner
//! products are the identity, the unweighted convention of the neighbour
//! graph pipelines this crate serves.

use crate::solve::{curl_cg, gradient_potential};
use crate::{Complex, ComplexError};

/// The three parts of an edge flow and the potential behind the gradient.
#[derive(Debug, Clone)]
pub struct Decomposition {
    /// Vertex potential, mean zero on every component; `grad = B1^T phi`.
    pub phi: Vec<f64>,
    pub grad: Vec<f64>,
    pub curl: Vec<f64>,
    pub harm: Vec<f64>,
    /// Conjugate gradient iterations spent on the curl.
    pub curl_iterations: usize,
    /// `|B2^T (r - B2 y)| / |B2^T r|` at the end, with `r = f - grad`.
    pub curl_residual: f64,
}

/// Controls for the curl solve.
#[derive(Debug, Clone)]
pub struct HodgeOptions {
    pub curl_tol: f64,
    pub curl_max_iter: usize,
}

impl Default for HodgeOptions {
    fn default() -> Self {
        Self {
            curl_tol: 1e-14,
            curl_max_iter: 100_000,
        }
    }
}

/// Decomposes `f`, one value per edge in the complex's sorted edge order.
pub fn decompose(
    complex: &Complex,
    f: &[f64],
    opts: &HodgeOptions,
) -> Result<Decomposition, ComplexError> {
    let m = complex.n_edges();
    if f.len() != m {
        return Err(ComplexError::LengthMismatch {
            what: "flow",
            got: f.len(),
            expected: m,
        });
    }
    let phi = gradient_potential(complex, f);
    let mut grad = vec![0.0; m];
    complex.b1t(&phi, &mut grad);
    let rest: Vec<f64> = f.iter().zip(&grad).map(|(a, b)| a - b).collect();
    let t = complex.n_triangles();
    let mut curl = vec![0.0; m];
    let (mut curl_iterations, mut curl_residual) = (0, 0.0);
    if t > 0 {
        let mut b = vec![0.0; t];
        complex.b2t(&rest, &mut b);
        let mut y = vec![0.0; t];
        let rep = curl_cg(complex, &b, opts.curl_tol, opts.curl_max_iter, &mut y);
        complex.b2(&y, &mut curl);
        curl_iterations = rep.iterations;
        curl_residual = rep.relative_residual;
    }
    let harm = rest.iter().zip(&curl).map(|(a, b)| a - b).collect();
    Ok(Decomposition {
        phi,
        grad,
        curl,
        harm,
        curl_iterations,
        curl_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn norm(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    fn complete(n: u32) -> Vec<[u32; 2]> {
        let mut e = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                e.push([a, b]);
            }
        }
        e
    }
    fn run(c: &Complex, f: &[f64]) -> Decomposition {
        decompose(c, f, &HodgeOptions::default()).unwrap()
    }

    #[test]
    fn pure_gradient_is_recovered() {
        let c = Complex::new(4, complete(4))
            .unwrap()
            .with_triangles(vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            .unwrap();
        let mut f = vec![0.0; 6];
        c.b1t(&[0.0, 1.0, 3.0, -2.0], &mut f);
        let d = run(&c, &f);
        assert!(dist(&d.grad, &f) < 1e-13);
        assert!(norm(&d.curl) < 1e-13 && norm(&d.harm) < 1e-13);
    }

    #[test]
    fn pure_circulation_on_a_cycle_is_harmonic() {
        let edges = vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]];
        let c = Complex::new(6, edges).unwrap();
        // Sorted edges: [0,1],[0,5],[1,2],[2,3],[3,4],[4,5]. Going round
        // 0→1→2→3→4→5→0 traverses [0,5] against its orientation.
        let f = vec![1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
        let d = run(&c, &f);
        assert!(dist(&d.harm, &f) < 1e-13);
        assert!(norm(&d.grad) < 1e-13 && norm(&d.curl) < 1e-13);
    }

    #[test]
    fn filled_triangle_turns_circulation_into_curl() {
        let c = Complex::new(3, complete(3))
            .unwrap()
            .with_triangles(vec![[0, 1, 2]])
            .unwrap();
        // Sorted edges [0,1],[0,2],[1,2]; round 0→1→2→0 is +1, -1, +1.
        let f = vec![1.0, -1.0, 1.0];
        let d = run(&c, &f);
        assert!(dist(&d.curl, &f) < 1e-13);
        assert!(norm(&d.harm) < 1e-13 && norm(&d.grad) < 1e-13);
    }

    #[test]
    fn parts_are_orthogonal_and_sum_to_f() {
        let base = Complex::new(6, complete(6)).unwrap();
        let tri: Vec<[u32; 3]> = crate::triangles_of(&base).into_iter().step_by(3).collect();
        let c = base.with_triangles(tri).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let f: Vec<f64> = (0..c.n_edges())
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let d = run(&c, &f);
        let ff = dot(&f, &f);
        assert!(dot(&d.grad, &d.curl).abs() < 1e-12 * ff);
        assert!(dot(&d.grad, &d.harm).abs() < 1e-12 * ff);
        assert!(dot(&d.curl, &d.harm).abs() < 1e-12 * ff);
        let sum: Vec<f64> = (0..f.len())
            .map(|i| d.grad[i] + d.curl[i] + d.harm[i])
            .collect();
        assert!(dist(&sum, &f) < 1e-14);
        let mut v = vec![0.0; c.n_vertices()];
        c.b1(&d.harm, &mut v);
        assert!(norm(&v) < 1e-11 * ff.sqrt());
        let mut t = vec![0.0; c.n_triangles()];
        c.b2t(&d.harm, &mut t);
        assert!(norm(&t) < 1e-11 * ff.sqrt());
    }

    #[test]
    fn disconnected_graph_is_grounded_per_component() {
        let c = Complex::new(7, vec![[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]]).unwrap();
        let f = vec![0.3, -1.2, 0.7, 2.0, 0.1, -0.4];
        let d = run(&c, &f);
        assert!(d.phi.iter().all(|x| x.is_finite()));
        let m0 = (d.phi[0] + d.phi[1] + d.phi[2]) / 3.0;
        let m1 = (d.phi[3] + d.phi[4] + d.phi[5]) / 3.0;
        assert!(m0.abs() < 1e-14 && m1.abs() < 1e-14);
        assert_eq!(d.phi[6], 0.0);
    }

    #[test]
    fn empty_complex_is_zero() {
        let c = Complex::new(3, vec![]).unwrap();
        let d = run(&c, &[]);
        assert!(d.grad.is_empty() && d.curl.is_empty() && d.harm.is_empty());
        assert_eq!(d.phi, vec![0.0; 3]);
    }

    #[test]
    fn wrong_flow_length_is_an_error() {
        let c = Complex::new(3, complete(3)).unwrap();
        assert!(matches!(
            decompose(&c, &[1.0], &HodgeOptions::default()),
            Err(crate::ComplexError::LengthMismatch { .. })
        ));
    }
}
