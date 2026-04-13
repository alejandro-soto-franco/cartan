//! Symmetric Positive Definite manifold SPD(N) with the **Bures-Wasserstein** metric.
//!
//! This is a second Riemannian structure on the SPD cone, distinct from the
//! affine-invariant metric in [`crate::spd`]. It is the quotient geometry of
//! the general linear group `GL(N)` by the orthogonal group `O(N)` acting on
//! the right, and coincides with the `L^2`-Wasserstein metric restricted to
//! centred Gaussian measures. It is the natural geometry for covariance
//! interpolation under optimal transport (McCann, Brenier) and for
//! vol-surface state spaces where the metric should penalise outright
//! movement of the distribution rather than relative re-parameterisation.
//!
//! ## Metric
//!
//! For `P ∈ SPD(N)` and `U, V ∈ T_P SPD(N) = Sym(N)`:
//!
//! ```text
//! <U, V>_P^{BW} = (1/2) tr(L_P[U] · V)
//! ```
//!
//! where `L_P[U]` is the solution of the Lyapunov equation
//! `P · L + L · P = U`. The inverse of `L_P` is `V ↦ PV + VP`, so equivalently:
//!
//! ```text
//! <U, V>_P^{BW} = (1/2) tr(U · L_P[V])
//! ```
//!
//! ## Geodesics
//!
//! The geodesic from `P` with initial velocity `U ∈ T_P SPD(N)` is
//!
//! ```text
//! γ(t) = (I + t · L_P[U])^T · P · (I + t · L_P[U]).
//! ```
//!
//! ## Exponential and logarithm
//!
//! ```text
//! Exp_P(U)  =  (I + L_P[U])^T · P · (I + L_P[U]),
//! Log_P(Q)  =  P · (T - I)  +  (T - I) · P,    where T = P^{-1/2} (P^{1/2} Q P^{1/2})^{1/2} P^{-1/2}.
//! ```
//!
//! `T` is the McCann-Brenier optimal transport map from the centred Gaussian
//! `N(0, P)` to `N(0, Q)` evaluated on the underlying R^N — a linear map
//! represented by a symmetric matrix. `Log_P` is well-defined globally:
//! Bures-Wasserstein SPD is geodesically complete and the exponential map
//! is a global diffeomorphism (no cut locus).
//!
//! ## Distance
//!
//! ```text
//! d_{BW}(P, Q)^2 = tr(P) + tr(Q) - 2 · tr((P^{1/2} · Q · P^{1/2})^{1/2}).
//! ```
//!
//! This is the squared 2-Wasserstein distance between `N(0, P)` and `N(0, Q)`.
//!
//! ## Curvature
//!
//! Bures-Wasserstein SPD has **non-negative** sectional curvature bounded
//! above by a dimensional constant (Takatsu, 2011). Contrast with the
//! affine-invariant metric which has non-positive curvature. Ricci and
//! sectional curvature impls are intentionally deferred pending a
//! Bismut-Elworthy-Li consumer that actually reads them.
//!
//! ## References
//!
//! - Bhatia, R., Jain, T., Lim, Y. "On the Bures-Wasserstein distance between
//!   positive definite matrices." *Expositiones Mathematicae* 37(2), 2019.
//! - Malagò, L., Montrucchio, L., Pistone, G. "Wasserstein Riemannian geometry
//!   of Gaussian densities." *Information Geometry* 1, 2018.
//! - Takatsu, A. "Wasserstein geometry of Gaussian measures." *Osaka J. Math.*
//!   48(4), 2011. (Curvature characterisation.)

use cartan_core::{CartanError, Manifold, Real, Retraction, VectorTransport};
use nalgebra::{DMatrix, SMatrix};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::util::sym::{sym_sqrt, sym_sqrt_inv, sym_symmetrize};

/// SPD(N) equipped with the Bures-Wasserstein (optimal-transport) metric.
///
/// Use this in preference to [`crate::Spd`] when the modelling intent is
/// "distance between Gaussian measures" rather than "affine-invariant
/// geometry on covariance matrices." Typical applications: vol-surface
/// regime interpolation, covariance transport under McCann optimal
/// transport, model averaging with a Wasserstein barycentre.
#[derive(Debug, Clone, Copy)]
pub struct SpdBuresWasserstein<const N: usize>;

// ─────────────────────────────────────────────────────────────────────────────
// Lyapunov solver
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the symmetric Lyapunov equation `P · X + X · P = U` for `X ∈ Sym(N)`.
///
/// Uses the eigendecomposition `P = V Λ V^T` (P is SPD, so Λ > 0). In the
/// eigenbasis the equation becomes `(λ_i + λ_j) X̃_ij = Ũ_ij`, which is solved
/// elementwise. Returns the solution in the ambient basis.
///
/// Panics if any eigenvalue of `P` is non-positive (should not happen when
/// `P` is genuinely SPD; caller is responsible for upstream validation).
fn solve_lyapunov_sym<const N: usize>(
    p: &SMatrix<Real, N, N>,
    u: &SMatrix<Real, N, N>,
) -> SMatrix<Real, N, N> {
    // Eigendecompose P = V Λ V^T via DMatrix (symmetric_eigen only defined for
    // specific const N in SMatrix; DMatrix path is generic over N).
    let dp = DMatrix::from_column_slice(N, N, p.as_slice());
    let eig = dp.symmetric_eigen();
    let lambda = &eig.eigenvalues;
    let v = &eig.eigenvectors;

    // U_tilde = V^T · U · V.
    let du = DMatrix::from_column_slice(N, N, u.as_slice());
    let u_tilde = v.transpose() * du * v;

    // X_tilde_ij = U_tilde_ij / (λ_i + λ_j).
    let mut x_tilde = u_tilde.clone();
    for i in 0..N {
        for j in 0..N {
            x_tilde[(i, j)] /= lambda[i] + lambda[j];
        }
    }

    // X = V · X_tilde · V^T, then copy back to SMatrix and symmetrise.
    let x_dm = v * x_tilde * v.transpose();
    let mut x_sm = SMatrix::<Real, N, N>::zeros();
    for i in 0..N {
        for j in 0..N {
            x_sm[(i, j)] = x_dm[(i, j)];
        }
    }
    sym_symmetrize(&x_sm)
}

// ─────────────────────────────────────────────────────────────────────────────
// Manifold
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Manifold for SpdBuresWasserstein<N> {
    type Point = SMatrix<Real, N, N>;
    type Tangent = SMatrix<Real, N, N>;

    fn dim(&self) -> usize {
        N * (N + 1) / 2
    }

    fn ambient_dim(&self) -> usize {
        N * N
    }

    fn injectivity_radius(&self, _p: &Self::Point) -> Real {
        // Bures-Wasserstein SPD is a geodesically complete Alexandrov space
        // of non-negative curvature; exp is a global diffeomorphism on the
        // SPD cone. No cut locus.
        Real::INFINITY
    }

    fn inner(&self, p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        // <U, V>_P = (1/2) tr(L_P[U] · V).
        let l_u = solve_lyapunov_sym(p, u);
        0.5 * (l_u * v).trace()
    }

    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        // Exp_P(V) = (I + L_P[V])^T · P · (I + L_P[V]).
        let l_v = solve_lyapunov_sym(p, v);
        let a = SMatrix::<Real, N, N>::identity() + l_v;
        let result = a.transpose() * p * a;
        sym_symmetrize(&result)
    }

    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> {
        // T = P^{-1/2} (P^{1/2} · Q · P^{1/2})^{1/2} · P^{-1/2}.
        let sqrt_p = sym_sqrt(p);
        let sqrt_p_inv = sym_sqrt_inv(p);
        let inner = sqrt_p * q * sqrt_p;
        let inner_sqrt = sym_sqrt(&inner);
        let t = sqrt_p_inv * inner_sqrt * sqrt_p_inv;
        // Log_P(Q) = P · (T - I) + (T - I) · P.
        let t_minus_i = t - SMatrix::<Real, N, N>::identity();
        let result = p * t_minus_i + t_minus_i * p;
        Ok(sym_symmetrize(&result))
    }

    fn project_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        // The tangent space at any P is Sym(N); projection = symmetrisation.
        sym_symmetrize(v)
    }

    fn project_point(&self, p: &Self::Point) -> Self::Point {
        // Project onto SPD cone: eigenclamp negative eigenvalues to a small
        // positive floor. Reuses the same approach as [`crate::Spd`].
        let dp = DMatrix::from_column_slice(N, N, p.as_slice());
        let eig = dp.symmetric_eigen();
        let floor: Real = 1e-12;
        let clamped: DMatrix<Real> = {
            let mut v = eig.eigenvalues.clone();
            for lam in v.iter_mut() {
                if *lam < floor {
                    *lam = floor;
                }
            }
            let diag = DMatrix::from_diagonal(&v);
            &eig.eigenvectors * diag * eig.eigenvectors.transpose()
        };
        let mut out = SMatrix::<Real, N, N>::zeros();
        for i in 0..N {
            for j in 0..N {
                out[(i, j)] = clamped[(i, j)];
            }
        }
        sym_symmetrize(&out)
    }

    fn zero_tangent(&self, _p: &Self::Point) -> Self::Tangent {
        SMatrix::<Real, N, N>::zeros()
    }

    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError> {
        // Symmetry.
        let asym = (p - p.transpose()).norm();
        if asym > 1e-8 {
            return Err(CartanError::NotOnManifold {
                constraint: "P = P^T".into(),
                violation: asym,
            });
        }
        // Positive definiteness via minimum eigenvalue.
        let dp = DMatrix::from_column_slice(N, N, p.as_slice());
        let min_eig = dp
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .cloned()
            .fold(Real::INFINITY, Real::min);
        if min_eig <= 0.0 {
            return Err(CartanError::NotOnManifold {
                constraint: "P > 0 (min eigenvalue)".into(),
                violation: -min_eig,
            });
        }
        Ok(())
    }

    fn check_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> {
        let asym = (v - v.transpose()).norm();
        if asym > 1e-8 {
            return Err(CartanError::NotInTangentSpace {
                constraint: "V = V^T".into(),
                violation: asym,
            });
        }
        Ok(())
    }

    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        // Wishart-like: draw a Gaussian matrix G and return G G^T / N + ε I.
        let mut g = SMatrix::<Real, N, N>::zeros();
        for i in 0..N {
            for j in 0..N {
                g[(i, j)] = StandardNormal.sample(rng);
            }
        }
        let gg = g * g.transpose();
        let eye = SMatrix::<Real, N, N>::identity();
        sym_symmetrize(&(gg * (1.0 / N as Real) + eye * 1e-3))
    }

    fn random_tangent<R: Rng>(&self, _p: &Self::Point, rng: &mut R) -> Self::Tangent {
        // Symmetric Gaussian (Wigner-like): (G + G^T) / √2.
        let mut g = SMatrix::<Real, N, N>::zeros();
        for i in 0..N {
            for j in 0..N {
                g[(i, j)] = StandardNormal.sample(rng);
            }
        }
        let sym = (g + g.transpose()) * (0.5 * Real::sqrt(2.0));
        sym_symmetrize(&sym)
    }
}

impl<const N: usize> Retraction for SpdBuresWasserstein<N> {
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        // For SPD cone the exponential is already a closed-form polynomial
        // in the Lyapunov operator, cheap enough to use as the retraction.
        self.exp(p, v)
    }

    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        // Exact inverse: retraction equals exp, so inverse retraction equals log.
        self.log(p, q)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector transport (differentiated retraction)
// ─────────────────────────────────────────────────────────────────────────────

/// Differentiated-retraction vector transport.
///
/// Parallel transport in the Bures-Wasserstein geometry does not admit a
/// clean closed form for general `N` (unlike the affine-invariant metric).
/// We expose the weaker `VectorTransport` instead, computed as the Fréchet
/// derivative of the retraction:
///
/// ```text
/// T_{P, V}(U) = (d/ds) Exp_P(V + s · U) |_{s=0}.
/// ```
///
/// For the Bures-Wasserstein exponential `Exp_P(V) = (I + L_P[V])^T · P · (I + L_P[V])`,
/// the directional derivative in the direction `U` works out to
///
/// ```text
/// T_{P, V}(U) = L_P[U] · P · (I + L_P[V]) + (I + L_P[V])^T · P · L_P[U].
/// ```
///
/// This lands in `Sym(N) = T_{Exp_P(V)} SPD(N)` by symmetry of the
/// expression. It satisfies the vector-transport axioms (linearity in `U`,
/// identity at `V = 0`) and is first-order accurate as an approximation to
/// exact parallel transport. Note that it is **not** an isometry in
/// general — downstream algorithms that need isometric transport (e.g. exact
/// conjugate-gradient beta) should resort to Schild's ladder or pole ladder.
///
/// The critical consumer in this stack is `cartan_stochastic::stratonovich_step`,
/// which only needs vector transport + Gram-Schmidt re-orthonormalisation
/// and so tolerates the non-isometry: the re-orthonormalisation absorbs the
/// discretisation and isometry drift at each step.
impl<const N: usize> VectorTransport for SpdBuresWasserstein<N> {
    fn vector_transport(
        &self,
        p: &Self::Point,
        direction: &Self::Tangent,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        let l_v = solve_lyapunov_sym(p, direction);
        let l_u = solve_lyapunov_sym(p, u);
        let a = SMatrix::<Real, N, N>::identity() + l_v;
        let result = l_u * p * a + a.transpose() * p * l_u;
        Ok(sym_symmetrize(&result))
    }
}

/// Bures-Wasserstein squared distance.
///
/// `d_{BW}(P, Q)^2 = tr(P) + tr(Q) - 2 · tr((P^{1/2} · Q · P^{1/2})^{1/2})`.
///
/// Provided as a standalone function because the default `Manifold::dist`
/// implementation routes through `log` + `norm`, which requires two
/// eigendecompositions and a Lyapunov solve. This closed form uses only the
/// square-root eigendecompositions and is faster.
pub fn bw_distance_sq<const N: usize>(
    p: &SMatrix<Real, N, N>,
    q: &SMatrix<Real, N, N>,
) -> Real {
    let sqrt_p = sym_sqrt(p);
    let inner = sqrt_p * q * sqrt_p;
    let inner_sqrt = sym_sqrt(&inner);
    p.trace() + q.trace() - 2.0 * inner_sqrt.trace()
}
