//! Wishart Brownian motion on the SPD cone.
//!
//! A Wishart process `X_t ∈ SPD(N)` is the canonical SPD-valued diffusion
//! with Itô dynamics
//!
//! ```text
//! dX_t = √X_t · dB_t + dB_t^T · √X_t  +  n · I · dt,
//! ```
//!
//! where `B_t ∈ R^{N×N}` is a standard matrix Brownian motion (entries i.i.d.
//! `N(0, 1)`) and `n` is the shape parameter (for the classical Wishart this
//! equals `N`). Under the affine-invariant metric the drift term is the
//! Christoffel correction that makes `X_t` a genuine Brownian motion on
//! `SPD(N)` with the Laplace-Beltrami generator; under the Bures-Wasserstein
//! metric the same SDE is a Riemannian Brownian motion up to a tangent-space
//! re-parameterisation.
//!
//! This helper is purely *stochastic* — it does not assume which Riemannian
//! structure the SPD cone is equipped with. For metric-consistent BM on
//! `Spd<N>` (affine-invariant) or `SpdBuresWasserstein<N>`, combine the
//! generic [`crate::stochastic_development`] with the chosen manifold.
//!
//! ## References
//!
//! - Bru, M.-F. "Wishart processes." *J. Theor. Probab.* 4, 1991.
//! - Graczyk, P., Malecki, J. "Multidimensional Yamada-Watanabe theorem and
//!   its applications to particle systems." *J. Math. Phys.* 54, 2013.

use cartan_core::Real;
use nalgebra::SMatrix;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// One Itô-Euler step of the Wishart process.
///
/// Given an SPD matrix `x` and a time increment `dt`, samples the next
/// state along the Wishart SDE with shape parameter `shape`.
///
/// The step is *not* projected back onto the SPD cone: for short `dt` the
/// drift term `n · I · dt` keeps the process in the interior of the cone
/// almost surely, but near the boundary (low eigenvalues) the caller may
/// wish to project with `Manifold::project_point`.
pub fn wishart_step<const N: usize, R: Rng + ?Sized>(
    x: &SMatrix<Real, N, N>,
    shape: Real,
    dt: Real,
    rng: &mut R,
) -> SMatrix<Real, N, N>
where
    StandardNormal: Distribution<Real>,
{
    // Draw matrix Brownian increment dB ∈ R^{N×N}, entries i.i.d. N(0, dt).
    let sqrt_dt = dt.sqrt();
    let mut db = SMatrix::<Real, N, N>::zeros();
    for i in 0..N {
        for j in 0..N {
            let z: Real = StandardNormal.sample(rng);
            db[(i, j)] = z * sqrt_dt;
        }
    }

    // √X via eigendecomposition of the symmetric input. Uses nalgebra's
    // DMatrix path to sidestep the Const<N>: ToTypenum requirement on
    // SMatrix::symmetric_eigen.
    let sqrt_x = {
        let dm = nalgebra::DMatrix::from_column_slice(N, N, x.as_slice());
        let eig = dm.symmetric_eigen();
        let floor: Real = 0.0;
        let mut vals = eig.eigenvalues.clone();
        for lam in vals.iter_mut() {
            if *lam < floor {
                *lam = floor;
            }
            *lam = lam.sqrt();
        }
        let diag = nalgebra::DMatrix::from_diagonal(&vals);
        let product = &eig.eigenvectors * diag * eig.eigenvectors.transpose();
        let mut out = SMatrix::<Real, N, N>::zeros();
        for i in 0..N {
            for j in 0..N {
                out[(i, j)] = product[(i, j)];
            }
        }
        out
    };

    // Diffusion: √X · dB + dB^T · √X.
    let diffusion = sqrt_x * db + db.transpose() * sqrt_x;
    // Drift: shape · I · dt.
    let drift = SMatrix::<Real, N, N>::identity() * (shape * dt);

    let mut next = x + diffusion + drift;
    // Enforce exact symmetry; the SDE theoretically preserves it, but
    // rounding accumulates.
    next = (next + next.transpose()) * 0.5;
    next
}
