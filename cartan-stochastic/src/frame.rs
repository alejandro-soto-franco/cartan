//! Orthonormal frames on Riemannian manifolds.
//!
//! An orthonormal frame at `p ∈ M` is a list of tangent vectors
//! `(e_1, …, e_n) ⊂ T_p M` with `<e_i, e_j>_p = δ_ij`, where `n = dim M`
//! is the intrinsic dimension. The set of all such frames over all points
//! of `M` is the orthonormal frame bundle `O(M)`, a principal `O(n)`-bundle.
//!
//! This file represents a single frame in extrinsic coordinates (a list of
//! `Manifold::Tangent`) and provides Gram-Schmidt orthonormalisation against
//! the Riemannian metric. The bundle structure itself is not materialised;
//! callers carry around a point `p` plus its frame `r`.

use cartan_core::{Manifold, Real};
use rand::Rng;

use crate::error::StochasticError;

/// An orthonormal basis of `T_p M` expressed in extrinsic ambient coordinates.
///
/// The length of `basis` equals the intrinsic dimension of the manifold.
/// Each element lies in `T_p M` (i.e. `check_tangent` passes) and the
/// collection is orthonormal with respect to the Riemannian inner product at `p`.
#[derive(Debug, Clone)]
pub struct OrthonormalFrame<M: Manifold> {
    /// The basis vectors, length `dim(M)`, each in `T_p M`.
    pub basis: Vec<M::Tangent>,
}

impl<M: Manifold> OrthonormalFrame<M> {
    /// Number of basis vectors (= intrinsic manifold dimension).
    #[inline]
    pub fn len(&self) -> usize {
        self.basis.len()
    }

    /// Whether the frame has zero basis vectors. Always false for well-formed
    /// frames on positive-dimensional manifolds.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.basis.is_empty()
    }

    /// Construct a frame from a pre-existing basis without orthonormalising.
    ///
    /// The caller asserts that the supplied basis is already orthonormal in
    /// `T_p M`. Use `from_candidates` instead if the basis is only a spanning
    /// set and needs Gram-Schmidt.
    pub fn from_orthonormal(basis: Vec<M::Tangent>) -> Self {
        Self { basis }
    }

    /// Build an orthonormal frame from a candidate spanning set via modified
    /// Gram-Schmidt against the Riemannian metric `<·,·>_p`.
    ///
    /// The candidate list must have length at least `dim(M)`. The first
    /// `dim(M)` vectors that survive orthonormalisation are retained; any
    /// excess candidates are discarded (they are linearly dependent on the
    /// preceding ones up to numerical tolerance).
    ///
    /// Returns an error if fewer than `dim(M)` linearly independent tangent
    /// vectors can be extracted — i.e. the candidate set does not span `T_p M`.
    pub fn from_candidates(
        manifold: &M,
        p: &M::Point,
        candidates: Vec<M::Tangent>,
        tol: Real,
    ) -> Result<Self, StochasticError> {
        let n = manifold.dim();
        if candidates.len() < n {
            return Err(StochasticError::FrameDimMismatch {
                expected: n,
                got: candidates.len(),
            });
        }
        let mut basis: Vec<M::Tangent> = Vec::with_capacity(n);
        for v in candidates {
            if basis.len() == n {
                break;
            }
            // Project v onto T_p M, then subtract components along existing
            // basis vectors (modified Gram-Schmidt, numerically stabler than
            // classical Gram-Schmidt when the spanning set is ill-conditioned).
            let mut u = manifold.project_tangent(p, &v);
            for e in &basis {
                let c = manifold.inner(p, &u, e);
                u = u - e.clone() * c;
            }
            let norm = manifold.norm(p, &u);
            if norm < tol {
                // This candidate is linearly dependent on the existing basis;
                // skip it and try the next candidate rather than failing,
                // since the caller may have supplied extras deliberately.
                continue;
            }
            basis.push(u * (1.0 / norm));
        }
        if basis.len() < n {
            return Err(StochasticError::GramSchmidtRankDeficient {
                index: basis.len(),
                norm: 0.0,
                threshold: tol,
            });
        }
        Ok(Self { basis })
    }

    /// Re-orthonormalise in place. Call this after parallel-transporting a
    /// frame along a discrete step: discretisation and projection drift
    /// cause the transported basis to slowly lose orthonormality.
    pub fn reorthonormalize(
        &mut self,
        manifold: &M,
        p: &M::Point,
        tol: Real,
    ) -> Result<(), StochasticError> {
        let taken = std::mem::take(&mut self.basis);
        let new = Self::from_candidates(manifold, p, taken, tol)?;
        self.basis = new.basis;
        Ok(())
    }
}

/// Sample a random orthonormal frame at `p`.
///
/// Draws `2 · dim(M)` i.i.d. Gaussian tangent vectors (the oversampling is
/// insurance against near-linear dependence in the first `dim(M)` draws) and
/// orthonormalises via Gram-Schmidt.
pub fn random_frame_at<M: Manifold, R: Rng>(
    manifold: &M,
    p: &M::Point,
    rng: &mut R,
) -> Result<OrthonormalFrame<M>, StochasticError> {
    let n = manifold.dim();
    let candidates: Vec<M::Tangent> = (0..(2 * n))
        .map(|_| manifold.random_tangent(p, rng))
        .collect();
    OrthonormalFrame::from_candidates(manifold, p, candidates, 1e-10)
}
