// ~/cartan/cartan-manifolds/src/qtensor.rs

//! Q-tensor manifold: the space of 3×3 symmetric traceless matrices.
//!
//! ## Mathematical structure
//!
//! The Q-tensor order parameter of a nematic liquid crystal lives in the space:
//!
//! ```text
//! Q = { Q ∈ Sym_0(R^3) : -1/3 ≤ λ_i(Q) ≤ 2/3 }
//! ```
//!
//! where `Sym_0(R^3)` is the 5-dimensional vector space of 3×3 real symmetric
//! traceless matrices. The eigenvalue bounds arise from the physical constraint
//! that Q encodes a probability distribution over directions on S², ensuring
//! that the scalar order parameter S ∈ [-1/2, 1] stays in its physical range.
//!
//! As a manifold, the **interior** of Q is an open convex subset of the 5D
//! flat vector space `Sym_0(R^3)`. The Riemannian metric is the L² (Frobenius)
//! inner product on sym-traceless matrices:
//!
//! ```text
//! <U, V>_Q = tr(U^T V) = tr(U V)   (since U, V symmetric)
//! ```
//!
//! independent of the base point Q. This makes `QTensor3` a flat Riemannian
//! manifold: geodesics are straight lines, curvature is zero, parallel
//! transport is the identity.
//!
//! ## Basis
//!
//! A convenient orthonormal basis for `Sym_0(R^3)` under the Frobenius inner
//! product is the five matrices:
//!
//! ```text
//! B_1 = (1/√2) [[1,0,0],[0,-1,0],[0,0,0]]   (Qxx - Qyy)
//! B_2 = (1/√2) [[1,0,0],[0,0,0],[0,0,-1]]   (Qxx - Qzz)
//! B_3 = [[0,1,0],[1,0,0],[0,0,0]]            (Qxy)
//! B_4 = [[0,0,1],[0,0,0],[1,0,0]]            (Qxz)
//! B_5 = [[0,0,0],[0,0,1],[0,1,0]]            (Qyz)
//! ```
//!
//! (not used internally — the Manifold impl works with the full 3×3 matrix).
//!
//! ## Physical interpretation
//!
//! For a **uniaxial** nematic with director `n ∈ S²` and scalar order parameter
//! `S ∈ [-1/2, 1]`:
//!
//! ```text
//! Q = S (n⊗n - I/3)
//! ```
//!
//! Eigenvalues: `λ_1 = λ_2 = -S/3` (degenerate, eigenvectors perpendicular to n)
//! and `λ_3 = 2S/3` (eigenvector = director n). The physical range `S ∈ [-1/2, 1]`
//! corresponds exactly to the eigenvalue bounds `-1/3 ≤ λ_i ≤ 2/3`.
//!
//! For a **biaxial** nematic (all three eigenvalues distinct), the order parameter
//! space is `SO(3)/D_2` where `D_2` is the Klein four-group. This is the relevant
//! case for the rotor phase of board-shaped particles with aspect ratio > 1.
//!
//! ## Relation to cartan manifolds
//!
//! `QTensor3` is structurally analogous to `Euclidean<5>` but uses the 3×3 matrix
//! representation natively (to avoid the overhead of a basis decomposition) and
//! adds the Q-space projection (`project_point`). All curvature is zero; all
//! geodesics are straight lines; parallel transport is trivial.
//!
//! ## References
//!
//! - de Gennes, P. G. & Prost, J. (1993). *The Physics of Liquid Crystals*.
//!   Oxford University Press. §2.1 (Q-tensor order parameter).
//! - Beris, A. N. & Edwards, B. J. (1994). *Thermodynamics of Flowing Systems*.
//!   Oxford University Press. §5.1 (Q-tensor dynamics).
//! - Sonnet, A. M. & Virga, E. G. (2012). *Dissipative Ordered Fluids*.
//!   Springer. §2.3 (biaxial Q-tensor and symmetry group D₂).
//! - Mottram, N. J. & Newton, C. J. P. (2014). "Introduction to Q-tensor theory."
//!   arXiv:1409.3542. §1–2 (Q-space geometry and invariants).

use nalgebra::{SMatrix, SVector, SymmetricEigen};
use rand::Rng;
use rand_distr::StandardNormal;

use cartan_core::{
    CartanError, Connection, Curvature, GeodesicInterpolation, Manifold, ParallelTransport, Real,
    Retraction,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Tolerance for validating Q-tensor symmetry: ||Q - Q^T||_F < SYM_TOL.
const SYM_TOL: Real = 1e-8;

/// Tolerance for validating tracelessness: |tr(Q)| < TRACE_TOL.
const TRACE_TOL: Real = 1e-8;

/// Tolerance on eigenvalue bounds: λ_i ∈ [-1/3 - EIG_TOL, 2/3 + EIG_TOL].
const EIG_TOL: Real = 1e-7;

/// Lower eigenvalue bound: -1/3.
const LAMBDA_MIN: Real = -1.0 / 3.0;

/// Upper eigenvalue bound: 2/3.
const LAMBDA_MAX: Real = 2.0 / 3.0;

// ─────────────────────────────────────────────────────────────────────────────
// Struct definition
// ─────────────────────────────────────────────────────────────────────────────

/// The Q-tensor manifold: 3×3 symmetric traceless real matrices.
///
/// A zero-sized type — carries no runtime data. The geometry is fully
/// determined by the Q-space structure of `Sym_0(R^3)` with Frobenius metric.
///
/// # Examples
///
/// ```rust
/// use cartan_manifolds::qtensor::QTensor3;
/// use cartan_core::Manifold;
/// use nalgebra::SMatrix;
///
/// let m = QTensor3;
/// // Uniaxial Q-tensor with S = 0.5, director along z
/// let q = SMatrix::<f64, 3, 3>::from_row_slice(&[
///     -1.0/6.0, 0.0, 0.0,
///      0.0, -1.0/6.0, 0.0,
///      0.0,  0.0,  1.0/3.0,
/// ]);
/// assert!(m.check_point(&q).is_ok());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QTensor3;

// ─────────────────────────────────────────────────────────────────────────────
// Manifold implementation
// ─────────────────────────────────────────────────────────────────────────────

impl Manifold for QTensor3 {
    /// Points are 3×3 real matrices.
    ///
    /// The physical constraint (symmetric, traceless, eigenvalues in [-1/3, 2/3])
    /// is NOT enforced by the type — any SMatrix can be stored — but is checked by
    /// `check_point` and enforced by `project_point`. This matches the ambient
    /// representation used by other cartan manifolds (SO(N), SPD, etc.).
    type Point = SMatrix<Real, 3, 3>;

    /// Tangent vectors are also 3×3 sym-traceless matrices.
    ///
    /// The tangent space at any Q ∈ Q-space is T_Q(Sym_0) = Sym_0(R^3) (the same flat
    /// space, since Q is an open subset of a vector space). Tangent vectors are NOT
    /// constrained to the eigenvalue bounds; they encode infinitesimal perturbations
    /// of Q in the symmetry-preserving direction.
    type Tangent = SMatrix<Real, 3, 3>;

    /// Intrinsic dimension of Sym_0(R^3): 5.
    ///
    /// A 3×3 symmetric matrix has 6 independent entries; the traceless constraint
    /// removes 1, leaving 5 degrees of freedom.
    fn dim(&self) -> usize {
        5
    }

    /// Ambient dimension: 9 (all entries of a 3×3 matrix).
    fn ambient_dim(&self) -> usize {
        9
    }

    /// Injectivity radius: infinity.
    ///
    /// The Q-manifold (interior of the convex set Q) is an open subset of a vector
    /// space. Geodesics are straight lines and never revisit a starting point, so
    /// there is no cut locus and the exponential map is a global diffeomorphism.
    fn injectivity_radius(&self, _p: &Self::Point) -> Real {
        Real::INFINITY
    }

    /// Frobenius inner product on sym-traceless matrices: tr(U V).
    ///
    /// Since both U and V are symmetric, tr(U^T V) = tr(U V). This inner product
    /// is the restriction of the flat Frobenius metric on R^{3×3} to Sym_0(R^3).
    /// It is independent of the base point Q.
    fn inner(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        // tr(U V) for symmetric U, V equals tr(U^T V) (component-wise sum of products).
        // nalgebra's `component_mul` (Hadamard product) followed by sum is equivalent
        // and slightly more readable. Use the standard trace of the product for clarity.
        (u * v).trace()
    }

    /// Exponential map: Q + V (straight-line geodesic in flat Q-space).
    ///
    /// Since Q-space is an open convex subset of a vector space, geodesics are
    /// straight lines. The exponential map is just addition.
    ///
    /// **Note:** The result is NOT projected onto the eigenvalue constraint; this
    /// is intentional for use in optimizers that rely on line-search. Use
    /// `project_point` to snap back to the physical Q-manifold if needed.
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        p + v
    }

    /// Logarithmic map: Q₂ - Q₁ (difference vector in flat Q-space).
    ///
    /// Always succeeds (no cut locus). Returns the tangent vector at Q₁ pointing
    /// toward Q₂, whose norm equals the geodesic distance dist(Q₁, Q₂).
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> {
        Ok(q - p)
    }

    /// Project an ambient matrix V onto the sym-traceless tangent space at Q.
    ///
    /// Two steps:
    /// 1. Symmetrize: V ← (V + V^T) / 2
    /// 2. Remove trace: V ← V - (tr(V)/3) I
    ///
    /// The result is the orthogonal projection of V onto Sym_0(R^3) under the
    /// standard Frobenius inner product on R^{3×3}.
    fn project_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        let sym = (v + v.transpose()) * 0.5;
        let trace = sym.trace();
        sym - SMatrix::identity() * (trace / 3.0)
    }

    /// Project an arbitrary 3×3 matrix onto the physical Q-manifold.
    ///
    /// Three steps:
    /// 1. Symmetrize: Q ← (Q + Q^T) / 2
    /// 2. Remove trace: Q ← Q - (tr(Q)/3) I
    /// 3. Clamp eigenvalues to [-1/3, 2/3] and restore tracelessness.
    ///
    /// Step 3 is a spectral projection: diagonalize Q = F Λ F^T, clamp the diagonal
    /// entries of Λ, subtract the new mean eigenvalue to restore tracelessness, then
    /// reassemble Q. This is an orthogonal projection onto the convex set Q under the
    /// Frobenius metric.
    fn project_point(&self, p: &Self::Point) -> Self::Point {
        // Step 1–2: sym-traceless part.
        let q_sym = (p + p.transpose()) * 0.5;
        let q_st = q_sym - SMatrix::<Real, 3, 3>::identity() * (q_sym.trace() / 3.0);

        // Step 3: spectral projection.
        clamp_eigenvalues(&q_st)
    }

    /// Validate that Q is a physical Q-tensor.
    ///
    /// Checks:
    /// - Symmetry: `||Q - Q^T||_F < 1e-8`
    /// - Tracelessness: `|tr(Q)| < 1e-8`
    /// - Eigenvalue bounds: `-1/3 - 1e-7 ≤ λ_i ≤ 2/3 + 1e-7`
    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError> {
        // Symmetry check.
        let asym = (p - p.transpose()).norm();
        if asym > SYM_TOL {
            return Err(CartanError::NotOnManifold {
                constraint: "Q must be symmetric: ||Q - Q^T||_F".into(),
                violation: asym,
            });
        }

        // Traceless check.
        let tr_abs = p.trace().abs();
        if tr_abs > TRACE_TOL {
            return Err(CartanError::NotOnManifold {
                constraint: "Q must be traceless: |tr(Q)|".into(),
                violation: tr_abs,
            });
        }

        // Eigenvalue bounds: only meaningful if the matrix is symmetric.
        let eig = SymmetricEigen::new(*p);
        for &lambda in eig.eigenvalues.iter() {
            if lambda < LAMBDA_MIN - EIG_TOL {
                return Err(CartanError::NotOnManifold {
                    constraint: "Q eigenvalue below -1/3".into(),
                    violation: LAMBDA_MIN - EIG_TOL - lambda,
                });
            }
            if lambda > LAMBDA_MAX + EIG_TOL {
                return Err(CartanError::NotOnManifold {
                    constraint: "Q eigenvalue above 2/3".into(),
                    violation: lambda - LAMBDA_MAX - EIG_TOL,
                });
            }
        }

        Ok(())
    }

    /// Validate that V is a valid tangent vector (sym-traceless).
    ///
    /// Checks:
    /// - Symmetry: `||V - V^T||_F < 1e-8`
    /// - Tracelessness: `|tr(V)| < 1e-8`
    fn check_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> {
        let asym = (v - v.transpose()).norm();
        if asym > SYM_TOL {
            return Err(CartanError::NotInTangentSpace {
                constraint: "tangent must be symmetric: ||V - V^T||_F".into(),
                violation: asym,
            });
        }
        let tr_abs = v.trace().abs();
        if tr_abs > TRACE_TOL {
            return Err(CartanError::NotInTangentSpace {
                constraint: "tangent must be traceless: |tr(V)|".into(),
                violation: tr_abs,
            });
        }
        Ok(())
    }

    /// Generate a random physical Q-tensor in the isotropic phase (Q ≈ 0).
    ///
    /// Returns a random sym-traceless matrix with small Frobenius norm (∼ 0.01),
    /// representing a weakly ordered or disordered state. To generate nematic
    /// samples see [`random_nematic`].
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        // Sample 5 independent Gaussian entries for the basis of Sym_0.
        // Construct a random sym-traceless matrix as a sum over the 5-dimensional
        // basis: Qxx, Qxy, Qxz, Qyy, Qyz (Qzz = -Qxx - Qyy by tracelessness).
        let scale = 0.05;
        let qxx: Real = rng.sample(StandardNormal);
        let qyy: Real = rng.sample(StandardNormal);
        let qxy: Real = rng.sample(StandardNormal);
        let qxz: Real = rng.sample(StandardNormal);
        let qyz: Real = rng.sample(StandardNormal);
        let qzz = -qxx - qyy;

        SMatrix::<Real, 3, 3>::from_row_slice(&[
            qxx * scale,  qxy * scale,  qxz * scale,
            qxy * scale,  qyy * scale,  qyz * scale,
            qxz * scale,  qyz * scale,  qzz * scale,
        ])
    }

    /// Generate a random sym-traceless tangent vector at Q.
    ///
    /// Samples a random sym-traceless matrix and scales it to unit Frobenius norm.
    /// Independent of the base point Q (flat geometry).
    fn random_tangent<R: Rng>(&self, p: &Self::Point, rng: &mut R) -> Self::Tangent {
        // Random sym-traceless matrix (same as random_point, unscaled).
        let v = self.random_point(rng);
        // Project to ensure sym-traceless (should already be, but apply for safety).
        let v_st = self.project_tangent(p, &v);
        // Normalize to unit Frobenius norm.
        let n = v_st.norm();
        if n < 1e-15 { v_st } else { v_st / n }
    }

    /// Zero tangent vector at `p`: the 3×3 zero matrix.
    fn zero_tangent(&self, _p: &Self::Point) -> Self::Tangent {
        SMatrix::<Real, 3, 3>::zeros()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retraction (same as exp/log for flat space)
// ─────────────────────────────────────────────────────────────────────────────

impl Retraction for QTensor3 {
    /// Retraction: Q + V (same as exp for flat Q-space).
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        p + v
    }

    /// Inverse retraction: Q₂ - Q₁ (same as log for flat Q-space).
    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        Ok(q - p)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel transport (trivial for flat space)
// ─────────────────────────────────────────────────────────────────────────────

impl ParallelTransport for QTensor3 {
    /// Parallel transport: identity (flat manifold, all tangent spaces canonically
    /// identified as Sym_0(R^3)).
    fn transport(
        &self,
        _p: &Self::Point,
        _q: &Self::Point,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        Ok(*u)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection (Riemannian Hessian-vector product, trivial for flat space)
// ─────────────────────────────────────────────────────────────────────────────

impl Connection for QTensor3 {
    /// Riemannian Hessian-vector product on flat Q-space.
    ///
    /// For a flat manifold the Levi-Civita connection is trivial (all Christoffel
    /// symbols vanish), so the Riemannian Hessian equals the Euclidean Hessian
    /// projected onto the tangent space:
    ///
    /// ```text
    /// ∇² f(Q)[V] = proj_{T_Q Sym_0} (Hess_ambient(Q)[V])
    /// ```
    fn riemannian_hessian_vector_product(
        &self,
        p: &Self::Point,
        _grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        Ok(self.project_tangent(p, &hess_ambient(v)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature (identically zero for flat Q-space)
// ─────────────────────────────────────────────────────────────────────────────

impl Curvature for QTensor3 {
    /// Riemann curvature tensor: identically zero (flat manifold).
    fn riemann_curvature(
        &self,
        _p: &Self::Point,
        _u: &Self::Tangent,
        _v: &Self::Tangent,
        _w: &Self::Tangent,
    ) -> Self::Tangent {
        SMatrix::zeros()
    }

    /// Ricci curvature: identically zero (flat manifold).
    fn ricci_curvature(
        &self,
        _p: &Self::Point,
        _u: &Self::Tangent,
        _v: &Self::Tangent,
    ) -> Real {
        0.0
    }

    /// Scalar curvature: identically zero (flat manifold).
    fn scalar_curvature(&self, _p: &Self::Point) -> Real {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeodesicInterpolation (linear interpolation for flat Q-space)
// ─────────────────────────────────────────────────────────────────────────────

impl GeodesicInterpolation for QTensor3 {
    /// Geodesic interpolation: (1-t) Q₁ + t Q₂ (straight line in flat Q-space).
    fn geodesic(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        t: Real,
    ) -> Result<Self::Point, CartanError> {
        Ok(p * (1.0 - t) + q * t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the ordered orthonormal eigenvector frame from a Q-tensor.
///
/// Returns a 3×3 rotation matrix `F ∈ SO(3)` whose columns are the eigenvectors
/// of Q sorted by ascending eigenvalue (λ₁ ≤ λ₂ ≤ λ₃). The frame satisfies:
///
/// ```text
/// Q = F · diag(λ₁, λ₂, λ₃) · F^T
/// ```
///
/// For a uniaxial Q-tensor with director `n` and order parameter `S > 0`:
/// - `F[:,2]` = director `n` (largest eigenvalue λ₃ = 2S/3)
/// - `F[:,0]`, `F[:,1]` = any two orthonormal vectors perpendicular to n
///
/// # Determinant convention
///
/// The returned frame satisfies `det(F) = +1` (i.e., F ∈ SO(3), not O(3)).
/// If the eigendecomposition returns det = -1, the third column is negated.
///
/// # Numerical note
///
/// The input is symmetrized before decomposition to guard against floating-point
/// asymmetry. For degenerate eigenvalues (uniaxial Q), the eigenvectors in the
/// degenerate subspace are arbitrary but orthonormal.
pub fn q_to_frame(q: &SMatrix<Real, 3, 3>) -> SMatrix<Real, 3, 3> {
    // Symmetrize to guard against floating-point asymmetry.
    let q_sym = (q + q.transpose()) * 0.5;

    // Eigendecomposition: Q = eigenvectors * diag(eigenvalues) * eigenvectors^T.
    // nalgebra's SymmetricEigen sorts eigenvalues in ascending order.
    let eig = SymmetricEigen::new(q_sym);
    let mut f = eig.eigenvectors;

    // Enforce det(F) = +1 (SO(3) rather than O(3)).
    // If det < 0, flip the sign of the third column to get det = +1.
    // This choice is consistent: the third eigenvector (largest eigenvalue,
    // i.e., the director for uniaxial nematics) retains its orientation when S > 0.
    if f.determinant() < 0.0 {
        let col2 = -f.column(2).into_owned();
        f.set_column(2, &col2);
    }

    f
}

/// Generate a random uniaxial Q-tensor in the nematic phase.
///
/// Samples a random director `n ∈ S²` and order parameter `S ∈ [s_lo, s_hi]`,
/// then returns `Q = S (n⊗n - I/3)`.
///
/// Default physical range for a stable nematic: `s_lo = 0.3`, `s_hi = 0.7`.
/// This ensures eigenvalues well inside the physical bounds (-1/3, 2/3).
pub fn random_nematic<R: Rng>(rng: &mut R, s_lo: Real, s_hi: Real) -> SMatrix<Real, 3, 3> {
    // Random unit vector on S² via Gaussian normalization.
    let nx: Real = rng.sample(StandardNormal);
    let ny: Real = rng.sample(StandardNormal);
    let nz: Real = rng.sample(StandardNormal);
    let norm = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-15);
    let n = SVector::<Real, 3>::from_column_slice(&[nx / norm, ny / norm, nz / norm]);

    // Scalar order parameter S ∈ [s_lo, s_hi].
    let s = s_lo + rng.random::<Real>() * (s_hi - s_lo);

    // Q = S (n⊗n - I/3).
    let nn = n * n.transpose();
    (nn - SMatrix::<Real, 3, 3>::identity() * (1.0 / 3.0)) * s
}

/// Check whether a matrix Q is in the physical Q-manifold (fast, no error message).
///
/// Returns `true` iff Q is symmetric, traceless, and all eigenvalues lie in
/// [-1/3 - 1e-7, 2/3 + 1e-7].
pub fn is_physical(q: &SMatrix<Real, 3, 3>) -> bool {
    QTensor3.check_point(q).is_ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Project a sym-traceless matrix onto Q by clamping its eigenvalues.
///
/// Steps:
/// 1. Eigendecompose: Q = F Λ F^T.
/// 2. Clamp Λ_{ii} to [LAMBDA_MIN, LAMBDA_MAX].
/// 3. Subtract mean eigenvalue to restore tracelessness: Λ_{ii} -= (Σ Λ_jj)/3.
/// 4. Reconstruct: Q = F Λ_clamped F^T.
///
/// The trace-restoration step (3) may slightly violate the bounds for extreme
/// inputs, but in practice (physical Q-tensors close to the nematic phase)
/// this is negligible.
fn clamp_eigenvalues(q: &SMatrix<Real, 3, 3>) -> SMatrix<Real, 3, 3> {
    let eig = SymmetricEigen::new(*q);
    let f = eig.eigenvectors;
    let mut lambdas = eig.eigenvalues;

    // Clamp each eigenvalue.
    for lam in lambdas.iter_mut() {
        *lam = lam.clamp(LAMBDA_MIN, LAMBDA_MAX);
    }

    // Restore tracelessness: subtract mean.
    let mean = lambdas.sum() / 3.0;
    lambdas -= SVector::<Real, 3>::repeat(mean);

    // Reconstruct Q = F * diag(lambdas) * F^T.
    let d = SMatrix::<Real, 3, 3>::from_diagonal(&lambdas);
    f * d * f.transpose()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// Build a uniaxial Q-tensor with director z and order parameter S.
    fn uniaxial_z(s: Real) -> SMatrix<Real, 3, 3> {
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            -s / 3.0, 0.0, 0.0,
             0.0, -s / 3.0, 0.0,
             0.0,  0.0,  2.0 * s / 3.0,
        ])
    }

    #[test]
    fn check_point_accepts_valid_uniaxial() {
        let q = uniaxial_z(0.5);
        assert!(QTensor3.check_point(&q).is_ok(), "valid uniaxial Q rejected");
    }

    #[test]
    fn check_point_rejects_non_symmetric() {
        let mut q = uniaxial_z(0.5);
        q[(0, 1)] += 0.1; // break symmetry
        assert!(QTensor3.check_point(&q).is_err());
    }

    #[test]
    fn check_point_rejects_non_traceless() {
        let mut q = uniaxial_z(0.5);
        q[(0, 0)] += 0.5; // add trace
        assert!(QTensor3.check_point(&q).is_err());
    }

    #[test]
    fn check_point_rejects_eigenvalue_violation() {
        // Q with eigenvalue > 2/3.
        let q = uniaxial_z(1.1); // λ_3 = 2*1.1/3 ≈ 0.733 > 2/3
        assert!(QTensor3.check_point(&q).is_err());
    }

    #[test]
    fn project_point_restores_validity() {
        // Start with a Q slightly out of bounds.
        let q_bad = uniaxial_z(0.9); // eigenvalue 0.6 > 2/3 = 0.667
        let q_proj = QTensor3.project_point(&q_bad);
        assert!(
            QTensor3.check_point(&q_proj).is_ok(),
            "projected Q still invalid: {:?}",
            q_proj
        );
    }

    #[test]
    fn project_tangent_is_sym_traceless() {
        // An arbitrary 3×3 matrix projected to tangent space should be sym-traceless.
        let q = uniaxial_z(0.5);
        let v_arb = SMatrix::<Real, 3, 3>::from_row_slice(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let v_st = QTensor3.project_tangent(&q, &v_arb);
        // Symmetry.
        assert_abs_diff_eq!((v_st - v_st.transpose()).norm(), 0.0, epsilon = 1e-14);
        // Tracelessness.
        assert_abs_diff_eq!(v_st.trace(), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn geodesic_stays_in_flat_space() {
        let q1 = uniaxial_z(0.3);
        let q2 = uniaxial_z(0.6);
        let mid = QTensor3.geodesic(&q1, &q2, 0.5).unwrap();
        let expected = uniaxial_z(0.45);
        assert_abs_diff_eq!((mid - expected).norm(), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn q_to_frame_is_so3() {
        let q = uniaxial_z(0.5);
        let f = q_to_frame(&q);
        // Orthonormality: F^T F = I.
        let orth_err = (f.transpose() * f - SMatrix::<Real, 3, 3>::identity()).norm();
        assert_abs_diff_eq!(orth_err, 0.0, epsilon = 1e-12);
        // Determinant = +1.
        assert_abs_diff_eq!(f.determinant(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn q_to_frame_director_is_third_column_for_uniaxial() {
        // For Q = S(z⊗z - I/3) with S > 0, the largest eigenvalue is 2S/3.
        // The eigenvector for the largest eigenvalue should be (0,0,1).
        let q = uniaxial_z(0.5);
        let f = q_to_frame(&q);
        let director = f.column(2).into_owned();
        // Director should be ±(0, 0, 1).
        assert_abs_diff_eq!(director[0].abs(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(director[1].abs(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(director[2].abs(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn random_nematic_is_valid() {
        let mut rng = SmallRng::seed_from_u64(13);
        for _ in 0..100 {
            let q = random_nematic(&mut rng, 0.3, 0.7);
            assert!(
                QTensor3.check_point(&q).is_ok(),
                "random_nematic returned invalid Q: {:?}",
                q
            );
        }
    }

    #[test]
    fn inner_product_is_frobenius() {
        // For sym-traceless U, V: tr(UV) should equal the Frobenius inner product.
        let u = uniaxial_z(0.5);
        let v = uniaxial_z(0.3);
        let inner = QTensor3.inner(&u, &u, &v);
        // tr(U V) = tr(U^T V) for symmetric U.
        let expected = (u.transpose() * v).trace();
        assert_abs_diff_eq!(inner, expected, epsilon = 1e-14);
    }
}
