// ~/cartan/cartan-core/src/curvature.rs

//! The `Curvature` trait: Riemann curvature tensor and derived quantities.
//!
//! Riemannian curvature measures how much the manifold deviates from being
//! flat. It appears in:
//! - Geodesic deviation (Jacobi equation)
//! - Error bounds for optimization algorithms (convergence rates depend on
//!   sectional curvature bounds)
//! - Comparison geometry (Toponogov's theorem, Bishop-Gromov volume comparison)
//!
//! ## Curvature quantities
//!
//! Starting from the Riemann curvature tensor R(X, Y)Z (a (1,3)-tensor),
//! one derives:
//!
//! - **Sectional curvature** K(sigma) for a 2-plane sigma = span{X, Y}:
//!     K(sigma) = <R(X,Y)Y, X> / (<X,X><Y,Y> - <X,Y>^2)
//!   Controls geodesic spreading: K > 0 makes geodesics converge (sphere),
//!   K < 0 makes them diverge (hyperbolic space), K = 0 flat (Euclidean).
//!
//! - **Ricci curvature** Ric(X, Y): trace of Z -> R(Z, X)Y.
//!   Appears in volume comparison and Ricci flow.
//!
//! - **Scalar curvature** s: trace of the Ricci tensor.
//!
//! ## Known values for v0.1 manifolds
//!
//! - Sphere S^{N-1}: K = 1 (constant sectional curvature)
//! - Hyperbolic H^N: K = -1 (constant sectional curvature)
//! - SO(N) (bi-invariant): K(X,Y) = (1/4)||[X,Y]||^2 / (||X||^2 ||Y||^2 - <X,Y>^2)
//! - SPD(N) (affine-invariant): K <= 0 (non-positive, Cartan-Hadamard manifold)
//! - Grassmann Gr(n,k): 0 <= K <= 2 (between 0 and 2)
//! - Euclidean R^N: K = 0 everywhere
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Chapter 4 (curvature tensor), Chapter 5 (sectional).
//! - Milnor. "Curvatures of Left-Invariant Metrics on Lie Groups." Advances in Math, 1976.
//! - Bhatia. "Positive Definite Matrices." Princeton, 2007. Chapter 6 (curvature of SPD).

use crate::connection::Connection;
use crate::Real;

/// A manifold with explicitly computable Riemannian curvature.
///
/// This trait provides the Riemann curvature tensor and derived scalar
/// quantities. It is useful for:
/// - Theoretical analysis (verifying curvature bounds for convergence proofs)
/// - Geodesic deviation simulation (Jacobi fields)
/// - Visualization (sectional curvature plots)
///
/// # Supertraiting Connection
///
/// Curvature requires Self: Connection because the curvature tensor is defined
/// via the Levi-Civita connection: R(X,Y)Z = nabla_X nabla_Y Z - nabla_Y nabla_X Z
/// - nabla_{[X,Y]} Z. Without the connection, curvature cannot be defined.
pub trait Curvature: Connection {
    /// The Riemann curvature tensor: R(u, v)w.
    ///
    /// Takes three tangent vectors u, v, w at p and returns a tangent vector
    /// at p. Defined via the Levi-Civita connection as:
    ///
    ///   R(u,v)w = nabla_u nabla_v w - nabla_v nabla_u w - nabla_{[u,v]} w
    ///
    /// Properties:
    /// - Anti-symmetry: R(u,v)w = -R(v,u)w
    /// - First Bianchi identity: R(u,v)w + R(v,w)u + R(w,u)v = 0
    /// - Symmetry: <R(u,v)w, z> = <R(w,z)u, v>
    ///
    /// Ref: do Carmo, Chapter 4, Definition 1.
    fn riemann_curvature(
        &self,
        p: &Self::Point,
        u: &Self::Tangent,
        v: &Self::Tangent,
        w: &Self::Tangent,
    ) -> Self::Tangent;

    /// Sectional curvature of the 2-plane spanned by u and v at p.
    ///
    /// K(u, v) = <R(u,v)v, u>_p / (<u,u>_p <v,v>_p - <u,v>_p^2)
    ///
    /// The denominator is the squared area of the parallelogram spanned by u, v.
    /// This quantity is well-defined only when u and v are linearly independent.
    ///
    /// The default implementation computes this from riemann_curvature() and inner().
    /// Manifolds with closed-form sectional curvature (sphere, hyperbolic) should
    /// override this with a direct formula to avoid the overhead of computing the
    /// full curvature tensor.
    fn sectional_curvature(
        &self,
        p: &Self::Point,
        u: &Self::Tangent,
        v: &Self::Tangent,
    ) -> Real {
        // Numerator: <R(u,v)v, u>_p
        // This is the standard formula for sectional curvature.
        let r_uvv = self.riemann_curvature(p, u, v, v);
        let numerator = self.inner(p, u, &r_uvv);

        // Denominator: ||u||^2 ||v||^2 - <u,v>^2
        // This is the squared area of the parallelogram (= 0 iff u,v are parallel).
        let uu = self.inner(p, u, u);
        let vv = self.inner(p, v, v);
        let uv = self.inner(p, u, v);
        let denominator = uu * vv - uv * uv;

        numerator / denominator
    }

    /// Ricci curvature tensor applied to tangent vectors u and v: Ric(u, v).
    ///
    /// The Ricci tensor is the trace of the full curvature tensor:
    ///   Ric(u, v) = sum_{i=1}^{n} <R(e_i, u)v, e_i>
    /// where {e_i} is an orthonormal basis for T_p M.
    ///
    /// Implementations should compute this directly for each manifold
    /// rather than summing over an explicit basis (which requires O(n)
    /// curvature evaluations).
    ///
    /// Known values:
    /// - Sphere S^{N-1}: Ric(u,v) = (N-2) <u,v>
    /// - SO(N): Ric(u,v) = (N-2)/4 * <u,v>  (for N >= 3)
    /// - Euclidean: Ric = 0
    fn ricci_curvature(
        &self,
        p: &Self::Point,
        u: &Self::Tangent,
        v: &Self::Tangent,
    ) -> Real;

    /// Scalar curvature at p.
    ///
    /// The scalar curvature is the trace of the Ricci tensor:
    ///   s(p) = sum_{i=1}^{n} Ric(e_i, e_i)
    /// where {e_i} is an orthonormal basis for T_p M.
    ///
    /// Known values (dimension n):
    /// - Sphere S^{N-1} (n = N-1): s = (N-1)(N-2)
    /// - Euclidean: s = 0
    fn scalar_curvature(&self, p: &Self::Point) -> Real;
}
