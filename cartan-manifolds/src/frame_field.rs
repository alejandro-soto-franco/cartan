// ~/cartan/cartan-manifolds/src/frame_field.rs

//! Frame fields on nematic Q-tensor configurations.
//!
//! ## Overview
//!
//! A *frame field* assigns an orthonormal frame `F ∈ SO(3)` to each vertex of a
//! discrete spatial grid. In a nematic simulation the frame encodes the local
//! molecular orientation: column 2 of `F` is the director (principal eigenvector
//! of the Q-tensor), and columns 0 and 1 span the transverse plane.
//!
//! ## Gauge ambiguity and D₂ fixing
//!
//! Diagonalizing a Q-tensor yields eigenvectors defined only up to sign flips
//! that preserve `det = +1`. The residual discrete symmetry is the Klein
//! four-group (dihedral group D₂):
//!
//! ```text
//! D₂ = { diag(+1,+1,+1), diag(-1,-1,+1), diag(-1,+1,-1), diag(+1,-1,-1) }
//! ```
//!
//! All four elements are diagonal SO(3) matrices (determinant = +1). For
//! uniaxial nematics only column 2 matters, so the relevant subgroup is
//! `Z₂ = { diag(+1,+1,+1), diag(-1,-1,+1) }` (sign flip of the two transverse
//! axes), which covers the `n ↔ -n` director ambiguity as well as the free
//! transverse orientation. The full D₂ routine handles both cases uniformly.
//!
//! [`FrameField3D::gauge_fix_chain`] removes this ambiguity along a 1D path
//! by choosing, at each vertex, the D₂ representative closest to the
//! previous frame under the rotation-angle metric on SO(3). Maximizing
//! `tr(R_prev^T · R_curr · g)` over the four elements is equivalent to
//! minimizing the geodesic distance.
//!
//! ## Usage
//!
//! ```rust
//! use cartan_manifolds::frame_field::FrameField3D;
//! use nalgebra::SMatrix;
//!
//! // Build from a slice of Q-tensors (here a single z-director uniaxial Q with S=0.6)
//! let q = SMatrix::<f64, 3, 3>::from_row_slice(&[
//!     -0.2, 0.0, 0.0,
//!      0.0,-0.2, 0.0,
//!      0.0, 0.0, 0.4,
//! ]);
//! let ff = FrameField3D::from_q_field(&[q]);
//! assert_eq!(ff.len(), 1);
//!
//! // The frame is in SO(3): F^T F ≈ I, det ≈ +1
//! let f = ff.frame_at(0);
//! let oto = f.transpose() * f;
//! // (oto ≈ identity checked in tests)
//! ```
//!
//! ## References
//!
//! - Mermin, N. D. (1979). "The topological theory of defects in ordered media."
//!   *Rev. Mod. Phys.* **51**, 591. §II.B (homotopy group classification).
//! - Nakahara, M. (2003). *Geometry, Topology and Physics*. IOP. §9 (principal bundles).
//! - Binysh, J. & Alexander, G. P. (2018). "Maxwell's theory of solid angle and the
//!   construction of knotted fields." *J. Phys. A* **51**, 385202. (frame fields and holonomy).

use nalgebra::SMatrix;

use cartan_core::Real;

use crate::qtensor::q_to_frame;

// ─────────────────────────────────────────────────────────────────────────────
// D₂ gauge elements
// ─────────────────────────────────────────────────────────────────────────────

/// The four D₂ right-multiplication elements as (g₀, g₁, g₂) sign triples.
///
/// Each triple represents the diagonal SO(3) matrix `diag(g₀, g₁, g₂)` with
/// `g₀·g₁·g₂ = +1`. Right-multiplying a frame `R` by such a matrix flips the
/// signs of the corresponding columns while keeping `R` in SO(3).
const D2_SIGNS: [(Real, Real, Real); 4] = [
    (1.0, 1.0, 1.0),   // identity
    (-1.0, -1.0, 1.0), // flip columns 0 and 1 (covers n ↔ -n for uniaxial)
    (-1.0, 1.0, -1.0), // flip columns 0 and 2
    (1.0, -1.0, -1.0), // flip columns 1 and 2
];

// ─────────────────────────────────────────────────────────────────────────────
// FrameField3D
// ─────────────────────────────────────────────────────────────────────────────

/// A frame field: an orthonormal frame `F ∈ SO(3)` at each grid vertex.
///
/// Frame `F` is stored as a 3×3 column-major matrix whose three columns
/// `(e₀, e₁, e₂)` are orthonormal. Column 2 (`e₂`) is the director (principal
/// eigenvector of the Q-tensor, i.e., the eigenvector for the largest eigenvalue).
///
/// See the [module documentation](self) for the gauge ambiguity and how to
/// remove it with [`gauge_fix_chain`](FrameField3D::gauge_fix_chain).
pub struct FrameField3D {
    /// Orthonormal frames, one per grid vertex.
    pub frames: Vec<SMatrix<Real, 3, 3>>,
}

impl FrameField3D {
    /// Construct a frame field from a slice of Q-tensors.
    ///
    /// Each Q-tensor is converted to an orthonormal frame via [`q_to_frame`]:
    /// eigendecompose, sort eigenvectors by eigenvalue ascending, enforce
    /// `det = +1` by negating column 2 if needed. The resulting frame has the
    /// director as column 2.
    ///
    /// No gauge fixing is applied. Call [`gauge_fix_chain`](Self::gauge_fix_chain)
    /// afterward if a smooth gauge is needed.
    pub fn from_q_field(q_values: &[SMatrix<Real, 3, 3>]) -> Self {
        let frames = q_values.iter().map(q_to_frame).collect();
        Self { frames }
    }

    /// Return the frame at vertex `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.frames.len()`.
    pub fn frame_at(&self, i: usize) -> &SMatrix<Real, 3, 3> {
        &self.frames[i]
    }

    /// Return the number of vertices.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the field contains no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Apply D₂ gauge fixing along the chain (vertex order 0, 1, …, n-1).
    ///
    /// The first frame is kept unchanged. For each subsequent vertex `i`,
    /// the algorithm finds the element `g ∈ D₂` that maximizes
    ///
    /// ```text
    /// tr( R_{i-1}^T · R_i · g )  =  M₀₀ g₀ + M₁₁ g₁ + M₂₂ g₂
    /// ```
    ///
    /// where `M = R_{i-1}^T · R_i`, and right-multiplies the frame by that `g`.
    /// This is equivalent to minimizing the geodesic distance (rotation angle)
    /// on SO(3) between consecutive frames, making the field as smooth as possible.
    ///
    /// Returns a new `FrameField3D` with gauge-fixed frames.
    pub fn gauge_fix_chain(&self) -> Self {
        if self.frames.is_empty() {
            return Self { frames: Vec::new() };
        }

        let mut fixed: Vec<SMatrix<Real, 3, 3>> = Vec::with_capacity(self.frames.len());
        fixed.push(self.frames[0]);

        for i in 1..self.frames.len() {
            let r_prev = &fixed[i - 1];
            let r_curr = &self.frames[i];
            fixed.push(d2_gauge_fix(r_prev, r_curr));
        }

        Self { frames: fixed }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D₂ gauge-fixing primitive
// ─────────────────────────────────────────────────────────────────────────────

/// Choose the D₂ representative of `r_curr` closest to `r_prev` in SO(3).
///
/// Finds `g = diag(g₀, g₁, g₂) ∈ D₂` maximizing
/// `tr(R_prev^T · R_curr · g)`, then returns `R_curr · g`.
///
/// The computation costs one 3×3 matrix multiply plus four scalar comparisons.
pub fn d2_gauge_fix(
    r_prev: &SMatrix<Real, 3, 3>,
    r_curr: &SMatrix<Real, 3, 3>,
) -> SMatrix<Real, 3, 3> {
    // M = R_prev^T · R_curr; only the diagonal of M·g is needed.
    let m = r_prev.transpose() * r_curr;
    let m00 = m[(0, 0)];
    let m11 = m[(1, 1)];
    let m22 = m[(2, 2)];

    // Find the D₂ element maximizing tr(M·g) = M₀₀g₀ + M₁₁g₁ + M₂₂g₂.
    let mut best_trace = Real::NEG_INFINITY;
    let mut best_g = (1.0_f64, 1.0_f64, 1.0_f64);

    for &(g0, g1, g2) in &D2_SIGNS {
        let t = m00 * g0 + m11 * g1 + m22 * g2;
        if t > best_trace {
            best_trace = t;
            best_g = (g0, g1, g2);
        }
    }

    // Apply: scale each column of r_curr by the corresponding sign.
    let (g0, g1, g2) = best_g;
    let mut result = *r_curr;
    for row in 0..3 {
        result[(row, 0)] *= g0;
        result[(row, 1)] *= g1;
        result[(row, 2)] *= g2;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qtensor::random_nematic;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    // Build a uniaxial Q-tensor with director (nx,ny,nz) (renormalized) and
    // scalar order parameter s.
    fn uniaxial_q(nx: Real, ny: Real, nz: Real, s: Real) -> SMatrix<Real, 3, 3> {
        let n = nalgebra::Vector3::new(nx, ny, nz).normalize();
        let nn = n * n.transpose();
        let eye = SMatrix::<Real, 3, 3>::identity();
        s * (nn - eye / 3.0)
    }

    #[test]
    fn from_q_field_length() {
        let mut rng = SmallRng::seed_from_u64(0);
        let qs: Vec<_> = (0..10)
            .map(|_| random_nematic(&mut rng, 0.1, 0.8))
            .collect();
        let ff = FrameField3D::from_q_field(&qs);
        assert_eq!(ff.len(), 10);
        assert!(!ff.is_empty());
    }

    #[test]
    fn empty_field() {
        let ff = FrameField3D::from_q_field(&[]);
        assert!(ff.is_empty());
        assert_eq!(ff.len(), 0);
        let fixed = ff.gauge_fix_chain();
        assert!(fixed.is_empty());
    }

    #[test]
    fn frame_is_so3_z_director() {
        let q = uniaxial_q(0.0, 0.0, 1.0, 0.6);
        let ff = FrameField3D::from_q_field(&[q]);
        let f = ff.frame_at(0);
        // F^T F ≈ I
        let oto = f.transpose() * f;
        assert_abs_diff_eq!(oto, SMatrix::<Real, 3, 3>::identity(), epsilon = 1e-12);
        // det ≈ +1
        assert_abs_diff_eq!(f.determinant(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn frame_is_so3_random() {
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..20 {
            let q = random_nematic(&mut rng, 0.05, 0.95);
            let ff = FrameField3D::from_q_field(&[q]);
            let f = ff.frame_at(0);
            let oto = f.transpose() * f;
            assert_abs_diff_eq!(oto, SMatrix::<Real, 3, 3>::identity(), epsilon = 1e-10);
            assert_abs_diff_eq!(f.determinant(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn gauge_fix_preserves_so3() {
        let mut rng = SmallRng::seed_from_u64(42);
        let qs: Vec<_> = (0..20)
            .map(|_| random_nematic(&mut rng, 0.2, 0.9))
            .collect();
        let ff = FrameField3D::from_q_field(&qs);
        let fixed = ff.gauge_fix_chain();

        for i in 0..20 {
            let f = fixed.frame_at(i);
            let oto = f.transpose() * f;
            assert_abs_diff_eq!(oto, SMatrix::<Real, 3, 3>::identity(), epsilon = 1e-10);
            assert_abs_diff_eq!(f.determinant(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn gauge_fix_trivial_same_frame() {
        // A constant field: every frame is the same. Gauge fixing must be a no-op.
        let q = uniaxial_q(0.0, 0.0, 1.0, 0.7);
        let qs = vec![q; 8];
        let ff = FrameField3D::from_q_field(&qs);
        let fixed = ff.gauge_fix_chain();
        for i in 0..8 {
            assert_abs_diff_eq!(ff.frame_at(i), fixed.frame_at(i), epsilon = 1e-12);
        }
    }

    #[test]
    fn d2_gauge_fix_identity_stays() {
        // If r_curr is already the D₂ representative closest to I, no change.
        let r = SMatrix::<Real, 3, 3>::identity();
        let fixed = d2_gauge_fix(&r, &r);
        assert_abs_diff_eq!(fixed, r, epsilon = 1e-12);
    }

    #[test]
    fn d2_gauge_fix_flips_back() {
        // r_prev = I; r_curr = diag(-1,-1,+1) (a D₂ element applied to I).
        // gauge_fix should detect this and return I.
        let r_prev = SMatrix::<Real, 3, 3>::identity();
        let mut r_curr = SMatrix::<Real, 3, 3>::identity();
        r_curr[(0, 0)] = -1.0;
        r_curr[(1, 1)] = -1.0;
        let fixed = d2_gauge_fix(&r_prev, &r_curr);
        assert_abs_diff_eq!(fixed, r_prev, epsilon = 1e-12);
    }

    #[test]
    fn d2_gauge_fix_all_four_elements() {
        // For each D₂ element g applied to I, gauge_fix(I, I·g) must return I.
        let id = SMatrix::<Real, 3, 3>::identity();
        for &(g0, g1, g2) in &D2_SIGNS {
            let mut r_curr = id;
            for row in 0..3 {
                r_curr[(row, 0)] *= g0;
                r_curr[(row, 1)] *= g1;
                r_curr[(row, 2)] *= g2;
            }
            let fixed = d2_gauge_fix(&id, &r_curr);
            assert_abs_diff_eq!(fixed, id, epsilon = 1e-12);
        }
    }

    #[test]
    fn gauge_fix_reduces_frame_jumps() {
        // Build a field where every other frame has a D₂ flip applied.
        // After gauge fixing, consecutive frame differences should be smaller.
        let q = uniaxial_q(1.0, 0.2, 0.05, 0.65);
        let mut rng = SmallRng::seed_from_u64(99);
        let qs: Vec<_> = (0..16)
            .map(|_| random_nematic(&mut rng, 0.3, 0.8))
            .collect();
        let ff = FrameField3D::from_q_field(&qs);

        // Manually flip every even frame (except 0) with diag(-1,-1,+1).
        let mut flipped_frames = ff.frames.clone();
        for i in (2..16).step_by(2) {
            for row in 0..3 {
                flipped_frames[i][(row, 0)] *= -1.0;
                flipped_frames[i][(row, 1)] *= -1.0;
            }
        }
        let ff_flipped = FrameField3D {
            frames: flipped_frames,
        };
        let fixed = ff_flipped.gauge_fix_chain();

        // Each fixed frame must still be in SO(3).
        for i in 0..16 {
            let f = fixed.frame_at(i);
            assert_abs_diff_eq!(f.determinant(), 1.0, epsilon = 1e-10);
        }

        // Sum of squared Frobenius norms of frame differences should be
        // no larger after fixing (fixing can only reduce or preserve jumps).
        let jump_before: f64 = (1..16)
            .map(|i| (ff_flipped.frames[i] - ff_flipped.frames[i - 1]).norm_squared())
            .sum();
        let jump_after: f64 = (1..16)
            .map(|i| (fixed.frames[i] - fixed.frames[i - 1]).norm_squared())
            .sum();
        assert!(jump_after <= jump_before + 1e-10);

        // Suppress unused variable warning from q (used to show intent, not in loop).
        let _ = q;
    }
}
