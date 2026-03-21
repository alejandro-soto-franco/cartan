// ~/cartan/cartan-geo/src/holonomy.rs

//! Holonomy-based topological defect detection for discrete Q-tensor fields.
//!
//! ## Mathematical background
//!
//! Let `A` be the connection 1-form on a principal `SO(3)` bundle over the
//! simulation grid, constructed from a discrete Q-tensor field. For an elementary
//! oriented plaquette `γ = (v₀, v₁, v₂, v₃, v₀)`, the **holonomy** is
//!
//! ```text
//! Hol_γ(A) = P exp( ∮_γ A ) ∈ SO(3)
//! ```
//!
//! In the discrete setting this path-ordered exponential becomes a product of
//! **transition matrices**: the SO(3) frame-to-frame rotation across each edge.
//! Each transition matrix is obtained by D₂ gauge-fixing the target frame to
//! the source frame and then computing the rotation that maps one to the other.
//!
//! Deviation of `Hol_γ(A)` from the identity `I` signals a topological charge
//! enclosed by `γ`:
//!
//! - `Hol_γ = I` (within tolerance): plaquette contains no defect.
//! - `Hol_γ ≈ rotation by π` about some axis: plaquette contains a `±1/2`
//!   disclination (nematic charge `s = ±1/2`).
//! - Other values indicate integer-charge disclinations or numerical noise.
//!
//! ## Advantages over zero-tracking
//!
//! This holonomy method detects defects without locating where `|Q| = 0`. It
//! is insensitive to core regularization details, works correctly for biaxial
//! nematics, and naturally accounts for the Z₂/D₂ director ambiguity via the
//! gauge-fixing step built into [`edge_transition`].
//!
//! ## Grid conventions
//!
//! Vertices are indexed linearly: vertex `(i, j)` in an `Nx × Ny` grid maps
//! to linear index `i * Ny + j`. Plaquettes are elementary squares. The four
//! vertices of plaquette `(i, j)` are:
//!
//! ```text
//! v₀ = (i,   j  )   v₁ = (i+1, j  )
//! v₃ = (i,   j+1)   v₂ = (i+1, j+1)
//! ```
//!
//! The default orientation is counter-clockwise: `v₀ → v₁ → v₂ → v₃ → v₀`.
//!
//! ## References
//!
//! - Nakahara, M. (2003). *Geometry, Topology and Physics*. IOP. §9.4
//!   (holonomy and curvature in principal bundles).
//! - Doostmohammadi, A. et al. (2018). "Active nematics." *Nature Commun.* **9**, 3246.
//!   §Supplementary (topological charge from holonomy).
//! - Binysh, J. & Alexander, G. P. (2018). "Maxwell's theory of solid angle."
//!   *J. Phys. A* **51**, 385202. (frame transport and defect charges).

use nalgebra::SMatrix;

use cartan_core::Real;
use cartan_manifolds::frame_field::d2_gauge_fix;

// ─────────────────────────────────────────────────────────────────────────────
// Edge transition matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the SO(3) transition matrix from frame `r_src` to frame `r_dst`.
///
/// The transition matrix `T` is defined by `r_dst · g ≈ r_src · T` where
/// `g ∈ D₂` is chosen to make `r_dst · g` as close to `r_src` as possible
/// (D₂ gauge fixing). Solving for `T`:
///
/// ```text
/// T = r_src^T · (r_dst · g)
/// ```
///
/// `T ∈ SO(3)` represents the minimal rotation needed to go from `r_src` to
/// the gauge-fixed `r_dst`, and is the discrete analogue of the connection
/// 1-form along the directed edge `(src → dst)`.
pub fn edge_transition(
    r_src: &SMatrix<Real, 3, 3>,
    r_dst: &SMatrix<Real, 3, 3>,
) -> SMatrix<Real, 3, 3> {
    let r_dst_fixed = d2_gauge_fix(r_src, r_dst);
    r_src.transpose() * r_dst_fixed
}

// ─────────────────────────────────────────────────────────────────────────────
// Plaquette holonomy
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the holonomy SO(3) matrix for a sequence of frames around a closed loop.
///
/// Given `n` frames `(F₀, F₁, …, F_{n-1})` forming a closed loop (edge sequence
/// `F₀→F₁→…→F_{n-1}→F₀`), the holonomy is the ordered product of transition
/// matrices:
///
/// ```text
/// Hol = T_{01} · T_{12} · … · T_{(n-1)0}
/// ```
///
/// where `T_{ij} = edge_transition(F_i, F_j)`.
///
/// For the standard 2D plaquette (4 vertices), pass frames in counter-clockwise
/// order: `[F₀, F₁, F₂, F₃]` corresponding to the loop `v₀→v₁→v₂→v₃→v₀`.
///
/// # Panics
///
/// Panics if `frames.len() < 2`.
pub fn loop_holonomy(frames: &[SMatrix<Real, 3, 3>]) -> SMatrix<Real, 3, 3> {
    assert!(frames.len() >= 2, "loop_holonomy requires at least 2 frames");

    let n = frames.len();
    let mut hol = SMatrix::<Real, 3, 3>::identity();
    for i in 0..n {
        let j = (i + 1) % n;
        hol *= edge_transition(&frames[i], &frames[j]);
    }
    hol
}

// ─────────────────────────────────────────────────────────────────────────────
// Defect detection scalar
// ─────────────────────────────────────────────────────────────────────────────

/// Measure the deviation of an SO(3) matrix from the identity.
///
/// Returns `||H - I||_F` (Frobenius norm of the difference). This equals
/// `2 sqrt(2) |sin(θ/2)|` where `θ` is the rotation angle of `H`.
///
/// | Value     | Interpretation                      |
/// |-----------|-------------------------------------|
/// | `≈ 0`     | No defect enclosed                  |
/// | `≈ 2√2`   | π rotation, i.e., ±1/2 disclination |
/// | `≈ 2√6`   | 2π/3 rotation                       |
///
/// Use [`rotation_angle`] for the angle itself.
pub fn holonomy_deviation(hol: &SMatrix<Real, 3, 3>) -> Real {
    (hol - SMatrix::<Real, 3, 3>::identity()).norm()
}

/// Extract the rotation angle `θ ∈ [0, π]` from an SO(3) matrix.
///
/// Uses the formula `cos θ = (tr(H) - 1) / 2`, clamped to avoid `acos` domain
/// errors from floating-point noise.
pub fn rotation_angle(hol: &SMatrix<Real, 3, 3>) -> Real {
    let cos_theta = ((hol.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    cos_theta.acos()
}

/// Returns `true` if the holonomy represents a `±1/2` disclination.
///
/// A half-integer disclination has holonomy equal to a rotation by `π`,
/// so `θ ∈ (π/2, π]` (liberal threshold) or `θ > threshold` (caller-supplied).
/// The threshold `π/2` is the midpoint between "no defect" (`θ = 0`) and
/// "half disclination" (`θ = π`).
///
/// Pass `threshold = Real::consts::PI / 2.0` for the default choice.
pub fn is_half_disclination(hol: &SMatrix<Real, 3, 3>, threshold: Real) -> bool {
    rotation_angle(hol) > threshold
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D grid holonomy scan
// ─────────────────────────────────────────────────────────────────────────────

/// A detected disclination in a 2D frame field.
#[derive(Debug, Clone, PartialEq)]
pub struct Disclination {
    /// Grid coordinates of the plaquette lower-left corner.
    pub plaquette: (usize, usize),
    /// Holonomy matrix of the plaquette loop.
    pub holonomy: SMatrix<Real, 3, 3>,
    /// Rotation angle of the holonomy (in radians).
    pub angle: Real,
}

/// Scan a 2D frame field for topological defects using the holonomy method.
///
/// `frames[i * ny + j]` is the SO(3) frame at grid vertex `(i, j)`.
/// Grid dimensions are `nx × ny`. The scan checks all `(nx-1) × (ny-1)`
/// elementary plaquettes.
///
/// For each plaquette `(i, j)` (lower-left corner), the loop is:
///
/// ```text
/// (i,j) → (i+1,j) → (i+1,j+1) → (i,j+1) → (i,j)
/// ```
///
/// Returns the list of plaquettes whose holonomy rotation angle exceeds
/// `threshold` (typically `π/2`).
pub fn scan_disclinations(
    frames: &[SMatrix<Real, 3, 3>],
    nx: usize,
    ny: usize,
    threshold: Real,
) -> Vec<Disclination> {
    assert_eq!(
        frames.len(),
        nx * ny,
        "frames.len() must equal nx * ny"
    );

    let idx = |i: usize, j: usize| i * ny + j;
    let mut result = Vec::new();

    for i in 0..(nx - 1) {
        for j in 0..(ny - 1) {
            let loop_frames = [
                frames[idx(i, j)],
                frames[idx(i + 1, j)],
                frames[idx(i + 1, j + 1)],
                frames[idx(i, j + 1)],
            ];
            let hol = loop_holonomy(&loop_frames);
            let angle = rotation_angle(&hol);
            if angle > threshold {
                result.push(Disclination {
                    plaquette: (i, j),
                    holonomy: hol,
                    angle,
                });
            }
        }
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use cartan_manifolds::frame_field::FrameField3D;
    use cartan_manifolds::qtensor::random_nematic;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use std::f64::consts::PI;

    fn identity_frame() -> SMatrix<Real, 3, 3> {
        SMatrix::<Real, 3, 3>::identity()
    }

    // Build an SO(3) rotation matrix about the z-axis by angle theta.
    fn rot_z(theta: Real) -> SMatrix<Real, 3, 3> {
        let c = theta.cos();
        let s = theta.sin();
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            c, -s, 0.0,
            s,  c, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    // Build an SO(3) rotation matrix about the x-axis by angle theta.
    fn rot_x(theta: Real) -> SMatrix<Real, 3, 3> {
        let c = theta.cos();
        let s = theta.sin();
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            1.0, 0.0, 0.0,
            0.0,  c, -s,
            0.0,  s,  c,
        ])
    }

    #[test]
    fn edge_transition_identity_loop() {
        // Transition from I to I should be I.
        let id = identity_frame();
        let t = edge_transition(&id, &id);
        assert_abs_diff_eq!(t, id, epsilon = 1e-12);
    }

    #[test]
    fn edge_transition_is_so3() {
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..20 {
            let q0 = random_nematic(&mut rng, 0.1, 0.9);
            let q1 = random_nematic(&mut rng, 0.1, 0.9);
            let ff = FrameField3D::from_q_field(&[q0, q1]);
            let t = edge_transition(ff.frame_at(0), ff.frame_at(1));
            // T^T T ≈ I
            let oto = t.transpose() * t;
            assert_abs_diff_eq!(oto, identity_frame(), epsilon = 1e-10);
            // det ≈ +1
            assert_abs_diff_eq!(t.determinant(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn loop_holonomy_trivial() {
        // A constant frame field: holonomy of any loop is I.
        let id = identity_frame();
        let frames = [id, id, id, id];
        let hol = loop_holonomy(&frames);
        assert_abs_diff_eq!(hol, id, epsilon = 1e-12);
        assert_abs_diff_eq!(holonomy_deviation(&hol), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn loop_holonomy_smooth_field() {
        // A slowly varying smooth field (small rotations between adjacent frames)
        // should have near-zero holonomy for small plaquettes.
        let eps = 0.05_f64;
        let f0 = identity_frame();
        let f1 = rot_z(eps);
        let f2 = rot_z(eps) * rot_x(eps);
        let f3 = rot_x(eps);
        let frames = [f0, f1, f2, f3];
        let hol = loop_holonomy(&frames);
        let angle = rotation_angle(&hol);
        // For a smooth field the holonomy is second-order in the step size.
        assert!(angle < 0.01, "angle = {angle}");
    }

    #[test]
    fn rotation_angle_identity() {
        let id = identity_frame();
        assert_abs_diff_eq!(rotation_angle(&id), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rotation_angle_pi() {
        // Rotation by pi about z: trace = cos(pi)*2 + 1 = -1, angle = pi.
        let r = rot_z(PI);
        assert_abs_diff_eq!(rotation_angle(&r), PI, epsilon = 1e-12);
    }

    #[test]
    fn holonomy_deviation_identity_is_zero() {
        let id = identity_frame();
        assert_abs_diff_eq!(holonomy_deviation(&id), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn holonomy_deviation_pi_rotation() {
        // Rotation by pi: ||R - I||_F = 2*sqrt(2) * |sin(pi/2)| = 2*sqrt(2).
        let r = rot_z(PI);
        let expected = 2.0 * 2.0_f64.sqrt();
        assert_abs_diff_eq!(holonomy_deviation(&r), expected, epsilon = 1e-10);
    }

    #[test]
    fn is_half_disclination_detection() {
        let threshold = PI / 2.0;
        let id = identity_frame();
        let r_pi = rot_z(PI);
        let r_small = rot_z(PI / 4.0);

        assert!(!is_half_disclination(&id, threshold));
        assert!(is_half_disclination(&r_pi, threshold));
        assert!(!is_half_disclination(&r_small, threshold));
    }

    #[test]
    fn scan_disclinations_empty_no_defects() {
        // A 3x3 grid with the same frame everywhere: no defects.
        let id = identity_frame();
        let frames: Vec<_> = (0..9).map(|_| id).collect();
        let defects = scan_disclinations(&frames, 3, 3, PI / 2.0);
        assert!(defects.is_empty());
    }

    #[test]
    fn scan_disclinations_planted_defect() {
        // Construct a genuine +1/2 disclination in a 2x2 grid (one plaquette).
        //
        // For a +1/2 disclination the director rotates by pi as you traverse the
        // plaquette loop. We model this with pure z-axis rotations:
        //
        //   (0,0) -> rot_z(0)      (1,0) -> rot_z(pi/4)
        //   (0,1) -> rot_z(3pi/4)  (1,1) -> rot_z(pi/2)
        //
        // The CCW loop (0,0)->(1,0)->(1,1)->(0,1)->(0,0) accumulates
        // four edge transitions of rot_z(pi/4) each (the last one via D2 gauge
        // fixing), giving holonomy = rot_z(pi) -- a pi rotation, i.e., a +1/2
        // disclination charge.
        //
        // Vertex linear index: i*ny + j = i*2 + j for ny=2.
        //   (0,0)=0, (1,0)=2, (0,1)=1, (1,1)=3
        let mut frames = vec![identity_frame(); 4];
        frames[0] = rot_z(0.0);            // (0,0)
        frames[2] = rot_z(PI / 4.0);       // (1,0)
        frames[3] = rot_z(PI / 2.0);       // (1,1)
        frames[1] = rot_z(3.0 * PI / 4.0); // (0,1)

        let defects = scan_disclinations(&frames, 2, 2, PI / 2.0);
        assert_eq!(defects.len(), 1, "expected exactly one defect in a 2x2 grid");
        assert!(
            defects[0].angle > PI / 2.0,
            "angle = {}",
            defects[0].angle
        );
        // For a +1/2 disclination the holonomy angle should be close to pi.
        assert_abs_diff_eq!(defects[0].angle, PI, epsilon = 1e-10);
    }

    #[test]
    fn scan_disclinations_dimensions_correct() {
        // A 4x5 grid has (4-1)*(5-1) = 12 plaquettes to check.
        // With all identity frames, there should be 0 defects but no panic.
        let id = identity_frame();
        let frames: Vec<_> = (0..20).map(|_| id).collect();
        let defects = scan_disclinations(&frames, 4, 5, PI / 2.0);
        assert!(defects.is_empty());
    }
}
