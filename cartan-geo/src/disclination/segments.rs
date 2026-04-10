// ~/cartan/cartan-geo/src/disclination/segments.rs

//! Disclination segment detection on a 3D Q-tensor field.
//!
//! ## Algorithm
//!
//! For each edge of the 3D Cartesian grid, compute the holonomy of the dual
//! loop surrounding that edge. If the holonomy rotation angle exceeds pi/2,
//! the edge carries a half-integer disclination charge.
//!
//! Each edge direction has a 4-face dual loop:
//! - x-directed edge (i,j,l): dual loop in the yz-plane
//! - y-directed edge (i,j,l): dual loop in the xz-plane
//! - z-directed edge (i,j,l): dual loop in the xy-plane
//!
//! Face-center frames are computed by averaging the Q-tensors at the 4
//! face-corner vertices and extracting an orthonormal eigenvector frame.
//!
//! ## References
//!
//! - Binysh, J. & Alexander, G. P. (2018). J. Phys. A 51, 385202.
//! - Smalyukh, I. I. (2010). Phys. Rev. Lett. 104, 097801.

use nalgebra::SMatrix;

use crate::holonomy::{is_half_disclination, loop_holonomy};

/// Minimal interface required by disclination scanning algorithms.
///
/// Implement this trait on any 3D Q-tensor field type to make it usable with
/// [`scan_disclination_lines_3d`]. Volterra's `QField3D` implements this trait.
pub trait QTensorField3D {
    /// Number of grid points along x.
    fn nx(&self) -> usize;
    /// Number of grid points along y.
    fn ny(&self) -> usize;
    /// Number of grid points along z.
    fn nz(&self) -> usize;
    /// Grid spacing (uniform, isotropic).
    fn dx(&self) -> f64;
    /// Linear index for vertex `(i, j, l)`.
    fn idx(&self, i: usize, j: usize, l: usize) -> usize;
    /// Q-tensor at linear vertex index `k` as a 3×3 traceless symmetric matrix.
    fn embed_matrix3(&self, k: usize) -> SMatrix<f64, 3, 3>;
}

/// Z2 orientation sign for a half-integer disclination charge.
#[derive(Debug, Clone, PartialEq)]
pub enum Sign {
    Positive,
    Negative,
}

/// Topological charge of a disclination segment.
///
/// Currently only Z2 half-integer charges are emitted by the scanner.
/// The `Anti` variant is reserved for Q8 (octahedral) extensions.
#[derive(Debug, Clone, PartialEq)]
pub enum DisclinationCharge {
    /// ±1/2 disclination (nematic half-integer charge).
    Half(Sign),
    /// Reserved for Q8 extension.
    Anti,
}

/// A single disclination segment: one grid edge pierced by a disclination line.
#[derive(Debug, Clone)]
pub struct DisclinationSegment {
    /// Linear vertex indices of the edge endpoints.
    pub edge: (usize, usize),
    /// Topological charge of the disclination.
    pub charge: DisclinationCharge,
    /// Midpoint of the edge in grid coordinates (units of dx).
    pub midpoint: [f64; 3],
}

/// Threshold angle for half-disclination detection: rotation angle > pi/2.
///
/// A pi rotation (half-disclination) gives angle = pi, which is > pi/2.
/// The identity (no defect) gives angle = 0. The threshold pi/2 is the
/// midpoint between them.
const HALF_DISC_THRESHOLD: f64 = std::f64::consts::FRAC_PI_2;

/// Average the Q-tensor at 4 face-corner vertices and extract an SO(3) frame.
///
/// The frame is the eigenvector matrix of the average Q (symmetric
/// eigendecomposition). The columns form an orthonormal basis ordered by
/// eigenvalue, suitable for holonomy computation.
fn face_center_frame<F: QTensorField3D>(q: &F, vs: [usize; 4]) -> SMatrix<f64, 3, 3> {
    let mut avg = SMatrix::<f64, 3, 3>::zeros();
    for &v in &vs {
        avg += q.embed_matrix3(v);
    }
    avg /= 4.0;
    // Symmetric eigendecomposition: columns of eigenvectors form an orthonormal frame
    avg.symmetric_eigen().eigenvectors
}

/// Scan a 3D Q-tensor field for disclination segments using the dual-loop holonomy method.
///
/// For each edge in the x, y, and z directions, the four adjacent face-centers
/// are visited in a fixed cyclic order. The holonomy of that 4-face loop is
/// computed. If the rotation angle exceeds pi/2, a `DisclinationSegment` is
/// recorded for that edge.
///
/// The field `q` must implement [`QTensorField3D`]. Volterra's `QField3D` and
/// the test helper [`UniformQGrid`] both implement this trait.
///
/// **Winding note:** The face traversal order (fa→fb→fc→fd) is clockwise when
/// viewed from the positive axis direction (not CCW). The `is_half_disclination`
/// check is purely based on rotation angle and is sign-independent, so detection
/// is unaffected. If sign discrimination (`Sign::Positive` vs `Sign::Negative`)
/// is ever implemented from the holonomy winding, the traversal order must be
/// corrected to CCW first.
///
/// Returns a `Vec<DisclinationSegment>` with one entry per pierced edge.
/// An empty vec indicates no disclinations were found.
pub fn scan_disclination_lines_3d<F: QTensorField3D>(q: &F) -> Vec<DisclinationSegment> {
    let nx = q.nx();
    let ny = q.ny();
    let nz = q.nz();
    let dx = q.dx();

    // Helper: periodic neighbor indices
    let ip_fn = |i: usize| (i + 1) % nx;
    let im_fn = |i: usize| (i + nx - 1) % nx;
    let jp_fn = |j: usize| (j + 1) % ny;
    let jm_fn = |j: usize| (j + ny - 1) % ny;
    let lp_fn = |l: usize| (l + 1) % nz;
    let lm_fn = |l: usize| (l + nz - 1) % nz;

    let mut segs = Vec::new();

    for i in 0..nx {
        let ip = ip_fn(i);
        let im = im_fn(i);
        for j in 0..ny {
            let jp = jp_fn(j);
            let jm = jm_fn(j);
            for l in 0..nz {
                let lp = lp_fn(l);
                let lm = lm_fn(l);

                // ── x-directed edge at (i,j,l) → (ip,j,l) ─────────────────
                // Dual loop in yz-plane (fa→fb→fc→fd is CW viewed from +x)
                {
                    // fa: face with corners (i,j,l), (i,j,lp), (i,jp,lp), (i,jp,l)
                    let fa = face_center_frame(
                        q,
                        [
                            q.idx(i, j, l),
                            q.idx(i, j, lp),
                            q.idx(i, jp, lp),
                            q.idx(i, jp, l),
                        ],
                    );
                    // fb: face with corners (i,j,lm), (i,j,l), (i,jp,l), (i,jp,lm)
                    let fb = face_center_frame(
                        q,
                        [
                            q.idx(i, j, lm),
                            q.idx(i, j, l),
                            q.idx(i, jp, l),
                            q.idx(i, jp, lm),
                        ],
                    );
                    // fc: face with corners (i,jm,lm), (i,jm,l), (i,j,l), (i,j,lm)
                    let fc = face_center_frame(
                        q,
                        [
                            q.idx(i, jm, lm),
                            q.idx(i, jm, l),
                            q.idx(i, j, l),
                            q.idx(i, j, lm),
                        ],
                    );
                    // fd: face with corners (i,jm,l), (i,jm,lp), (i,j,lp), (i,j,l)
                    let fd = face_center_frame(
                        q,
                        [
                            q.idx(i, jm, l),
                            q.idx(i, jm, lp),
                            q.idx(i, j, lp),
                            q.idx(i, j, l),
                        ],
                    );

                    let frames = [fa, fb, fc, fd];
                    let hol = loop_holonomy(&frames);
                    if is_half_disclination(&hol, HALF_DISC_THRESHOLD) {
                        let v0 = q.idx(i, j, l);
                        let v1 = q.idx(ip, j, l);
                        segs.push(DisclinationSegment {
                            edge: (v0, v1),
                            charge: DisclinationCharge::Half(Sign::Positive),
                            midpoint: [(i as f64 + 0.5) * dx, j as f64 * dx, l as f64 * dx],
                        });
                    }
                }

                // ── y-directed edge at (i,j,l) → (i,jp,l) ─────────────────
                // Dual loop in xz-plane (fa→fb→fc→fd is CW viewed from +y)
                {
                    // fa: corners (i,j,l), (ip,j,l), (ip,j,lp), (i,j,lp)
                    let fa = face_center_frame(
                        q,
                        [
                            q.idx(i, j, l),
                            q.idx(ip, j, l),
                            q.idx(ip, j, lp),
                            q.idx(i, j, lp),
                        ],
                    );
                    // fb: corners (im,j,l), (i,j,l), (i,j,lp), (im,j,lp)
                    let fb = face_center_frame(
                        q,
                        [
                            q.idx(im, j, l),
                            q.idx(i, j, l),
                            q.idx(i, j, lp),
                            q.idx(im, j, lp),
                        ],
                    );
                    // fc: corners (im,j,lm), (i,j,lm), (i,j,l), (im,j,l)
                    let fc = face_center_frame(
                        q,
                        [
                            q.idx(im, j, lm),
                            q.idx(i, j, lm),
                            q.idx(i, j, l),
                            q.idx(im, j, l),
                        ],
                    );
                    // fd: corners (i,j,lm), (ip,j,lm), (ip,j,l), (i,j,l)
                    let fd = face_center_frame(
                        q,
                        [
                            q.idx(i, j, lm),
                            q.idx(ip, j, lm),
                            q.idx(ip, j, l),
                            q.idx(i, j, l),
                        ],
                    );

                    let frames = [fa, fb, fc, fd];
                    let hol = loop_holonomy(&frames);
                    if is_half_disclination(&hol, HALF_DISC_THRESHOLD) {
                        let v0 = q.idx(i, j, l);
                        let v1 = q.idx(i, jp, l);
                        segs.push(DisclinationSegment {
                            edge: (v0, v1),
                            charge: DisclinationCharge::Half(Sign::Positive),
                            midpoint: [i as f64 * dx, (j as f64 + 0.5) * dx, l as f64 * dx],
                        });
                    }
                }

                // ── z-directed edge at (i,j,l) → (i,j,lp) ─────────────────
                // Dual loop in xy-plane (fa→fb→fc→fd is CW viewed from +z)
                {
                    // fa: corners (i,j,l), (i,jp,l), (ip,jp,l), (ip,j,l)
                    let fa = face_center_frame(
                        q,
                        [
                            q.idx(i, j, l),
                            q.idx(i, jp, l),
                            q.idx(ip, jp, l),
                            q.idx(ip, j, l),
                        ],
                    );
                    // fb: corners (i,jm,l), (i,j,l), (ip,j,l), (ip,jm,l)
                    let fb = face_center_frame(
                        q,
                        [
                            q.idx(i, jm, l),
                            q.idx(i, j, l),
                            q.idx(ip, j, l),
                            q.idx(ip, jm, l),
                        ],
                    );
                    // fc: corners (im,jm,l), (im,j,l), (i,j,l), (i,jm,l)
                    let fc = face_center_frame(
                        q,
                        [
                            q.idx(im, jm, l),
                            q.idx(im, j, l),
                            q.idx(i, j, l),
                            q.idx(i, jm, l),
                        ],
                    );
                    // fd: corners (im,j,l), (im,jp,l), (i,jp,l), (i,j,l)
                    let fd = face_center_frame(
                        q,
                        [
                            q.idx(im, j, l),
                            q.idx(im, jp, l),
                            q.idx(i, jp, l),
                            q.idx(i, j, l),
                        ],
                    );

                    let frames = [fa, fb, fc, fd];
                    let hol = loop_holonomy(&frames);
                    if is_half_disclination(&hol, HALF_DISC_THRESHOLD) {
                        let v0 = q.idx(i, j, l);
                        let v1 = q.idx(i, j, lp);
                        segs.push(DisclinationSegment {
                            edge: (v0, v1),
                            charge: DisclinationCharge::Half(Sign::Positive),
                            midpoint: [i as f64 * dx, j as f64 * dx, (l as f64 + 0.5) * dx],
                        });
                    }
                }
            }
        }
    }

    segs
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal `QTensorField3D` implementation for unit testing.
///
/// Stores a single Q-tensor value that is returned for every grid site.
/// Useful for verifying that a uniform field produces no disclinations.
#[cfg(test)]
pub struct UniformQGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    /// Fixed traceless symmetric Q tensor (3×3) returned for every vertex.
    pub qtensor: SMatrix<f64, 3, 3>,
}

#[cfg(test)]
impl QTensorField3D for UniformQGrid {
    fn nx(&self) -> usize {
        self.nx
    }
    fn ny(&self) -> usize {
        self.ny
    }
    fn nz(&self) -> usize {
        self.nz
    }
    fn dx(&self) -> f64 {
        self.dx
    }
    fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        ((i % self.nx) * self.ny + (j % self.ny)) * self.nz + (l % self.nz)
    }
    fn embed_matrix3(&self, _k: usize) -> SMatrix<f64, 3, 3> {
        self.qtensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_q(nx: usize, ny: usize, nz: usize) -> UniformQGrid {
        // Uniaxial Q tensor along x: Q = S(n⊗n − I/3), n = [1,0,0], S = 0.6
        let s = 0.6_f64;
        let qtensor = SMatrix::<f64, 3, 3>::new(
            2.0 / 3.0 * s,
            0.0,
            0.0,
            0.0,
            -1.0 / 3.0 * s,
            0.0,
            0.0,
            0.0,
            -1.0 / 3.0 * s,
        );
        UniformQGrid {
            nx,
            ny,
            nz,
            dx: 1.0,
            qtensor,
        }
    }

    #[test]
    fn test_scan_no_defects_uniform() {
        // A perfectly uniform Q field has no disclinations.
        let q = uniform_q(8, 8, 8);
        let segs = scan_disclination_lines_3d(&q);
        assert!(
            segs.is_empty(),
            "Uniform Q field should have no disclination segments, got {}",
            segs.len()
        );
    }

    #[test]
    fn test_disclination_charge_enum() {
        let c = DisclinationCharge::Half(Sign::Positive);
        assert!(matches!(c, DisclinationCharge::Half(_)));
    }
}
