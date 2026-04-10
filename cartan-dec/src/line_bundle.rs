//! Complex line bundle sections for k-atic fields on 2-manifolds.
//!
//! On a 2-manifold, each tangent space is isomorphic to C. A k-atic director
//! (k-fold rotational symmetry) lives in the line bundle L_k with connection
//! k*omega, where omega is the Levi-Civita connection 1-form.
//!
//! This module provides:
//! - [`Section`]: complex section of L_k (one `Complex<f64>` per vertex)
//! - [`ConnectionAngles`]: discrete Levi-Civita connection on primal and dual edges
//! - [`BochnerLaplacian`]: sparse Hermitian Laplacian on L_k
//! - [`defect_charges`]: exact discrete topological charge per face
//!
//! ## References
//!
//! - Zhu, Saintillan, Chern (2024). "Active nematic fluids on Riemannian
//!   2-manifolds." arXiv:2405.06044. Sections 2.1 and 3.

use num_complex::Complex;
use sprs::{CsMat, TriMat};

use crate::hodge::HodgeStar;
use crate::mesh::Mesh;
use cartan_core::Manifold;

// ─────────────────────────────────────────────────────────────────────────────
// Section<K>: complex section of the line bundle L_k
// ─────────────────────────────────────────────────────────────────────────────

/// Complex section of the line bundle L_k on a 2-manifold.
///
/// Stores one `Complex<f64>` per vertex. The const generic `K` is the
/// k-atic order: K=1 for tangent vectors (spin-1), K=2 for nematics (spin-2),
/// K=4 for tetratics, K=6 for hexatics.
///
/// Transport along an edge with connection angle Omega multiplies by
/// `exp(-i * K * Omega)`.
#[derive(Debug, Clone)]
pub struct Section<const K: u32> {
    /// One complex number per vertex.
    pub values: Vec<Complex<f64>>,
}

impl<const K: u32> Section<K> {
    /// Zero section on `nv` vertices.
    pub fn zeros(nv: usize) -> Self {
        Self {
            values: vec![Complex::new(0.0, 0.0); nv],
        }
    }

    /// Uniform section: same value at every vertex.
    pub fn uniform(nv: usize, z: Complex<f64>) -> Self {
        Self {
            values: vec![z; nv],
        }
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.values.len()
    }

    /// Pointwise addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len());
        Self {
            values: self
                .values
                .iter()
                .zip(&other.values)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Pointwise scalar multiplication (complex).
    pub fn scale(&self, s: Complex<f64>) -> Self {
        Self {
            values: self.values.iter().map(|z| z * s).collect(),
        }
    }

    /// Pointwise scalar multiplication (real).
    pub fn scale_real(&self, s: f64) -> Self {
        Self {
            values: self.values.iter().map(|z| z * s).collect(),
        }
    }

    /// Pointwise norm |z| at each vertex.
    pub fn norms(&self) -> Vec<f64> {
        self.values.iter().map(|z| z.norm()).collect()
    }

    /// Mean norm over all vertices.
    pub fn mean_norm(&self) -> f64 {
        let n = self.values.len() as f64;
        self.values.iter().map(|z| z.norm()).sum::<f64>() / n
    }

    /// L2 inner product: sum_v conj(a_v) * b_v * dual_area_v.
    pub fn l2_inner(&self, other: &Self, dual_areas: &[f64]) -> Complex<f64> {
        self.values
            .iter()
            .zip(&other.values)
            .zip(dual_areas)
            .map(|((a, b), &area)| a.conj() * b * area)
            .sum()
    }

    /// Normalise each vertex to unit norm. Vertices with |z| < eps are left unchanged.
    pub fn normalise(&mut self, eps: f64) {
        for z in &mut self.values {
            let n = z.norm();
            if n > eps {
                *z /= n;
            }
        }
    }

    /// Scalar order parameter: 2|z| at each vertex (for K=2 nematics).
    pub fn scalar_order(&self) -> Vec<f64> {
        self.values.iter().map(|z| 2.0 * z.norm()).collect()
    }

    /// Mean scalar order parameter.
    pub fn mean_scalar_order(&self) -> f64 {
        let s = self.scalar_order();
        s.iter().sum::<f64>() / s.len() as f64
    }

    /// Convert to real (q1, q2) representation: q1 = Re(z), q2 = Im(z).
    pub fn to_real_components(&self) -> (Vec<f64>, Vec<f64>) {
        let q1 = self.values.iter().map(|z| z.re).collect();
        let q2 = self.values.iter().map(|z| z.im).collect();
        (q1, q2)
    }

    /// Construct from real (q1, q2) components: z = q1 + i*q2.
    pub fn from_real_components(q1: &[f64], q2: &[f64]) -> Self {
        assert_eq!(q1.len(), q2.len());
        Self {
            values: q1
                .iter()
                .zip(q2)
                .map(|(&r, &i)| Complex::new(r, i))
                .collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionAngles: discrete Levi-Civita connection
// ─────────────────────────────────────────────────────────────────────────────

/// Discrete Levi-Civita connection angles on a triangle mesh.
///
/// Stores two sets of angles:
/// - `primal[e]` (Omega_e): vertex-to-vertex transport along primal edge e.
///   Used for vertex-based Laplacians and defect charges.
/// - `dual[e]` (Omega_{e*}): face-to-face transport along dual edge e*.
///   Used for face-based vector Laplacians in Stokes solvers.
///
/// The connection angle Omega along an edge is the rotation angle between
/// the reference frames at the two endpoints, measured via the logarithmic map.
#[derive(Debug, Clone)]
pub struct ConnectionAngles {
    /// Primal edge connection angles (one per edge, vertex-to-vertex).
    pub primal: Vec<f64>,
    /// Dual edge connection angles (one per edge, face-to-face).
    pub dual: Vec<f64>,
}

impl ConnectionAngles {
    /// Compute connection angles from a triangle mesh on a manifold.
    ///
    /// For each primal edge (v_i, v_j), Omega_e is the angle between the
    /// reference frames at v_i and v_j. Specifically, we measure the angle
    /// of the tangent vector log(v_i, v_j) in v_i's frame, and the angle of
    /// log(v_j, v_i) in v_j's frame, then Omega_e = angle_at_j - angle_at_i - pi.
    ///
    /// For dual edges, Omega_{e*} is computed from the first edge vector of
    /// each face (the reference frame direction) transported across the shared edge.
    pub fn from_mesh<M: Manifold>(mesh: &Mesh<M, 3, 2>, manifold: &M) -> Self {
        let ne = mesh.n_boundaries();
        let mut primal = vec![0.0; ne];
        let mut dual = vec![0.0; ne];

        for e in 0..ne {
            let [vi, vj] = mesh.boundaries[e];
            let pi = &mesh.vertices[vi];
            let pj = &mesh.vertices[vj];

            // Primal connection: vertex-to-vertex.
            // Log maps give tangent vectors in each vertex's frame.
            let log_ij = manifold
                .log(pi, pj)
                .unwrap_or_else(|_| manifold.zero_tangent(pi));
            let log_ji = manifold
                .log(pj, pi)
                .unwrap_or_else(|_| manifold.zero_tangent(pj));

            // Compute angles of these tangent vectors in a local 2D frame.
            // For a 2-manifold in R^3, we use the first incident triangle to
            // define a consistent reference direction at each vertex.
            let angle_i = tangent_angle_2d(manifold, mesh, vi, &log_ij);
            let angle_j = tangent_angle_2d(manifold, mesh, vj, &log_ji);
            primal[e] = angle_j - angle_i - std::f64::consts::PI;

            // Dual connection: face-to-face.
            let cofaces = &mesh.boundary_simplices[e];
            if cofaces.len() == 2 {
                let angle_f0 = face_reference_angle(manifold, mesh, cofaces[0], e);
                let angle_f1 = face_reference_angle(manifold, mesh, cofaces[1], e);
                dual[e] = angle_f1 - angle_f0 - std::f64::consts::PI;
            }
        }

        Self { primal, dual }
    }
}

/// Compute the angle of a tangent vector at vertex `v` relative to the
/// reference frame defined by the first incident edge.
fn tangent_angle_2d<M: Manifold>(
    manifold: &M,
    mesh: &Mesh<M, 3, 2>,
    v: usize,
    tangent: &M::Tangent,
) -> f64 {
    let pv = &mesh.vertices[v];

    // Reference direction: log to the first neighbor via the first incident edge.
    let first_edge = mesh.vertex_boundaries[v][0];
    let [e0, e1] = mesh.boundaries[first_edge];
    let neighbor = if e0 == v { e1 } else { e0 };
    let ref_dir = manifold
        .log(pv, &mesh.vertices[neighbor])
        .unwrap_or_else(|_| manifold.zero_tangent(pv));

    // Compute angle between ref_dir and tangent using inner products.
    let rr = manifold.inner(pv, &ref_dir, &ref_dir);
    let tt = manifold.inner(pv, tangent, tangent);
    let rt = manifold.inner(pv, &ref_dir, tangent);

    if rr < 1e-30 || tt < 1e-30 {
        return 0.0;
    }

    let cos_a = rt / (rr.sqrt() * tt.sqrt());

    // For the sine, we need the "cross product" in 2D tangent space.
    // Use Gram determinant: sin(a) = sqrt(rr*tt - rt^2) / (|r|*|t|).
    // Sign from the orientation: we need a second basis vector.
    // Use the second incident edge to determine orientation.
    let second_edge = if mesh.vertex_boundaries[v].len() > 1 {
        mesh.vertex_boundaries[v][1]
    } else {
        first_edge
    };
    let [s0, s1] = mesh.boundaries[second_edge];
    let neighbor2 = if s0 == v { s1 } else { s0 };
    let ref_dir2 = manifold
        .log(pv, &mesh.vertices[neighbor2])
        .unwrap_or_else(|_| manifold.zero_tangent(pv));

    // The "cross product" sign: project tangent onto ref_dir2 component
    // perpendicular to ref_dir.
    let r2r = manifold.inner(pv, &ref_dir2, &ref_dir);
    let r2t = manifold.inner(pv, &ref_dir2, tangent);
    let r2_perp_component = r2t - r2r * rt / rr;

    // The reference cross component tells us the sign.
    let r2r2 = manifold.inner(pv, &ref_dir2, &ref_dir2);
    let ref_cross = r2r2 - r2r * r2r / rr; // |ref_dir2_perp|^2

    let sin_sign = if ref_cross > 1e-30 {
        r2_perp_component.signum()
    } else {
        1.0
    };

    let sin_a = sin_sign * (1.0 - cos_a.clamp(-1.0, 1.0).powi(2)).sqrt();
    sin_a.atan2(cos_a)
}

/// Compute the angle of the shared edge in a face's reference frame.
fn face_reference_angle<M: Manifold>(
    manifold: &M,
    mesh: &Mesh<M, 3, 2>,
    face: usize,
    edge: usize,
) -> f64 {
    let simplex = &mesh.simplices[face];
    let [e0, e1] = mesh.boundaries[edge];

    // The shared edge direction as seen from simplex[0].
    let p0 = &mesh.vertices[simplex[0]];
    let edge_at_p0 = if e0 == simplex[0] {
        manifold
            .log(p0, &mesh.vertices[e1])
            .unwrap_or_else(|_| manifold.zero_tangent(p0))
    } else if e1 == simplex[0] {
        manifold
            .log(p0, &mesh.vertices[e0])
            .unwrap_or_else(|_| manifold.zero_tangent(p0))
    } else {
        // Edge doesn't touch simplex[0]; use e0 endpoint.
        manifold
            .log(p0, &mesh.vertices[e0])
            .unwrap_or_else(|_| manifold.zero_tangent(p0))
    };

    tangent_angle_2d(manifold, mesh, simplex[0], &edge_at_p0)
}

// ─────────────────────────────────────────────────────────────────────────────
// BochnerLaplacian<K>: sparse complex Laplacian on L_k
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse Hermitian Bochner Laplacian on the line bundle L_k.
///
/// Assembled from cotangent weights and connection transport phases.
/// Negative semi-definite by construction.
///
/// Entry for edge (i,j) with cotangent weight w_e and connection angle Omega_e:
/// - Off-diagonal: -w_e * exp(-i*K*Omega_e)
/// - Diagonal: sum of incident cotangent weights
/// - Normalised by star_0^{-1} (inverse dual area) on the left.
#[derive(Debug, Clone)]
pub struct BochnerLaplacian<const K: u32> {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Sparse matrix (nv x nv, complex).
    matrix: CsMat<Complex<f64>>,
}

impl<const K: u32> BochnerLaplacian<K> {
    /// Assemble the Bochner Laplacian from mesh data.
    ///
    /// Uses cotangent weights from the Hodge star (star1) and primal
    /// connection angles. The result is star_0^{-1} * L where L is the
    /// raw stiffness matrix.
    pub fn from_mesh_data(
        mesh: &Mesh<impl Manifold, 3, 2>,
        hodge: &HodgeStar,
        connection: &ConnectionAngles,
    ) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let star0 = hodge.star0();
        let star1 = hodge.star1();

        // Build triplets for the sparse matrix.
        let mut triplets = TriMat::new((nv, nv));

        // Precompute inverse dual areas for row scaling.
        let inv_star0: Vec<f64> = (0..nv)
            .map(|i| {
                if star0[i].abs() > 1e-30 {
                    -1.0 / star0[i]
                } else {
                    0.0
                }
            })
            .collect();

        for e in 0..ne {
            let [vi, vj] = mesh.boundaries[e];
            let w = star1[e];
            let omega = connection.primal[e];
            let phase = Complex::from_polar(1.0, -(K as f64) * omega);

            // Off-diagonal: star_0^{-1} * (-w) * exp(-iK*Omega)
            triplets.add_triplet(vi, vj, inv_star0[vi] * (-w * phase));
            triplets.add_triplet(vj, vi, inv_star0[vj] * (-w * phase.conj()));

            // Diagonal contributions: star_0^{-1} * w
            triplets.add_triplet(vi, vi, Complex::new(inv_star0[vi] * w, 0.0));
            triplets.add_triplet(vj, vj, Complex::new(inv_star0[vj] * w, 0.0));
        }

        let raw = triplets.to_csc();

        Self {
            n_vertices: nv,
            matrix: raw,
        }
    }

    /// Apply the Laplacian to a section: result = Delta * z.
    pub fn apply(&self, z: &Section<K>) -> Section<K> {
        let input: Vec<Complex<f64>> = z.values.clone();
        let mut output = vec![Complex::new(0.0, 0.0); self.n_vertices];

        // Sparse matrix-vector multiply.
        for (col, col_view) in self.matrix.outer_iterator().enumerate() {
            for (row, &val) in col_view.iter() {
                output[row] += val * input[col];
            }
        }

        Section { values: output }
    }

    /// Reference to the underlying sparse matrix.
    pub fn matrix(&self) -> &CsMat<Complex<f64>> {
        &self.matrix
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Defect charges
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the topological charge of a k-atic section at each face.
///
/// Per-face charge (paper 1 Eq. 50):
/// Z_f = (1/2piK) * sum_{e < f} arg(z_j / (exp(-iK*Omega_e) * z_i))
///       + (1/2pi) * A_f * K_gauss
///
/// The discrete Poincare-Hopf theorem holds exactly:
/// sum_f Z_f = chi(M) = 2 - 2*genus.
///
/// `gaussian_curvature_per_face[f]` = A_f * K_f (integrated Gaussian curvature).
pub fn defect_charges<const K: u32>(
    section: &Section<K>,
    connection: &ConnectionAngles,
    mesh: &Mesh<impl Manifold, 3, 2>,
    gaussian_curvature_per_face: &[f64],
) -> Vec<f64> {
    let nf = mesh.n_simplices();
    let mut charges = vec![0.0; nf];
    let two_pi = 2.0 * std::f64::consts::PI;
    let k = K as f64;

    for f in 0..nf {
        let simplex = &mesh.simplices[f];
        let mut winding = 0.0;

        // Sum over the 3 edges of this face.
        for local_edge in 0..3 {
            let vi = simplex[local_edge];
            let vj = simplex[(local_edge + 1) % 3];

            // Find the global edge index for (vi, vj).
            let edge_idx = find_edge(mesh, vi, vj);
            let omega = connection.primal[edge_idx];

            // Oriented: if mesh.boundaries[edge_idx] = [vi, vj], use omega as-is.
            // If reversed, negate omega.
            let [e0, _e1] = mesh.boundaries[edge_idx];
            let oriented_omega = if e0 == vi { omega } else { -omega };

            let zi = section.values[vi];
            let zj = section.values[vj];

            if zi.norm() < 1e-30 || zj.norm() < 1e-30 {
                continue;
            }

            // Transport z_i to z_j's frame: z_i_transported = exp(-iK*Omega) * z_i
            let phase = Complex::from_polar(1.0, -k * oriented_omega);
            let transported = phase * zi;

            // Angle difference: arg(z_j / transported)
            let ratio = zj / transported;
            winding += ratio.arg();
        }

        charges[f] = winding / (two_pi * k) + gaussian_curvature_per_face[f] / two_pi;
    }

    charges
}

/// Find the edge index connecting vertices vi and vj.
fn find_edge(mesh: &Mesh<impl Manifold, 3, 2>, vi: usize, vj: usize) -> usize {
    let key = if vi < vj { [vi, vj] } else { [vj, vi] };
    for &e in &mesh.vertex_boundaries[vi] {
        let [e0, e1] = mesh.boundaries[e];
        let sorted = if e0 < e1 { [e0, e1] } else { [e1, e0] };
        if sorted == key {
            return e;
        }
    }
    panic!("edge ({vi}, {vj}) not found in mesh");
}

// ─────────────────────────────────────────────────────────────────────────────
// Veronese map: Section<2> <-> Q-tensor
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a nematic section (K=2) to Q-tensor components (q1, q2).
///
/// The Veronese map sends z = |z| exp(i*theta) to Q = z (x) z, which in
/// the traceless symmetric representation gives:
/// - q1 = Re(z^2) = |z|^2 * cos(2*theta) = Re(z)^2 - Im(z)^2
/// - q2 = Im(z^2) = |z|^2 * sin(2*theta) = 2 * Re(z) * Im(z)
///
/// Wait: for K=2, the section stores z such that the director angle is theta,
/// and z = |z| exp(i*theta). The Q-tensor is Q = S/2 * [[cos 2t, sin 2t],[sin 2t, -cos 2t]].
/// With our convention q1 = (Q_xx - Q_yy)/2, q2 = Q_xy:
/// q1 = S/2 * cos(2t), q2 = S/2 * sin(2t).
/// And S/2 = |z|, so q1 = |z| cos(2t), q2 = |z| sin(2t).
/// But z = |z| exp(it), so z^2 = |z|^2 exp(2it).
/// Actually, with |z| = S/2: Re(z) = (S/2) cos(t), Im(z) = (S/2) sin(t).
/// z^2 = (S/2)^2 exp(2it) = (S^2/4)(cos 2t + i sin 2t).
/// And q1 = (S/2) cos 2t = (S/2) * Re(exp(2it)), q2 = (S/2) * Im(exp(2it)).
/// So (q1, q2) = (Re(z), Im(z)) directly! Since z already encodes the
/// doubled angle via K=2.
///
/// Returns (q1_per_vertex, q2_per_vertex).
pub fn section_to_q_components(section: &Section<2>) -> (Vec<f64>, Vec<f64>) {
    section.to_real_components()
}

/// Construct a nematic section from Q-tensor components.
///
/// z = q1 + i*q2.
pub fn q_components_to_section(q1: &[f64], q2: &[f64]) -> Section<2> {
    Section::<2>::from_real_components(q1, q2)
}
