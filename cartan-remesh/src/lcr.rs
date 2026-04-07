// ~/cartan/cartan-remesh/src/lcr.rs

//! Length-cross-ratio (LCR) conformal regularization.
//!
//! The length-cross-ratio of an interior edge with diamond vertices {i, j, k, l}
//! (where i,j are the edge endpoints and k,l are the opposite vertices of the
//! two adjacent triangles) is:
//!
//!   lcr = dist(i,l) * dist(j,k) / (dist(k,i) * dist(l,j))
//!
//! For a Delaunay triangulation of a flat surface, all interior LCRs equal 1.0.
//! The LCR spring energy penalises deviation from a reference LCR snapshot,
//! encouraging conformal mesh quality during remeshing.

use cartan_core::Manifold;
use cartan_dec::Mesh;

/// Compute the length-cross-ratio of an edge in the mesh.
///
/// For an interior edge (two adjacent triangles), the diamond has vertices
/// {i, j, k, l} where [i, j] is the edge and k, l are the opposite vertices
/// of the two adjacent triangles. The LCR is:
///
///   lcr = dist(i, l) * dist(j, k) / (dist(k, i) * dist(l, j))
///
/// Returns 1.0 for boundary edges (only one adjacent face), since no
/// meaningful cross-ratio exists without a full diamond.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn length_cross_ratio<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    edge: usize,
) -> f64 {
    assert!(edge < mesh.n_boundaries(), "edge index out of bounds");

    let adjacent = &mesh.boundary_simplices[edge];
    if adjacent.len() < 2 {
        // Boundary edge: no full diamond, return neutral LCR.
        return 1.0;
    }

    let [vi, vj] = mesh.boundaries[edge];

    // Find opposite vertex k in the first adjacent triangle.
    let tri_0 = mesh.simplices[adjacent[0]];
    let vk = tri_0
        .iter()
        .copied()
        .find(|&v| v != vi && v != vj)
        .expect("triangle must have an opposite vertex");

    // Find opposite vertex l in the second adjacent triangle.
    let tri_1 = mesh.simplices[adjacent[1]];
    let vl = tri_1
        .iter()
        .copied()
        .find(|&v| v != vi && v != vj)
        .expect("triangle must have an opposite vertex");

    let pi = &mesh.vertices[vi];
    let pj = &mesh.vertices[vj];
    let pk = &mesh.vertices[vk];
    let pl = &mesh.vertices[vl];

    let dist_il = manifold.dist(pi, pl).unwrap_or(0.0);
    let dist_jk = manifold.dist(pj, pk).unwrap_or(0.0);
    let dist_ki = manifold.dist(pk, pi).unwrap_or(0.0);
    let dist_lj = manifold.dist(pl, pj).unwrap_or(0.0);

    let denom = dist_ki * dist_lj;
    if denom < 1e-30 {
        return 1.0;
    }
    (dist_il * dist_jk) / denom
}

/// Capture reference LCR values for every edge in the mesh.
///
/// Returns a vector of length `mesh.n_boundaries()` where entry `e` is the
/// length-cross-ratio of edge `e`. Boundary edges get LCR = 1.0.
///
/// This snapshot is used as the reference configuration for the LCR spring
/// energy, so that subsequent remeshing steps can penalise conformal
/// distortion relative to the initial mesh.
pub fn capture_reference_lcrs<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
) -> Vec<f64> {
    (0..mesh.n_boundaries())
        .map(|e| length_cross_ratio(mesh, manifold, e))
        .collect()
}

/// Total LCR spring energy measuring conformal distortion from a reference.
///
/// The energy sums over all interior edges:
///
///   E = 0.5 * kst * sum_e ((lcr_e - ref_e)^2 / ref_e^2)
///
/// where `lcr_e` is the current length-cross-ratio of edge `e`, `ref_e` is
/// the reference value from the initial mesh, and `kst` is the spring
/// stiffness. Boundary edges (ref = 1.0, current = 1.0) contribute zero.
///
/// # Panics
///
/// Panics if `ref_lcrs.len() != mesh.n_boundaries()`.
pub fn lcr_spring_energy<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    ref_lcrs: &[f64],
    kst: f64,
) -> f64 {
    assert_eq!(
        ref_lcrs.len(),
        mesh.n_boundaries(),
        "ref_lcrs length must match edge count"
    );

    let mut energy = 0.0;
    for (e, &ref_val) in ref_lcrs.iter().enumerate() {
        if ref_val.abs() < 1e-30 {
            continue;
        }
        let lcr_e = length_cross_ratio(mesh, manifold, e);
        let diff = lcr_e - ref_val;
        energy += (diff * diff) / (ref_val * ref_val);
    }
    0.5 * kst * energy
}

/// Per-vertex gradient of the LCR spring energy.
///
/// Returns a vector of length `mesh.n_vertices()` containing one tangent
/// vector per vertex. Currently returns zero tangent vectors for all
/// vertices. Analytical gradients of the LCR energy with respect to vertex
/// positions require differentiating geodesic distances through the manifold
/// exponential map, which is a future extension.
///
/// # Panics
///
/// Panics if `ref_lcrs.len() != mesh.n_boundaries()`.
pub fn lcr_spring_gradient<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    ref_lcrs: &[f64],
    _kst: f64,
) -> Vec<M::Tangent> {
    assert_eq!(
        ref_lcrs.len(),
        mesh.n_boundaries(),
        "ref_lcrs length must match edge count"
    );

    (0..mesh.n_vertices())
        .map(|v| manifold.zero_tangent(&mesh.vertices[v]))
        .collect()
}
