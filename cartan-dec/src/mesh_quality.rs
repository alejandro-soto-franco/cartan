//! Mesh quality predicates and improvement operators for DEC correctness.
//!
//! DEC requires:
//! - **Delaunay**: non-negative cotangent weights (star_1 >= 0).
//! - **Well-centered**: positive dual cell volumes (star_0 > 0).
//!
//! This module provides predicates to check these properties and operators
//! to improve mesh quality via edge flips and Lloyd smoothing.

use cartan_core::Manifold;
use crate::mesh::Mesh;

/// Compute the angle at vertex `v_idx` within simplex `s` on manifold `m`.
///
/// For a triangle with vertices (a, b, c), the angle at `a` is the angle
/// between edges ab and ac measured in the tangent space at `a`.
///
/// Returns the angle in radians.
pub fn angle_at_vertex<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
    s: usize,
    v_idx: usize,
) -> f64 {
    let simplex = &mesh.simplices[s];

    // Find the position of v_idx in the simplex.
    let local = simplex
        .iter()
        .position(|&v| v == v_idx)
        .expect("v_idx not in simplex");

    // The other two vertices.
    let other: Vec<usize> = simplex
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != local)
        .map(|(_, &v)| v)
        .collect();

    let v0 = &mesh.vertices[v_idx];
    let u = manifold
        .log(v0, &mesh.vertices[other[0]])
        .unwrap_or_else(|_| manifold.zero_tangent(v0));
    let v = manifold
        .log(v0, &mesh.vertices[other[1]])
        .unwrap_or_else(|_| manifold.zero_tangent(v0));

    let uu = manifold.inner(v0, &u, &u);
    let vv = manifold.inner(v0, &v, &v);
    let uv = manifold.inner(v0, &u, &v);

    let cos_angle = uv / (uu.sqrt() * vv.sqrt());
    cos_angle.clamp(-1.0, 1.0).acos()
}

/// Check whether a triangle mesh is intrinsic Delaunay.
///
/// A mesh is Delaunay if for every interior edge, the sum of opposite
/// angles in the two adjacent triangles is <= pi. Boundary edges (with
/// only one adjacent triangle) are always Delaunay.
pub fn is_delaunay<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
) -> bool {
    assert_eq!(K, 3, "is_delaunay requires triangle meshes (K=3)");

    for e in 0..mesh.n_boundaries() {
        let cofaces = &mesh.boundary_simplices[e];
        if cofaces.len() != 2 {
            continue;
        }
        let edge = &mesh.boundaries[e];
        let alpha = opposite_angle(mesh, manifold, cofaces[0], edge);
        let beta = opposite_angle(mesh, manifold, cofaces[1], edge);
        if alpha + beta > std::f64::consts::PI + 1e-10 {
            return false;
        }
    }
    true
}

/// Angle opposite to edge `edge` in simplex `s`.
///
/// The opposite vertex is the one in `s` that is not in `edge`.
fn opposite_angle<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
    s: usize,
    edge: &[usize; B],
) -> f64 {
    let simplex = &mesh.simplices[s];
    let opp = simplex
        .iter()
        .find(|&&v| !edge.contains(&v))
        .expect("simplex must have a vertex not in edge");
    angle_at_vertex(mesh, manifold, s, *opp)
}

/// Check whether a triangle mesh is well-centered.
///
/// A mesh is well-centered if the circumcenter of every simplex lies
/// strictly inside the simplex. For 2D triangles, this is equivalent
/// to all angles being acute (< pi/2).
///
/// Uses the existing `Mesh::check_well_centered` implementation.
pub fn is_well_centered<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
) -> bool {
    mesh.check_well_centered(manifold).is_ok()
}

/// Summary of mesh quality metrics.
#[derive(Debug, Clone)]
pub struct MeshQualityReport {
    /// Smallest angle in the mesh (radians).
    pub min_angle: f64,
    /// Largest angle in the mesh (radians).
    pub max_angle: f64,
    /// Number of edges violating the Delaunay condition.
    pub non_delaunay_edges: usize,
    /// Number of simplices whose circumcenter lies outside the simplex.
    pub non_well_centered_simplices: usize,
    /// Total number of vertices.
    pub n_vertices: usize,
    /// Total number of edges.
    pub n_edges: usize,
    /// Total number of simplices.
    pub n_simplices: usize,
    /// Whether the mesh is Delaunay.
    pub is_delaunay: bool,
    /// Whether the mesh is well-centered.
    pub is_well_centered: bool,
}

/// Compute a full quality report for a triangle mesh.
pub fn quality_report<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
) -> MeshQualityReport {
    assert_eq!(K, 3, "quality_report requires triangle meshes (K=3)");

    let mut min_angle = f64::MAX;
    let mut max_angle = 0.0_f64;

    for s in 0..mesh.n_simplices() {
        for &v in &mesh.simplices[s] {
            let a = angle_at_vertex(mesh, manifold, s, v);
            min_angle = min_angle.min(a);
            max_angle = max_angle.max(a);
        }
    }

    let mut non_delaunay = 0;
    for e in 0..mesh.n_boundaries() {
        let cofaces = &mesh.boundary_simplices[e];
        if cofaces.len() != 2 {
            continue;
        }
        let edge = &mesh.boundaries[e];
        let alpha = opposite_angle(mesh, manifold, cofaces[0], edge);
        let beta = opposite_angle(mesh, manifold, cofaces[1], edge);
        if alpha + beta > std::f64::consts::PI + 1e-10 {
            non_delaunay += 1;
        }
    }

    let mut non_wc = 0;
    for s in 0..mesh.n_simplices() {
        let all_acute = mesh.simplices[s]
            .iter()
            .all(|&v| angle_at_vertex(mesh, manifold, s, v) < std::f64::consts::FRAC_PI_2 + 1e-10);
        if !all_acute {
            non_wc += 1;
        }
    }

    MeshQualityReport {
        min_angle,
        max_angle,
        non_delaunay_edges: non_delaunay,
        non_well_centered_simplices: non_wc,
        n_vertices: mesh.n_vertices(),
        n_edges: mesh.n_boundaries(),
        n_simplices: mesh.n_simplices(),
        is_delaunay: non_delaunay == 0,
        is_well_centered: non_wc == 0,
    }
}

/// Flip an interior edge in a triangle mesh.
///
/// Given edge e between two triangles, replaces the shared edge with the
/// opposite diagonal. Rebuilds topology afterward.
///
/// Panics if the edge is a boundary edge (only one adjacent triangle).
fn edge_flip<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    e: usize,
) {
    let cofaces = mesh.boundary_simplices[e].clone();
    assert_eq!(cofaces.len(), 2, "cannot flip a boundary edge");
    let t0 = cofaces[0];
    let t1 = cofaces[1];
    let edge = mesh.boundaries[e];

    let opp0 = mesh.simplices[t0]
        .iter()
        .find(|&&v| !edge.contains(&v))
        .copied()
        .expect("opposite vertex in t0");
    let opp1 = mesh.simplices[t1]
        .iter()
        .find(|&&v| !edge.contains(&v))
        .copied()
        .expect("opposite vertex in t1");

    // Replace the two triangles with the flipped diagonal.
    mesh.simplices[t0] = [edge[0], opp1, opp0];
    mesh.simplices[t1] = [opp1, edge[1], opp0];

    mesh.rebuild_topology();
}

/// Convert a triangle mesh to intrinsic Delaunay via edge flips.
///
/// Iterates over all interior edges, flipping any edge whose opposite
/// angles sum to > pi. Repeats until no more flips are needed.
///
/// The number of vertices and triangles is preserved (no Steiner points).
pub fn make_delaunay<M: Manifold>(
    mut mesh: Mesh<M, 3, 2>,
    manifold: &M,
) -> Mesh<M, 3, 2> {
    let max_iterations = mesh.n_boundaries() * mesh.n_boundaries();

    for _ in 0..max_iterations {
        let mut flipped = false;
        for e in 0..mesh.n_boundaries() {
            let cofaces = &mesh.boundary_simplices[e];
            if cofaces.len() != 2 {
                continue;
            }
            let edge = &mesh.boundaries[e];
            let alpha = opposite_angle(&mesh, manifold, cofaces[0], edge);
            let beta = opposite_angle(&mesh, manifold, cofaces[1], edge);
            if alpha + beta > std::f64::consts::PI + 1e-10 {
                edge_flip(&mut mesh, e);
                flipped = true;
                break; // Restart scan after topology change.
            }
        }
        if !flipped {
            break;
        }
    }

    mesh
}
