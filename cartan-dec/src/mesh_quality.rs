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
