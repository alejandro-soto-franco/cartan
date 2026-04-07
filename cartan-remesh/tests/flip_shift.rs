// ~/cartan/cartan-remesh/tests/flip_shift.rs

use approx::assert_relative_eq;
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_remesh::{flip_edge, shift_vertex, RemeshError};
use nalgebra::SVector;

/// Build a diamond mesh: 4 vertices, 2 triangles sharing edge (1,2).
///
///     0
///    / \
///   1---2
///    \ /
///     3
fn diamond_mesh() -> Mesh<Euclidean<2>, 3, 2> {
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 1.0]),
        SVector::from([-1.0, 0.0]),
        SVector::from([1.0, 0.0]),
        SVector::from([0.0, -1.0]),
    ];
    let triangles = vec![[1, 2, 0], [2, 1, 3]];
    Mesh::from_simplices(&manifold, vertices, triangles)
}

/// Find the edge index for the boundary connecting vertices a and b.
fn find_edge(mesh: &Mesh<Euclidean<2>, 3, 2>, a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    mesh.boundaries
        .iter()
        .position(|&[i, j]| i == lo && j == hi)
}

// ── flip_edge tests ──────────────────────────────────────────────────────

#[test]
fn flip_edge_flips_shared_diagonal() {
    // The diamond has two triangles sharing edge (1,2). The opposite vertices
    // are 0 (top) and 3 (bottom). The angles at 0 and 3 are each 90 degrees
    // (right angles in a unit diamond), so the opposite angle sum is pi.
    // We need a non-Delaunay configuration to trigger the flip.
    //
    // Stretch the diamond vertically so opposite angles become obtuse:
    //   0 at (0, 2), 3 at (0, -2) => angles at 0 and 3 are each
    //   arctan(1/2) + arctan(1/2) ~ 53 deg, still < 90.
    //
    // Instead, squish horizontally: 1 at (-0.3, 0), 2 at (0.3, 0).
    // Angle at 0: the two edges from 0 to 1 and 0 to 2 subtend a narrow angle.
    // Angle at 3: similarly narrow. Their sum should still be < pi.
    //
    // For the flip to trigger, we need the angle sum > pi. This happens when
    // the quad is "concave" in Delaunay terms. Create a thin, tall diamond.
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 0.3]),   // 0 (top, close to edge)
        SVector::from([-1.0, 0.0]),  // 1 (left)
        SVector::from([1.0, 0.0]),   // 2 (right)
        SVector::from([0.0, -0.3]),  // 3 (bottom, close to edge)
    ];
    let triangles = vec![[1, 2, 0], [2, 1, 3]];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let result = flip_edge(&mut mesh, &manifold, edge);
    assert!(result.is_ok(), "flip should succeed on non-Delaunay diamond");

    let log = result.unwrap();
    assert_eq!(log.flips.len(), 1);
    assert_eq!(mesh.n_simplices(), 2, "flip preserves face count");

    // After flip, edge (1,2) should no longer exist; edge (0,3) should exist.
    assert!(
        find_edge(&mesh, 0, 3).is_some(),
        "new diagonal (0,3) must exist after flip"
    );
}

#[test]
fn flip_edge_preserves_euler_characteristic() {
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 0.3]),
        SVector::from([-1.0, 0.0]),
        SVector::from([1.0, 0.0]),
        SVector::from([0.0, -0.3]),
    ];
    let triangles = vec![[1, 2, 0], [2, 1, 3]];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);
    let euler_before = mesh.euler_characteristic();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let _ = flip_edge(&mut mesh, &manifold, edge).expect("flip should succeed");

    assert_eq!(
        euler_before,
        mesh.euler_characteristic(),
        "Euler characteristic must be preserved by flip"
    );
}

#[test]
fn flip_edge_rejects_boundary_edge() {
    // In the diamond mesh, edges (0,1), (0,2), (1,3), (2,3) are boundary edges
    // (each has exactly 1 adjacent face).
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();

    let boundary_edge = find_edge(&mesh, 0, 1).expect("edge (0,1) must exist");
    let result = flip_edge(&mut mesh, &manifold, boundary_edge);

    assert!(
        matches!(result, Err(RemeshError::BoundaryEdge { .. })),
        "flip should reject boundary edges"
    );
}

#[test]
fn flip_edge_rejects_already_delaunay() {
    // A regular (square) diamond: opposite angles sum to exactly pi.
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let result = flip_edge(&mut mesh, &manifold, edge);

    assert!(
        matches!(result, Err(RemeshError::AlreadyDelaunay { .. })),
        "flip should reject edges that already satisfy Delaunay (angle sum <= pi)"
    );
}

// ── shift_vertex tests ───────────────────────────────────────────────────

#[test]
fn shift_vertex_moves_toward_centroid() {
    // Build a mesh where vertex 4 (interior) is offset from the centroid of
    // its neighbors, so the Laplacian shift pulls it toward the centroid.
    //
    //   0---1
    //   |\ /|
    //   | 4  |
    //   |/ \|
    //   3---2
    //
    // Outer ring at the unit square corners, interior vertex 4 offset from center.
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 1.0]),  // 0 (top-left)
        SVector::from([1.0, 1.0]),  // 1 (top-right)
        SVector::from([1.0, 0.0]),  // 2 (bottom-right)
        SVector::from([0.0, 0.0]),  // 3 (bottom-left)
        SVector::from([0.3, 0.3]),  // 4 (interior, offset from centroid (0.5, 0.5))
    ];
    let triangles = vec![
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);

    let old_pos = mesh.vertices[4];
    let centroid = SVector::from([0.5, 0.5]);

    let log = shift_vertex(&mut mesh, &manifold, 4);

    let new_pos = mesh.vertices[4];
    assert_eq!(log.shifts.len(), 1);
    assert_eq!(log.shifts[0].vertex, 4);

    // The vertex should have moved closer to the centroid of its neighbors.
    let old_dist = (old_pos - centroid).norm();
    let new_dist = (new_pos - centroid).norm();
    assert!(
        new_dist < old_dist,
        "shift should move vertex closer to neighbor centroid: old_dist={old_dist}, new_dist={new_dist}"
    );
}

#[test]
fn shift_vertex_regular_mesh_negligible_displacement() {
    // When the interior vertex is already at the centroid of its neighbors,
    // the Laplacian displacement should be (near) zero.
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 1.0]),  // 0
        SVector::from([1.0, 1.0]),  // 1
        SVector::from([1.0, 0.0]),  // 2
        SVector::from([0.0, 0.0]),  // 3
        SVector::from([0.5, 0.5]),  // 4 (exactly at centroid)
    ];
    let triangles = vec![
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);

    let old_pos = mesh.vertices[4];
    let _ = shift_vertex(&mut mesh, &manifold, 4);
    let new_pos = mesh.vertices[4];

    let displacement = (new_pos - old_pos).norm();
    assert_relative_eq!(displacement, 0.0, epsilon = 1e-12);
}
