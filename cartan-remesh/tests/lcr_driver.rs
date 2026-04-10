// ~/cartan/cartan-remesh/tests/lcr_driver.rs

//! Tests for LCR (length-cross-ratio) functions and the adaptive remesh driver.

use approx::assert_relative_eq;
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_remesh::{
    RemeshConfig, capture_reference_lcrs, lcr_spring_energy, lcr_spring_gradient,
    length_cross_ratio, needs_remesh,
};
use nalgebra::SVector;

/// Build a diamond mesh: 4 vertices, 2 triangles sharing edge (1,2).
///
///     0
///    / \
///   1---2
///    \ /
///     3
///
/// In this diamond, the interior edge is (1,2) and the two opposite vertices
/// are 0 and 3. Because the diamond is symmetric (a rotated square), the
/// length-cross-ratio of the interior edge is exactly 1.0.
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

/// Find the index of the edge with endpoints (a, b), where the edge is stored
/// with the lower-index vertex first.
fn find_edge(mesh: &Mesh<Euclidean<2>, 3, 2>, a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    mesh.boundaries
        .iter()
        .position(|&[i, j]| i == lo && j == hi)
}

// ─── LCR tests ──────────────────────────────────────────────────────────────

#[test]
fn lcr_interior_edge_symmetric_diamond() {
    // A symmetric diamond has LCR = 1.0 for the interior edge.
    // Diamond vertices: 0=(0,1), 1=(-1,0), 2=(1,0), 3=(0,-1).
    // Interior edge (1,2): opposite vertices are 0 and 3.
    // dist(1,3) = dist((-1,0),(0,-1)) = sqrt(2)
    // dist(2,0) = dist((1,0),(0,1)) = sqrt(2)
    // dist(0,1) = dist((0,1),(-1,0)) = sqrt(2)
    // dist(3,2) = dist((0,-1),(1,0)) = sqrt(2)
    // lcr = sqrt(2)*sqrt(2) / (sqrt(2)*sqrt(2)) = 1.0
    let manifold = Euclidean::<2>;
    let mesh = diamond_mesh();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let lcr = length_cross_ratio(&mesh, &manifold, edge);
    assert_relative_eq!(lcr, 1.0, epsilon = 1e-12);
}

#[test]
fn lcr_boundary_edge_returns_one() {
    // Boundary edges (only one adjacent face) should return LCR = 1.0.
    let manifold = Euclidean::<2>;
    let mesh = diamond_mesh();

    // Find a boundary edge (any edge that is not the interior edge 1-2).
    // Edges of the diamond: (0,1), (0,2), (1,2), (1,3), (2,3).
    // Interior: (1,2). Boundary: all others.
    let edge_01 = find_edge(&mesh, 0, 1).expect("edge (0,1) must exist");
    let lcr = length_cross_ratio(&mesh, &manifold, edge_01);
    assert_relative_eq!(lcr, 1.0, epsilon = 1e-12);
}

#[test]
fn capture_reference_lcrs_has_correct_length() {
    let manifold = Euclidean::<2>;
    let mesh = diamond_mesh();
    let ref_lcrs = capture_reference_lcrs(&mesh, &manifold);
    assert_eq!(ref_lcrs.len(), mesh.n_boundaries());
}

#[test]
fn lcr_spring_energy_zero_when_current_matches_reference() {
    // If we capture reference LCRs and immediately compute energy, it should be 0.
    let manifold = Euclidean::<2>;
    let mesh = diamond_mesh();
    let ref_lcrs = capture_reference_lcrs(&mesh, &manifold);
    let energy = lcr_spring_energy(&mesh, &manifold, &ref_lcrs, 1.0);
    assert_relative_eq!(energy, 0.0, epsilon = 1e-12);
}

#[test]
fn lcr_spring_gradient_returns_correct_length() {
    let manifold = Euclidean::<2>;
    let mesh = diamond_mesh();
    let ref_lcrs = capture_reference_lcrs(&mesh, &manifold);
    let grad = lcr_spring_gradient(&mesh, &manifold, &ref_lcrs, 1.0);
    assert_eq!(grad.len(), mesh.n_vertices());
}

#[test]
fn lcr_spring_energy_nonzero_for_distorted_mesh() {
    // Distort the diamond by moving vertex 3 and check that energy is positive.
    let manifold = Euclidean::<2>;
    let mesh = diamond_mesh();
    let ref_lcrs = capture_reference_lcrs(&mesh, &manifold);

    // Build a distorted version: move vertex 3 from (0,-1) to (0.5, -0.5).
    let vertices = vec![
        SVector::from([0.0, 1.0]),
        SVector::from([-1.0, 0.0]),
        SVector::from([1.0, 0.0]),
        SVector::from([0.5, -0.5]),
    ];
    let triangles = vec![[1, 2, 0], [2, 1, 3]];
    let distorted = Mesh::from_simplices(&manifold, vertices, triangles);

    let energy = lcr_spring_energy(&distorted, &manifold, &ref_lcrs, 1.0);
    assert!(
        energy > 0.0,
        "energy should be positive for a distorted mesh"
    );
}

// ─── Driver tests ───────────────────────────────────────────────────────────

#[test]
fn needs_remesh_false_for_well_sized_mesh() {
    // A unit square grid with n=4 has edges of length ~0.25 to ~0.354.
    // Set bounds that comfortably contain those lengths.
    let manifold = Euclidean::<2>;
    let mesh = Mesh::<Euclidean<2>, 3, 2>::unit_square_grid(4);
    let nv = mesh.n_vertices();
    let mean_h = vec![0.0; nv];
    let gauss_k = vec![0.0; nv];

    let config = RemeshConfig {
        min_edge_length: 0.01,
        max_edge_length: 2.0,
        curvature_scale: 10.0,
        ..RemeshConfig::default()
    };

    assert!(
        !needs_remesh(&mesh, &manifold, &mean_h, &gauss_k, &config),
        "well-sized mesh with generous bounds should not need remeshing"
    );
}

#[test]
fn needs_remesh_true_when_max_edge_length_very_small() {
    // Set max_edge_length to something tiny so every edge exceeds it.
    let manifold = Euclidean::<2>;
    let mesh = Mesh::<Euclidean<2>, 3, 2>::unit_square_grid(2);
    let nv = mesh.n_vertices();
    let mean_h = vec![0.0; nv];
    let gauss_k = vec![0.0; nv];

    let config = RemeshConfig {
        min_edge_length: 0.0,
        max_edge_length: 0.001,
        curvature_scale: 10.0,
        ..RemeshConfig::default()
    };

    assert!(
        needs_remesh(&mesh, &manifold, &mean_h, &gauss_k, &config),
        "mesh with edges exceeding max_edge_length should need remeshing"
    );
}

#[test]
fn needs_remesh_true_when_min_edge_length_very_large() {
    // Set min_edge_length to something huge so every edge is below it.
    let manifold = Euclidean::<2>;
    let mesh = Mesh::<Euclidean<2>, 3, 2>::unit_square_grid(2);
    let nv = mesh.n_vertices();
    let mean_h = vec![0.0; nv];
    let gauss_k = vec![0.0; nv];

    let config = RemeshConfig {
        min_edge_length: 100.0,
        max_edge_length: 200.0,
        curvature_scale: 10.0,
        ..RemeshConfig::default()
    };

    assert!(
        needs_remesh(&mesh, &manifold, &mean_h, &gauss_k, &config),
        "mesh with edges below min_edge_length should need remeshing"
    );
}
