// ~/cartan/cartan-remesh/tests/split_collapse.rs

use approx::assert_relative_eq;
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_remesh::{collapse_edge, split_edge, RemeshError};
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

fn total_area(mesh: &Mesh<Euclidean<2>, 3, 2>, manifold: &Euclidean<2>) -> f64 {
    (0..mesh.n_simplices())
        .map(|t| mesh.triangle_area(manifold, t))
        .sum()
}

fn find_edge(mesh: &Mesh<Euclidean<2>, 3, 2>, a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    mesh.boundaries
        .iter()
        .position(|&[i, j]| i == lo && j == hi)
}

#[test]
fn split_edge_adds_one_vertex() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let nv_before = mesh.n_vertices();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let log = split_edge(&mut mesh, &manifold, edge);

    assert_eq!(mesh.n_vertices(), nv_before + 1);
    assert_eq!(log.splits.len(), 1);
    assert_eq!(log.splits[0].new_vertex, nv_before);
}

#[test]
fn split_edge_replaces_two_faces_with_four() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let nf_before = mesh.n_simplices();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let _ = split_edge(&mut mesh, &manifold, edge);

    assert_eq!(mesh.n_simplices(), nf_before + 2);
}

#[test]
fn split_edge_preserves_total_area() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let area_before = total_area(&mesh, &manifold);
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let _ = split_edge(&mut mesh, &manifold, edge);

    let area_after = total_area(&mesh, &manifold);
    assert_relative_eq!(area_before, area_after, epsilon = 1e-12);
}

#[test]
fn split_edge_new_vertex_at_midpoint() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let log = split_edge(&mut mesh, &manifold, edge);

    let new_v = log.splits[0].new_vertex;
    let pos = &mesh.vertices[new_v];
    assert_relative_eq!(pos[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(pos[1], 0.0, epsilon = 1e-12);
}

#[test]
fn split_edge_preserves_euler() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let euler_before = mesh.euler_characteristic();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let _ = split_edge(&mut mesh, &manifold, edge);

    assert_eq!(euler_before, mesh.euler_characteristic());
}

#[test]
fn collapse_edge_removes_one_vertex_and_two_faces() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let nv_before = mesh.n_vertices();
    let nf_before = mesh.n_simplices();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let log = collapse_edge(&mut mesh, &manifold, edge, 0.5)
        .expect("collapse should succeed on diamond mesh");

    assert_eq!(mesh.n_vertices(), nv_before - 1);
    assert_eq!(mesh.n_simplices(), nf_before - 2);
    assert_eq!(log.collapses.len(), 1);
    assert_eq!(log.collapses[0].removed_faces.len(), 2);
}

#[test]
fn collapse_edge_surviving_vertex_at_midpoint() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let log = collapse_edge(&mut mesh, &manifold, edge, 0.5)
        .expect("collapse should succeed");

    let survivor = log.collapses[0].surviving_vertex;
    let pos = &mesh.vertices[survivor];
    assert_relative_eq!(pos[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(pos[1], 0.0, epsilon = 1e-12);
}

#[test]
fn collapse_edge_rejects_foldover() {
    // Build a mesh where collapsing edge (1,3) would flip triangle [1,4,0].
    //
    //       0=(0,2)
    //      / \
    //     /   \
    //  1=(-1,0)--2=(1,0)
    //     \   /
    //      \ /
    //   3=(0,-1)
    //      |
    //   4=(-0.8, 0.5)  <-- positioned so that moving 1 to midpoint(-1,0),(0,-1)=(-0.5,-0.5)
    //                       causes triangle [1,4,0] to flip orientation.
    //
    // Triangle [1,4,0] = [(-1,0), (-0.8,0.5), (0,2)]:
    //   area_before = 0.5 * ((0.2)(2) - (0.5)(1)) = 0.5*(0.4-0.5) = -0.05  (CW)
    // After collapse, vertex 1 moves to (-0.5,-0.5):
    //   [(-0.5,-0.5), (-0.8,0.5), (0,2)]:
    //   area_after = 0.5*((-0.3)(2.5) - (1.0)(0.5)) = 0.5*(-0.75-0.5) = -0.625  (still CW)
    //
    // That's same sign. Let me try a different configuration.
    // We need a triangle where moving the survivor inverts the orientation.
    //
    // Simpler approach: place vertex 4 such that the triangle [1,4,0] is barely
    // CCW with vertex 1 at (-1,0), but becomes CW when 1 moves to (-0.5,-0.5).
    //
    //   4 = (-0.6, 0.1): triangle [1,4,0] = [(-1,0),(-0.6,0.1),(0,2)]
    //   ab = (0.4, 0.1), ac = (1, 2)
    //   area = 0.5*(0.4*2 - 0.1*1) = 0.5*(0.8-0.1) = 0.35 (CCW)
    //   After: [(-0.5,-0.5),(-0.6,0.1),(0,2)]
    //   ab = (-0.1, 0.6), ac = (0.5, 2.5)
    //   area = 0.5*((-0.1)*2.5 - 0.6*0.5) = 0.5*(-0.25-0.3) = -0.275 (CW!)
    //   Foldover detected!
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 2.0]),    // 0
        SVector::from([-1.0, 0.0]),   // 1
        SVector::from([1.0, 0.0]),    // 2
        SVector::from([0.0, -1.0]),   // 3
        SVector::from([-0.6, 0.1]),   // 4
    ];
    let triangles = vec![[1, 2, 0], [2, 1, 3], [1, 3, 4], [1, 4, 0]];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);
    let edge13 = find_edge(&mesh, 1, 3).expect("edge (1,3) must exist");

    let result = collapse_edge(&mut mesh, &manifold, edge13, 0.0);
    assert!(
        matches!(result, Err(RemeshError::Foldover { .. })),
        "collapse should be rejected: triangle [1,4,0] flips when 1 moves to midpoint"
    );
}
