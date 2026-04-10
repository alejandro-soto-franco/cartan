use cartan_dec::mesh_quality::{is_delaunay, angle_at_vertex};
use cartan_dec::FlatMesh;
use cartan_manifolds::euclidean::Euclidean;

#[test]
fn test_unit_square_grid_is_delaunay() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    assert!(
        is_delaunay(&mesh, &manifold),
        "unit_square_grid should be Delaunay"
    );
}

#[test]
fn test_non_delaunay_detected() {
    // Four vertices forming a thin rhombus where the diagonal creates
    // opposite angles summing to > pi.
    let verts = vec![
        [0.0, 0.0],   // 0: left
        [1.0, 0.1],   // 1: top (close to horizontal)
        [2.0, 0.0],   // 2: right
        [1.0, -0.1],  // 3: bottom (close to horizontal)
    ];
    // Triangles sharing edge (0, 2): the "bad" diagonal of the thin rhombus.
    let tris = vec![[0, 2, 1], [0, 3, 2]];
    let mesh = FlatMesh::from_triangles(verts, tris);
    let manifold = Euclidean::<2>;
    assert!(
        !is_delaunay(&mesh, &manifold),
        "thin rhombus with bad diagonal should not be Delaunay"
    );
}

#[test]
fn test_angle_at_vertex_equilateral() {
    let h = (3.0_f64).sqrt() / 2.0;
    let verts = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
    ];
    let tris = vec![[0, 1, 2]];
    let mesh = FlatMesh::from_triangles(verts, tris);
    let manifold = Euclidean::<2>;

    // All angles should be pi/3 (60 degrees).
    for &v in &[0, 1, 2] {
        let angle = angle_at_vertex(&mesh, &manifold, 0, v);
        assert!(
            (angle - std::f64::consts::FRAC_PI_3).abs() < 1e-10,
            "angle at vertex {v} = {angle}, expected pi/3"
        );
    }
}
