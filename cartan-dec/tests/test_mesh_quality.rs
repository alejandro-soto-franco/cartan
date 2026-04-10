use cartan_dec::mesh_quality::{is_delaunay, is_well_centered, angle_at_vertex, quality_report};
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

#[test]
fn test_equilateral_is_well_centered() {
    let h = (3.0_f64).sqrt() / 2.0;
    let verts = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
    ];
    let tris = vec![[0, 1, 2]];
    let mesh = FlatMesh::from_triangles(verts, tris);
    let manifold = Euclidean::<2>;
    assert!(is_well_centered(&mesh, &manifold));
}

#[test]
fn test_obtuse_is_not_well_centered() {
    let verts = vec![
        [0.0, 0.0],
        [4.0, 0.0],
        [0.1, 0.1],
    ];
    let tris = vec![[0, 1, 2]];
    let mesh = FlatMesh::from_triangles(verts, tris);
    let manifold = Euclidean::<2>;
    assert!(!is_well_centered(&mesh, &manifold));
}

#[test]
fn test_quality_report_unit_square() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let report = quality_report(&mesh, &manifold);

    assert!(report.is_delaunay);
    assert_eq!(report.non_delaunay_edges, 0);
    assert!(report.n_vertices > 0);
    assert!(report.min_angle > 0.0);
    assert!(report.max_angle < std::f64::consts::PI);
    // unit_square_grid uses right triangles (45-45-90), so max angle ~ pi/2.
    assert!(
        (report.max_angle - std::f64::consts::FRAC_PI_2).abs() < 0.01,
        "expected max angle ~pi/2, got {}",
        report.max_angle
    );
}
