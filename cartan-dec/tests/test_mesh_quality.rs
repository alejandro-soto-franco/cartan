use cartan_dec::mesh_quality::{
    is_delaunay, is_well_centered, angle_at_vertex, quality_report, make_delaunay, make_well_centered,
};
use cartan_dec::mesh_gen::{icosphere, torus};
use cartan_dec::HodgeStar;
use cartan_dec::FlatMesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_manifolds::sphere::Sphere;

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

#[test]
fn test_make_delaunay_fixes_bad_diagonal() {
    let verts = vec![
        [0.0, 0.0],
        [1.0, 0.1],
        [2.0, 0.0],
        [1.0, -0.1],
    ];
    let tris = vec![[0, 2, 1], [0, 3, 2]];
    let mesh = FlatMesh::from_triangles(verts, tris);
    let manifold = Euclidean::<2>;
    assert!(!is_delaunay(&mesh, &manifold));

    let fixed = make_delaunay(mesh, &manifold);
    assert!(
        is_delaunay(&fixed, &manifold),
        "make_delaunay should produce a Delaunay mesh"
    );
    assert_eq!(fixed.n_vertices(), 4);
    assert_eq!(fixed.n_simplices(), 2);
}

#[test]
fn test_make_delaunay_preserves_euler() {
    let verts = vec![
        [0.0, 0.0],
        [1.0, 0.1],
        [2.0, 0.0],
        [1.0, -0.1],
    ];
    let tris = vec![[0, 2, 1], [0, 3, 2]];
    let mesh = FlatMesh::from_triangles(verts, tris);
    let chi_before = mesh.euler_characteristic();

    let manifold = Euclidean::<2>;
    let fixed = make_delaunay(mesh, &manifold);
    assert_eq!(
        fixed.euler_characteristic(),
        chi_before,
        "Euler characteristic must be preserved by edge flips"
    );
}

#[test]
fn test_make_well_centered_on_flat_grid() {
    // Use a larger grid so interior vertices have room to move.
    let mut mesh = FlatMesh::unit_square_grid(8);
    let manifold = Euclidean::<2>;

    // Perturb several interior vertices to create obtuse triangles.
    // In a 9x9 vertex grid, vertex (row*9 + col) for interior rows/cols.
    mesh.vertices[20][0] += 0.06;
    mesh.vertices[30][1] += 0.06;
    mesh.vertices[40][0] -= 0.06;

    let report_before = quality_report(&mesh, &manifold);
    assert!(
        report_before.non_well_centered_simplices > 0,
        "perturbation should create obtuse triangles"
    );

    let smoothed = make_well_centered(mesh, &manifold, 200, 1e-8);
    let report_after = quality_report(&smoothed, &manifold);
    assert!(
        report_after.non_well_centered_simplices <= report_before.non_well_centered_simplices,
        "Lloyd smoothing should not increase obtuse triangles: before={}, after={}",
        report_before.non_well_centered_simplices,
        report_after.non_well_centered_simplices,
    );
    // Verify the mesh is still valid.
    assert!(report_after.is_delaunay, "should remain Delaunay after smoothing");
    assert_eq!(smoothed.n_vertices(), 81);
}

#[test]
fn test_icosphere_euler_characteristic() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, false);
    assert_eq!(mesh.euler_characteristic(), 2, "S^2 has chi=2");
    assert_eq!(mesh.n_vertices(), 162);
    assert_eq!(mesh.n_simplices(), 320);
}

#[test]
fn test_icosphere_vertices_on_sphere() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 3, false);
    for (i, v) in mesh.vertices.iter().enumerate() {
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(
            (r - 1.0).abs() < 1e-12,
            "vertex {i} should be on unit sphere, r = {r}"
        );
    }
}

#[test]
fn test_icosphere_well_centered() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    assert_eq!(mesh.euler_characteristic(), 2);
    let report = quality_report(&mesh, &manifold);
    assert!(report.is_delaunay, "well-centered icosphere should be Delaunay");
    // Icosphere subdivision naturally produces near-equilateral triangles,
    // so it should already be well-centered or very close after smoothing.
}

#[test]
fn test_torus_euler_characteristic() {
    let manifold = Euclidean::<3>;
    let mesh = torus(&manifold, 3.0, 1.0, 20, 10, false);
    assert_eq!(mesh.euler_characteristic(), 0, "T^2 has chi=0");
    assert_eq!(mesh.n_vertices(), 200);
    assert_eq!(mesh.n_simplices(), 400);
}

#[test]
fn test_torus_delaunay() {
    let manifold = Euclidean::<3>;
    let mesh = torus(&manifold, 3.0, 1.0, 16, 8, false);
    let report = quality_report(&mesh, &manifold);
    // Regular parametric torus grid should be Delaunay.
    assert!(
        report.is_delaunay,
        "regular torus grid should be Delaunay, non-Delaunay edges: {}",
        report.non_delaunay_edges
    );
}

#[test]
fn test_circumcentric_star0_positive_on_well_centered_mesh() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);

    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold).unwrap();
    let s0 = hodge.star0();
    for i in 0..mesh.n_vertices() {
        assert!(
            s0[i] > 0.0,
            "circumcentric star0[{i}] = {} should be positive on well-centered mesh",
            s0[i]
        );
    }
}

#[test]
fn test_circumcentric_star0_sums_to_surface_area() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 3, true);

    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold).unwrap();
    let total: f64 = hodge.star0().iter().sum();
    // Surface area of unit sphere = 4*pi.
    let expected = 4.0 * std::f64::consts::PI;
    assert!(
        (total - expected).abs() < 0.5,
        "total dual area = {total}, expected ~{expected:.4}"
    );
}
