use cartan_dec::extrinsic::{ExtrinsicOperators, FaceData};
use cartan_dec::mesh_gen::icosphere;
use cartan_manifolds::sphere::Sphere;

#[test]
fn test_face_data_normals_unit_length() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let face_data = FaceData::from_mesh(&mesh);

    for (f, n) in face_data.normals.iter().enumerate() {
        let len = n.norm();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "face {f}: normal length = {len}, expected 1.0"
        );
    }
}

#[test]
fn test_face_data_areas_positive() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let face_data = FaceData::from_mesh(&mesh);

    for (f, &a) in face_data.areas.iter().enumerate() {
        assert!(a > 0.0, "face {f}: area = {a}, should be positive");
    }

    // Total area should approximate 4*pi (unit sphere).
    let total: f64 = face_data.areas.iter().sum();
    assert!(
        (total - 4.0 * std::f64::consts::PI).abs() < 0.5,
        "total face area = {total}, expected ~{:.4}",
        4.0 * std::f64::consts::PI
    );
}

#[test]
fn test_fem_grads_partition_of_unity() {
    // For any face, sum of FEM gradients should be zero (partition of unity).
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let face_data = FaceData::from_mesh(&mesh);

    for f in 0..mesh.n_simplices() {
        let sum = face_data.fem_grads[f][0] + face_data.fem_grads[f][1] + face_data.fem_grads[f][2];
        assert!(
            sum.norm() < 1e-10,
            "face {f}: sum of FEM grads = {}, should be zero",
            sum.norm()
        );
    }
}

#[test]
fn test_projectors_idempotent() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let face_data = FaceData::from_mesh(&mesh);

    for (f, p) in face_data.projectors.iter().enumerate() {
        let pp = p * p;
        let diff = (pp - p).norm();
        assert!(
            diff < 1e-10,
            "face {f}: P^2 - P norm = {diff}, projector not idempotent"
        );
    }
}

#[test]
fn test_projectors_kill_normal() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let face_data = FaceData::from_mesh(&mesh);

    for (f, (p, n)) in face_data
        .projectors
        .iter()
        .zip(&face_data.normals)
        .enumerate()
    {
        let pn = p * n;
        assert!(
            pn.norm() < 1e-10,
            "face {f}: P*N norm = {}, should be zero",
            pn.norm()
        );
    }
}

#[test]
fn test_extrinsic_operators_dimensions() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let ops = ExtrinsicOperators::from_mesh(&mesh);

    let nv = mesh.n_vertices();
    assert_eq!(ops.div.rows(), nv);
    assert_eq!(ops.div.cols(), 3 * nv);
    assert_eq!(ops.grad.rows(), 3 * nv);
    assert_eq!(ops.grad.cols(), nv);
    assert_eq!(ops.viscosity_lap.rows(), 3 * nv);
    assert_eq!(ops.viscosity_lap.cols(), 3 * nv);
}

#[test]
fn test_viscosity_lap_symmetric() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let ops = ExtrinsicOperators::from_mesh(&mesh);

    let n = 3 * mesh.n_vertices();
    // Check symmetry: L[i,j] == L[j,i] for sampled entries.
    for col_view in ops.viscosity_lap.outer_iterator().enumerate() {
        let (col, view) = col_view;
        for (row, &val) in view.iter() {
            if let Some(&val_t) = ops.viscosity_lap.get(col, row) {
                assert!(
                    (val - val_t).abs() < 1e-10,
                    "L[{row},{col}] = {val}, L[{col},{row}] = {val_t}"
                );
            }
        }
    }
}

#[test]
fn test_viscosity_lap_kernel_contains_translations() {
    // Rigid translations (constant velocity) should be in the kernel of L.
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let ops = ExtrinsicOperators::from_mesh(&mesh);

    let nv = mesh.n_vertices();

    // Translation in x: U = [1,0,0] at every vertex.
    let mut u_x = vec![0.0; 3 * nv];
    for v in 0..nv {
        u_x[v * 3] = 1.0;
    }

    let lu = ops.apply_viscosity_lap(&u_x);
    let norm: f64 = lu.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        norm < 1e-6,
        "L * (translation in x) should be ~zero, got norm = {norm}"
    );
}

#[test]
fn test_grad_is_neg_div_transpose() {
    // GRAD should be -DIV^T. Test by applying to known vectors.
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let ops = ExtrinsicOperators::from_mesh(&mesh);

    let nv = mesh.n_vertices();

    // <DIV * u, p> should equal <u, -GRAD * p> for arbitrary u, p.
    let u: Vec<f64> = (0..3 * nv).map(|i| (i as f64 * 0.1).sin()).collect();
    let p: Vec<f64> = (0..nv).map(|i| (i as f64 * 0.3).cos()).collect();

    let div_u = ops.apply_div(&u);
    let grad_p = ops.apply_grad(&p);

    let lhs: f64 = div_u.iter().zip(&p).map(|(d, p)| d * p).sum();
    let rhs: f64 = u.iter().zip(&grad_p).map(|(u, g)| u * (-g)).sum();

    assert!(
        (lhs - rhs).abs() < 1e-8,
        "<DIV u, p> = {lhs}, <u, -GRAD p> = {rhs}, diff = {}",
        (lhs - rhs).abs()
    );
}
