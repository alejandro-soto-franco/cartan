// cartan-dec/tests/integration.rs
//
// End-to-end tests for cartan-dec: mesh topology, DEC identities,
// operator correctness, and physical sanity checks.

use cartan_dec::{
    ExteriorDerivative, FlatMesh, HodgeStar, Mesh, Operators, apply_divergence,
    apply_divergence_generic, apply_scalar_advection, apply_scalar_advection_generic,
};
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::DVector;

// ─────────────────────────────────────────────────────────────────────────────
// Mesh topology tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unit_square_grid_topology() {
    // n=2: 3x3 = 9 vertices, 2*2=4 squares = 8 triangles.
    let mesh = FlatMesh::unit_square_grid(2);
    assert_eq!(mesh.n_vertices(), 9);
    assert_eq!(mesh.n_simplices(), 8);
    // Euler characteristic for a disk: V - E + T = 1.
    assert_eq!(mesh.euler_characteristic(), 1);
}

#[test]
fn unit_square_grid_1_topology() {
    // n=1: 2x2 = 4 vertices, 2 triangles.
    let mesh = FlatMesh::unit_square_grid(1);
    assert_eq!(mesh.n_vertices(), 4);
    assert_eq!(mesh.n_simplices(), 2);
    assert_eq!(mesh.euler_characteristic(), 1);
}

#[test]
fn mesh_triangle_areas_positive() {
    let mesh = FlatMesh::unit_square_grid(4);
    for t in 0..mesh.n_simplices() {
        let area = mesh.triangle_area_flat(t);
        assert!(area > 0.0, "triangle {t} has non-positive area: {area}");
    }
}

#[test]
fn mesh_total_area_is_one() {
    let mesh = FlatMesh::unit_square_grid(4);
    let total: f64 = (0..mesh.n_simplices())
        .map(|t| mesh.triangle_area_flat(t))
        .sum();
    assert!(
        (total - 1.0).abs() < 1e-14,
        "total area = {total}, expected 1.0"
    );
}

#[test]
fn mesh_edge_lengths_positive() {
    let mesh = FlatMesh::unit_square_grid(4);
    for e in 0..mesh.n_boundaries() {
        let len = mesh.edge_length_flat(e);
        assert!(len > 0.0, "edge {e} has non-positive length: {len}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exterior derivative: d1 o d0 = 0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn exterior_derivative_exactness() {
    let mesh = FlatMesh::unit_square_grid(4);
    let ext = ExteriorDerivative::from_mesh(&mesh);
    let err = ext.check_exactness();
    assert!(err < 1e-14, "d1 * d0 is not zero: max entry = {err:.2e}");
}

#[test]
fn d0_dimensions() {
    let mesh = FlatMesh::unit_square_grid(3);
    let ext = ExteriorDerivative::from_mesh(&mesh);
    assert_eq!(ext.d0().rows(), mesh.n_boundaries());
    assert_eq!(ext.d0().cols(), mesh.n_vertices());
}

#[test]
fn d1_dimensions() {
    let mesh = FlatMesh::unit_square_grid(3);
    let ext = ExteriorDerivative::from_mesh(&mesh);
    assert_eq!(ext.d1().rows(), mesh.n_simplices());
    assert_eq!(ext.d1().cols(), mesh.n_boundaries());
}

// ─────────────────────────────────────────────────────────────────────────────
// Hodge star: positivity and dimensions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hodge_star0_positive() {
    let mesh = FlatMesh::unit_square_grid(4);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    for (v, &w) in hodge.star0().iter().enumerate() {
        assert!(w > 0.0, "star0[{v}] = {w}, expected positive");
    }
}

#[test]
fn hodge_star1_nonnegative() {
    // On a right-triangle uniform grid, diagonal edges have coincident circumcenters
    // (not well-centered), so their dual edge length is zero -> star1 = 0 is allowed.
    // Assert non-negative (no negative weights).
    let mesh = FlatMesh::unit_square_grid(4);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    for (e, &w) in hodge.star1().iter().enumerate() {
        assert!(w >= 0.0, "star1[{e}] = {w}, expected non-negative");
    }
}

#[test]
fn hodge_star2_positive() {
    let mesh = FlatMesh::unit_square_grid(4);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    for (t, &w) in hodge.star2().iter().enumerate() {
        assert!(w > 0.0, "star2[{t}] = {w}, expected positive");
    }
}

#[test]
fn hodge_star0_total_area_is_one() {
    // sum star0[v] = total area of the domain (sum of dual cell areas = domain area).
    let mesh = FlatMesh::unit_square_grid(4);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    let total: f64 = hodge.star0().iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-14,
        "sum of star0 = {total}, expected 1.0"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Laplace-Beltrami operator
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn laplace_beltrami_kills_constants() {
    // The Laplacian of a constant function is zero.
    let mesh = FlatMesh::unit_square_grid(4);
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();

    let f = DVector::from_element(nv, 3.14);
    let lf = ops.apply_laplace_beltrami(&f);

    let max_err = lf.abs().max();
    assert!(
        max_err < 1e-12,
        "Delta(const) != 0: max |Delta f| = {max_err:.2e}"
    );
}

#[test]
fn laplace_beltrami_is_positive_semidefinite() {
    // The DEC Laplacian corresponds to the *positive* operator -Delta_smooth.
    // So <f, Delta f>_{star0} = f^T d0^T star1 d0 f = ||star1^{1/2} d0 f||^2 >= 0.
    let n = 8;
    let mesh = FlatMesh::unit_square_grid(n);
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();

    let f: DVector<f64> = DVector::from_fn(nv, |v, _| {
        let p = mesh.vertex(v);
        (std::f64::consts::PI * p.x).sin() * (std::f64::consts::PI * p.y).sin()
    });

    let lf = ops.apply_laplace_beltrami(&f);

    // <f, Delta f>_{star0} = sum_v f[v] * (Delta f)[v] * star0[v] >= 0.
    let f_dot_lf: f64 = f
        .iter()
        .zip(lf.iter())
        .zip(ops.mass0.iter())
        .map(|((fi, lfi), mi)| fi * lfi * mi)
        .sum();

    assert!(
        f_dot_lf >= -1e-10,
        "<f, Delta f>_{{star0}} = {f_dot_lf:.6e}, expected >= 0"
    );
}

#[test]
fn laplace_beltrami_linear_function_interior_vanishes() {
    // On a flat domain, Delta(ax + by) = 0 exactly (linear functions are harmonic).
    let n = 4;
    let mesh = FlatMesh::unit_square_grid(n);
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();

    // f(x, y) = 2x + 3y
    let f: DVector<f64> = DVector::from_fn(nv, |v, _| {
        let p = mesh.vertex(v);
        2.0 * p.x + 3.0 * p.y
    });

    let lf = ops.apply_laplace_beltrami(&f);

    // Interior vertices only (boundary vertices are influenced by their
    // one-sided neighborhood, so their Laplacian may not vanish exactly).
    let mut max_interior_err = 0.0f64;
    for v in 0..nv {
        let p = mesh.vertex(v);
        let is_boundary = p.x < 1e-10 || p.x > 1.0 - 1e-10 || p.y < 1e-10 || p.y > 1.0 - 1e-10;
        if !is_boundary {
            max_interior_err = max_interior_err.max(lf[v].abs());
        }
    }

    assert!(
        max_interior_err < 1e-12,
        "Delta(linear) at interior: max err = {max_interior_err:.2e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Divergence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn divergence_of_constant_field_vanishes() {
    // div(c) = 0 for any constant vector field.
    let mesh = FlatMesh::unit_square_grid(4);
    let ext = ExteriorDerivative::from_mesh(&mesh);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();

    // u = (1, 0) everywhere.
    let mut u = DVector::<f64>::zeros(2 * nv);
    for v in 0..nv {
        u[v] = 1.0; // u_x = 1
    }

    let div_u = apply_divergence(&mesh, &ext, &hodge, &u);
    let max_interior_err = {
        let mut m = 0.0f64;
        for v in 0..nv {
            let p = mesh.vertex(v);
            let is_boundary = p.x < 1e-10 || p.x > 1.0 - 1e-10 || p.y < 1e-10 || p.y > 1.0 - 1e-10;
            if !is_boundary {
                m = m.max(div_u[v].abs());
            }
        }
        m
    };

    assert!(
        max_interior_err < 1e-12,
        "div(const) interior: max = {max_interior_err:.2e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Advection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn advection_of_constant_field_vanishes() {
    // (u . nabla) c = 0 for any constant scalar field.
    let mesh = FlatMesh::unit_square_grid(4);
    let nv = mesh.n_vertices();

    let f = DVector::<f64>::from_element(nv, 5.0);
    let mut u = DVector::<f64>::zeros(2 * nv);
    for v in 0..nv {
        u[v] = 1.0; // u_x = 1
        u[nv + v] = 0.5; // u_y = 0.5
    }

    let adv = apply_scalar_advection(&mesh, &f, &u);
    let max_err = adv.abs().max();
    assert!(
        max_err < 1e-13,
        "advection of constant: max |result| = {max_err:.2e}"
    );
}

#[test]
fn advection_generic_constant_field_vanishes() {
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let nv = mesh.n_vertices();

    let f = DVector::<f64>::from_element(nv, 5.0);
    let u: Vec<SVector<f64, 2>> = vec![SVector::<f64, 2>::new(1.0, 0.5); nv];

    let adv = apply_scalar_advection_generic(&mesh, &manifold, &f, &u);
    let max_err = adv.abs().max();
    assert!(
        max_err < 1e-13,
        "generic advection of constant: max = {max_err:.2e}"
    );
}

#[test]
fn advection_generic_matches_old() {
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let nv = mesh.n_vertices();

    let f: DVector<f64> = DVector::from_fn(nv, |v, _| {
        let p = mesh.vertex(v);
        (std::f64::consts::PI * p.x).sin() * (std::f64::consts::PI * p.y).cos()
    });

    let mut u_old = DVector::<f64>::zeros(2 * nv);
    let mut u_new: Vec<SVector<f64, 2>> = Vec::with_capacity(nv);
    for v in 0..nv {
        let p = mesh.vertex(v);
        u_old[v] = p.x;
        u_old[nv + v] = -p.y;
        u_new.push(SVector::<f64, 2>::new(p.x, -p.y));
    }

    let adv_old = apply_scalar_advection(&mesh, &f, &u_old);
    let adv_new = apply_scalar_advection_generic(&mesh, &manifold, &f, &u_new);

    let diff = (&adv_old - &adv_new).norm();
    assert!(
        diff < 1e-12,
        "generic vs old advection: diff = {diff}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Divergence (generic)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn divergence_generic_constant_field_vanishes() {
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ext = ExteriorDerivative::from_mesh(&mesh);
    let hodge = HodgeStar::from_mesh(&mesh, &manifold);
    let nv = mesh.n_vertices();

    let u: Vec<SVector<f64, 2>> = vec![SVector::<f64, 2>::new(1.0, 0.0); nv];

    let div_u = apply_divergence_generic(&mesh, &manifold, &ext, &hodge, &u);
    let max_interior_err = {
        let mut m = 0.0f64;
        for v in 0..nv {
            let p = mesh.vertex(v);
            let is_boundary =
                p.x < 1e-10 || p.x > 1.0 - 1e-10 || p.y < 1e-10 || p.y > 1.0 - 1e-10;
            if !is_boundary {
                m = m.max(div_u[v].abs());
            }
        }
        m
    };

    assert!(
        max_interior_err < 1e-12,
        "generic div(const) interior: max = {max_interior_err:.2e}"
    );
}

#[test]
fn divergence_generic_matches_old() {
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ext = ExteriorDerivative::from_mesh(&mesh);
    let hodge = HodgeStar::from_mesh(&mesh, &manifold);
    let nv = mesh.n_vertices();

    let mut u_old = DVector::<f64>::zeros(2 * nv);
    let mut u_new: Vec<SVector<f64, 2>> = Vec::with_capacity(nv);
    for v in 0..nv {
        let p = mesh.vertex(v);
        u_old[v] = p.x;
        u_old[nv + v] = -p.y;
        u_new.push(SVector::<f64, 2>::new(p.x, -p.y));
    }

    let div_old = apply_divergence(&mesh, &ext, &hodge, &u_old);
    let div_new = apply_divergence_generic(&mesh, &manifold, &ext, &hodge, &u_new);

    let diff = (&div_old - &div_new).norm();
    assert!(
        diff < 1e-12,
        "generic vs old divergence: diff = {diff}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Bochner and Lichnerowicz
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bochner_laplacian_flat_kills_constant_vector() {
    let mesh = FlatMesh::unit_square_grid(4);
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();

    // u = (1, 2) everywhere.
    let mut u = DVector::<f64>::zeros(2 * nv);
    for v in 0..nv {
        u[v] = 1.0;
        u[nv + v] = 2.0;
    }

    let lu = ops.apply_bochner_laplacian(&u, None);
    let max_err = lu.abs().max();
    assert!(
        max_err < 1e-12,
        "Bochner(const vector): max err = {max_err:.2e}"
    );
}

#[test]
fn lichnerowicz_laplacian_flat_kills_constant_tensor() {
    let mesh = FlatMesh::unit_square_grid(4);
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();

    // Q = [[1, 0.5], [0.5, -1]] everywhere (traceless symmetric).
    let mut q = DVector::<f64>::zeros(3 * nv);
    for v in 0..nv {
        q[v] = 1.0; // Q_xx
        q[nv + v] = 0.5; // Q_xy
        q[2 * nv + v] = -1.0; // Q_yy
    }

    let lq = ops.apply_lichnerowicz_laplacian(&q, None);
    let max_err = lq.abs().max();
    assert!(
        max_err < 1e-12,
        "Lichnerowicz(const tensor): max err = {max_err:.2e}"
    );
}

// -------------------------------------------------------------------------
// Sparse exterior derivative tests
// -------------------------------------------------------------------------

#[test]
fn sparse_exterior_matches_dense() {
    // The sparse exterior derivative must match the dense one to machine epsilon.
    use nalgebra::DMatrix;

    let mesh = FlatMesh::unit_square_grid(4);
    let ext = ExteriorDerivative::from_mesh_sparse(&mesh);

    let nv = mesh.n_vertices();
    let ne = mesh.n_boundaries();
    let nt = mesh.n_simplices();

    // Build expected dense d0
    let mut d0_dense = DMatrix::<f64>::zeros(ne, nv);
    for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
        d0_dense[(e, i)] = -1.0;
        d0_dense[(e, j)] = 1.0;
    }

    // Check d[0]
    let d0_sp = ext.d0();
    for r in 0..ne {
        for c in 0..nv {
            let sp_val = d0_sp.get(r, c).copied().unwrap_or(0.0);
            let dn_val = d0_dense[(r, c)];
            assert!(
                (sp_val - dn_val).abs() < 1e-15,
                "d0[{r},{c}]: sparse={sp_val}, dense={dn_val}"
            );
        }
    }

    // Build expected dense d1
    let mut d1_dense = DMatrix::<f64>::zeros(nt, ne);
    for (t, (local_e, local_s)) in mesh
        .simplex_boundary_ids
        .iter()
        .zip(mesh.boundary_signs.iter())
        .enumerate()
    {
        for k in 0..3 {
            d1_dense[(t, local_e[k])] = local_s[k];
        }
    }

    // Check d[1]
    let d1_sp = ext.d1();
    for r in 0..nt {
        for c in 0..ne {
            let sp_val = d1_sp.get(r, c).copied().unwrap_or(0.0);
            let dn_val = d1_dense[(r, c)];
            assert!(
                (sp_val - dn_val).abs() < 1e-15,
                "d1[{r},{c}]: sparse={sp_val}, dense={dn_val}"
            );
        }
    }
}

#[test]
fn sparse_exterior_exactness() {
    // d[k+1] * d[k] = 0 for all k.
    let mesh = FlatMesh::unit_square_grid(4);
    let ext = ExteriorDerivative::from_mesh_sparse(&mesh);
    let max_err = ext.check_exactness();
    assert!(
        max_err < 1e-14,
        "sparse exactness: max entry of d1*d0 = {max_err:.2e}"
    );
}

#[test]
fn sparse_exterior_k_generic_dimensions() {
    // d has K-1 entries. For K=3: d[0] is (n_edges x n_verts), d[1] is (n_faces x n_edges).
    let mesh = FlatMesh::unit_square_grid(3);
    let ext = ExteriorDerivative::from_mesh_sparse(&mesh);
    assert_eq!(ext.degree(), 2);
    assert_eq!(ext.d0().rows(), mesh.n_boundaries());
    assert_eq!(ext.d0().cols(), mesh.n_vertices());
    assert_eq!(ext.d1().rows(), mesh.n_simplices());
    assert_eq!(ext.d1().cols(), mesh.n_boundaries());
}

// -------------------------------------------------------------------------
// K-generic Operators
// -------------------------------------------------------------------------

#[test]
fn operators_generic_laplace_kills_constants() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
    let nv = mesh.n_vertices();

    let f = DVector::from_element(nv, 3.14);
    let lf = ops.apply_laplace_beltrami(&f);

    let max_err = lf.abs().max();
    assert!(
        max_err < 1e-12,
        "generic Delta(const) != 0: max = {max_err:.2e}"
    );
}

#[test]
fn operators_generic_laplace_positive_semidefinite() {
    let mesh = FlatMesh::unit_square_grid(8);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
    let nv = mesh.n_vertices();

    let f: DVector<f64> = DVector::from_fn(nv, |v, _| {
        let p = mesh.vertex(v);
        (std::f64::consts::PI * p.x).sin() * (std::f64::consts::PI * p.y).sin()
    });

    let lf = ops.apply_laplace_beltrami(&f);
    let f_dot_lf: f64 = f
        .iter()
        .zip(lf.iter())
        .zip(ops.mass[0].iter())
        .map(|((fi, lfi), mi)| fi * lfi * mi)
        .sum();

    assert!(
        f_dot_lf >= -1e-10,
        "<f, Lf>_{{star0}} = {f_dot_lf:.6e}, expected >= 0"
    );
}

#[test]
fn operators_backward_compat_default_type() {
    let mesh = FlatMesh::unit_square_grid(3);
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();
    let f = DVector::from_element(nv, 1.0);
    let lf = ops.apply_laplace_beltrami(&f);
    assert!(lf.abs().max() < 1e-12);
}

// -------------------------------------------------------------------------
// K-generic Hodge star
// -------------------------------------------------------------------------

#[test]
fn hodge_star_generic_matches_flat() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let hodge_flat = HodgeStar::from_mesh(&mesh, &manifold);
    let hodge_generic = HodgeStar::from_mesh_generic(&mesh, &manifold).unwrap();

    let diff0 = (hodge_flat.star0() - hodge_generic.star_k(0)).norm();
    assert!(diff0 < 1e-12, "star0 diff = {diff0}");

    let diff1 = (hodge_flat.star1() - hodge_generic.star_k(1)).norm();
    assert!(diff1 < 1e-12, "star1 diff = {diff1}");

    let diff2 = (hodge_flat.star2() - hodge_generic.star_k(2)).norm();
    assert!(diff2 < 1e-12, "star2 diff = {diff2}");
}

#[test]
fn hodge_star_k_inv_roundtrip() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let hodge = HodgeStar::from_mesh_generic(&mesh, &manifold).unwrap();

    for k in 0..3 {
        let s = hodge.star_k(k);
        let sinv = hodge.star_k_inv(k);
        for i in 0..s.len() {
            if s[i].abs() > 1e-30 {
                let product = s[i] * sinv[i];
                assert!(
                    (product - 1.0).abs() < 1e-12,
                    "star[{k}][{i}] * star_inv[{k}][{i}] = {product}"
                );
            }
        }
    }
}

#[test]
fn hodge_star_generic_sphere_positive() {
    use cartan_manifolds::sphere::Sphere;
    use nalgebra::SVector;

    let manifold = Sphere::<3>;
    let verts: Vec<SVector<f64, 3>> = vec![
        SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        SVector::<f64, 3>::new(-1.0, 0.0, 0.0),
        SVector::<f64, 3>::new(0.0, 1.0, 0.0),
        SVector::<f64, 3>::new(0.0, -1.0, 0.0),
        SVector::<f64, 3>::new(0.0, 0.0, 1.0),
        SVector::<f64, 3>::new(0.0, 0.0, -1.0),
    ];
    let tris = vec![
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [0, 5, 2], [2, 5, 1], [1, 5, 3], [3, 5, 0],
    ];
    let mesh = Mesh::from_simplices(&manifold, verts, tris);
    let hodge = HodgeStar::from_mesh_generic(&mesh, &manifold).unwrap();

    for k in 0..3 {
        for (i, &val) in hodge.star_k(k).iter().enumerate() {
            assert!(
                val > 0.0,
                "star[{k}][{i}] = {val}, expected positive on S^2 octahedron"
            );
        }
    }
}

// -------------------------------------------------------------------------
// K-generic geometric primitives
// -------------------------------------------------------------------------

#[test]
fn simplex_volume_triangle_matches_triangle_area() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    for t in 0..mesh.n_simplices() {
        let generic = mesh.simplex_volume(&manifold, t);
        let specific = mesh.triangle_area(&manifold, t);
        assert!(
            (generic - specific).abs() < 1e-14,
            "simplex {t}: generic={generic}, specific={specific}"
        );
    }
}

#[test]
fn simplex_circumcenter_triangle_matches_circumcenter() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    for t in 0..mesh.n_simplices() {
        let generic = mesh.simplex_circumcenter(&manifold, t);
        let specific = mesh.circumcenter(&manifold, t);
        let diff = (generic - specific).norm();
        assert!(
            diff < 1e-14,
            "simplex {t}: circumcenter diff = {diff}"
        );
    }
}

#[test]
fn boundary_volume_matches_edge_length() {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    for e in 0..mesh.n_boundaries() {
        let generic = mesh.boundary_volume(&manifold, e);
        let specific = mesh.edge_length(&manifold, e);
        assert!(
            (generic - specific).abs() < 1e-14,
            "boundary {e}: generic={generic}, specific={specific}"
        );
    }
}

#[test]
fn regular_tet_volume() {
    // A regular tetrahedron with edge length 1 has volume sqrt(2)/12.
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 6.0, (2.0_f64 / 3.0).sqrt());

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    let vol = mesh.simplex_volume(&manifold, 0);
    let expected = (2.0_f64).sqrt() / 12.0;
    assert!(
        (vol - expected).abs() < 1e-12,
        "regular tet volume: got {vol}, expected {expected}"
    );
}

#[test]
fn regular_tet_circumcenter_is_centroid() {
    // For a regular tetrahedron, the circumcenter equals the centroid.
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 6.0, (2.0_f64 / 3.0).sqrt());

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    let cc = mesh.simplex_circumcenter(&manifold, 0);
    let centroid = (SVector::<f64, 3>::new(0.0, 0.0, 0.0)
        + SVector::<f64, 3>::new(1.0, 0.0, 0.0)
        + SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0)
        + SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 6.0, (2.0_f64 / 3.0).sqrt()))
        * 0.25;
    let diff = (cc - centroid).norm();
    assert!(
        diff < 1e-12,
        "regular tet circumcenter: diff from centroid = {diff}"
    );
}

// -------------------------------------------------------------------------
// Adjacency map tests
// -------------------------------------------------------------------------

#[test]
fn adjacency_handshaking_lemma() {
    // Handshaking lemma: sum of vertex degrees (in boundaries) = 2 * n_boundaries.
    // Each boundary has exactly B=2 vertices, so each boundary contributes 2 to the total degree.
    let mesh = FlatMesh::unit_square_grid(4);
    let total_degree: usize = mesh.vertex_boundaries.iter().map(|vb| vb.len()).sum();
    assert_eq!(
        total_degree,
        2 * mesh.n_boundaries(),
        "handshaking lemma: sum(deg) = {} != 2*E = {}",
        total_degree,
        2 * mesh.n_boundaries()
    );
}

#[test]
fn adjacency_interior_edges_have_two_cofaces() {
    // On a unit_square_grid(4), interior edges have exactly 2 adjacent triangles.
    // Boundary edges have exactly 1.
    let mesh = FlatMesh::unit_square_grid(4);
    for (e, cofaces) in mesh.boundary_simplices.iter().enumerate() {
        let [i, j] = mesh.boundaries[e];
        let pi = mesh.vertex(i);
        let pj = mesh.vertex(j);
        let on_boundary = (pi.x < 1e-10 && pj.x < 1e-10)
            || (pi.x > 1.0 - 1e-10 && pj.x > 1.0 - 1e-10)
            || (pi.y < 1e-10 && pj.y < 1e-10)
            || (pi.y > 1.0 - 1e-10 && pj.y > 1.0 - 1e-10);
        if on_boundary {
            assert_eq!(
                cofaces.len(),
                1,
                "boundary edge {e} has {} cofaces, expected 1",
                cofaces.len()
            );
        } else {
            assert_eq!(
                cofaces.len(),
                2,
                "interior edge {e} has {} cofaces, expected 2",
                cofaces.len()
            );
        }
    }
}

#[test]
fn adjacency_vertex_simplices_consistent() {
    // Every simplex that contains vertex v must appear in vertex_simplices[v].
    let mesh = FlatMesh::unit_square_grid(3);
    for (t, simplex) in mesh.simplices.iter().enumerate() {
        for &v in simplex {
            assert!(
                mesh.vertex_simplices[v].contains(&t),
                "simplex {t} contains vertex {v} but vertex_simplices[{v}] = {:?}",
                mesh.vertex_simplices[v]
            );
        }
    }
}

#[test]
fn adjacency_edge_faces_convenience() {
    // edge_faces returns (face_a, Some(face_b)) for interior edges, (face_a, None) for boundary.
    let mesh = FlatMesh::unit_square_grid(3);
    for e in 0..mesh.n_boundaries() {
        let (fa, fb) = mesh.edge_faces(e);
        assert!(fa < mesh.n_simplices());
        if let Some(fb_val) = fb {
            assert!(fb_val < mesh.n_simplices());
            assert_ne!(fa, fb_val);
        }
    }
}

#[test]
fn adjacency_rebuild_matches_initial() {
    // rebuild_adjacency() must produce the same maps as from_simplices.
    let mesh_orig = FlatMesh::unit_square_grid(3);
    let mut mesh_rebuilt = mesh_orig.clone();
    mesh_rebuilt.rebuild_adjacency();
    assert_eq!(mesh_orig.vertex_boundaries, mesh_rebuilt.vertex_boundaries);
    assert_eq!(mesh_orig.vertex_simplices, mesh_rebuilt.vertex_simplices);
    assert_eq!(mesh_orig.boundary_simplices, mesh_rebuilt.boundary_simplices);
}

// -------------------------------------------------------------------------
// K=4 tet mesh smoke tests
// -------------------------------------------------------------------------

#[test]
fn tet_mesh_k4_adjacency() {
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.0, 1.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    assert_eq!(mesh.n_vertices(), 4);
    assert_eq!(mesh.n_simplices(), 1);
    assert_eq!(mesh.n_boundaries(), 4);

    // Each vertex is in 3 boundary faces.
    for v in 0..4 {
        assert_eq!(
            mesh.vertex_boundaries[v].len(),
            3,
            "vertex {v} has {} boundary faces, expected 3",
            mesh.vertex_boundaries[v].len()
        );
    }

    // Each vertex is in 1 simplex.
    for v in 0..4 {
        assert_eq!(mesh.vertex_simplices[v].len(), 1);
    }

    // Each boundary face has exactly 1 co-simplex.
    for b in 0..4 {
        assert_eq!(mesh.boundary_simplices[b].len(), 1);
    }

    // Euler characteristic: V - B + S = 4 - 4 + 1 = 1
    assert_eq!(mesh.euler_characteristic(), 1);
}

#[test]
fn tet_mesh_k4_exterior_dimensions() {
    // For K=4: d[0] is (n_faces x n_vertices), d[1] is (n_tets x n_faces).
    // Note: exactness d[1]*d[0]=0 does NOT hold for K=4 because our 2-level
    // mesh stores faces (2-simplices) as boundaries, skipping edges. The
    // chain complex requires all intermediate simplices for exactness.
    // For K=3 (triangle meshes), boundaries = edges and exactness holds.
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.0, 1.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    let ext = ExteriorDerivative::from_mesh_sparse_generic(&mesh);
    assert_eq!(ext.degree(), 2);
    // d[0]: 4 faces x 4 vertices
    assert_eq!(ext.d0().rows(), 4);
    assert_eq!(ext.d0().cols(), 4);
    // d[1]: 1 tet x 4 faces
    assert_eq!(ext.d1().rows(), 1);
    assert_eq!(ext.d1().cols(), 4);
}

#[test]
fn tet_mesh_k4_simplex_volume() {
    // Volume of the standard simplex [0,0,0],[1,0,0],[0,1,0],[0,0,1] = 1/6.
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.0, 1.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    let vol = mesh.simplex_volume(&manifold, 0);
    let expected = 1.0 / 6.0;
    assert!(
        (vol - expected).abs() < 1e-14,
        "standard tet volume: got {vol}, expected {expected}"
    );
}
