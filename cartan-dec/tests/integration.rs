// cartan-dec/tests/integration.rs
//
// End-to-end tests for cartan-dec: mesh topology, DEC identities,
// operator correctness, and physical sanity checks.

use cartan_dec::{
    ExteriorDerivative, FlatMesh, HodgeStar, Operators, apply_divergence, apply_scalar_advection,
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
    assert_eq!(ext.d0.nrows(), mesh.n_boundaries());
    assert_eq!(ext.d0.ncols(), mesh.n_vertices());
}

#[test]
fn d1_dimensions() {
    let mesh = FlatMesh::unit_square_grid(3);
    let ext = ExteriorDerivative::from_mesh(&mesh);
    assert_eq!(ext.d1.nrows(), mesh.n_simplices());
    assert_eq!(ext.d1.ncols(), mesh.n_boundaries());
}

// ─────────────────────────────────────────────────────────────────────────────
// Hodge star: positivity and dimensions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hodge_star0_positive() {
    let mesh = FlatMesh::unit_square_grid(4);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    for (v, &w) in hodge.star0.iter().enumerate() {
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
    for (e, &w) in hodge.star1.iter().enumerate() {
        assert!(w >= 0.0, "star1[{e}] = {w}, expected non-negative");
    }
}

#[test]
fn hodge_star2_positive() {
    let mesh = FlatMesh::unit_square_grid(4);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    for (t, &w) in hodge.star2.iter().enumerate() {
        assert!(w > 0.0, "star2[{t}] = {w}, expected positive");
    }
}

#[test]
fn hodge_star0_total_area_is_one() {
    // sum star0[v] = total area of the domain (sum of dual cell areas = domain area).
    let mesh = FlatMesh::unit_square_grid(4);
    let hodge = HodgeStar::from_mesh(&mesh, &Euclidean::<2>);
    let total: f64 = hodge.star0.iter().sum();
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
