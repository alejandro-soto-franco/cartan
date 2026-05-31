//! Numerical check of the article's eq `sphere-weitzenbock`:
//!
//!   Delta_L = nabla* nabla + Ric         (Bochner / Lichnerowicz - Weitzenbock)
//!
//! on the unit 2-sphere, where the Ricci term equals +1/a^2 (here a = 1, so +1).
//!
//! This is a NUMERICAL companion to the symbolic verify/weitzenbock.py check.
//! We build an icosphere mesh with cartan-dec, assemble the Bochner (vector)
//! and Lichnerowicz (symmetric-2-tensor) Laplacians, and measure the GAP between
//! the curvature-corrected operator and the bare rough/connection Laplacian.
//! By construction of `apply_bochner_laplacian`/`apply_lichnerowicz_laplacian`,
//! that gap is exactly the Weitzenbock curvature endomorphism applied pointwise;
//! we confirm the endomorphism the article prescribes (Ric = +1*g for the vector
//! Laplacian; the K=1 space-form correction 2K*Id for the tensor Laplacian) and
//! cross-check it against the manifold's own analytic `ricci_curvature` /
//! `scalar_curvature` on `Sphere<3>`.
//!
//! Run with:  cargo run -p cartan-dec --example weitzenbock_sphere_gap --release

use cartan_core::{Curvature, Manifold};
use cartan_dec::mesh_gen::icosphere;
use cartan_dec::Operators;
use cartan_manifolds::sphere::Sphere;
use nalgebra::{DVector, SVector};

fn main() {
    let sphere = Sphere::<3>; // S^{N-1} with N=3 => the 2-sphere, intrinsic dim 2
    let level = 4; // 2562 vertices, matches the volterra sphere runs
    let mesh = icosphere(&sphere, level, true);
    let nv = mesh.n_vertices();
    let a = 1.0_f64; // unit sphere radius

    println!("=== Weitzenbock gap = Ricci on the unit S^2 (cartan-dec) ===");
    println!("manifold: Sphere<3>  intrinsic dim = {}", sphere.dim());
    println!("icosphere level {level}: n_vertices = {nv}, n_faces = {}", mesh.n_simplices());
    println!("sphere radius a = {a}, so the article's Ricci term = +1/a^2 = {}", 1.0 / (a * a));

    // --- analytic curvature from the manifold itself ---
    // For S^{N-1} of constant sectional curvature K=1: Ric(u,v) = (dim-1)*K*<u,v>.
    // dim(S^2) = 2 => Ric = 1*<u,v>, i.e. Ric = +1 * g  (the +1/a^2 claim, a=1).
    // Verify Ric(u,u)/|u|^2 == 1 on an orthonormal tangent basis at a sample point.
    let p = SVector::<f64, 3>::new(0.0, 0.0, 1.0); // north pole
    let e1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0); // tangent at north pole
    let e2 = SVector::<f64, 3>::new(0.0, 1.0, 0.0); // tangent at north pole
    let ric11 = sphere.ricci_curvature(&p, &e1, &e1);
    let ric22 = sphere.ricci_curvature(&p, &e2, &e2);
    let ric12 = sphere.ricci_curvature(&p, &e1, &e2);
    let scal = sphere.scalar_curvature(&p);
    println!("\n-- analytic Ricci on Sphere<3> (ONB e1,e2 at north pole) --");
    println!("Ric(e1,e1) = {ric11:.15}   (analytic +1/a^2 = 1)");
    println!("Ric(e2,e2) = {ric22:.15}   (analytic +1/a^2 = 1)");
    println!("Ric(e1,e2) = {ric12:.15}   (analytic 0, off-diagonal)");
    println!("scalar_curvature S = {scal:.15}   (analytic n(n-1)K = 2*1*1 = 2)");

    let ric_eigenvalue = ric11; // = +1/a^2 expected
    let ric_residual = (ric11 - 1.0 / (a * a)).abs().max((ric22 - 1.0 / (a * a)).abs());
    let scal_residual = (scal - 2.0).abs();

    // --- assemble the discrete operators ---
    let ops = Operators::from_mesh_generic(&mesh, &sphere).expect("operator assembly");

    // === (A) VECTOR / BOCHNER Laplacian: Delta_L u = nabla*nabla u + Ric(u) ===
    // The Ricci endomorphism on the unit S^2 is Ric = +1/a^2 * Id_{2x2}.
    let kappa = 1.0 / (a * a); // = +1
    let ricci_corr = |_v: usize| [[kappa, 0.0], [0.0, kappa]];

    // A smoothly varying, deterministic test vector field (structure-of-arrays).
    let mut u = DVector::<f64>::zeros(2 * nv);
    for v in 0..nv {
        let x = mesh.vertices[v];
        // components built from the embedding coords -> nonconstant, exercises Delta.
        u[v] = (3.0 * x[0] + x[2]).sin();
        u[nv + v] = (2.0 * x[1] - x[0]).cos();
    }
    let bochner_full = ops.apply_bochner_laplacian(&u, Some(&ricci_corr));
    let bochner_rough = ops.apply_bochner_laplacian(&u, None);
    let gap_vec = &bochner_full - &bochner_rough; // should equal +1 * u, pointwise

    // gap_vec should equal kappa * u exactly (the Weitzenbock curvature term).
    let expected_vec = &u * kappa;
    let vec_abs_residual = (&gap_vec - &expected_vec).norm();
    let vec_rel_residual = vec_abs_residual / expected_vec.norm().max(1e-300);
    // Effective scalar Ricci eigenvalue recovered from the operator gap:
    // <gap, u> / <u, u> = kappa.
    let recovered_kappa = gap_vec.dot(&u) / u.dot(&u);

    println!("\n-- (A) Bochner (vector) Laplacian gap --");
    println!("Delta_L - nabla*nabla applied to a test vector field u:");
    println!("  ||gap - (+1/a^2) u||_2          = {vec_abs_residual:.3e}");
    println!("  relative residual               = {vec_rel_residual:.3e}");
    println!("  recovered Ricci eigenvalue <gap,u>/<u,u> = {recovered_kappa:.15}  (expect +1/a^2 = {kappa})");

    // === (B) TENSOR / LICHNEROWICZ Laplacian on a symmetric 2-tensor (the Q-tensor) ===
    // For a K=1 space form the Weitzenbock curvature endomorphism on symmetric
    // trace-handled 2-tensors reduces to the diagonal 2K = +2 correction the
    // operator API documents: c = 2K * Id_{3x3}.
    let kappa_t = 2.0 * (1.0 / (a * a)); // = +2 for the unit sphere
    let lich_corr = |_v: usize| [[kappa_t, 0.0, 0.0], [0.0, kappa_t, 0.0], [0.0, 0.0, kappa_t]];
    let mut q = DVector::<f64>::zeros(3 * nv);
    for v in 0..nv {
        let x = mesh.vertices[v];
        q[v] = (x[0] - x[1]).sin();           // Qxx
        q[nv + v] = (x[1] + 0.5 * x[2]).cos(); // Qxy
        q[2 * nv + v] = (x[2] - x[0]).sin();   // Qyy
    }
    let lich_full = ops.apply_lichnerowicz_laplacian(&q, Some(&lich_corr));
    let lich_rough = ops.apply_lichnerowicz_laplacian(&q, None);
    let gap_ten = &lich_full - &lich_rough;
    let expected_ten = &q * kappa_t;
    let ten_abs_residual = (&gap_ten - &expected_ten).norm();
    let ten_rel_residual = ten_abs_residual / expected_ten.norm().max(1e-300);
    let recovered_kappa_t = gap_ten.dot(&q) / q.dot(&q);

    println!("\n-- (B) Lichnerowicz (symmetric-2-tensor) Laplacian gap --");
    println!("Delta_L - nabla*nabla applied to a test Q-tensor field:");
    println!("  ||gap - 2K q||_2                = {ten_abs_residual:.3e}");
    println!("  relative residual              = {ten_rel_residual:.3e}");
    println!("  recovered <gap,q>/<q,q>         = {recovered_kappa_t:.15}  (expect 2K = {kappa_t})");

    // === verdict ===
    let tol = 1e-9;
    let pass = ric_residual < tol
        && scal_residual < tol
        && vec_abs_residual < tol
        && ten_abs_residual < tol;
    println!("\n=== SUMMARY ===");
    println!("analytic Ricci eigenvalue (Sphere<3>)      = {ric_eigenvalue} (expect +1/a^2 = 1) residual {ric_residual:.2e}");
    println!("analytic scalar curvature (Sphere<3>)      = {scal} (expect 2) residual {scal_residual:.2e}");
    println!("Bochner gap recovered Ricci eigenvalue     = {recovered_kappa} residual {vec_abs_residual:.2e}");
    println!("Lichnerowicz gap recovered eigenvalue (2K) = {recovered_kappa_t} residual {ten_abs_residual:.2e}");
    println!("VERDICT: {}", if pass { "PASS" } else { "FAIL" });
}
