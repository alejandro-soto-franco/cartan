//! Weitzenböck curvature terms of the Bochner and Lichnerowicz Laplacians on the
//! unit 2-sphere, assembled with cartan-dec.
//!
//!   Bochner (vectors):        Delta u = nabla*nabla u + Ric(u),    Ric = K g
//!   Lichnerowicz (Q-tensors): Delta_L Q = nabla*nabla Q + 2nK Q - 2K tr(Q) g
//!
//! On S^2 (n = 2, K = 1/a^2) the Ricci term is +1/a^2 and the Lichnerowicz term
//! is 4K on traceless Q and 0 on the trace part.
//!
//! What this example checks, and what it does not:
//! - The curvature values come from the manifold itself: K and Ric are read off
//!   `Sphere<3>` (`sectional_curvature`, `ricci_curvature`) and fed through the
//!   space-form formulas, so the endomorphism is derived independently of the
//!   numbers printed below.
//! - The operator gap (curvature-corrected minus rough Laplacian) equals the
//!   endomorphism pointwise by construction of the API. That part is a
//!   consistency check of the callback plumbing, not a test of the geometry.
//! - The geometric content (the 4K, the vanishing trace part) is tested
//!   symbolically in `space_form_lichnerowicz_correction`'s unit test.
//!
//! Run with:  cargo run -p cartan-dec --example weitzenbock_sphere_gap --release

use cartan_core::{Curvature, Manifold};
use cartan_dec::mesh_gen::icosphere;
use cartan_dec::{Operators, space_form_lichnerowicz_correction};
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
    println!(
        "icosphere level {level}: n_vertices = {nv}, n_faces = {}",
        mesh.n_simplices()
    );
    println!(
        "sphere radius a = {a}, so the article's Ricci term = +1/a^2 = {}",
        1.0 / (a * a)
    );

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
    let ric_residual = (ric11 - 1.0 / (a * a))
        .abs()
        .max((ric22 - 1.0 / (a * a)).abs());
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
    println!(
        "  recovered Ricci eigenvalue <gap,u>/<u,u> = {recovered_kappa:.15}  (expect +1/a^2 = {kappa})"
    );

    // === (B) TENSOR / LICHNEROWICZ Laplacian on a traceless symmetric 2-tensor ===
    // Curvature from the manifold: K = sectional curvature of the (e1, e2) plane.
    let k_sec = sphere.sectional_curvature(&p, &e1, &e2);
    let lich = space_form_lichnerowicz_correction(k_sec);
    let lich_corr = |_v: usize| lich;
    let kappa_t = 4.0 * k_sec; // eigenvalue on traceless Q (= 4/a^2 on the unit sphere)
    let mut q = DVector::<f64>::zeros(3 * nv);
    for v in 0..nv {
        let x = mesh.vertices[v];
        q[v] = (x[0] - x[1]).sin(); // Qxx
        q[nv + v] = (x[1] + 0.5 * x[2]).cos(); // Qxy
        q[2 * nv + v] = -q[v]; // Qyy = -Qxx (traceless, like a nematic Q)
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
    println!("  ||gap - 4K q||_2                = {ten_abs_residual:.3e}");
    println!("  relative residual              = {ten_rel_residual:.3e}");
    println!(
        "  recovered <gap,q>/<q,q>         = {recovered_kappa_t:.15}  (expect 4K = {kappa_t})"
    );

    // === verdict ===
    let tol = 1e-9;
    let pass = ric_residual < tol
        && scal_residual < tol
        && vec_abs_residual < tol
        && ten_abs_residual < tol;
    println!("\n=== SUMMARY ===");
    println!(
        "analytic Ricci eigenvalue (Sphere<3>)      = {ric_eigenvalue} (expect +1/a^2 = 1) residual {ric_residual:.2e}"
    );
    println!(
        "analytic scalar curvature (Sphere<3>)      = {scal} (expect 2) residual {scal_residual:.2e}"
    );
    println!(
        "Bochner gap recovered Ricci eigenvalue     = {recovered_kappa} residual {vec_abs_residual:.2e}"
    );
    println!(
        "Lichnerowicz gap recovered eigenvalue (4K) = {recovered_kappa_t} residual {ten_abs_residual:.2e}"
    );
    println!("VERDICT: {}", if pass { "PASS" } else { "FAIL" });
}
