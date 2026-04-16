//! End-to-end pipeline wall-clock benchmark.
//!
//! Times the blog's fractured-sandstone capstone stage-by-stage:
//!   1. Homogenise at 7 sampled depths (Mori-Tanaka on penny-crack RVEs)
//!   2. SPD geodesic interpolation across depths
//!   3. Slab tet mesh generation (PeriodicCubeMeshBuilder stretched to slab)
//!   4. Anisotropic Darcy solve (top/bottom Dirichlet, lateral no-flow, dense LU)
//!   5. Hausdorff refinement-vs-analytic check
//!   6. Full-field cell-problem cross-check at z=H/2
//!
//! Writes a JSON-line report: one row per stage with wall-clock in nanoseconds
//! plus the derived physical metrics (k_eff, Hausdorff distance, anisotropy
//! ratio). Paired with `benchmarks/python/bench_pipeline_echoes.py` which
//! times just stage 1 through ECHOES (the only stage ECHOES can do).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use nalgebra::{Matrix3, Unit, Vector3};

use cartan_homog::{
    fullfield::{FullField, hausdorff, macroscale::SlabProblem,
                mesh::{PeriodicCubeMeshBuilder, PeriodicCubeMeshBuilderOpts},
                reliability_indicator_order2},
    rve::{Phase, Rve},
    schemes::{MoriTanaka, Scheme, SchemeOpts},
    shapes::{PennyCrack, Sphere},
    tensor::{Order2, TensorOrder},
};

// Problem parameters (match the capstone test in cartan-homog-valid).
const L_X: f64 = 100.0;
const L_Y: f64 = 100.0;
const H:   f64 = 200.0;
const K0:  f64 = 1.0e-13;
const RHO0: f64 = 0.2;
const AMPL: f64 = 0.5;
const ELL:  f64 = 40.0;

fn crack_density(z: f64) -> f64 {
    RHO0 * (1.0 + AMPL * (2.0 * core::f64::consts::PI * z / ELL).sin())
}

fn rve_at_depth(z: f64) -> Rve<Order2> {
    let rho = crack_density(z);
    let mut rve = Rve::<Order2>::new();
    rve.add_phase(Phase {
        name: "MATRIX".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(K0), fraction: 1.0 - rho,
    });
    rve.add_phase(Phase {
        name: "CRACK".into(),
        shape: Arc::new(PennyCrack::new(Unit::new_normalize(Vector3::z()), rho)),
        property: Order2::scalar(K0 * 1.0e-6),
        fraction: rho,
    });
    rve.set_matrix("MATRIX");
    rve
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(||
        "benchmarks/results/pipeline_capstone.jsonl".into());
    let out = File::create(&out_path).expect("create output file");
    let mut wr = BufWriter::new(out);

    let depths: [f64; 7] = [
        H / 8.0, H / 4.0, 3.0 * H / 8.0, H / 2.0, 5.0 * H / 8.0, 3.0 * H / 4.0, 7.0 * H / 8.0,
    ];

    let t_total = Instant::now();

    // Stage 1: homogenisation at 7 depths.
    let t_stage = Instant::now();
    let mut k_by_depth: Vec<(f64, Matrix3<f64>)> = Vec::new();
    for &z in &depths {
        let rve = rve_at_depth(z);
        let e = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
        k_by_depth.push((z, e.tensor));
    }
    let ns_stage1 = t_stage.elapsed().as_nanos();
    writeln!(wr, r#"{{"stage":1,"name":"mori_tanaka_at_depths","n_calls":{},"ns":{}}}"#,
             depths.len(), ns_stage1).unwrap();

    // Stage 2: SPD geodesic interpolation (compute midpoint between first and last).
    let t_stage = Instant::now();
    let k_mid = Order2::spd_geodesic_step(&k_by_depth[0].1, &k_by_depth[6].1, 0.5).unwrap();
    let eig_mid = k_mid.symmetric_eigen();
    let ns_stage2 = t_stage.elapsed().as_nanos();
    writeln!(wr, r#"{{"stage":2,"name":"spd_geodesic_interp","ns":{},"mid_eig_min":{},"mid_eig_max":{}}}"#,
             ns_stage2,
             eig_mid.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min),
             eig_mid.eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .unwrap();

    // Stage 3: slab mesh generation.
    let t_stage = Instant::now();
    let k_of_z = Box::new(|z: f64| -> Matrix3<f64> {
        let rve = rve_at_depth(z);
        MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap().tensor
    });
    let prob = SlabProblem { l_x: L_X, l_y: L_Y, h: H, k_of_z,
        p_top: 0.0, p_bot: 1.0, resolution: 6 };
    let ns_stage3 = t_stage.elapsed().as_nanos();
    writeln!(wr, r#"{{"stage":3,"name":"slab_problem_setup","resolution":6,"ns":{}}}"#, ns_stage3).unwrap();

    // Stage 4: Darcy solve (macroscale).
    let t_stage = Instant::now();
    let sol = prob.solve().unwrap();
    let ns_stage4 = t_stage.elapsed().as_nanos();
    writeln!(wr, r#"{{"stage":4,"name":"darcy_solve","n_tets":{},"ns":{},"k_eff_zz":{}}}"#,
             sol.mesh.n_simplices(), ns_stage4, sol.k_eff_macro[(2, 2)]).unwrap();

    // Stage 5: Hausdorff gate.
    let t_stage = Instant::now();
    let rho_prime = |z: f64| -> f64 {
        let k = 2.0 * core::f64::consts::PI / ELL;
        RHO0 * AMPL * k * (k * z).cos()
    };
    let threshold = 0.5 * RHO0 * AMPL * 2.0 * core::f64::consts::PI / ELL;
    let builder = PeriodicCubeMeshBuilder::new(&PeriodicCubeMeshBuilderOpts {
        resolution: 8, refine_depth: 0,
    });
    let (unit_mesh, _) = builder.build().unwrap();
    let slab_bary: Vec<Vector3<f64>> = unit_mesh.simplices.iter().map(|tet| {
        let v: [Vector3<f64>; 4] = [
            unit_mesh.vertices[tet[0]], unit_mesh.vertices[tet[1]],
            unit_mesh.vertices[tet[2]], unit_mesh.vertices[tet[3]],
        ];
        let b = (v[0] + v[1] + v[2] + v[3]) / 4.0;
        Vector3::new(b.x * L_X, b.y * L_Y, b.z * H)
    }).collect();
    let refined = hausdorff::refined_barycentres(&slab_bary, rho_prime, threshold);
    let analytic = hausdorff::analytic_transition_points(L_X, L_Y, H, 4, 200, rho_prime, threshold);
    let d_h = hausdorff::one_sided_hausdorff(&refined, &analytic);
    let ns_stage5 = t_stage.elapsed().as_nanos();
    writeln!(wr, r#"{{"stage":5,"name":"hausdorff_gate","n_refined":{},"n_analytic":{},"d_h":{},"ns":{}}}"#,
             refined.len(), analytic.len(), d_h, ns_stage5).unwrap();

    // Stage 6: full-field vs mean-field cross-check at z=H/2.
    let t_stage = Instant::now();
    let z = H / 2.0;
    let rho = crack_density(z);
    let mut rve_ff = Rve::<Order2>::new();
    rve_ff.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(K0), fraction: 1.0 - rho });
    rve_ff.add_phase(Phase { name: "I".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(K0 * 1.0e-6), fraction: rho });
    rve_ff.set_matrix("M");

    let mut ff = FullField::<Order2>::new_with_resolution(8);
    ff.tol = 1e-6;
    ff.max_iter = 20_000;
    let e_ff = ff.homogenize(&rve_ff).unwrap();
    let e_mf = MoriTanaka.homogenize(&rve_ff, &SchemeOpts::default()).unwrap();
    let d_ai = reliability_indicator_order2(&e_ff.tensor, &e_mf.tensor).unwrap();
    let ns_stage6 = t_stage.elapsed().as_nanos();
    writeln!(wr, r#"{{"stage":6,"name":"full_field_vs_mf","resolution":8,"d_ai":{},"ns":{}}}"#,
             d_ai, ns_stage6).unwrap();

    let ns_total = t_total.elapsed().as_nanos();
    writeln!(wr, r#"{{"stage":"total","name":"end_to_end","ns":{},"ms":{}}}"#,
             ns_total, ns_total as f64 / 1.0e6).unwrap();

    wr.flush().unwrap();
    eprintln!("Pipeline capstone: {} ms total, {} stages timed",
              ns_total / 1_000_000, 6);
}
