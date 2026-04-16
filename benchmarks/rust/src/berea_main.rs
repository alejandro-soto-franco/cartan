//! Berea sandstone real-data benchmark.
//!
//! Runs cartan-homog's mean-field schemes on published Berea parameters and
//! compares against published measurements and FEM-computed reference values.
//!
//! **Inputs (all from published literature):**
//!
//! | Quantity | Value | Source |
//! |---|---|---|
//! | Porosity ϕ | 0.195 | Andrä et al. 2013, mean of 3-team segmentation (0.184-0.209) |
//! | Mineral bulk modulus K₀ | 39.75 GPa | Zimmerman 1991 |
//! | Mineral shear modulus G₀ | 31.34 GPa | Zimmerman 1991 |
//! | Pore fluid | dry (vacuum) | K_pore = G_pore ≈ 0 |
//!
//! **Cross-check targets:**
//!
//! | Target | Value | Source |
//! |---|---|---|
//! | Drained K @ 10 MPa | 6.6 GPa | Hart 1995 (laboratory measurement) |
//! | FEM-computed K @ ϕ=0.22 | ≈ 13.0 GPa | Arns et al. 2002 |
//! | Hashin-Shtrikman lower bound | computed | Hashin-Shtrikman 1963 |
//! | Hashin-Shtrikman upper bound | computed | Hashin-Shtrikman 1963 |
//!
//! The benchmark emits a JSON summary: each cartan scheme's predicted K and μ,
//! the HS envelope, the comparison against measured/FEM targets, and the
//! wall-clock per scheme call.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use cartan_homog::{
    rve::{Phase, Rve},
    schemes::{
        Differential, Dilute, DiluteStress, Maxwell, MoriTanaka,
        ReussBound, Scheme, SchemeOpts, SelfConsistent, VoigtBound,
    },
    shapes::Sphere,
    tensor::Order4,
};

// Berea parameters, from published literature.
const PHI:        f64 = 0.195;    // Andrä et al. 2013 (3-team segmentation mean)
const K_MINERAL:  f64 = 39.75;    // Zimmerman 1991, in GPa
const MU_MINERAL: f64 = 31.34;    // Zimmerman 1991, in GPa
const K_PORE:     f64 = 1.0e-6;   // dry pore: vacuum
const MU_PORE:    f64 = 1.0e-6;

// Cross-check targets (published Berea values).
const K_MEASURED_DRAINED: f64 = 6.6;   // Hart 1995, drained @ 10 MPa effective stress
const K_FEM_ARNS_2002:    f64 = 13.0;  // Arns 2002 FEM (at φ=0.22, slight porosity offset)

fn hs_bulk_bounds(k1: f64, mu1: f64, k2: f64, mu2: f64, phi2: f64) -> (f64, f64) {
    // Standard HS formula for 2-phase isotropic bulk modulus.
    let hs = |k_inner: f64, _mu_inner: f64, k_outer: f64, mu_outer: f64, phi_outer: f64| {
        k_outer + phi_outer / (1.0 / (k_inner - k_outer)
                               + 3.0 * (1.0 - phi_outer) / (3.0 * k_outer + 4.0 * mu_outer))
    };
    if k1 < k2 {
        let lower = hs(k2, mu2, k1, mu1, phi2);
        let upper = hs(k1, mu1, k2, mu2, 1.0 - phi2);
        (lower.min(upper), lower.max(upper))
    } else {
        let lower = hs(k1, mu1, k2, mu2, 1.0 - phi2);
        let upper = hs(k2, mu2, k1, mu1, phi2);
        (lower.min(upper), lower.max(upper))
    }
}

fn extract_bulk(c: &nalgebra::SMatrix<f64, 6, 6>) -> f64 {
    let (j, _) = Order4::iso_projectors();
    (*c * j).trace() / (3.0 * j.trace())
}

fn extract_shear(c: &nalgebra::SMatrix<f64, 6, 6>) -> f64 {
    let (_, k) = Order4::iso_projectors();
    (*c * k).trace() / (2.0 * k.trace())
}

const WARMUP: usize = 3;
const REPS:   usize = 20;

fn time_and_extract<S: Scheme<Order4>>(scheme: &S, rve: &Rve<Order4>, opts: &SchemeOpts)
    -> (f64, f64, u128)
{
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..WARMUP { let _ = scheme.homogenize(rve, opts); }
    let mut last_k = 0.0;
    let mut last_g = 0.0;
    for _ in 0..REPS {
        let t0 = Instant::now();
        let e = scheme.homogenize(rve, opts).unwrap();
        times.push(t0.elapsed().as_nanos());
        last_k = extract_bulk(&e.tensor);
        last_g = extract_shear(&e.tensor);
    }
    times.sort_unstable();
    (last_k, last_g, times[REPS / 2])
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(||
        "benchmarks/results/berea_cartan.jsonl".into());
    let out = File::create(&out_path).expect("create output file");
    let mut wr = BufWriter::new(out);

    // Build the Berea RVE.
    let mut rve = Rve::<Order4>::new();
    rve.add_phase(Phase {
        name: "MINERAL".into(), shape: Arc::new(Sphere),
        property: Order4::iso_stiff(K_MINERAL, MU_MINERAL),
        fraction: 1.0 - PHI,
    });
    rve.add_phase(Phase {
        name: "PORE".into(), shape: Arc::new(Sphere),
        property: Order4::iso_stiff(K_PORE, MU_PORE),
        fraction: PHI,
    });
    rve.set_matrix("MINERAL");

    let opts = SchemeOpts { max_iter: 500, rel_tol: 1e-10, ..SchemeOpts::default() };

    // Run each scheme, time it, extract k_eff and mu_eff.
    type BenchFn<'a> = &'a dyn Fn() -> (f64, f64, u128);
    let schemes: &[(&str, BenchFn)] = &[
        ("VOIGT",  &|| time_and_extract(&VoigtBound,   &rve, &opts)),
        ("REUSS",  &|| time_and_extract(&ReussBound,   &rve, &opts)),
        ("DIL",    &|| time_and_extract(&Dilute,       &rve, &opts)),
        ("DILD",   &|| time_and_extract(&DiluteStress, &rve, &opts)),
        ("MT",     &|| time_and_extract(&MoriTanaka,   &rve, &opts)),
        ("SC",     &|| time_and_extract(&SelfConsistent, &rve, &opts)),
        ("MAX",    &|| time_and_extract(&Maxwell,      &rve, &opts)),
        ("DIFF",   &|| time_and_extract(&Differential::default(), &rve, &opts)),
    ];

    let (hs_lo, hs_hi) = hs_bulk_bounds(K_MINERAL, MU_MINERAL, K_PORE, MU_PORE, PHI);

    writeln!(wr, r#"{{"meta":{{"phi":{PHI},"K_mineral":{K_MINERAL},"mu_mineral":{MU_MINERAL},"K_measured_drained":{K_MEASURED_DRAINED},"K_fem_arns_2002":{K_FEM_ARNS_2002},"hs_lower":{hs_lo},"hs_upper":{hs_hi}}}}}"#).unwrap();

    println!("\nBerea sandstone mean-field benchmark");
    println!("  porosity = {PHI}");
    println!("  mineral  (K, μ) = ({K_MINERAL}, {MU_MINERAL}) GPa");
    println!("  HS bounds on K_eff: [{hs_lo:.3}, {hs_hi:.3}] GPa");
    println!("  measured (Hart 1995)     K_drained = {K_MEASURED_DRAINED} GPa");
    println!("  FEM      (Arns 2002)     K_fem     ≈ {K_FEM_ARNS_2002} GPa (at φ=0.22)");
    println!();
    println!("  {:<6} {:>10} {:>10} {:>14} {:>14}",
             "scheme", "K (GPa)", "μ (GPa)", "Δ vs measured", "time (ns)");
    for (name, fun) in schemes {
        let (k, g, ns) = fun();
        let in_hs = k >= hs_lo && k <= hs_hi;
        let mark = if in_hs { "✓" } else { "!" };
        let delta_meas = (k - K_MEASURED_DRAINED) / K_MEASURED_DRAINED * 100.0;
        println!("  {:<6} {:>10.3} {:>10.3} {:>12.1}%   {:>12}  {}",
                 name, k, g, delta_meas, ns, mark);
        writeln!(wr, r#"{{"scheme":"{}","k_eff":{},"mu_eff":{},"in_hs":{},"delta_vs_measured_pct":{},"ns":{}}}"#,
                 name, k, g, in_hs, delta_meas, ns).unwrap();
    }

    wr.flush().unwrap();
    println!("\n  ✓ = inside HS bounds,  ! = outside");
}
