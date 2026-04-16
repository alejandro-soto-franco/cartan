//! Homogenisation benchmark binary.
//!
//! Times cartan-homog across a sweep of (scheme × shape × volume fraction) and
//! emits a JSON-line result per case. Paired with `benchmarks/python/bench_homog.py`
//! which times ECHOES on the same matrix, for head-to-head comparison.
//!
//! ```bash
//! cargo run --release --bin cartan-bench-homog -- \
//!     --out ../results/homog_cartan.jsonl
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use nalgebra::{Unit, Vector3};

use cartan_homog::{
    rve::{Phase, Rve},
    schemes::{
        AsymmetricSc, Dilute, DiluteStress, Differential, Maxwell, MoriTanaka,
        PonteCastanedaWillis, ReussBound, Scheme, SchemeOpts, SelfConsistent, VoigtBound,
    },
    shapes::{PennyCrack, Sphere, Spheroid},
    tensor::{Order2, Order4, TensorOrder},
};

const WARMUP: usize = 3;
const REPS: usize = 50;

#[derive(Parser)]
struct Args {
    /// Output file for JSON-line results.
    #[arg(long, default_value = "homog_cartan.jsonl")]
    out: String,

    /// Whether to include high-cost iterative schemes (SC, ASC, DIFF).
    #[arg(long, default_value_t = true)]
    iterative: bool,
}

fn main() {
    let args = Args::parse();
    let out = File::create(&args.out).expect("create output file");
    let mut wr = BufWriter::new(out);

    // Sweep parameters.
    let fractions_sphere = [0.05, 0.10, 0.20, 0.30, 0.40];
    let fractions_spheroid = [0.10, 0.20, 0.30];
    let aspects = [0.1_f64, 10.0];   // oblate and prolate
    let densities_crack = [0.05, 0.15, 0.30];

    // Order2: iso matrix (k=1.0), inclusion k=5.0.
    for &phi in &fractions_sphere {
        for scheme in scheme_list(args.iterative) {
            let rve = rve_sphere_o2(phi);
            bench_case(&mut wr, "O2", &scheme, &rve, phi, "sphere", None);
        }
    }
    for &aspect in &aspects {
        for &phi in &fractions_spheroid {
            for scheme in scheme_list(args.iterative) {
                let rve = rve_spheroid_o2(phi, aspect);
                let shape_label = if aspect < 1.0 { "oblate" } else { "prolate" };
                bench_case(&mut wr, "O2", &scheme, &rve, phi, shape_label, Some(aspect));
            }
        }
    }
    for &rho in &densities_crack {
        for scheme in scheme_list(args.iterative) {
            let rve = rve_crack_o2(rho);
            bench_case(&mut wr, "O2", &scheme, &rve, rho, "crack", None);
        }
    }

    // Order4: iso matrix (k=72, mu=32), inclusion (k=5, mu=2).
    for &phi in &fractions_sphere {
        for scheme in scheme_list(args.iterative) {
            let rve = rve_sphere_o4(phi);
            bench_case_o4(&mut wr, "O4", &scheme, &rve, phi, "sphere");
        }
    }

    wr.flush().unwrap();
    eprintln!("Wrote {} to {}", std::fs::metadata(&args.out).unwrap().len(), args.out);
}

fn scheme_list(iterative: bool) -> Vec<String> {
    let mut v = vec![
        "VOIGT".into(), "REUSS".into(),
        "DIL".into(), "DILD".into(),
        "MT".into(), "MAX".into(), "PCW".into(),
    ];
    if iterative {
        v.push("SC".into());
        v.push("ASC".into());
        v.push("DIFF".into());
    }
    v
}

fn rve_sphere_o2(phi: f64) -> Rve<Order2> {
    let mut r = Rve::<Order2>::new();
    r.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(1.0), fraction: 1.0 - phi });
    r.add_phase(Phase { name: "I".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(5.0), fraction: phi });
    r.set_matrix("M");
    r
}

fn rve_spheroid_o2(phi: f64, aspect: f64) -> Rve<Order2> {
    let mut r = Rve::<Order2>::new();
    r.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(1.0), fraction: 1.0 - phi });
    r.add_phase(Phase { name: "I".into(),
        shape: Arc::new(Spheroid::new(Unit::new_normalize(Vector3::z()), aspect)),
        property: Order2::scalar(5.0), fraction: phi });
    r.set_matrix("M");
    r
}

fn rve_crack_o2(rho: f64) -> Rve<Order2> {
    let mut r = Rve::<Order2>::new();
    r.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(1.0), fraction: 1.0 - rho });
    r.add_phase(Phase { name: "C".into(),
        shape: Arc::new(PennyCrack::new(Unit::new_normalize(Vector3::z()), rho)),
        property: Order2::scalar(5.0e-6), fraction: rho });
    r.set_matrix("M");
    r
}

fn rve_sphere_o4(phi: f64) -> Rve<Order4> {
    let mut r = Rve::<Order4>::new();
    r.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
        property: Order4::iso_stiff(72.0, 32.0), fraction: 1.0 - phi });
    r.add_phase(Phase { name: "I".into(), shape: Arc::new(Sphere),
        property: Order4::iso_stiff(5.0, 2.0), fraction: phi });
    r.set_matrix("M");
    r
}

fn bench_case<W: Write>(
    wr: &mut W, order: &str, scheme: &str, rve: &Rve<Order2>,
    param: f64, shape: &str, aspect: Option<f64>,
) {
    let opts = SchemeOpts::default();
    let (median_ns, result_k11) = time_scheme(scheme, rve, &opts);
    let record = serde_json::json!({
        "library": "cartan",
        "order": order,
        "scheme": scheme,
        "shape": shape,
        "aspect": aspect,
        "param": param,
        "median_ns": median_ns,
        "k_eff_11": result_k11,
    });
    writeln!(wr, "{record}").unwrap();
}

fn bench_case_o4<W: Write>(
    wr: &mut W, order: &str, scheme: &str, rve: &Rve<Order4>,
    param: f64, shape: &str,
) {
    let opts = SchemeOpts::default();
    let (median_ns, result_c11) = time_scheme_o4(scheme, rve, &opts);
    let record = serde_json::json!({
        "library": "cartan",
        "order": order,
        "scheme": scheme,
        "shape": shape,
        "aspect": serde_json::Value::Null,
        "param": param,
        "median_ns": median_ns,
        "k_eff_11": result_c11,
    });
    writeln!(wr, "{record}").unwrap();
}

fn time_scheme(scheme: &str, rve: &Rve<Order2>, opts: &SchemeOpts) -> (Option<u128>, Option<f64>) {
    for _ in 0..WARMUP {
        if dispatch(scheme, rve, opts).is_err() { return (None, None); }
    }
    let mut times: Vec<u128> = Vec::with_capacity(REPS);
    let mut last = 0.0_f64;
    for _ in 0..REPS {
        let t0 = Instant::now();
        match dispatch(scheme, rve, opts) {
            Ok(e) => {
                times.push(t0.elapsed().as_nanos());
                last = e.tensor[(0, 0)];
            }
            Err(_) => return (None, None),
        }
    }
    times.sort_unstable();
    (Some(times[REPS / 2]), Some(last))
}

fn time_scheme_o4(scheme: &str, rve: &Rve<Order4>, opts: &SchemeOpts) -> (Option<u128>, Option<f64>) {
    for _ in 0..WARMUP {
        if dispatch_o4(scheme, rve, opts).is_err() { return (None, None); }
    }
    let mut times: Vec<u128> = Vec::with_capacity(REPS);
    let mut last = 0.0_f64;
    for _ in 0..REPS {
        let t0 = Instant::now();
        match dispatch_o4(scheme, rve, opts) {
            Ok(e) => {
                times.push(t0.elapsed().as_nanos());
                last = e.tensor[(0, 0)];
            }
            Err(_) => return (None, None),
        }
    }
    times.sort_unstable();
    (Some(times[REPS / 2]), Some(last))
}

fn dispatch(scheme: &str, rve: &Rve<Order2>, opts: &SchemeOpts)
    -> Result<cartan_homog::schemes::Effective<Order2>, cartan_homog::HomogError> {
    match scheme {
        "VOIGT" => VoigtBound.homogenize(rve, opts),
        "REUSS" => ReussBound.homogenize(rve, opts),
        "DIL"   => Dilute.homogenize(rve, opts),
        "DILD"  => DiluteStress.homogenize(rve, opts),
        "MT"    => MoriTanaka.homogenize(rve, opts),
        "SC"    => SelfConsistent.homogenize(rve, opts),
        "ASC"   => AsymmetricSc.homogenize(rve, opts),
        "MAX"   => Maxwell.homogenize(rve, opts),
        "PCW"   => PonteCastanedaWillis.homogenize(rve, opts),
        "DIFF"  => Differential::default().homogenize(rve, opts),
        _ => panic!("unknown scheme {scheme}"),
    }
}

fn dispatch_o4(scheme: &str, rve: &Rve<Order4>, opts: &SchemeOpts)
    -> Result<cartan_homog::schemes::Effective<Order4>, cartan_homog::HomogError> {
    match scheme {
        "VOIGT" => VoigtBound.homogenize(rve, opts),
        "REUSS" => ReussBound.homogenize(rve, opts),
        "DIL"   => Dilute.homogenize(rve, opts),
        "DILD"  => DiluteStress.homogenize(rve, opts),
        "MT"    => MoriTanaka.homogenize(rve, opts),
        "SC"    => SelfConsistent.homogenize(rve, opts),
        "ASC"   => AsymmetricSc.homogenize(rve, opts),
        "MAX"   => Maxwell.homogenize(rve, opts),
        "PCW"   => PonteCastanedaWillis.homogenize(rve, opts),
        "DIFF"  => Differential::default().homogenize(rve, opts),
        _ => panic!("unknown scheme {scheme}"),
    }
}
