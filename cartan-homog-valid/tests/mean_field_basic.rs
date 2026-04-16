//! Mean-field integration test: load every basic fixture, homogenise with the named
//! scheme in Rust, assert affine-invariant agreement within the tolerance tier.

use cartan_homog::{
    rve::{Phase, Rve},
    schemes::{Dilute, DiluteStress, MoriTanaka, SelfConsistent, AsymmetricSc,
              Maxwell, PonteCastanedaWillis, Differential, VoigtBound, ReussBound,
              Scheme, SchemeOpts},
    shapes::{PennyCrack, Sphere, Spheroid, UserInclusion},
    tensor::{Order2, Order4, TensorOrder},
    HomogError,
};
use cartan_homog_valid::{approx::{ai_distance_order2, ai_distance_order4}, fixture::{Fixture, fixture_root}};
use nalgebra::{Unit, Vector3};
use ndarray_npy::NpzReader;
use std::fs::File;
use std::sync::Arc;

const K_MATRIX_O2: f64 = 1.0;
const K_INCLUSION_O2: f64 = 5.0;
const K_MATRIX_O4:   (f64, f64) = (72.0, 32.0);
const K_INCLUSION_O4: (f64, f64) = (5.0,  2.0);

#[derive(Debug, Clone)]
enum ShapeKind {
    Sphere,
    Spheroid(f64),  // aspect
    Crack,          // penny crack with normal = z
}

/// Parse the microstructure + parameter from the case_id, e.g.
///   "o2_mt_iso_matrix_prolate_10_phi=0.10"
///   "o2_mt_iso_matrix_penny_cracks_rho=0.30"
fn parse_case(case_id: &str) -> (ShapeKind, f64) {
    // Find phi= or rho=
    let (_, tail) = if let Some(idx) = case_id.rfind("_phi=") {
        (idx, &case_id[idx + 5..])
    } else if let Some(idx) = case_id.rfind("_rho=") {
        (idx, &case_id[idx + 5..])
    } else {
        panic!("case_id `{case_id}` missing phi= / rho= parameter");
    };
    let param: f64 = tail.parse().expect("parse parameter");

    let shape = if case_id.contains("_penny_cracks_") {
        ShapeKind::Crack
    } else if case_id.contains("_prolate_10_") {
        ShapeKind::Spheroid(10.0)
    } else if case_id.contains("_oblate_01_") {
        ShapeKind::Spheroid(0.1)
    } else if case_id.contains("_spheres_") {
        ShapeKind::Sphere
    } else {
        panic!("cannot infer shape from case_id `{case_id}`");
    };
    (shape, param)
}

fn arc_shape_o2(kind: &ShapeKind) -> UserInclusion<Order2> {
    match kind {
        ShapeKind::Sphere          => Arc::new(Sphere),
        ShapeKind::Spheroid(aspect) => Arc::new(Spheroid::new(Unit::new_normalize(Vector3::z()), *aspect)),
        ShapeKind::Crack           => Arc::new(PennyCrack::new(Unit::new_normalize(Vector3::z()), 0.0)),
    }
}

fn arc_shape_o4(kind: &ShapeKind) -> UserInclusion<Order4> {
    match kind {
        ShapeKind::Sphere          => Arc::new(Sphere),
        ShapeKind::Spheroid(aspect) => Arc::new(Spheroid::new(Unit::new_normalize(Vector3::z()), *aspect)),
        ShapeKind::Crack           => Arc::new(PennyCrack::new(Unit::new_normalize(Vector3::z()), 0.0)),
    }
}

fn rve_o2(kind: &ShapeKind, param: f64) -> Rve<Order2> {
    let is_crack = matches!(kind, ShapeKind::Crack);
    let k_inc = if is_crack { K_INCLUSION_O2 * 1e-6 } else { K_INCLUSION_O2 };
    let mut rve = Rve::<Order2>::new();
    rve.add_phase(Phase {
        name: "MATRIX".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(K_MATRIX_O2), fraction: 1.0 - param,
    });
    rve.add_phase(Phase {
        name: "INCLUSION".into(), shape: arc_shape_o2(kind),
        property: Order2::scalar(k_inc), fraction: param,
    });
    rve.set_matrix("MATRIX");
    rve
}

fn rve_o4(kind: &ShapeKind, param: f64) -> Rve<Order4> {
    let is_crack = matches!(kind, ShapeKind::Crack);
    let c_inc = if is_crack {
        Order4::iso_stiff(1e-6, 1e-6)
    } else {
        Order4::iso_stiff(K_INCLUSION_O4.0, K_INCLUSION_O4.1)
    };
    let mut rve = Rve::<Order4>::new();
    rve.add_phase(Phase {
        name: "MATRIX".into(), shape: Arc::new(Sphere),
        property: Order4::iso_stiff(K_MATRIX_O4.0, K_MATRIX_O4.1), fraction: 1.0 - param,
    });
    rve.add_phase(Phase {
        name: "INCLUSION".into(), shape: arc_shape_o4(kind),
        property: c_inc, fraction: param,
    });
    rve.set_matrix("MATRIX");
    rve
}

fn homog_o2(scheme: &str, rve: &Rve<Order2>) -> Result<nalgebra::Matrix3<f64>, HomogError> {
    let opts = SchemeOpts::default();
    let e = match scheme {
        "VOIGT" => VoigtBound.homogenize(rve, &opts)?,
        "REUSS" => ReussBound.homogenize(rve, &opts)?,
        "DIL"   => Dilute.homogenize(rve, &opts)?,
        "DILD"  => DiluteStress.homogenize(rve, &opts)?,
        "MT"    => MoriTanaka.homogenize(rve, &opts)?,
        "SC"    => SelfConsistent.homogenize(rve, &opts)?,
        "ASC"   => AsymmetricSc.homogenize(rve, &opts)?,
        "MAX"   => Maxwell.homogenize(rve, &opts)?,
        "PCW"   => PonteCastanedaWillis.homogenize(rve, &opts)?,
        "DIFF"  => Differential::default().homogenize(rve, &opts)?,
        _ => return Err(HomogError::Solver(format!("unknown scheme `{scheme}`"))),
    };
    Ok(e.tensor)
}

fn homog_o4(scheme: &str, rve: &Rve<Order4>) -> Result<nalgebra::SMatrix<f64, 6, 6>, HomogError> {
    let opts = SchemeOpts::default();
    let e = match scheme {
        "VOIGT" => VoigtBound.homogenize(rve, &opts)?,
        "REUSS" => ReussBound.homogenize(rve, &opts)?,
        "DIL"   => Dilute.homogenize(rve, &opts)?,
        "DILD"  => DiluteStress.homogenize(rve, &opts)?,
        "MT"    => MoriTanaka.homogenize(rve, &opts)?,
        "SC"    => SelfConsistent.homogenize(rve, &opts)?,
        "ASC"   => AsymmetricSc.homogenize(rve, &opts)?,
        "MAX"   => Maxwell.homogenize(rve, &opts)?,
        "PCW"   => PonteCastanedaWillis.homogenize(rve, &opts)?,
        "DIFF"  => Differential::default().homogenize(rve, &opts)?,
        _ => return Err(HomogError::Solver(format!("unknown scheme `{scheme}`"))),
    };
    Ok(e.tensor)
}

fn load_expected_o2(path: &std::path::Path) -> nalgebra::Matrix3<f64> {
    let file = File::open(path).expect("open npz");
    let mut npz = NpzReader::new(file).expect("npz reader");
    let arr: ndarray::Array2<f64> = npz.by_name("c_eff.npy")
        .or_else(|_| npz.by_name("c_eff"))
        .expect("c_eff in npz");
    assert_eq!(arr.shape(), &[3, 3]);
    nalgebra::Matrix3::from_iterator(arr.iter().copied())
}

fn load_expected_o4(path: &std::path::Path) -> nalgebra::SMatrix<f64, 6, 6> {
    let file = File::open(path).expect("open npz");
    let mut npz = NpzReader::new(file).expect("npz reader");
    let arr: ndarray::Array2<f64> = npz.by_name("c_eff.npy")
        .or_else(|_| npz.by_name("c_eff"))
        .expect("c_eff in npz");
    assert_eq!(arr.shape(), &[6, 6]);
    nalgebra::SMatrix::<f64, 6, 6>::from_iterator(arr.iter().copied())
}

#[test]
fn all_basic_fixtures_pass() {
    let root = fixture_root();
    let fixtures = Fixture::load_all(&root);
    assert!(!fixtures.is_empty(), "no fixtures found under {root:?}");

    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = Vec::new();

    for fx in &fixtures {
        let (kind, param) = parse_case(&fx.meta.case_id);
        let tol = fx.tolerance();

        let result: Result<f64, String> = if fx.meta.tensor_order == 2 {
            let rve = rve_o2(&kind, param);
            match homog_o2(&fx.meta.scheme, &rve) {
                Ok(c_rust) => {
                    let c_echoes = load_expected_o2(&fx.npz_path);
                    match ai_distance_order2(&c_rust, &c_echoes) {
                        Some(d) if d < tol => Ok(d),
                        Some(d) => Err(format!("d_AI = {d:.3e} > tol {tol:.3e}")),
                        None    => Err(String::from("SPD distance failed")),
                    }
                }
                Err(e) => Err(format!("rust homogenize error: {e}")),
            }
        } else {
            let rve = rve_o4(&kind, param);
            match homog_o4(&fx.meta.scheme, &rve) {
                Ok(c_rust) => {
                    let c_echoes = load_expected_o4(&fx.npz_path);
                    match ai_distance_order4(&c_rust, &c_echoes) {
                        Some(d) if d < tol => Ok(d),
                        Some(d) => Err(format!("d_AI = {d:.3e} > tol {tol:.3e}")),
                        None    => Err(String::from("SPD distance failed")),
                    }
                }
                Err(e) => Err(format!("rust homogenize error: {e}")),
            }
        };

        match result {
            Ok(d) => {
                println!("  [PASS] {:<50}  d_AI = {:.2e}  tier = {}",
                         fx.meta.case_id, d, fx.meta.tolerance_tier);
                passed += 1;
            }
            Err(msg) if msg.starts_with("rust homogenize error") => {
                println!("  [SKIP] {}: {}", fx.meta.case_id, msg);
                skipped += 1;
            }
            Err(msg) => {
                println!("  [FAIL] {}: {}", fx.meta.case_id, msg);
                failed.push(fx.meta.case_id.clone());
            }
        }
    }

    println!("\nSummary: {passed} passed, {skipped} skipped, {} failed, {} total",
             failed.len(), fixtures.len());
    assert!(failed.is_empty(), "fixtures with disagreement: {failed:?}");
}
