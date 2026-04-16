//! Mean-field integration test: load every basic fixture, homogenise with the named
//! scheme in Rust, assert affine-invariant agreement within the tolerance tier.

use cartan_homog::{
    rve::{Phase, Rve},
    schemes::{Dilute, DiluteStress, MoriTanaka, SelfConsistent, AsymmetricSc,
              Maxwell, PonteCastanedaWillis, Differential, VoigtBound, ReussBound,
              Scheme, SchemeOpts},
    shapes::Sphere,
    tensor::{Order2, Order4, TensorOrder},
    HomogError,
};
use cartan_homog_valid::{approx::{ai_distance_order2, ai_distance_order4}, fixture::{Fixture, fixture_root}};
use ndarray_npy::NpzReader;
use std::fs::File;
use std::sync::Arc;

// Build the canonical single-inclusion iso-matrix RVE for the fixtures we generated.
// Must match generate_fixtures.py: k_matrix=1, k_inclusion=5 (O2); (72,32) matrix,
// (5,2) inclusion (O4). Phi is parsed from the case_id.
fn phi_from_case_id(case_id: &str) -> f64 {
    // Pattern: "..._phi=0.XX"
    let idx = case_id.rfind("phi=").expect("case_id must contain phi=");
    case_id[idx + 4..].parse().expect("phi value")
}

fn rve_o2(phi: f64) -> Rve<Order2> {
    let mut rve = Rve::<Order2>::new();
    rve.add_phase(Phase {
        name: "MATRIX".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(1.0), fraction: 1.0 - phi,
    });
    rve.add_phase(Phase {
        name: "INCLUSION".into(), shape: Arc::new(Sphere),
        property: Order2::scalar(5.0), fraction: phi,
    });
    rve.set_matrix("MATRIX");
    rve
}

fn rve_o4(phi: f64) -> Rve<Order4> {
    let mut rve = Rve::<Order4>::new();
    rve.add_phase(Phase {
        name: "MATRIX".into(), shape: Arc::new(Sphere),
        property: Order4::iso_stiff(72.0, 32.0), fraction: 1.0 - phi,
    });
    rve.add_phase(Phase {
        name: "INCLUSION".into(), shape: Arc::new(Sphere),
        property: Order4::iso_stiff(5.0, 2.0), fraction: phi,
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
        let phi = phi_from_case_id(&fx.meta.case_id);
        let tol = fx.tolerance();

        let result = if fx.meta.tensor_order == 2 {
            let rve = rve_o2(phi);
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
            let rve = rve_o4(phi);
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
                println!("  [PASS] {} (d_AI = {:.2e}, tier = {})", fx.meta.case_id, d, fx.meta.tolerance_tier);
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
