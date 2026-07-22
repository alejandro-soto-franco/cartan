//! Two-phase RVE through the mean-field schemes.
//!
//! Takes a matrix with spherical inclusions of a contrasting conductivity and
//! reports what each scheme predicts for the effective property, alongside the
//! Voigt and Reuss bounds that bracket any admissible answer.
//!
//! Run with:
//!
//! ```text
//! cargo run -p cartan-homog --example rve_to_effective
//! ```

use std::sync::Arc;

use cartan_homog::{
    AsymmetricSc, Dilute, Maxwell, MoriTanaka, Order2, Phase, ReussBound, Rve, Scheme,
    SchemeOpts, SelfConsistent, Sphere, TensorOrder, VoigtBound,
};

/// Conductivity of the surrounding matrix.
const K_MATRIX: f64 = 1.0;
/// Conductivity of the inclusions, five times the matrix.
const K_INCLUSION: f64 = 5.0;

fn build_rve(volume_fraction: f64) -> Rve<Order2> {
    let mut rve = Rve::<Order2>::new();
    rve.add_phase(Phase {
        name: "MATRIX".into(),
        shape: Arc::new(Sphere),
        property: Order2::scalar(K_MATRIX),
        fraction: 1.0 - volume_fraction,
    });
    rve.add_phase(Phase {
        name: "INCLUSION".into(),
        shape: Arc::new(Sphere),
        property: Order2::scalar(K_INCLUSION),
        fraction: volume_fraction,
    });
    rve.set_matrix("MATRIX");
    rve
}

/// Isotropic effective conductivity, read off the Kelvin-Mandel diagonal.
fn k_eff<S: Scheme<Order2>>(scheme: &S, rve: &Rve<Order2>) -> Option<f64> {
    scheme
        .homogenize(rve, &SchemeOpts::default())
        .ok()
        .map(|eff| eff.tensor[(0, 0)])
}

fn main() {
    println!(
        "Two-phase RVE: spherical inclusions (k = {K_INCLUSION}) in a matrix (k = {K_MATRIX}).\n"
    );
    println!(
        "{:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "phi", "Reuss", "Dilute", "MT", "SC", "ASC", "Voigt"
    );
    println!("{}", "-".repeat(66));

    for step in 0..=8 {
        let phi = step as f64 * 0.05;
        let rve = build_rve(phi);

        let reuss = k_eff(&ReussBound, &rve);
        let voigt = k_eff(&VoigtBound, &rve);
        let dilute = k_eff(&Dilute, &rve);
        let mt = k_eff(&MoriTanaka, &rve);
        let sc = k_eff(&SelfConsistent, &rve);
        let asc = k_eff(&AsymmetricSc, &rve);

        let fmt = |v: Option<f64>| match v {
            Some(x) => format!("{x:8.5}"),
            None => "       -".to_string(),
        };

        println!(
            "{phi:>6.2}  {}  {}  {}  {}  {}  {}",
            fmt(reuss),
            fmt(dilute),
            fmt(mt),
            fmt(sc),
            fmt(asc),
            fmt(voigt)
        );

        // Voigt and Reuss bracket every admissible scheme. This is the
        // cheapest available check that a prediction is physical, and it is
        // asserted rather than merely printed.
        if let (Some(lo), Some(hi)) = (reuss, voigt) {
            for (name, value) in [("Dilute", dilute), ("MT", mt), ("SC", sc), ("ASC", asc)] {
                if let Some(v) = value {
                    assert!(
                        v >= lo - 1e-12 && v <= hi + 1e-12,
                        "{name} at phi = {phi}: {v} escapes the Reuss-Voigt bracket [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    println!(
        "\nAll schemes stayed inside the Reuss-Voigt bracket at every volume fraction."
    );

    // Maxwell's scheme is the other classical single-inclusion estimate. At
    // dilute concentrations it should agree closely with Mori-Tanaka, since
    // both reduce to the same first-order expansion.
    let dilute_rve = build_rve(0.02);
    if let (Some(mt), Some(maxwell)) = (
        k_eff(&MoriTanaka, &dilute_rve),
        k_eff(&Maxwell, &dilute_rve),
    ) {
        println!(
            "\nAt phi = 0.02, Mori-Tanaka gives {mt:.6} and Maxwell gives {maxwell:.6}, \
             differing by {:.2e}.",
            (mt - maxwell).abs()
        );
    }
}
