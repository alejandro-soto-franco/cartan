use cartan_feec::cochain::Cochain;
use cartan_maxwell::{cfl_dt, coboundary_matrix, FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver};
use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

/// Helper: sparse-matrix times dense vector (sprs does not impl Mul<DVector>).
fn spmv(m: &sprs::CsMat<f64>, v: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut result = nalgebra::DVector::zeros(m.rows());
    for (&val, (row, col)) in m.iter() {
        result[row] += val * v[col];
    }
    result
}

#[test]
fn synchronized_energy_is_tightly_conserved_on_static_cavity() {
    // Same mesh and initial conditions as the Task 5 energy test, but using the
    // synchronized half-step energy (tighter +-5% tolerance justified by better stagger match).
    let (complex, coords) = CartesianMeshInfo::new_unit(2, 6).compute_coord_complex();
    let base = coords.to_edge_lengths(&complex);
    let driver = FlrwDriver::static_metric(complex.clone(), base);
    let l0 = driver.lengths_at(0.0);

    let dt = 0.5 * cfl_dt(&l0);
    let mut evolver = MaxwellEvolver::new(&driver, dt);

    let d1 = coboundary_matrix(&complex, 1);
    let seed = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| ((i + 3) as f64).recip());
    let b0_coeffs = spmv(&d1, &seed);
    let e0 = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| 0.05 * (i as f64).cos());
    let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0_coeffs));

    let u0 = evolver.step_with_energy(&mut state, None);
    assert!(u0 > 0.0, "initial synchronized energy must be positive");

    let mut umin = u0;
    let mut umax = u0;
    for _ in 1..200 {
        let u = evolver.step_with_energy(&mut state, None);
        umin = umin.min(u);
        umax = umax.max(u);
    }
    // Tighter band: synchronized energy conserved within +-5%.
    assert!(umax / u0 < 1.05, "synchronized energy grew: umax/u0 = {}", umax / u0);
    assert!(umin / u0 > 0.95, "synchronized energy decayed: umin/u0 = {}", umin / u0);
}
