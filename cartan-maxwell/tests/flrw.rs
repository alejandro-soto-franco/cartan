use cartan_feec::assemble::assemble_galmat;
use cartan_feec::cochain::Cochain;
use cartan_feec::operators::HodgeMassElmat;
use cartan_maxwell::{cfl_dt, coboundary_matrix, FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver};
use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

/// Sparse matrix-vector multiply helper for integration tests.
fn spmv(m: &sprs::CsMat<f64>, v: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut result = nalgebra::DVector::zeros(m.rows());
    for (&val, (row, col)) in m.iter() {
        result[row] += val * v[col];
    }
    result
}

#[test]
fn flrw_expansion_is_stable_and_flux_conserving() {
    let (complex, coords) = CartesianMeshInfo::new_unit(3, 2).compute_coord_complex();
    let base = coords.to_edge_lengths(&complex);
    // Gentle expansion a(t) = 1 + 0.1 t.
    let driver = FlrwDriver::new(complex.clone(), base, Box::new(|t| 1.0 + 0.1 * t));

    let dt = 0.5 * cfl_dt(&driver.lengths_at(0.0));
    let mut evolver = MaxwellEvolver::new(&driver, dt);

    let d1 = coboundary_matrix(&complex, 1);
    let seed = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| ((i + 2) as f64).recip());
    let b0 = spmv(&d1, &seed);
    let e0 = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| 0.05 * (i as f64).sin());
    let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0));

    let mut max_resid = 0.0f64;
    let mut last_energy = f64::NAN;
    for step in 0..100 {
        evolver.step(&mut state, None);
        max_resid = max_resid.max(evolver.magnetic_gauss_residual(&state));
        let t = evolver.time();
        let lt = driver.lengths_at(t);
        let m1 = assemble_galmat(&complex, &lt, HodgeMassElmat::new(3, 1));
        let m2 = assemble_galmat(&complex, &lt, HodgeMassElmat::new(3, 2));
        let u = state.energy(&m1, &m2);
        assert!(u.is_finite(), "energy diverged at step {step}");
        last_energy = u;
    }
    // Structural law is exact even on the moving background.
    assert!(max_resid < 1e-9, "flux not conserved under expansion: {max_resid:e}");
    // Field stayed bounded (no instability) over the expansion.
    assert!(last_energy.is_finite() && last_energy >= 0.0);
}
