use nalgebra_sparse::CsrMatrix;
use formoniq::whitney_complex::WhitneyComplex;
use derham::cochain::Cochain;
use cartan_maxwell::{cfl_dt, coboundary_matrix, FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver};
use simplicial::r#gen::cartesian::CartesianGrid;

#[test]
fn static_cavity_energy_has_no_secular_growth() {
    let (complex, coords) = CartesianGrid::new_unit(2, 6).triangulate();
    let base = coords.to_edge_lengths_sq(&complex);
    let driver = FlrwDriver::static_metric(complex.clone(), base);
    let l0 = driver.lengths_sq_at(0.0);
    let m1 = CsrMatrix::from(&WhitneyComplex::new(&complex, &l0).mass(1));
    let m2 = CsrMatrix::from(&WhitneyComplex::new(&complex, &l0).mass(2));

    let dt = 0.5 * cfl_dt(&l0);
    let mut evolver = MaxwellEvolver::new(&driver, dt);

    let d1 = coboundary_matrix(&complex, 1);
    let seed = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| ((i + 3) as f64).recip());
    let b0 = &d1 * &seed;
    let e0 = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| 0.05 * (i as f64).cos());
    let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0));

    let u0 = state.energy(&m1, &m2);
    let mut umin = u0;
    let mut umax = u0;
    for _ in 0..400 {
        evolver.step(&mut state, None);
        let u = state.energy(&m1, &m2);
        umin = umin.min(u);
        umax = umax.max(u);
    }
    // Bounded oscillation: no run-away. Band stays within 20% of the initial energy.
    assert!(umax / u0 < 1.20, "energy grew: umax/u0 = {}", umax / u0);
    assert!(umin / u0 > 0.80, "energy decayed: umin/u0 = {}", umin / u0);
}
