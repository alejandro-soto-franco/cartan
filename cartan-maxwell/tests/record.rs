use nalgebra_sparse::CsrMatrix;
use formoniq::whitney_complex::WhitneyComplex;
use derham::cochain::Cochain;
use cartan_io::run::RunWriter;
use cartan_maxwell::{cfl_dt, coboundary_matrix, FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver};
use simplicial::r#gen::cartesian::CartesianGrid;
use simplicial::geometry::coord::mesh::MeshCoords;

#[test]
fn flrw_run_produces_valid_run_dir() {
    let (complex, coords0) = CartesianGrid::new_unit(2, 3).triangulate();
    let base = coords0.to_edge_lengths_sq(&complex);
    let driver = FlrwDriver::new(complex.clone(), base, Box::new(|t| 1.0 + 0.1 * t));
    let dt = cfl_dt(&driver.lengths_sq_at(0.0));
    let mut evolver = MaxwellEvolver::new(&driver, dt);

    let d1 = coboundary_matrix(&complex, 1);
    let seed = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| ((i + 1) as f64).recip());
    let b0_coeffs = &d1 * &seed;
    let e0 = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| 0.05 * (i as f64).cos());
    let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0_coeffs));

    let dir = std::env::temp_dir().join("cartan_maxwell_record_test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut run = RunWriter::new(&dir).unwrap();
    let stride = 2;
    let mut recorded = 0;
    for k in 0..6 {
        if k % stride == 0 {
            let t = evolver.time();
            let a = driver.scale_factor(t);
            let coords_t = MeshCoords::new(coords0.matrix() * a);
            let lt = driver.lengths_sq_at(t);
            let m1 = CsrMatrix::from(&WhitneyComplex::new(&complex, &lt).mass(1));
            let m2 = CsrMatrix::from(&WhitneyComplex::new(&complex, &lt).mass(2));
            let energy = state.energy(&m1, &m2);
            let resid = evolver.magnetic_gauss_residual(&state);
            run.push_frame(t, &complex, &coords_t, &state.b, &state.e, energy, resid).unwrap();
            recorded += 1;
        }
        evolver.step(&mut state, None);
    }
    run.finish().unwrap();

    assert!(dir.join("frames.pvd").exists());
    assert!(dir.join("diagnostics.csv").exists());
    assert!(dir.join("blender/motion.mdd").exists());
    let csv = std::fs::read_to_string(dir.join("diagnostics.csv")).unwrap();
    assert_eq!(csv.lines().count(), recorded + 1); // header + recorded rows
}
