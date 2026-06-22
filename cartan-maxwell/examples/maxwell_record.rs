//! Record a short FLRW Maxwell run to a run_dir consumable by cartan-viz.
//!
//! Usage: maxwell_record [out_dir]
//!
//! Defaults to `out/maxwell_run` if no argument is given.

use cartan_feec::assemble::assemble_galmat;
use cartan_feec::cochain::Cochain;
use cartan_feec::operators::HodgeMassElmat;
use cartan_io::run::RunWriter;
use cartan_maxwell::{cfl_dt, coboundary_matrix, FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver};
use cartan_simplicial::geometry::coord::mesh::MeshCoords;
use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

/// Return a scaled copy of `coords0`: all coordinates multiplied by `a`.
fn scaled_coords(
    coords0: &MeshCoords,
    a: f64,
) -> MeshCoords {
    MeshCoords::new(coords0.matrix() * a)
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out/maxwell_run".to_string());
    let out_path = std::path::Path::new(&out_dir);

    // Build a 2D unit-square mesh with 4 subdivisions per side.
    let n = 4;
    let (complex, coords0) = CartesianMeshInfo::new_unit(2, n).compute_coord_complex();
    let base = coords0.to_edge_lengths(&complex);

    // Gentle cosmological expansion: a(t) = 1 + 0.1 t.
    let driver = FlrwDriver::new(complex.clone(), base, Box::new(|t| 1.0 + 0.1 * t));
    let dt = cfl_dt(&driver.lengths_at(0.0));
    let mut evolver = MaxwellEvolver::new(&driver, dt);

    // Seed a closed B = d1 * seed (so d2 B = 0 by exactness).
    let d1 = coboundary_matrix(&complex, 1);
    let seed = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| ((i + 1) as f64).recip());
    // sprs does not impl Mul<DVector>; use manual sparse-vector multiply.
    let mut b0_coeffs = nalgebra::DVector::zeros(d1.rows());
    for (&val, (row, col)) in d1.iter() {
        b0_coeffs[row] += val * seed[col];
    }

    // Smooth initial electric field.
    let e0 = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| 0.05 * (i as f64).cos());
    let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0_coeffs));

    let mut run = RunWriter::new(out_path).expect("failed to create run directory");

    let nsteps = 40;
    let stride = 4;

    for k in 0..nsteps {
        if k % stride == 0 {
            let t = evolver.time();
            let a = driver.scale_factor(t);
            let coords_t = scaled_coords(&coords0, a);
            let lt = driver.lengths_at(t);
            let m1 = assemble_galmat(&complex, &lt, HodgeMassElmat::new(2, 1));
            let m2 = assemble_galmat(&complex, &lt, HodgeMassElmat::new(2, 2));
            let energy = state.energy(&m1, &m2);
            let resid = evolver.magnetic_gauss_residual(&state);
            run.push_frame(t, &complex, &coords_t, &state.b, &state.e, energy, resid)
                .expect("failed to write frame");
        }
        evolver.step(&mut state, None);
    }

    run.finish().expect("failed to finalize run directory");
    println!("wrote run_dir: {}", out_path.display());
}
