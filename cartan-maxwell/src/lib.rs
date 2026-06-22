//! Maxwell evolution on a prescribed evolving simplicial-Riemannian background.
//!
//! E is a 1-cochain (edges), B is a 2-cochain (faces). Faraday `dB/dt = -d1 E`
//! is metric-free and exact; all metric and motion live in the time-dependent
//! Hodge masses `M_1(t)`, `M_2(t)` supplied by a [`driver::MetricDriver`].
//!
//! ```
//! use cartan_feec::cochain::Cochain;
//! use cartan_maxwell::{cfl_dt, coboundary_matrix, FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver};
//! use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;
//!
//! // The identical evolver runs on a 2D or 3D spatial complex.
//! for spatial_dim in [2usize, 3] {
//!     let (complex, coords) = CartesianMeshInfo::new_unit(spatial_dim, 2).compute_coord_complex();
//!     let base = coords.to_edge_lengths(&complex);
//!     let driver = FlrwDriver::static_metric(complex.clone(), base);
//!     let dt = cfl_dt(&driver.lengths_at(0.0));
//!     let mut evolver = MaxwellEvolver::new(&driver, dt);
//!     let d1 = coboundary_matrix(&complex, 1);
//!     let seed = nalgebra::DVector::from_element(complex.nsimplices(1), 1.0);
//!     // Compute d1 * seed via manual sparse-vector multiply (sprs does not impl Mul<DVector>).
//!     let mut b_coeffs = nalgebra::DVector::zeros(d1.rows());
//!     for (&val, (row, col)) in d1.iter() { b_coeffs[row] += val * seed[col]; }
//!     let b = Cochain::new(2, b_coeffs);
//!     let e = Cochain::new(1, nalgebra::DVector::zeros(complex.nsimplices(1)));
//!     let mut state = MaxwellState::new(e, b);
//!     evolver.step(&mut state, None);
//!     assert!(evolver.magnetic_gauss_residual(&state) < 1e-9);
//! }
//! ```

pub mod driver;
pub mod state;
pub mod evolver;

pub use driver::{FlrwDriver, MetricDriver};
pub use evolver::{cfl_dt, coboundary_matrix, MaxwellEvolver};
pub use state::MaxwellState;
