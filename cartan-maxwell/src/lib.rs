//! Maxwell evolution on a prescribed evolving simplicial-Riemannian background.
//!
//! E is a 1-cochain (edges), B is a 2-cochain (faces). Faraday `dB/dt = -d1 E`
//! is metric-free and exact; all metric and motion live in the time-dependent
//! Hodge masses `M_1(t)`, `M_2(t)` supplied by a [`driver::MetricDriver`].
//!
//! Geometry is carried as squared edge lengths ([`MeshLengthsSq`]), the Regge
//! primitive: the per-cell metric is linear in them and indefinite signatures
//! stay representable.
//!
//! [`MeshLengthsSq`]: simplicial::geometry::metric::mesh::MeshLengthsSq
//!
//! ```
//! use cartan_maxwell::{cfl_dt, coboundary_matrix, FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver};
//! use derham::cochain::Cochain;
//! use simplicial::r#gen::cartesian::CartesianGrid;
//!
//! // The identical evolver runs on a 2D or 3D spatial complex.
//! for spatial_dim in [2usize, 3] {
//!     let (complex, coords) = CartesianGrid::new_unit(spatial_dim, 2).triangulate();
//!     let base = coords.to_edge_lengths_sq(&complex);
//!     let driver = FlrwDriver::static_metric(complex.clone(), base);
//!     let dt = cfl_dt(&driver.lengths_sq_at(0.0));
//!     let mut evolver = MaxwellEvolver::new(&driver, dt);
//!     let d1 = coboundary_matrix(&complex, 1);
//!     let seed = nalgebra::DVector::from_element(complex.nsimplices(1), 1.0);
//!     let b = Cochain::new(2, &d1 * &seed);
//!     let e = Cochain::new(1, nalgebra::DVector::zeros(complex.nsimplices(1)));
//!     let mut state = MaxwellState::new(e, b);
//!     evolver.step(&mut state, None);
//!     assert!(evolver.magnetic_gauss_residual(&state) < 1e-9);
//! }
//! ```

pub mod driver;
pub mod evolver;
pub mod state;

pub use driver::{FlrwDriver, MetricDriver};
pub use evolver::{cfl_dt, coboundary_matrix, MaxwellEvolver};
pub use state::MaxwellState;
