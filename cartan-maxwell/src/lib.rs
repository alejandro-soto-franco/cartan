//! Maxwell evolution on a prescribed evolving simplicial-Riemannian background.
//!
//! E is a 1-cochain (edges), B is a 2-cochain (faces). Faraday `dB/dt = -d1 E`
//! is metric-free and exact; all metric and motion live in the time-dependent
//! Hodge masses `M_1(t)`, `M_2(t)` supplied by a [`driver::MetricDriver`].

pub mod driver;
pub mod state;
pub mod evolver;

pub use driver::{FlrwDriver, MetricDriver};
pub use evolver::{cfl_dt, coboundary_matrix, MaxwellEvolver};
pub use state::MaxwellState;
