//! # Homogenisation
//!
//! Given a microstructure of several phases, homogenisation produces the
//! effective property a macroscopic model should use. The schemes here are
//! SPD-manifold-native: effective tensors live in the positive-definite cone,
//! and iterating along its geodesics keeps every intermediate physically
//! admissible rather than merely converging to something admissible.
//!
//! ```
//! use cartan::homog::{Rve, Phase, Order2, TensorOrder, Sphere, MoriTanaka, Scheme, SchemeOpts};
//! use std::sync::Arc;
//!
//! let mut rve = Rve::<Order2>::new();
//! rve.add_phase(Phase {
//!     name: "MATRIX".into(),
//!     shape: Arc::new(Sphere),
//!     property: Order2::scalar(1.0),
//!     fraction: 0.8,
//! });
//! rve.add_phase(Phase {
//!     name: "INCLUSION".into(),
//!     shape: Arc::new(Sphere),
//!     property: Order2::scalar(5.0),
//!     fraction: 0.2,
//! });
//! rve.set_matrix("MATRIX");
//!
//! let eff = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
//!
//! // The effective conductivity lies between the two phase values, as any
//! // admissible scheme must place it.
//! let k = eff.tensor[(0, 0)];
//! assert!(k > 1.0 && k < 5.0);
//! ```
//!
//! ## Choosing a scheme
//!
//! `VoigtBound` and `ReussBound` are the arithmetic and harmonic averages: the
//! widest bracket, valid for any microstructure. `MoriTanaka` assumes
//! inclusions embedded in a distinguished matrix and is the usual default.
//! `SelfConsistent` treats every phase symmetrically, which suits polycrystals
//! and materials near a percolation threshold where no phase is the matrix.
//!
//! All the interaction-corrected schemes respect the Hashin-Shtrikman bounds,
//! which the test suite checks rather than assumes.
//!
//! ## On embedded targets
//!
//! The mean-field schemes run at the `alloc` tier, with one difference:
//! `SelfConsistent` damps its fixed-point iteration linearly rather than along
//! SPD geodesics, because the geodesic step needs an eigen decomposition that
//! requires std. It reaches the same fixed point more slowly. The `full-field`
//! cell-problem solver needs std outright.
