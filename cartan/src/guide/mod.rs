//! Worked chapters. Every code block is a doctest, run by `cargo test --doc`,
//! so nothing here can drift from the API.
//!
//! Read them in order for a tour of the three regimes: geometry at points,
//! at fields, and along paths.
//!
//! | chapter | regime | needs |
//! |---|---|---|
//! | [`getting_started`] | points | nothing |
//! | [`manifolds`] | points | nothing |
//! | [`optimisation`] | points | nothing |
//! | `fields` | fields | `dec` |
//! | `bundles` | fields | `dec` |
//! | `stochastic` | paths | `stochastic` |
//! | `homogenisation` | fields | `homog` |
//! | `interop` | export | `io` |
//!
//! The gated chapters are listed without links because they are absent from a
//! build that does not enable their feature. docs.rs enables all of them.

pub mod getting_started;
pub mod manifolds;
pub mod optimisation;

#[cfg(feature = "dec")]
#[cfg_attr(docsrs, doc(cfg(feature = "dec")))]
pub mod fields;

#[cfg(feature = "dec")]
#[cfg_attr(docsrs, doc(cfg(feature = "dec")))]
pub mod bundles;

#[cfg(feature = "stochastic")]
#[cfg_attr(docsrs, doc(cfg(feature = "stochastic")))]
pub mod stochastic;

#[cfg(feature = "homog")]
#[cfg_attr(docsrs, doc(cfg(feature = "homog")))]
pub mod homogenisation;

#[cfg(feature = "io")]
#[cfg_attr(docsrs, doc(cfg(feature = "io")))]
pub mod interop;
