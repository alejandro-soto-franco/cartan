//! Mean-field and full-field homogenisation of random media, generic over tensor order.
//!
//! See `docs/superpowers/specs/2026-04-16-cartan-echoes-validation.md` for design.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;

pub use error::HomogError;

// Modules below are declared as they land in Phases 2-10. Task boundary markers:
//   Phase 2 (traits):   tensor, kelvin_mandel
//   Phase 3 (shapes):   shapes
//   Phase 4 (rve):      rve
//   Phase 5 (schemes):  schemes
//   Phase 8 (γ):        stochastic (feature = "stochastic")
//   Phase 9 (β):        fullfield  (feature = "full-field")
