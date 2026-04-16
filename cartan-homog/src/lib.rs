//! Mean-field and full-field homogenisation of random media, generic over tensor order.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
pub mod tensor;
pub mod kelvin_mandel;
pub mod shapes;

pub use error::HomogError;
pub use tensor::{Order2, Order4, TensorOrder};
pub use shapes::{Shape, Sphere, Spheroid, PennyCrack, Ellipsoid, SphereNLayers, IntegrationOpts, UserInclusion};

// Modules that land in later phases:
//   Phase 4 rve, Phase 5 schemes, Phase 8 stochastic, Phase 9 fullfield
