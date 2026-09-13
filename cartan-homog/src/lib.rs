//! Mean-field and full-field homogenisation of random media, generic over tensor order.
//!
//! # Feature tiers
//!
//! The crate builds on `no_std` targets with `default-features = false,
//! features = ["alloc"]`. One numerical behaviour differs at that tier:
//! `SelfConsistent` damps its fixed-point iteration linearly rather than along
//! SPD geodesics, because the geodesic step needs an eigen decomposition that
//! requires std. It converges to the same fixed point, more slowly.
//!
//! `full-field`, `stochastic` and `gpu-fft` all require std.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
mod float;
pub mod kelvin_mandel;
pub mod rve;
pub mod schemes;
pub mod shapes;
pub mod tensor;

#[cfg(feature = "stochastic")]
pub mod stochastic;

#[cfg(feature = "full-field")]
pub mod fullfield;

pub use error::HomogError;
pub use rve::{Phase, RefMedium, Rve};
pub use schemes::{
    AsymmetricSc, Differential, DifferentialCompliance, Dilute, DiluteStress, Effective, Maxwell,
    MoriTanaka, PonteCastanedaWillis, ReussBound, Scheme, SchemeOpts, SelfConsistent, VoigtBound,
};
pub use shapes::{
    Ellipsoid, IntegrationOpts, PennyCrack, Shape, Sphere, SphereNLayers, Spheroid, UserInclusion,
};
pub use tensor::{Order2, Order4, TensorOrder};
