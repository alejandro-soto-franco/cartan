//! Mean-field and full-field homogenisation of random media, generic over tensor order.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
pub mod tensor;
pub mod kelvin_mandel;
pub mod shapes;
pub mod rve;
pub mod schemes;

#[cfg(feature = "stochastic")]
pub mod stochastic;

pub use error::HomogError;
pub use tensor::{Order2, Order4, TensorOrder};
pub use shapes::{Shape, Sphere, Spheroid, PennyCrack, Ellipsoid, SphereNLayers, IntegrationOpts, UserInclusion};
pub use rve::{Phase, RefMedium, Rve};
pub use schemes::{
    Scheme, SchemeOpts, Effective,
    VoigtBound, ReussBound, Dilute, DiluteStress,
    MoriTanaka, SelfConsistent, AsymmetricSc,
    Maxwell, PonteCastanedaWillis, Differential,
};
