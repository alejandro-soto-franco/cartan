//! Homogenisation schemes. Each concrete scheme is a zero-sized type that
//! implements `Scheme<O>` (generic over tensor order) where possible.

use crate::{error::HomogError, rve::Rve, shapes::IntegrationOpts, tensor::TensorOrder};
use alloc::vec::Vec;

pub mod bounds;
pub mod dilute;
pub mod mori_tanaka;
pub mod self_consistent;
pub mod asymmetric_sc;
pub mod maxwell_pcw;
pub mod differential;

pub use bounds::{ReussBound, VoigtBound};
pub use dilute::{Dilute, DiluteStress};
pub use mori_tanaka::MoriTanaka;
pub use self_consistent::SelfConsistent;
pub use asymmetric_sc::AsymmetricSc;
pub use maxwell_pcw::{Maxwell, PonteCastanedaWillis};
pub use differential::{Differential, DifferentialCompliance};

#[derive(Clone, Debug, PartialEq)]
pub struct SchemeOpts {
    pub max_iter: usize,
    pub rel_tol: f64,
    pub damping: f64,
    pub store_concentration: bool,
    pub spd_iteration: bool,
    pub integration: IntegrationOpts,
}

impl Default for SchemeOpts {
    fn default() -> Self {
        Self {
            max_iter: 100, rel_tol: 1e-10, damping: 1.0,
            store_concentration: false, spd_iteration: true,
            integration: IntegrationOpts::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Effective<O: TensorOrder> {
    pub tensor: O::KmMatrix,
    pub concentration: Option<Vec<O::KmMatrix>>,
    pub iterations: Option<usize>,
    pub residual: Option<f64>,
}

pub trait Scheme<O: TensorOrder> {
    fn homogenize(&self, rve: &Rve<O>, opts: &SchemeOpts) -> Result<Effective<O>, HomogError>;
}
