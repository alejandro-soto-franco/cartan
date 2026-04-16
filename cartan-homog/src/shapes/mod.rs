//! Shape trait and concrete inclusion kinds.
//!
//! Each `Shape<O>` provides the Hill polarization tensor `P(C_ref, shape)` and the
//! default dilute concentration `A_dil = (I + P : (C_phase - C_ref))^{-1}`.

use crate::{error::HomogError, tensor::TensorOrder};
use alloc::sync::Arc;

pub mod opts;
pub mod sphere;
pub mod spheroid;
pub mod crack;
pub mod ellipsoid;
pub mod sphere_nlayers;

pub use opts::IntegrationOpts;
pub use sphere::Sphere;
pub use spheroid::Spheroid;
pub use crack::PennyCrack;
pub use ellipsoid::{carlson_rd, Ellipsoid};
pub use sphere_nlayers::SphereNLayers;

pub trait Shape<O: TensorOrder>: Send + Sync + core::fmt::Debug {
    fn hill(&self, c_ref: &O::KmMatrix, opts: &IntegrationOpts)
        -> Result<O::KmMatrix, HomogError>;

    fn concentration_dilute(
        &self,
        c_ref: &O::KmMatrix,
        c_phase: &O::KmMatrix,
        opts: &IntegrationOpts,
    ) -> Result<O::KmMatrix, HomogError> {
        let p = self.hill(c_ref, opts)?;
        let delta_c = O::sub(c_phase, c_ref);
        let p_dc = O::mat_mul(&p, &delta_c);
        let arg = O::add(&O::identity(), &p_dc);
        O::inverse(&arg)
    }
}

/// User-supplied shape via trait object; first-class alongside built-ins.
pub type UserInclusion<O> = Arc<dyn Shape<O>>;
