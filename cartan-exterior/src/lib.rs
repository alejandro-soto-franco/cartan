//! Dimension-generic numerical exterior algebra.
//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.

pub mod combo;
pub mod gramian;

pub use gramian::Gramian;

/// Intrinsic dimension of the linear space underlying the exterior algebra.
pub type Dim = usize;
/// Exterior grade (form degree) k in Λᵏ.
pub type ExteriorGrade = usize;
