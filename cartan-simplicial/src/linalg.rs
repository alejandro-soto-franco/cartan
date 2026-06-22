//! Dense linear-algebra type aliases used throughout the ported geometry code.
//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.

pub use nalgebra as na;

/// Dense column-major matrix of f64.
pub type Matrix = na::DMatrix<f64>;
/// Dense column vector of f64.
pub type Vector = na::DVector<f64>;
/// Dense row vector of f64.
pub type RowVector = na::RowDVector<f64>;
