//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.

pub mod simplex;
pub mod mesh;
pub mod quadrature;

use crate::linalg::{RowVector, Vector};

pub type Coord = Vector;
pub type TangentVector = Vector;
pub type CoTangentVector = RowVector;
pub type CoordRef<'a> = &'a nalgebra::DVectorView<'a, f64>;
