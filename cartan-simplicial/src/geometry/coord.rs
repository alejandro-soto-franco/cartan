//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.

pub mod simplex;
pub mod mesh;

use crate::linalg::{RowVector, Vector};

pub type Coord = Vector;
pub type TangentVector = Vector;
pub type CoTangentVector = RowVector;
