//! Intrinsic Regge metric geometry on simplicial complexes.
//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use crate::Dim;
use cartan_exterior::combo::factorialf;

pub mod metric;

/// Volume of the reference d-simplex (standard unit simplex in R^d).
pub fn refsimp_vol(dim: Dim) -> f64 {
    factorialf(dim).recip()
}
