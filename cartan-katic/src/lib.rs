//! k-atic order parameters on Riemannian manifolds.
//!
//! A k-atic order parameter on `(M^3, g)` is a section of
//! `Spin(M) x_{SU(2)} (SU(2)/H^)`, with `H^` the binary lift of the molecular
//! point group `H` in `SO(3)`. Since `SU(2)` is simply connected,
//! `pi_1(SO(3)/H) = H^`, so defect charges live in the lift and are
//! non-abelian beyond the cyclic case.
//!
//! The state is a rotor with amplitudes; the energy is a polynomial in the
//! `H^`-invariant tensors, which are single-valued functions of the rotor.

pub mod error;
pub mod fiber;
pub mod group;

pub use error::KaticError;
pub use fiber::{KaticElement, KaticFiber};
pub use group::{
    AxialApolar, AxialPolar, BinaryIcosahedral, BinaryOctahedral, BinaryTetrahedral, Cyclic,
    Dicyclic, GroupTable, PointGroupKind, SymmetryGroup, closure,
};
