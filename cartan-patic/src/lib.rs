//! p-atic order parameters on Riemannian manifolds.
//!
//! A p-atic order parameter on `(M^3, g)` is a section of
//! `Spin(M) x_{SU(2)} (SU(2)/H^)`, with `H^` the binary lift of the molecular
//! point group `H` in `SO(3)`. Since `SU(2)` is simply connected,
//! `pi_1(SO(3)/H) = H^`, so defect charges live in the lift and are
//! non-abelian beyond the cyclic case.
//!
//! The state is a rotor with amplitudes; the energy is a polynomial in the
//! `H^`-invariant tensors, which are single-valued functions of the rotor.

pub mod active;
pub mod advect;
pub mod boundary;
pub mod complex3;
pub mod defect;
pub mod energy;
pub mod error;
pub mod fiber;
pub mod flow;
pub mod geometry;
pub mod group;
pub mod invariant;
pub mod knot;
pub mod quasipotential;
pub mod selection;
pub mod simulation;
pub mod spin;
pub mod stokes;

pub use active::active_force;
pub use boundary::{Boundary, rotor_taking_z_to};
pub use complex3::Complex3;
pub use defect::{DefectField, PiercedFace};
pub use energy::{Energy, State};
pub use error::PaticError;
pub use fiber::{PaticElement, PaticFiber};
pub use flow::{DiscreteGradientFlow, ExplicitFlow};
pub use geometry::{Geometry3, TetData, mass0, mass1, mass2};
pub use group::{
    AxialApolar, AxialPolar, BinaryIcosahedral, BinaryOctahedral, BinaryTetrahedral, Cyclic,
    Dicyclic, GroupTable, PointGroupKind, SymmetryGroup, closure,
};
pub use invariant::{InvariantBasis, harmonic_basis, monomials, separating_degree};
