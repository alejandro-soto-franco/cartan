//! # cartan-stochastic
//!
//! Stochastic analysis primitives on Riemannian manifolds.
//!
//! This crate provides the foundation that downstream crates (`hsu`, `bismut`,
//! `elworthy`, `malliavin`) need to do probability, SDE integration, and
//! pathwise-derivative computation on manifolds — independent of the
//! underlying manifold type, as long as it implements the `cartan-core`
//! `Manifold + ParallelTransport + Retraction` trait stack.
//!
//! The architectural purpose is to **prevent primitive duplication** across
//! the Hsu / Bismut / Elworthy / Malliavin stack. Horizontal lift, orthonormal
//! frame bundle, Stratonovich development, and stochastic-development by
//! Euler-Maruyama are all defined once here.
//!
//! ## Concepts
//!
//! **Orthonormal frame bundle `O(M)`**: the total space of orthonormal bases
//! of the tangent spaces of `M`. A point in `O(M)` is a pair `(p, r)` where
//! `p ∈ M` and `r = (e_1, …, e_n)` is an orthonormal basis of `T_p M`.
//!
//! **Horizontal lift**: given a tangent vector `u ∈ T_p M`, a curve in `O(M)`
//! whose velocity projects to `u` and whose frame evolves by parallel transport.
//! Implemented as a right action of `R^n` on the frame bundle via
//! `(p, r) · ξ = (γ(1), r̃)` where `γ` is the exponential of `Σ ξ_i e_i` and
//! `r̃` is the parallel transport of `r` along `γ`.
//!
//! **Stochastic development (Eells-Elworthy-Malliavin)**: solve the SDE on
//! `O(M)` driven by Euclidean Brownian motion `W_t`, with Stratonovich
//! differential `∂_t (p, r) = H_i(p, r) ∘ dW^i_t` where `H_i` is the
//! horizontal lift of the `i`-th frame vector. The projection to `M` is
//! Brownian motion on `M` with the Laplace-Beltrami generator.
//!
//! ## Minimum trait requirements
//!
//! Any manifold implementing `Manifold + ParallelTransport + Retraction` can
//! host a stochastic development. Exact exponentials are not required;
//! retraction suffices at the cost of higher-order discretisation error.
//!
//! ## References
//!
//! - Hsu, Elton P. *Stochastic Analysis on Manifolds.* AMS, 2002. Chapter 2
//!   (horizontal lift and anti-development), Chapter 3 (Brownian motion
//!   via orthonormal frame bundle).
//! - Eells, J. and Elworthy, K. D. *Wiener integration on certain manifolds.*
//!   Problems in Non-Linear Analysis, 1971.
//! - Elworthy, K. D. *Stochastic Differential Equations on Manifolds.*
//!   Cambridge LMS Lecture Notes 70, 1982.

#![deny(missing_docs)]

pub mod development;
pub mod error;
pub mod frame;
pub mod horizontal;
pub mod sde;
pub mod wishart;

pub use development::{stochastic_development, DevelopmentPath};
pub use error::StochasticError;
pub use frame::{random_frame_at, OrthonormalFrame};
pub use horizontal::horizontal_velocity;
pub use sde::{stratonovich_step, StratonovichDevelopment};
pub use wishart::wishart_step;
