//! Error types for stochastic-analysis primitives.

use thiserror::Error;

/// Errors raised by cartan-stochastic operations.
#[derive(Debug, Error)]
pub enum StochasticError {
    /// A frame construction produced a basis of the wrong dimension.
    #[error("frame dim mismatch: expected {expected}, got {got}")]
    FrameDimMismatch {
        /// The expected dimension (intrinsic dim of the manifold).
        expected: usize,
        /// The observed dimension.
        got: usize,
    },

    /// A driving noise vector did not match the frame dimension.
    #[error("noise dim mismatch: frame has {frame_dim} vectors, got dW of length {noise_dim}")]
    NoiseDimMismatch {
        /// The frame's number of basis vectors.
        frame_dim: usize,
        /// The supplied noise vector length.
        noise_dim: usize,
    },

    /// Gram-Schmidt orthonormalisation failed because the candidate basis was
    /// numerically rank-deficient (encountered a vector of near-zero norm
    /// after subtracting prior projections).
    #[error("gram-schmidt failure at index {index}: residual norm {norm} below {threshold}")]
    GramSchmidtRankDeficient {
        /// The basis index at which orthonormalisation collapsed.
        index: usize,
        /// The observed residual norm.
        norm: f64,
        /// The threshold below which re-orthonormalisation is deemed to have failed.
        threshold: f64,
    },

    /// A cartan-core operation (typically `log` at the cut locus) failed
    /// inside a stochastic-analysis routine.
    #[error("cartan-core error: {0}")]
    Cartan(#[from] cartan_core::CartanError),
}
