//! Error type for cartan-cuda. All fallible public APIs return
//! `Result<_, CudaError>`.
//!
//! Written out by hand rather than derived. Host-side code in this crate is
//! compiled by the CUDA backend, so every dependency it carries is one more
//! thing that has to survive that path, and a derive earns nothing here.

use core::fmt;

use cuda_core::DriverError;
use cuda_host::EmbeddedModuleError;

#[derive(Debug)]
pub enum CudaError {
    /// A driver call failed. Error 803 is the usual one: the loaded kernel
    /// module and the userspace driver are different versions, which a reboot
    /// after a driver upgrade fixes.
    Driver(DriverError),

    /// The PTX compiled from this crate could not be loaded onto the device.
    Module(EmbeddedModuleError),

    /// A batch was not the shape the kernel indexes.
    Shape {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    /// `dim` was zero, which would divide by zero in the per-element kernels.
    ZeroDim,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver call failed: {e}"),
            Self::Module(e) => write!(f, "loading the compiled module failed: {e}"),
            Self::Shape {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected {expected} elements, got {got}"),
            Self::ZeroDim => write!(f, "dim must be at least 1"),
        }
    }
}

impl std::error::Error for CudaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(e) => Some(e),
            Self::Module(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DriverError> for CudaError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<EmbeddedModuleError> for CudaError {
    fn from(e: EmbeddedModuleError) -> Self {
        Self::Module(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_error_displays() {
        let e = CudaError::Shape {
            what: "tangent batch",
            expected: 12,
            got: 9,
        };
        assert_eq!(format!("{e}"), "tangent batch: expected 12 elements, got 9");
    }

    #[test]
    fn zero_dim_displays() {
        assert_eq!(format!("{}", CudaError::ZeroDim), "dim must be at least 1");
    }
}
