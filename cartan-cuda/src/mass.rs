//! A Galerkin Hodge mass resident on the device.
//!
//! The operator data is uploaded once. A Krylov iteration applies the same
//! operator tens of times, so uploading it per application would dominate the
//! solve and measure the bus rather than the kernel.

use cuda_core::{DeviceBuffer, LaunchConfig};

use crate::device::Device;
use crate::error::CudaError;

/// `M_k` in element form, held in device memory.
pub struct DeviceHodgeMass {
    offsets: DeviceBuffer<u32>,
    entries: DeviceBuffer<u32>,
    dofs: DeviceBuffer<u32>,
    elmats: DeviceBuffer<f64>,
    nlocal: u32,
    ndofs: usize,
}

impl DeviceHodgeMass {
    /// Upload an operator. The four slices are exactly what `cartan-matfree`
    /// exposes: `GatherMap::offsets`, `GatherMap::entries`, `HostMass::dof_map`
    /// and `HostMass::elmats`.
    pub fn new(
        device: &Device,
        offsets: &[u32],
        entries: &[u32],
        dofs: &[u32],
        elmats: &[f64],
        nlocal: usize,
    ) -> Result<Self, CudaError> {
        let ndofs = offsets.len().checked_sub(1).ok_or(CudaError::Shape {
            what: "gather offsets",
            expected: 1,
            got: 0,
        })?;
        if nlocal == 0 || dofs.len() % nlocal != 0 {
            return Err(CudaError::Shape {
                what: "cell degree-of-freedom map",
                expected: dofs.len().next_multiple_of(nlocal.max(1)),
                got: dofs.len(),
            });
        }
        let ncells = dofs.len() / nlocal;
        if elmats.len() != ncells * nlocal * nlocal {
            return Err(CudaError::Shape {
                what: "element matrices",
                expected: ncells * nlocal * nlocal,
                got: elmats.len(),
            });
        }

        let stream = device.stream();
        Ok(Self {
            offsets: DeviceBuffer::from_host(stream, offsets)?,
            entries: DeviceBuffer::from_host(stream, entries)?,
            dofs: DeviceBuffer::from_host(stream, dofs)?,
            elmats: DeviceBuffer::from_host(stream, elmats)?,
            nlocal: nlocal as u32,
            ndofs,
        })
    }

    /// Degrees of freedom the operator acts on.
    pub fn ndofs(&self) -> usize {
        self.ndofs
    }

    /// `y = M x`, uploading `x` and downloading the result.
    pub fn apply(&self, device: &Device, x: &[f64]) -> Result<Vec<f64>, CudaError> {
        self.apply_repeated(device, x, 1)
    }

    /// Apply the operator `reps` times to the same input, downloading once.
    ///
    /// Every repetition writes the same answer, so the result is independent of
    /// `reps`. The point is timing: it separates the kernel from the transfer
    /// that a host-facing call cannot avoid.
    pub fn apply_repeated(
        &self,
        device: &Device,
        x: &[f64],
        reps: usize,
    ) -> Result<Vec<f64>, CudaError> {
        if x.len() != self.ndofs {
            return Err(CudaError::Shape {
                what: "input vector",
                expected: self.ndofs,
                got: x.len(),
            });
        }
        let stream = device.stream();
        let x_dev = DeviceBuffer::from_host(stream, x)?;
        let mut out = DeviceBuffer::<f64>::zeroed(stream, self.ndofs)?;

        for _ in 0..reps {
            // SAFETY: the launch is sized to `ndofs`, and the kernel returns
            // early past that bound, so `offsets[i + 1]` stays inside the
            // `ndofs + 1` allocation. `entries` is indexed over the run the
            // offsets delimit, `dofs` and `elmats` were length-checked against
            // `nlocal` in the constructor, and `x` is indexed by degree of
            // freedom, all of which are below `ndofs`.
            unsafe {
                device.module().hodge_mass_apply(
                    stream,
                    LaunchConfig::for_num_elems(self.ndofs as u32),
                    &self.offsets,
                    &self.entries,
                    &self.dofs,
                    &self.elmats,
                    &x_dev,
                    self.nlocal,
                    self.ndofs as u32,
                    &mut out,
                )?;
            }
        }

        Ok(out.to_host_vec(stream)?)
    }
}
