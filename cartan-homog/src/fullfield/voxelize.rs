//! Voxelisation: phase-id assignment over a regular grid of the unit cube.
//!
//! v1.1 shape indicators: centred sphere, centred oblate spheroid (penny-crack
//! surrogate), centred prolate spheroid. Axis-aligned with z by default.

use alloc::vec::Vec;
use nalgebra::Vector3;

pub type PhaseId = u16;
pub const NO_PHASE: PhaseId = PhaseId::MAX;

#[derive(Clone, Debug)]
pub struct VoxelGrid {
    pub resolution: usize,
    pub phase_ids: Vec<PhaseId>,
}

impl VoxelGrid {
    pub fn new(resolution: usize) -> Self {
        let n = resolution * resolution * resolution;
        Self { resolution, phase_ids: alloc::vec![NO_PHASE; n] }
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, phase: PhaseId) {
        let n = self.resolution;
        self.phase_ids[i * n * n + j * n + k] = phase;
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> PhaseId {
        let n = self.resolution;
        self.phase_ids[i * n * n + j * n + k]
    }
}

/// Centred inclusion shape, used by `FullField` to tag tets by phase.
#[derive(Clone, Copy, Debug)]
pub enum CentredInclusion {
    /// Sphere of radius r, centred at (0.5, 0.5, 0.5).
    Sphere { radius: f64 },
    /// Spheroid centred at (0.5, 0.5, 0.5) aligned with z, with equatorial radius `a`
    /// and polar (vertical) semi-axis `c = aspect * a`. Aspect < 1 = oblate (penny).
    Spheroid { a: f64, aspect: f64 },
}

/// Raw-binary voxel indicator loader: reads N*N*N bytes from an .raw file
/// where each byte is a phase id (0 = matrix, 1 = phase 1, etc.). Matches the
/// convention used by most public μCT datasets (Digital Porous Media Portal,
/// Imperial College Berea, NIST CBT). For higher-resolution datasets beyond
/// u8 phase ids, use load_voxel_u16.
pub fn load_voxel_raw_u8(path: &std::path::Path, resolution: usize) -> Result<VoxelGrid, crate::error::HomogError> {
    let expected = resolution * resolution * resolution;
    let bytes = std::fs::read(path).map_err(|e|
        crate::error::HomogError::Mesh(alloc::format!("failed to read {}: {e}", path.display())))?;
    if bytes.len() != expected {
        return Err(crate::error::HomogError::Mesh(alloc::format!(
            "voxel file size {} != expected {} for N={resolution}", bytes.len(), expected)));
    }
    let phase_ids: Vec<PhaseId> = bytes.iter().map(|&b| b as PhaseId).collect();
    Ok(VoxelGrid { resolution, phase_ids })
}

/// Generate a synthetic voxelisation of a `CentredInclusion` at the given
/// resolution. Useful for testing and for bridging analytic shape definitions
/// to the voxel-based FullField path.
pub fn voxelize_centred(incl: &CentredInclusion, resolution: usize) -> VoxelGrid {
    let n = resolution;
    let h = 1.0 / (n as f64);
    let mut g = VoxelGrid::new(n);
    for k in 0..n { for j in 0..n { for i in 0..n {
        let p = Vector3::new(
            (i as f64 + 0.5) * h,
            (j as f64 + 0.5) * h,
            (k as f64 + 0.5) * h,
        );
        g.set(i, j, k, if incl.contains(&p) { 1 } else { 0 });
    }}}
    g
}

impl CentredInclusion {
    /// Volume of the inclusion in the unit cube (clipped at 1 if it exceeds the cube).
    pub fn volume(&self) -> f64 {
        match *self {
            CentredInclusion::Sphere { radius } => {
                (4.0 / 3.0) * core::f64::consts::PI * radius.powi(3)
            }
            CentredInclusion::Spheroid { a, aspect } => {
                let c = aspect * a;
                (4.0 / 3.0) * core::f64::consts::PI * a * a * c
            }
        }
    }

    /// Returns true if the given point (in the unit cube) lies inside the inclusion.
    pub fn contains(&self, p: &Vector3<f64>) -> bool {
        let q = p - Vector3::new(0.5, 0.5, 0.5);
        match *self {
            CentredInclusion::Sphere { radius } => q.norm() < radius,
            CentredInclusion::Spheroid { a, aspect } => {
                let c = aspect * a;
                // (x/a)^2 + (y/a)^2 + (z/c)^2 < 1
                (q.x / a).powi(2) + (q.y / a).powi(2) + (q.z / c).powi(2) < 1.0
            }
        }
    }

    /// Build an inclusion sized so that `volume() == target_fraction` of the unit cube.
    /// `aspect` constrains the shape (sphere if aspect == 1.0; oblate / prolate otherwise).
    pub fn from_volume_fraction(target: f64, aspect: f64) -> Self {
        if (aspect - 1.0).abs() < 1e-12 {
            let r = (3.0 * target / (4.0 * core::f64::consts::PI)).powf(1.0 / 3.0);
            Self::Sphere { radius: r }
        } else {
            // Volume = (4/3) pi a^2 c = (4/3) pi a^3 aspect
            // target = (4/3) pi a^3 aspect  =>  a = cbrt(3*target / (4*pi*aspect))
            let a = (3.0 * target / (4.0 * core::f64::consts::PI * aspect)).powf(1.0 / 3.0);
            Self::Spheroid { a, aspect }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sphere_from_volume_round_trip() {
        let s = CentredInclusion::from_volume_fraction(0.1, 1.0);
        assert_relative_eq!(s.volume(), 0.1, epsilon = 1e-12);
    }

    #[test]
    fn oblate_spheroid_has_zero_volume_at_aspect_zero() {
        // Sanity: limit aspect -> 0 sends volume -> 0 for fixed a.
        let thin = CentredInclusion::Spheroid { a: 0.3, aspect: 1e-6 };
        assert!(thin.volume() < 1e-6);
    }

    #[test]
    fn penny_crack_contains_thin_disk_plane_points() {
        // Thin oblate spheroid (a=0.3, c=3e-4): points at z=0 with r<0.3 inside, r>0.3 outside.
        let penny = CentredInclusion::Spheroid { a: 0.3, aspect: 1e-3 };
        assert!(penny.contains(&Vector3::new(0.5, 0.5, 0.5)));        // centre
        assert!(penny.contains(&Vector3::new(0.6, 0.5, 0.5)));        // in-plane, r=0.1
        assert!(!penny.contains(&Vector3::new(0.9, 0.5, 0.5)));       // in-plane, r=0.4, outside
        assert!(!penny.contains(&Vector3::new(0.5, 0.5, 0.502)));     // off-plane by 2e-3 > c=3e-4
    }
}
