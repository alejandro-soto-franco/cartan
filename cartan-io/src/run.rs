//! Run directory orchestrator tying vtu, pvd, csv, obj, and mdd together.
//!
//! `RunWriter` writes a complete `run_dir/` consumable by cartan-viz:
//! - `frame_XXXX.vtu` per recorded frame
//! - `frames.pvd` ParaView collection
//! - `diagnostics.csv` time series
//! - `blender/base.obj` static mesh
//! - `blender/motion.mdd` vertex animation cache

use std::io;
use std::path::{Path, PathBuf};

use derham::cochain::Cochain;
use simplicial::geometry::coord::mesh::MeshCoords;
use simplicial::topology::complex::Complex;

use crate::blender::{write_mdd, write_obj};
use crate::diagnostics::DiagnosticsCsv;
use crate::feec_fields::{cell_scalar_from_cochain, cell_vectors_from_cochain};
use crate::pvd::write_pvd;
use crate::vtu::write_vtu;

/// Orchestrates writing a complete run directory for a Maxwell simulation.
pub struct RunWriter {
    dir: PathBuf,
    pvd_entries: Vec<(f64, String)>,
    diag: DiagnosticsCsv,
    mdd_frames: Vec<Vec<[f32; 3]>>,
    frame_counter: usize,
    wrote_base_obj: bool,
}

impl RunWriter {
    /// Create a new RunWriter rooted at `dir`.
    ///
    /// Creates `dir` and `dir/blender` on disk, and opens the diagnostics CSV.
    pub fn new(dir: &Path) -> io::Result<RunWriter> {
        std::fs::create_dir_all(dir)?;
        std::fs::create_dir_all(dir.join("blender"))?;
        let diag =
            DiagnosticsCsv::new(&dir.join("diagnostics.csv"), &["time", "energy", "magnetic_flux_residual"])?;
        Ok(RunWriter {
            dir: dir.to_path_buf(),
            pvd_entries: Vec::new(),
            diag,
            mdd_frames: Vec::new(),
            frame_counter: 0,
            wrote_base_obj: false,
        })
    }

    /// Record one simulation frame.
    ///
    /// Writes `frame_{n:04}.vtu`, appends a diagnostics row, records an MDD
    /// frame from `coords`, and (on the first call) writes `blender/base.obj`.
    ///
    /// One frame is an irreducible bundle of geometry, the two field cochains,
    /// and two scalar diagnostics, so the wide argument list is intentional.
    #[allow(clippy::too_many_arguments)]
    pub fn push_frame(
        &mut self,
        time: f64,
        complex: &Complex,
        coords: &MeshCoords,
        b: &Cochain,
        e: &Cochain,
        energy: f64,
        residual: f64,
    ) -> io::Result<()> {
        let n = self.frame_counter;
        let frame_name = format!("frame_{:04}.vtu", n);
        let frame_path = self.dir.join(&frame_name);

        let b_vals = cell_scalar_from_cochain(complex, coords, b);
        let e_vals = cell_vectors_from_cochain(complex, coords, e);

        write_vtu(
            &frame_path,
            complex,
            coords,
            &[],
            &[],
            &[("B", &b_vals)],
            &[("E", &e_vals)],
        )?;

        self.pvd_entries.push((time, frame_name));
        self.diag.push_row(&[time, energy, residual])?;

        // Collect MDD frame: x, y, z per vertex (pad z=0 for 2D).
        let mdd_frame: Vec<[f32; 3]> = coords
            .coord_iter()
            .map(|v| {
                let x = v[0] as f32;
                let y = v[1] as f32;
                let z = if coords.dim() >= 3 { v[2] as f32 } else { 0.0 };
                [x, y, z]
            })
            .collect();
        self.mdd_frames.push(mdd_frame);

        // Write static base OBJ from the first frame's topology and coordinates.
        if !self.wrote_base_obj {
            write_obj(&self.dir.join("blender").join("base.obj"), complex, coords)?;
            self.wrote_base_obj = true;
        }

        self.frame_counter += 1;
        Ok(())
    }

    /// Finalize the run directory.
    ///
    /// Writes `frames.pvd`, flushes `diagnostics.csv`, and writes
    /// `blender/motion.mdd` from all recorded frames.
    pub fn finish(self) -> io::Result<()> {
        write_pvd(&self.dir.join("frames.pvd"), &self.pvd_entries)
            .map_err(|e| io::Error::other(e.to_string()))?;

        self.diag.finish()?;

        let times: Vec<f32> = self.pvd_entries.iter().map(|(t, _)| *t as f32).collect();
        write_mdd(&self.dir.join("blender").join("motion.mdd"), &self.mdd_frames, &times)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use derham::cochain::Cochain;
    use simplicial::r#gen::cartesian::CartesianGrid;

    #[test]
    fn writes_a_complete_run_dir() {
        let (complex, coords) = CartesianGrid::new_unit(2, 2).triangulate();
        let dir = std::env::temp_dir().join("cartan_run_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut run = RunWriter::new(&dir).unwrap();
        for k in 0..3 {
            let e = Cochain::new(1, nalgebra::DVector::from_element(complex.nsimplices(1), 0.1 * k as f64));
            let b = Cochain::new(2, nalgebra::DVector::from_element(complex.nsimplices(2), 0.2));
            run.push_frame(k as f64 * 0.1, &complex, &coords, &b, &e, 1.0, 0.0).unwrap();
        }
        run.finish().unwrap();
        assert!(dir.join("frames.pvd").exists());
        assert!(dir.join("frame_0000.vtu").exists());
        assert!(dir.join("frame_0002.vtu").exists());
        assert!(dir.join("diagnostics.csv").exists());
        assert!(dir.join("blender/base.obj").exists());
        assert!(dir.join("blender/motion.mdd").exists());
        let csv = std::fs::read_to_string(dir.join("diagnostics.csv")).unwrap();
        assert_eq!(csv.lines().count(), 4); // header + 3 rows
    }
}
