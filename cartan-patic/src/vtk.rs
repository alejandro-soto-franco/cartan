//! VTK XML output, for PyVista and ParaView.
//!
//! Two files per frame: an unstructured grid of the tetrahedra with the fields
//! at its points, and a polydata of the disclination lines.
//!
//! ASCII rather than base64 binary. The meshes that get looked at are small,
//! the files stay readable, and a reader can check a value by eye. `cartan-io`
//! has a binary writer for production-sized output, built on the simplicial
//! complex type rather than this crate's.
//!
//! ## Fields
//!
//! - `director`: the frame's principal axis, `R e_z`. Defined at any harmonic
//!   degree, and at degree 2 it is the nematic director.
//! - `frame_x`, `frame_y`: the other two axes, so a glyph can show the full
//!   frame rather than one axis.
//! - `order`: the norm of the invariant tensor, a scalar order measure that
//!   works at any degree.
//! - `amplitude_i`: the amplitudes themselves.
//! - `velocity`: the flow, reconstructed at the vertices.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

use cartan_core::rotor::Rotor3;

use crate::complex3::Complex3;
use crate::energy::{Energy, State};
use crate::geometry::Geometry3;

/// Point fields for one frame.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// `(name, components, values)` with `values.len() == components * n_points`.
    pub point_data: Vec<(String, usize, Vec<f64>)>,
}

impl Snapshot {
    /// The standard fields for a state: frame axes, order, amplitudes.
    #[must_use]
    pub fn from_state(e: &Energy, state: &State) -> Self {
        let nv = state.n_vertices();
        let n_amp = e.basis().n_amplitudes();
        let mut director = Vec::with_capacity(3 * nv);
        let mut fx = Vec::with_capacity(3 * nv);
        let mut fy = Vec::with_capacity(3 * nv);
        let mut order = Vec::with_capacity(nv);
        let mut amps = vec![Vec::with_capacity(nv); n_amp];

        for v in 0..nv {
            let r: Rotor3 = state.rotors[v];
            director.extend_from_slice(&r.rotate_vec([0.0, 0.0, 1.0]));
            fx.extend_from_slice(&r.rotate_vec([1.0, 0.0, 0.0]));
            fy.extend_from_slice(&r.rotate_vec([0.0, 1.0, 0.0]));
            let t = e.tensor(&r, state.amps(v));
            order.push(t.norm());
            for (i, a) in state.amps(v).iter().enumerate() {
                amps[i].push(*a);
            }
        }

        let mut point_data = vec![
            ("director".to_string(), 3, director),
            ("frame_x".to_string(), 3, fx),
            ("frame_y".to_string(), 3, fy),
            ("order".to_string(), 1, order),
        ];
        for (i, a) in amps.into_iter().enumerate() {
            point_data.push((format!("amplitude_{i}"), 1, a));
        }
        Self { point_data }
    }

    /// Attach a vector field at the points.
    #[must_use]
    pub fn with_vectors(mut self, name: &str, values: &[[f64; 3]]) -> Self {
        let flat: Vec<f64> = values.iter().flat_map(|v| v.iter().copied()).collect();
        self.point_data.push((name.to_string(), 3, flat));
        self
    }

    /// Attach a scalar field at the points.
    #[must_use]
    pub fn with_scalars(mut self, name: &str, values: &[f64]) -> Self {
        self.point_data.push((name.to_string(), 1, values.to_vec()));
        self
    }
}

fn data_array(out: &mut String, name: &str, comps: usize, values: &[f64]) {
    let _ = write!(
        out,
        "        <DataArray type=\"Float64\" Name=\"{name}\" NumberOfComponents=\"{comps}\" format=\"ascii\">\n          "
    );
    for v in values {
        let _ = write!(out, "{v:.10e} ");
    }
    out.push_str("\n        </DataArray>\n");
}

/// Write the tetrahedral mesh and its point fields as a `.vtu`.
///
/// # Errors
///
/// Propagates any filesystem error from creating or writing the file.
pub fn write_vtu(path: &Path, c: &Complex3, g: &Geometry3, snap: &Snapshot) -> io::Result<()> {
    let nv = c.n_vertices();
    for (name, comps, values) in &snap.point_data {
        assert_eq!(
            values.len(),
            comps * nv,
            "field {name} has {} values for {nv} points at {comps} components",
            values.len()
        );
    }

    let mut s = String::with_capacity(1 << 16);
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    s.push_str("  <UnstructuredGrid>\n");
    let _ = writeln!(
        s,
        "    <Piece NumberOfPoints=\"{nv}\" NumberOfCells=\"{}\">",
        c.n_tets()
    );

    s.push_str("      <Points>\n");
    let flat: Vec<f64> = g
        .positions()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    data_array(&mut s, "Points", 3, &flat);
    s.push_str("      </Points>\n");

    s.push_str("      <Cells>\n");
    s.push_str(
        "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          ",
    );
    for t in c.tets() {
        for v in t {
            let _ = write!(s, "{v} ");
        }
    }
    s.push_str("\n        </DataArray>\n");
    s.push_str("        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          ");
    for i in 1..=c.n_tets() {
        let _ = write!(s, "{} ", 4 * i);
    }
    s.push_str("\n        </DataArray>\n");
    s.push_str("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ");
    for _ in 0..c.n_tets() {
        s.push_str("10 ");
    }
    s.push_str("\n        </DataArray>\n");
    s.push_str("      </Cells>\n");

    if !snap.point_data.is_empty() {
        s.push_str("      <PointData>\n");
        for (name, comps, values) in &snap.point_data {
            data_array(&mut s, name, *comps, values);
        }
        s.push_str("      </PointData>\n");
    }

    s.push_str("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
    fs::write(path, s)
}

/// Write a set of polylines as a `.vtp`, for the disclination lines.
///
/// # Errors
///
/// Propagates any filesystem error from creating or writing the file.
pub fn write_lines_vtp(path: &Path, curves: &[Vec<[f64; 3]>]) -> io::Result<()> {
    let n_points: usize = curves.iter().map(Vec::len).sum();
    let mut s = String::with_capacity(1 << 14);
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    s.push_str("  <PolyData>\n");
    let _ = writeln!(
        s,
        "    <Piece NumberOfPoints=\"{n_points}\" NumberOfLines=\"{}\">",
        curves.len()
    );

    s.push_str("      <Points>\n");
    let flat: Vec<f64> = curves
        .iter()
        .flat_map(|c| c.iter().flat_map(|p| p.iter().copied()))
        .collect();
    data_array(&mut s, "Points", 3, &flat);
    s.push_str("      </Points>\n");

    s.push_str("      <Lines>\n");
    s.push_str(
        "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          ",
    );
    for i in 0..n_points {
        let _ = write!(s, "{i} ");
    }
    s.push_str("\n        </DataArray>\n");
    s.push_str("        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          ");
    let mut acc = 0usize;
    for c in curves {
        acc += c.len();
        let _ = write!(s, "{acc} ");
    }
    s.push_str("\n        </DataArray>\n");
    s.push_str("      </Lines>\n");

    // One id per line, so a viewer can colour them apart.
    s.push_str("      <CellData>\n");
    let ids: Vec<f64> = (0..curves.len()).map(|i| i as f64).collect();
    data_array(&mut s, "line_id", 1, &ids);
    s.push_str("      </CellData>\n");

    s.push_str("    </Piece>\n  </PolyData>\n</VTKFile>\n");
    fs::write(path, s)
}

/// Write a ParaView collection so a frame sequence plays as a time series.
///
/// # Errors
///
/// Propagates any filesystem error from creating or writing the file.
pub fn write_pvd(path: &Path, frames: &[(f64, String)]) -> io::Result<()> {
    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    s.push_str("  <Collection>\n");
    for (t, file) in frames {
        let _ = writeln!(
            s,
            "    <DataSet timestep=\"{t}\" group=\"\" part=\"0\" file=\"{file}\"/>"
        );
    }
    s.push_str("  </Collection>\n</VTKFile>\n");
    fs::write(path, s)
}
