//! VTK UnstructuredGrid (.vtu) writer for runtime-dimensional simplicial complexes.

use std::io::{self, Write};
use std::path::Path;
use cartan_simplicial::topology::complex::Complex;
use cartan_simplicial::geometry::coord::mesh::MeshCoords;
use crate::xml::{encode_f64_le, encode_i64_le};

/// Write a VTK UnstructuredGrid file (.vtu) for a simplicial complex.
///
/// `point_scalars`: (name, values) pairs with `values.len() == nvertices`.
/// `point_vectors`: (name, values) pairs with `values.len() == 3 * nvertices`.
/// `cell_scalars`:  (name, values) pairs with `values.len() == ncells`.
/// `cell_vectors`:  (name, values) pairs with `values.len() == 3 * ncells`.
///
/// All DataArrays are encoded as little-endian binary base64 (VTK header_type UInt64),
/// mirroring the convention in `vtp.rs`.
pub fn write_vtu(
    path: &Path,
    complex: &Complex,
    coords: &MeshCoords,
    point_scalars: &[(&str, &[f64])],
    point_vectors: &[(&str, &[f64])],
    cell_scalars:  &[(&str, &[f64])],
    cell_vectors:  &[(&str, &[f64])],
) -> io::Result<()> {
    let nv    = coords.nvertices();
    let ncells = complex.nsimplices(complex.dim());
    let cdim  = complex.dim();

    // Validate array lengths (panic with clear message on mismatch).
    for (name, vals) in point_scalars {
        assert_eq!(vals.len(), nv, "point scalar '{name}': expected {nv} values, got {}", vals.len());
    }
    for (name, vals) in point_vectors {
        assert_eq!(vals.len(), 3 * nv, "point vector '{name}': expected {} values, got {}", 3 * nv, vals.len());
    }
    for (name, vals) in cell_scalars {
        assert_eq!(vals.len(), ncells, "cell scalar '{name}': expected {ncells} values, got {}", vals.len());
    }
    for (name, vals) in cell_vectors {
        assert_eq!(vals.len(), 3 * ncells, "cell vector '{name}': expected {} values, got {}", 3 * ncells, vals.len());
    }

    // Points: x,y,z interleaved (pad z=0 for 2D).
    let mut pts = Vec::with_capacity(nv * 3);
    for coord in coords.coord_iter() {
        pts.push(coord[0]);
        pts.push(coord[1]);
        pts.push(if cdim >= 3 { coord[2] } else { 0.0 });
    }

    // Connectivity: vertex indices per cell.
    // Offsets: cumulative count per cell (each cell has cdim+1 vertices).
    // Types: VTK cell type id per cell.
    let verts_per_cell = cdim + 1;
    let vtk_type: i64 = match cdim {
        1 => 3,  // VTK_LINE
        2 => 5,  // VTK_TRIANGLE
        3 => 10, // VTK_TETRA
        d => panic!("write_vtu: unsupported complex dim {d}"),
    };

    let mut conn = Vec::with_capacity(ncells * verts_per_cell);
    let mut offs = Vec::with_capacity(ncells);
    let mut types = Vec::with_capacity(ncells);

    for (k, cell) in complex.cells().handle_iter().enumerate() {
        for v in cell.iter() {
            conn.push(v as i64);
        }
        offs.push(((k + 1) * verts_per_cell) as i64);
        types.push(vtk_type);
    }

    let mut f = io::BufWriter::new(std::fs::File::create(path)?);

    write!(f, r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{nv}" NumberOfCells="{ncells}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="binary">{pts_enc}</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="binary">{conn_enc}</DataArray>
        <DataArray type="Int64" Name="offsets" format="binary">{offs_enc}</DataArray>
        <DataArray type="Int64" Name="types" format="binary">{types_enc}</DataArray>
      </Cells>
      <PointData>
"#,
        nv       = nv,
        ncells   = ncells,
        pts_enc  = encode_f64_le(&pts),
        conn_enc = encode_i64_le(&conn),
        offs_enc = encode_i64_le(&offs),
        types_enc = encode_i64_le(&types),
    )?;

    for (name, vals) in point_scalars {
        writeln!(
            f,
            r#"        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="binary">{enc}</DataArray>"#,
            name = name,
            enc  = encode_f64_le(vals),
        )?;
    }
    for (name, vals) in point_vectors {
        writeln!(
            f,
            r#"        <DataArray type="Float64" Name="{name}" NumberOfComponents="3" format="binary">{enc}</DataArray>"#,
            name = name,
            enc  = encode_f64_le(vals),
        )?;
    }

    write!(f, r#"      </PointData>
      <CellData>
"#)?;

    for (name, vals) in cell_scalars {
        writeln!(
            f,
            r#"        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="binary">{enc}</DataArray>"#,
            name = name,
            enc  = encode_f64_le(vals),
        )?;
    }
    for (name, vals) in cell_vectors {
        writeln!(
            f,
            r#"        <DataArray type="Float64" Name="{name}" NumberOfComponents="3" format="binary">{enc}</DataArray>"#,
            name = name,
            enc  = encode_f64_le(vals),
        )?;
    }

    write!(f, r#"      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#)?;

    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    #[test]
    fn writes_wellformed_vtu_for_unit_square() {
        let (complex, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
        let ncells = complex.nsimplices(2);
        let cell_scalar: Vec<f64> = (0..ncells).map(|i| i as f64).collect();
        let cell_vec: Vec<f64> = (0..3 * ncells).map(|i| i as f64).collect();
        let dir = std::env::temp_dir().join("cartan_vtu_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.vtu");
        write_vtu(&path, &complex, &coords, &[], &[], &[("B", &cell_scalar)], &[("E", &cell_vec)]).unwrap();
        let xml = std::fs::read_to_string(&path).unwrap();
        assert!(xml.contains("type=\"UnstructuredGrid\""));
        assert!(xml.contains(&format!("NumberOfPoints=\"{}\"", coords.nvertices())));
        assert!(xml.contains(&format!("NumberOfCells=\"{}\"", ncells)));
        assert!(xml.contains("Name=\"B\""));
        assert!(xml.contains("Name=\"E\"") && xml.contains("NumberOfComponents=\"3\""));
        // triangle cell type id 5 appears in the types DataArray (decode-independent smoke: file nonempty)
        assert!(xml.len() > 200);
    }
}
