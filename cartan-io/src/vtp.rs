//! VTK PolyData (.vtp) writer for cartan-dec triangle surface meshes.

use std::io::Write;
use std::path::Path;
use cartan_core::Manifold;
use cartan_dec::mesh::Mesh;
use nalgebra::SVector;
use crate::xml::{encode_f64_le, encode_i64_le};

pub enum Field {
    Scalar { name: String, values: Vec<f64> },
    /// A vector field on the surface mesh.
    ///
    /// When `nematic` is true the field is a headless director (draw it
    /// double-ended). The DataArray Name is written as `{name}__nematic` so
    /// the suffix survives a round-trip through pyvista and the renderer can
    /// detect it without relying on non-standard XML attributes.
    Vector { name: String, values: Vec<f64>, nematic: bool },
}

pub fn write_vtp<M>(
    path: &Path,
    mesh: &Mesh<M, 3, 2>,
    fields: &[Field],
) -> Result<(), Box<dyn std::error::Error>>
where
    M: Manifold<Point = SVector<f64, 3>>,
{
    let nv = mesh.n_vertices();
    let nt = mesh.n_simplices();

    // Points: x,y,z interleaved per vertex.
    let mut pts = Vec::with_capacity(nv * 3);
    for v in &mesh.vertices {
        pts.push(v[0]);
        pts.push(v[1]);
        pts.push(v[2]);
    }

    // Triangle connectivity + cumulative offsets.
    let mut conn = Vec::with_capacity(nt * 3);
    let mut offs = Vec::with_capacity(nt);
    for (t, tri) in mesh.simplices.iter().enumerate() {
        conn.push(tri[0] as i64);
        conn.push(tri[1] as i64);
        conn.push(tri[2] as i64);
        offs.push(((t + 1) * 3) as i64);
    }

    // Length validation: return errors instead of panicking on mismatch.
    for field in fields {
        match field {
            Field::Scalar { name, values } => {
                if values.len() != nv {
                    return Err(format!(
                        "scalar field '{name}': expected {nv} values, got {}",
                        values.len()
                    )
                    .into());
                }
            }
            Field::Vector { name, values, .. } => {
                if values.len() != 3 * nv {
                    return Err(format!(
                        "vector field '{name}': expected {} values, got {}",
                        3 * nv,
                        values.len()
                    )
                    .into());
                }
            }
        }
    }

    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(
        f,
        r#"<?xml version="1.0"?>
<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <PolyData>
    <Piece NumberOfPoints="{nv}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{nt}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="binary">{pts_enc}</DataArray>
      </Points>
      <Polys>
        <DataArray type="Int64" Name="connectivity" format="binary">{conn_enc}</DataArray>
        <DataArray type="Int64" Name="offsets" format="binary">{offs_enc}</DataArray>
      </Polys>
      <PointData>
"#,
        nv = nv,
        nt = nt,
        pts_enc = encode_f64_le(&pts),
        conn_enc = encode_i64_le(&conn),
        offs_enc = encode_i64_le(&offs),
    )?;

    for field in fields {
        match field {
            Field::Scalar { name, values } => {
                write!(
                    f,
                    r#"        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="binary">{enc}</DataArray>
"#,
                    name = name,
                    enc = encode_f64_le(values),
                )?;
            }
            Field::Vector { name, values, nematic } => {
                let da_name = if *nematic {
                    format!("{name}__nematic")
                } else {
                    name.clone()
                };
                write!(
                    f,
                    r#"        <DataArray type="Float64" Name="{da_name}" NumberOfComponents="3" format="binary">{enc}</DataArray>
"#,
                    da_name = da_name,
                    enc = encode_f64_le(values),
                )?;
            }
        }
    }

    write!(
        f,
        r#"      </PointData>
    </Piece>
  </PolyData>
</VTKFile>
"#
    )?;
    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_dec::mesh::Mesh;
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    #[test]
    fn writes_triangle_surface_with_scalar() {
        // One triangle in the z=0 plane, embedded in R^3.
        let verts = vec![
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 0.0, 0.0),
            SVector::<f64, 3>::new(0.0, 1.0, 0.0),
        ];
        let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(
            &Euclidean::<3>,
            verts,
            vec![[0, 1, 2]],
        );
        let f = Field::Scalar {
            name: "temp".into(),
            values: vec![1.0, 2.0, 3.0],
        };
        let tmp = std::env::temp_dir().join("ff_tri.vtp");
        write_vtp(&tmp, &mesh, &[f]).unwrap();
        let body = std::fs::read_to_string(&tmp).unwrap();
        assert!(body.contains(r#"NumberOfPoints="3""#));
        assert!(body.contains(r#"NumberOfPolys="1""#));
        assert!(body.contains(r#"Name="temp""#));
    }

    #[test]
    fn writes_vector_and_nematic_fields() {
        let verts = vec![
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 0.0, 0.0),
            SVector::<f64, 3>::new(0.0, 1.0, 0.0),
        ];
        let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(
            &Euclidean::<3>,
            verts,
            vec![[0, 1, 2]],
        );

        // 3 vertices x 3 components = 9 values.
        let vec_field = Field::Vector {
            name: "velocity".into(),
            values: vec![1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
            nematic: false,
        };
        let nem_field = Field::Vector {
            name: "director".into(),
            values: vec![1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
            nematic: true,
        };

        let tmp = std::env::temp_dir().join("ff_vec.vtp");
        write_vtp(&tmp, &mesh, &[vec_field, nem_field]).unwrap();
        let body = std::fs::read_to_string(&tmp).unwrap();
        assert!(body.contains(r#"NumberOfComponents="3""#));
        // Nematic director uses the __nematic name suffix (survives VTK round-trip).
        assert!(body.contains(r#"Name="director__nematic""#));
        // Non-nematic vector keeps its plain name.
        assert!(body.contains(r#"Name="velocity""#));
        // No non-standard attribute should appear.
        assert!(!body.contains(r#"attribute="nematic""#));

        // Wrong-length scalar must return Err, not panic.
        let bad_scalar = Field::Scalar {
            name: "bad".into(),
            values: vec![1.0, 2.0], // only 2 values for 3-vertex mesh
        };
        let tmp2 = std::env::temp_dir().join("ff_bad.vtp");
        let result = write_vtp(&tmp2, &mesh, &[bad_scalar]);
        assert!(result.is_err());
    }
}
