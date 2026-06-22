//! Blender OBJ and MDD vertex-cache exporter for mesh animation.
//!
//! MDD binary layout ported from formoniq (author-permitted):
//! [nframes u32 BE][nverts u32 BE][times f32 BE ...][xyz f32 BE per vertex per frame]

use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::path::Path;

use cartan_simplicial::geometry::coord::mesh::MeshCoords;
use cartan_simplicial::topology::complex::Complex;

/// Write a Wavefront OBJ file for the top-dimensional faces of `complex`.
///
/// Vertices are written as `v x y z` (z=0 for 2D meshes).
/// Faces are written as `f i j k ...` (1-indexed) for each top-dimensional cell.
pub fn write_obj(path: &Path, complex: &Complex, coords: &MeshCoords) -> std::io::Result<()> {
    let mut s = String::new();
    for v in coords.coord_iter() {
        let x = v[0];
        let y = v[1];
        let z = if coords.dim() >= 3 { v[2] } else { 0.0 };
        writeln!(s, "v {:.6} {:.6} {:.6}", x, y, z).unwrap();
    }
    for cell in complex.cells().handle_iter() {
        let indices: Vec<String> = cell.iter().map(|i| (i + 1).to_string()).collect();
        writeln!(s, "f {}", indices.join(" ")).unwrap();
    }
    std::fs::write(path, s)
}

/// Write a Blender MDD vertex-cache file.
///
/// Format (all big-endian): `[nframes u32][nverts u32][times f32 x nframes]
/// [xyz f32 x nverts x nframes]`.
///
/// Ported from formoniq `write_mdd_file` (author-permitted).
pub fn write_mdd(path: &Path, frames: &[Vec<[f32; 3]>], times: &[f32]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    let nframes = frames.len() as u32;
    writer.write_all(&nframes.to_be_bytes())?;
    let nvertices = frames[0].len() as u32;
    writer.write_all(&nvertices.to_be_bytes())?;

    for &time in times {
        writer.write_all(&time.to_be_bytes())?;
    }
    for vertices in frames {
        for vertex in vertices {
            for comp in vertex {
                writer.write_all(&comp.to_be_bytes())?;
            }
        }
    }
    Ok(())
}

/// Write a Blender MDD animation from a sequence of `MeshCoords` frames.
///
/// Each frame is converted to `Vec<[f32; 3]>` (z=0 for 2D). Times are
/// downcast from f64 to f32.
pub fn write_mesh_animation(
    path: &Path,
    coords_frames: &[MeshCoords],
    times: &[f64],
) -> std::io::Result<()> {
    let mdd_frames: Vec<Vec<[f32; 3]>> = coords_frames
        .iter()
        .map(|coords| {
            coords
                .coord_iter()
                .map(|v| {
                    let x = v[0] as f32;
                    let y = v[1] as f32;
                    let z = if coords.dim() >= 3 { v[2] as f32 } else { 0.0 };
                    [x, y, z]
                })
                .collect()
        })
        .collect();
    let time_frames: Vec<f32> = times.iter().map(|&t| t as f32).collect();
    write_mdd(path, &mdd_frames, &time_frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    #[test]
    fn mdd_header_encodes_counts_big_endian() {
        let dir = std::env::temp_dir().join("cartan_mdd_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("motion.mdd");
        let frames = vec![
            vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0f32, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ];
        write_mdd(&path, &frames, &[0.0, 1.0]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], &2u32.to_be_bytes()); // nframes
        assert_eq!(&bytes[4..8], &2u32.to_be_bytes()); // nvertices
    }

    #[test]
    fn obj_has_vertices_and_faces() {
        let (complex, coords) = CartesianMeshInfo::new_unit(2, 1).compute_coord_complex();
        let dir = std::env::temp_dir().join("cartan_obj_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("base.obj");
        write_obj(&path, &complex, &coords).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), coords.nvertices());
        assert_eq!(text.lines().filter(|l| l.starts_with("f ")).count(), complex.nsimplices(2));
    }
}
