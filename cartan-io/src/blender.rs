//! Blender OBJ and MDD vertex-cache exporter for mesh animation.
//!
//! MDD is the NewTek Motion Designer vertex cache that Blender's Mesh Cache
//! modifier reads. Every field is big-endian, and the layout is fully
//! determined by two counts read from the header:
//!
//! | offset | type          | meaning              |
//! |--------|---------------|----------------------|
//! | 0      | `u32`         | frame count `F`      |
//! | 4      | `u32`         | vertex count `V`     |
//! | 8      | `f32 * F`     | time of each frame   |
//! | 8+4F   | `f32 * 3*V*F` | xyz, vertex-major within a frame |
//!
//! There is no per-frame vertex count, so `V` is fixed for the whole cache:
//! a mesh whose vertex count changes between frames cannot be represented,
//! and [`write_mdd`] rejects that input rather than writing a file that
//! silently misparses.

use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::path::Path;

use simplicial::geometry::coord::mesh::MeshCoords;
use simplicial::topology::complex::Complex;

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
        let indices: Vec<String> = cell.simplex().iter().map(|i| (i + 1).to_string()).collect();
        writeln!(s, "f {}", indices.join(" ")).unwrap();
    }
    std::fs::write(path, s)
}

/// Write a Blender MDD vertex-cache file, per the layout in the module docs.
///
/// Errors with `InvalidInput` if `times` disagrees with `frames` in length, or
/// if the frames do not all carry the same vertex count, since neither is
/// representable in the format.
pub fn write_mdd(path: &Path, frames: &[Vec<[f32; 3]>], times: &[f32]) -> std::io::Result<()> {
    let invalid = |msg: &str| std::io::Error::new(std::io::ErrorKind::InvalidInput, msg.to_owned());

    if frames.len() != times.len() {
        return Err(invalid("MDD needs exactly one time per frame"));
    }
    let nvertices = match frames.first() {
        Some(first) => first.len(),
        // An empty cache is well-formed: F = 0, V = 0, no payload.
        None => 0,
    };
    if frames.iter().any(|f| f.len() != nvertices) {
        return Err(invalid("MDD requires a constant vertex count across frames"));
    }

    // Header (8 bytes) + times (4F) + positions (12 V F).
    let mut buf: Vec<u8> = Vec::with_capacity(8 + 4 * frames.len() + 12 * nvertices * frames.len());
    buf.extend_from_slice(&(frames.len() as u32).to_be_bytes());
    buf.extend_from_slice(&(nvertices as u32).to_be_bytes());
    for &time in times {
        buf.extend_from_slice(&time.to_be_bytes());
    }
    for frame in frames {
        for [x, y, z] in frame {
            buf.extend_from_slice(&x.to_be_bytes());
            buf.extend_from_slice(&y.to_be_bytes());
            buf.extend_from_slice(&z.to_be_bytes());
        }
    }

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    writer.write_all(&buf)?;
    writer.flush()
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
    use simplicial::r#gen::cartesian::CartesianGrid;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn mdd_header_encodes_counts_big_endian() {
        let path = tmp("cartan_mdd_test").join("motion.mdd");
        let frames = vec![
            vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0f32, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ];
        write_mdd(&path, &frames, &[0.0, 1.0]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], &2u32.to_be_bytes()); // nframes
        assert_eq!(&bytes[4..8], &2u32.to_be_bytes()); // nvertices
        // Header + 2 times + 2 frames * 2 vertices * 3 components.
        assert_eq!(bytes.len(), 8 + 2 * 4 + 2 * 2 * 3 * 4);
    }

    #[test]
    fn mdd_rejects_ragged_frames() {
        let path = tmp("cartan_mdd_ragged").join("motion.mdd");
        let frames = vec![
            vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0f32, 0.0, 0.0]],
        ];
        assert!(write_mdd(&path, &frames, &[0.0, 1.0]).is_err());
    }

    #[test]
    fn mdd_rejects_time_count_mismatch() {
        let path = tmp("cartan_mdd_times").join("motion.mdd");
        let frames = vec![vec![[0.0f32, 0.0, 0.0]]];
        assert!(write_mdd(&path, &frames, &[0.0, 1.0]).is_err());
    }

    #[test]
    fn obj_has_vertices_and_faces() {
        let (complex, coords) = CartesianGrid::new_unit(2, 1).triangulate();
        let path = tmp("cartan_obj_test").join("base.obj");
        write_obj(&path, &complex, &coords).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text.lines().filter(|l| l.starts_with("v ")).count(),
            coords.nvertices()
        );
        assert_eq!(
            text.lines().filter(|l| l.starts_with("f ")).count(),
            complex.nsimplices(2)
        );
    }
}
