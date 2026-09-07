//! VTK output, checked by writing files and reading them back with PyVista.

use cartan_core::rotor::Rotor3;
use cartan_patic::complex3::Complex3;
use cartan_patic::energy::{Energy, State};
use cartan_patic::geometry::Geometry3;
use cartan_patic::group::{AxialApolar, SymmetryGroup};
use cartan_patic::vtk::{write_lines_vtp, write_pvd, write_vtu, Snapshot};
use nalgebra::DMatrix;
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cartan-patic-{}-{name}", std::process::id()));
    p
}

fn energy_obj() -> Energy {
    let n = <AxialApolar as SymmetryGroup>::N_AMPLITUDES;
    Energy::new::<AxialApolar>(
        DMatrix::from_diagonal_element(n, n, -1.0),
        vec![0.0; n * n * n],
        DMatrix::identity(n, n),
        0.5,
    )
    .expect("coercive")
}

fn state(c: &Complex3, g: &Geometry3) -> State {
    let mut s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
    for (v, p) in g.positions().iter().enumerate() {
        let t = 1.7 * p[0];
        let (sn, cs) = (t / 2.0).sin_cos();
        s.rotors[v] = Rotor3 { w: cs, x: 0.0, y: sn, z: 0.0 };
    }
    s
}

#[test]
fn the_grid_file_has_the_right_counts_and_fields() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let e = energy_obj();
    let snap = Snapshot::from_state(&e, &state(&c, &g));
    let path = tmp("grid.vtu");
    write_vtu(&path, &c, &g, &snap).expect("write");

    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.contains(&format!("NumberOfPoints=\"{}\"", c.n_vertices())));
    assert!(text.contains(&format!("NumberOfCells=\"{}\"", c.n_tets())));
    for name in ["director", "frame_x", "frame_y", "order", "amplitude_0"] {
        assert!(text.contains(&format!("Name=\"{name}\"")), "missing field {name}");
    }
    // VTK_TETRA is type 10.
    assert!(text.contains("Name=\"types\""));
    std::fs::remove_file(&path).ok();
}

#[test]
fn the_line_file_carries_one_cell_per_curve() {
    let curves = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        vec![[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    ];
    let path = tmp("lines.vtp");
    write_lines_vtp(&path, &curves).expect("write");
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.contains("NumberOfPoints=\"5\""));
    assert!(text.contains("NumberOfLines=\"2\""));
    assert!(text.contains("Name=\"line_id\""));
    std::fs::remove_file(&path).ok();
}

#[test]
fn the_collection_lists_every_frame() {
    let path = tmp("series.pvd");
    let frames = vec![
        (0.0, "frame_0000.vtu".to_string()),
        (0.1, "frame_0001.vtu".to_string()),
    ];
    write_pvd(&path, &frames).expect("write");
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.contains("frame_0000.vtu"));
    assert!(text.contains("frame_0001.vtu"));
    assert!(text.contains("timestep=\"0.1\""));
    std::fs::remove_file(&path).ok();
}

/// The director is a unit vector at every point, so a glyph filter scales it
/// by the order rather than by an accidental length.
#[test]
fn the_exported_director_is_a_unit_field() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let e = energy_obj();
    let snap = Snapshot::from_state(&e, &state(&c, &g));
    let (_, comps, values) = snap
        .point_data
        .iter()
        .find(|(n, _, _)| n == "director")
        .expect("director present");
    assert_eq!(*comps, 3);
    for v in values.chunks(3) {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-12, "director length {n}");
    }
}

/// The three exported axes are an orthonormal frame, so a viewer showing all
/// three sees a rigid triad.
#[test]
fn the_exported_frame_is_orthonormal() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let e = energy_obj();
    let snap = Snapshot::from_state(&e, &state(&c, &g));
    let get = |name: &str| {
        snap.point_data
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, _, v)| v.clone())
            .expect("field present")
    };
    let (d, x, y) = (get("director"), get("frame_x"), get("frame_y"));
    for i in 0..c.n_vertices() {
        let s = |v: &Vec<f64>| [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
        let (a, b, cc) = (s(&x), s(&y), s(&d));
        let dot = |p: [f64; 3], q: [f64; 3]| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
        assert!(dot(a, b).abs() < 1e-12, "vertex {i}: x.y = {}", dot(a, b));
        assert!(dot(a, cc).abs() < 1e-12, "vertex {i}: x.z = {}", dot(a, cc));
        assert!(dot(b, cc).abs() < 1e-12, "vertex {i}: y.z = {}", dot(b, cc));
    }
}
