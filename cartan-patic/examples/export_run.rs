//! Run the coupled loop and write a VTK series for PyVista or ParaView.
//!
//! `cargo run -p cartan-patic --release --example export_run -- <out-dir>`

use cartan_core::rotor::Rotor3;
use cartan_patic::advect::vertex_velocity;
use cartan_patic::boundary::Boundary;
use cartan_patic::complex3::Complex3;
use cartan_patic::defect::DefectField;
use cartan_patic::energy::{Energy, State};
use cartan_patic::geometry::Geometry3;
use cartan_patic::group::{AxialApolar, SymmetryGroup};
use cartan_patic::knot::curves;
use cartan_patic::simulation::Simulation;
use cartan_patic::spin::Incidence;
use cartan_patic::vtk::{Snapshot, write_lines_vtp, write_pvd, write_vtu};
use nalgebra::DMatrix;
use std::path::PathBuf;

fn incidence_of(c: &Complex3) -> Incidence {
    let mut tris = Vec::new();
    for t in 0..c.n_tets() {
        for f in c.tet_triangles(t) {
            let tri = c.triangle(f);
            if !tris.contains(&tri) {
                tris.push(tri);
            }
        }
    }
    Incidence::from_triangles(c.n_vertices(), &tris)
}

/// A straight half-charge disclination along the axis of the box.
fn seeded(g: &Geometry3, n_amp: usize) -> State {
    let r = core::f64::consts::FRAC_1_SQRT_2;
    let mut s = State::uniform(g.positions().len(), Rotor3::IDENTITY, &vec![0.8; n_amp]);
    for (v, p) in g.positions().iter().enumerate() {
        let phi = (p[1] - 0.5).atan2(p[0] - 0.5);
        let theta = phi / 2.0;
        let (st, ct) = theta.sin_cos();
        s.rotors[v] = Rotor3 {
            w: r,
            x: -r * st,
            y: r * ct,
            z: 0.0,
        };
    }
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out/patic_run".to_string())
        .into();
    std::fs::create_dir_all(&dir)?;

    let n = 4;
    let c = Complex3::cube_grid(n);
    let g = Geometry3::cube_grid(n);
    let b = Boundary::extract(&c, &g);
    let inc = incidence_of(&c);
    let n_amp = <AxialApolar as SymmetryGroup>::N_AMPLITUDES;
    let e = Energy::new::<AxialApolar>(
        DMatrix::from_diagonal_element(n_amp, n_amp, -1.0),
        vec![0.0; n_amp * n_amp * n_amp],
        DMatrix::identity(n_amp, n_amp),
        0.5,
    )?;

    let sim = Simulation::new(&c, &g, &inc, &e, 1.0, 3.0, 2e-3).with_no_slip(b.edges());
    let mut state = seeded(&g, n_amp);

    let frames = 24;
    let mut entries = Vec::with_capacity(frames);
    for k in 0..frames {
        let u = sim.velocity(&state)?;
        let vel = vertex_velocity(&c, &g, u.as_slice());
        let snap = Snapshot::from_state(&e, &state).with_vectors("velocity", &vel);

        let name = format!("frame_{k:04}.vtu");
        write_vtu(&dir.join(&name), &c, &g, &snap)?;
        entries.push((k as f64 * 2e-3, name));

        let d = DefectField::detect::<AxialApolar>(&c, &state.rotors, 1e-8);
        write_lines_vtp(&dir.join(format!("lines_{k:04}.vtp")), &curves(&c, &g, &d))?;

        let r = sim.step::<AxialApolar>(&mut state)?;
        if k % 8 == 0 {
            println!(
                "frame {k:3}  energy {:+.6e}  speed {:.3e}  defect faces {}",
                r.energy,
                r.speed,
                d.pierced().len()
            );
        }
    }
    write_pvd(&dir.join("series.pvd"), &entries)?;
    println!("wrote {frames} frames to {}", dir.display());
    Ok(())
}
