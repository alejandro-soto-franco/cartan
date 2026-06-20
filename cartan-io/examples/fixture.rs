//! Fixture generator for flowforms integration tests.
//!
//! Usage: cargo run -p cartan-io --example fixture -- <output-dir>
//!
//! Writes into <output-dir>:
//!   surface.vtp   - one triangle mesh with a scalar field

use std::path::PathBuf;
use nalgebra::SVector;
use cartan_dec::mesh::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_io::{write_vtp, Field};

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: fixture <output-dir>")
        .into();

    std::fs::create_dir_all(&out_dir).expect("could not create output dir");

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
    let scalar = Field::Scalar {
        name: "temp".into(),
        values: vec![1.0, 2.0, 3.0],
    };
    // Director field: (1, 0, 0) per vertex, nematic (headless). Name will be
    // written as "director__nematic" so the Python loader can detect it via
    // the __nematic suffix convention.
    let director = Field::Vector {
        name: "director".into(),
        values: vec![1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0],
        nematic: true,
    };

    write_vtp(&out_dir.join("surface.vtp"), &mesh, &[scalar, director]).expect("write surface.vtp");

    println!("fixtures written to {}", out_dir.display());
}
