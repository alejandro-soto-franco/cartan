//! Mesh generators for common manifolds.
//!
//! All generators return `Mesh<M, 3, 2>` (triangle meshes). The `well_centered`
//! flag applies Lloyd smoothing + Delaunay flips to guarantee DEC correctness.

use std::collections::HashMap;

use nalgebra::SVector;

use cartan_manifolds::euclidean::Euclidean;
use cartan_manifolds::sphere::Sphere;

use crate::mesh::Mesh;
use crate::mesh_quality::{is_well_centered, make_delaunay, make_well_centered};

/// Generate an icosahedral subdivision of the unit sphere.
///
/// # Parameters
///
/// - `manifold`: the `Sphere<3>` manifold.
/// - `level`: refinement level (0 = 12 vertices, 1 = 42, 2 = 162, 3 = 642, 4 = 2562).
/// - `well_centered`: if true, apply Delaunay flips + Lloyd smoothing to ensure
///   all circumcenters lie inside their triangles.
pub fn icosphere(manifold: &Sphere<3>, level: usize, well_centered: bool) -> Mesh<Sphere<3>, 3, 2> {
    let (mut vertices, mut triangles) = icosahedron_seed();

    for _ in 0..level {
        let (v, t) = subdivide_sphere(&vertices, &triangles);
        vertices = v;
        triangles = t;
    }

    let points: Vec<SVector<f64, 3>> = vertices
        .iter()
        .map(|v| SVector::<f64, 3>::new(v[0], v[1], v[2]))
        .collect();

    let mut mesh = Mesh::from_simplices_generic(manifold, points, triangles);

    if well_centered {
        mesh = make_delaunay(mesh, manifold);
        if !is_well_centered(&mesh, manifold) {
            mesh = make_well_centered(mesh, manifold, 50, 1e-8);
        }
    }

    mesh
}

/// Generate a triangulated torus embedded in R^3.
///
/// # Parameters
///
/// - `manifold`: the `Euclidean<3>` ambient space.
/// - `major_radius`: distance from center of tube to center of torus (R).
/// - `minor_radius`: tube radius (r).
/// - `n_major`: number of divisions around the major circle.
/// - `n_minor`: number of divisions around the minor circle.
/// - `well_centered`: if true, apply Delaunay flips + Lloyd smoothing.
pub fn torus(
    manifold: &Euclidean<3>,
    major_radius: f64,
    minor_radius: f64,
    n_major: usize,
    n_minor: usize,
    well_centered: bool,
) -> Mesh<Euclidean<3>, 3, 2> {
    let nv = n_major * n_minor;
    let mut vertices = Vec::with_capacity(nv);

    for i in 0..n_major {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / n_major as f64;
        for j in 0..n_minor {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / n_minor as f64;
            let x = (major_radius + minor_radius * phi.cos()) * theta.cos();
            let y = (major_radius + minor_radius * phi.cos()) * theta.sin();
            let z = minor_radius * phi.sin();
            vertices.push(SVector::<f64, 3>::new(x, y, z));
        }
    }

    let mut triangles = Vec::with_capacity(2 * nv);
    for i in 0..n_major {
        for j in 0..n_minor {
            let v00 = i * n_minor + j;
            let v10 = ((i + 1) % n_major) * n_minor + j;
            let v01 = i * n_minor + (j + 1) % n_minor;
            let v11 = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor;
            triangles.push([v00, v10, v01]);
            triangles.push([v10, v11, v01]);
        }
    }

    let mut mesh = Mesh::from_simplices_generic(manifold, vertices, triangles);

    if well_centered {
        mesh = make_delaunay(mesh, manifold);
        if !is_well_centered(&mesh, manifold) {
            mesh = make_well_centered(mesh, manifold, 50, 1e-8);
        }
    }

    mesh
}

/// Compute analytical Gaussian curvature at each vertex of a torus mesh.
///
/// K(phi) = cos(phi) / (r * (R + r * cos(phi)))
pub fn torus_gaussian_curvature(
    major_radius: f64,
    minor_radius: f64,
    n_major: usize,
    n_minor: usize,
) -> Vec<f64> {
    let mut curvatures = Vec::with_capacity(n_major * n_minor);
    for _i in 0..n_major {
        for j in 0..n_minor {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / n_minor as f64;
            let k = phi.cos() / (minor_radius * (major_radius + minor_radius * phi.cos()));
            curvatures.push(k);
        }
    }
    curvatures
}

/// Seed icosahedron: 12 vertices, 20 faces.
fn icosahedron_seed() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = 1.0 / norm;
    let b = phi / norm;

    let vertices = vec![
        [-a, b, 0.0],
        [a, b, 0.0],
        [-a, -b, 0.0],
        [a, -b, 0.0],
        [0.0, -a, b],
        [0.0, a, b],
        [0.0, -a, -b],
        [0.0, a, -b],
        [b, 0.0, -a],
        [b, 0.0, a],
        [-b, 0.0, -a],
        [-b, 0.0, a],
    ];

    let triangles = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    (vertices, triangles)
}

/// Subdivide each triangle into 4 by inserting edge midpoints, projecting to unit sphere.
fn subdivide_sphere(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut new_verts = vertices.to_vec();
    let mut midpoint_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut new_tris = Vec::with_capacity(triangles.len() * 4);

    for &[v0, v1, v2] in triangles {
        let a = get_midpoint(v0, v1, &mut new_verts, &mut midpoint_cache);
        let b = get_midpoint(v1, v2, &mut new_verts, &mut midpoint_cache);
        let c = get_midpoint(v2, v0, &mut new_verts, &mut midpoint_cache);

        new_tris.push([v0, a, c]);
        new_tris.push([v1, b, a]);
        new_tris.push([v2, c, b]);
        new_tris.push([a, b, c]);
    }

    (new_verts, new_tris)
}

fn get_midpoint(
    v0: usize,
    v1: usize,
    verts: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<(usize, usize), usize>,
) -> usize {
    let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let mid = [
        (verts[v0][0] + verts[v1][0]) / 2.0,
        (verts[v0][1] + verts[v1][1]) / 2.0,
        (verts[v0][2] + verts[v1][2]) / 2.0,
    ];
    let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    let projected = [mid[0] / len, mid[1] / len, mid[2] / len];
    let idx = verts.len();
    verts.push(projected);
    cache.insert(key, idx);
    idx
}
