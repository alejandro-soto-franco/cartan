//! Boundary extraction, anchoring, and no-slip.
//!
//! A face is on the boundary when exactly one tetrahedron contains it. The
//! outward normal is fixed by that tetrahedron: it points away from the fourth
//! vertex.
//!
//! The Euler characteristic of the extracted boundary is the check that the
//! extraction is right. For a ball it must be 2, and a face miscounted in
//! either direction breaks that.

use std::collections::HashMap;

use cartan_core::rotor::Rotor3;

use crate::complex3::Complex3;
use crate::geometry::Geometry3;

/// The boundary of a tetrahedral complex.
#[derive(Clone, Debug)]
pub struct Boundary {
    faces: Vec<usize>,
    edges: Vec<usize>,
    vertices: Vec<usize>,
    /// Outward unit normal per entry of `faces`.
    normals: Vec<[f64; 3]>,
    /// Area per entry of `faces`.
    areas: Vec<f64>,
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Boundary {
    /// Extract the boundary.
    #[must_use]
    pub fn extract(c: &Complex3, g: &Geometry3) -> Self {
        let mut count: HashMap<usize, Vec<usize>> = HashMap::new();
        for t in 0..c.n_tets() {
            for f in c.tet_triangles(t) {
                count.entry(f).or_default().push(t);
            }
        }
        let mut faces: Vec<usize> = count
            .iter()
            .filter(|(_, ts)| ts.len() == 1)
            .map(|(f, _)| *f)
            .collect();
        faces.sort_unstable();

        let p = g.positions();
        let mut normals = Vec::with_capacity(faces.len());
        let mut areas = Vec::with_capacity(faces.len());
        for &f in &faces {
            let tri = c.triangle(f);
            let t = count[&f][0];
            let tet = c.tets()[t];
            let apex = *tet
                .iter()
                .find(|v| !tri.contains(v))
                .expect("a tetrahedron has four vertices");
            let n = cross(&sub(&p[tri[1]], &p[tri[0]]), &sub(&p[tri[2]], &p[tri[0]]));
            let len = dot(&n, &n).sqrt();
            areas.push(0.5 * len);
            let mut u = [n[0] / len, n[1] / len, n[2] / len];
            // Point away from the apex.
            if dot(&u, &sub(&p[apex], &p[tri[0]])) > 0.0 {
                u = [-u[0], -u[1], -u[2]];
            }
            normals.push(u);
        }

        let mut edges: Vec<usize> = Vec::new();
        let mut vertices: Vec<usize> = Vec::new();
        for &f in &faces {
            let tri = c.triangle(f);
            for (a, b) in [(0, 1), (1, 2), (0, 2)] {
                let e = c.edge_of(&[tri[a], tri[b]]);
                if !edges.contains(&e) {
                    edges.push(e);
                }
            }
            for v in tri {
                if !vertices.contains(&v) {
                    vertices.push(v);
                }
            }
        }
        edges.sort_unstable();
        vertices.sort_unstable();
        Self {
            faces,
            edges,
            vertices,
            normals,
            areas,
        }
    }

    /// Boundary face indices.
    #[must_use]
    pub fn faces(&self) -> &[usize] {
        &self.faces
    }
    /// Boundary edge indices.
    #[must_use]
    pub fn edges(&self) -> &[usize] {
        &self.edges
    }
    /// Boundary vertex indices.
    #[must_use]
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    /// Outward unit normals, one per boundary face.
    #[must_use]
    pub fn normals(&self) -> &[[f64; 3]] {
        &self.normals
    }
    /// Areas, one per boundary face.
    #[must_use]
    pub fn areas(&self) -> &[f64] {
        &self.areas
    }

    /// Euler characteristic of the boundary surface, `V - E + F`.
    #[must_use]
    pub fn euler_characteristic(&self) -> i64 {
        self.vertices.len() as i64 - self.edges.len() as i64 + self.faces.len() as i64
    }

    /// The area-weighted average outward normal at each boundary vertex.
    ///
    /// A vertex on an edge or corner of a polyhedral domain has no single
    /// normal, so the average is what a homeotropic condition can use there.
    #[must_use]
    pub fn vertex_normals(&self, c: &Complex3) -> HashMap<usize, [f64; 3]> {
        let mut acc: HashMap<usize, [f64; 3]> = HashMap::new();
        for (i, &f) in self.faces.iter().enumerate() {
            for v in c.triangle(f) {
                let e = acc.entry(v).or_insert([0.0; 3]);
                for (k, ek) in e.iter_mut().enumerate() {
                    *ek += self.normals[i][k] * self.areas[i];
                }
            }
        }
        for n in acc.values_mut() {
            let len = dot(n, n).sqrt().max(1e-300);
            for nk in n.iter_mut() {
                *nk /= len;
            }
        }
        acc
    }
}

/// The rotor whose reference director `e_z` is taken to `n`.
///
/// Used to prescribe homeotropic anchoring, where the director is the surface
/// normal.
#[must_use]
pub fn rotor_taking_z_to(n: [f64; 3]) -> Rotor3 {
    let z = [0.0, 0.0, 1.0];
    let c = dot(&z, &n);
    if c > 1.0 - 1e-12 {
        return Rotor3::IDENTITY;
    }
    if c < -1.0 + 1e-12 {
        // A half turn about any perpendicular axis.
        return Rotor3 {
            w: 0.0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
    }
    let axis = cross(&z, &n);
    let len = dot(&axis, &axis).sqrt();
    let theta = c.clamp(-1.0, 1.0).acos();
    let (s, cw) = (theta / 2.0).sin_cos();
    Rotor3 {
        w: cw,
        x: s * axis[0] / len,
        y: s * axis[1] / len,
        z: s * axis[2] / len,
    }
}

/// Rapini-Papoular surface energy and its gradient.
///
/// `W * sum over boundary faces of area * |T - T_s|^2`, with `T_s` the
/// prescribed order parameter at each boundary vertex and `T` the current one.
/// The face integral is approximated by the vertex average, which is exact for
/// a field that is constant on the face and second-order otherwise.
///
/// Kept as a separate contribution rather than folded into `Energy`, so a
/// caller adds it to the bulk and elastic terms explicitly and the boundary
/// stays out of the interior functional's signature.
pub mod anchoring {
    use nalgebra::DVector;

    use super::Boundary;
    use crate::complex3::Complex3;
    use crate::energy::{Energy, State};

    /// Surface energy of the current state against a prescribed one.
    #[must_use]
    pub fn energy(
        c: &Complex3,
        b: &Boundary,
        e: &Energy,
        state: &State,
        prescribed: &[DVector<f64>],
        w: f64,
    ) -> f64 {
        let mut acc = 0.0;
        for (i, &f) in b.faces().iter().enumerate() {
            for v in c.triangle(f) {
                let t = e.tensor(&state.rotors[v], state.amps(v));
                let d = &t - &prescribed[v];
                acc += w * b.areas()[i] * d.dot(&d) / 3.0;
            }
        }
        acc
    }

    /// Gradient of the surface energy in the same layout as
    /// [`Energy::gradient`]: three rotation components then the amplitudes,
    /// per vertex.
    #[must_use]
    pub fn gradient(
        c: &Complex3,
        b: &Boundary,
        e: &Energy,
        state: &State,
        prescribed: &[DVector<f64>],
        w: f64,
    ) -> Vec<f64> {
        let n = e.basis().n_amplitudes();
        let block = 3 + n;
        let mut g = vec![0.0; state.n_vertices() * block];
        for (i, &f) in b.faces().iter().enumerate() {
            for v in c.triangle(f) {
                let rho = e.basis().rep(&state.rotors[v]);
                let t = e.tensor(&state.rotors[v], state.amps(v));
                let d = &t - &prescribed[v];
                let scale = 2.0 * w * b.areas()[i] / 3.0;
                for a in 0..3 {
                    g[v * block + a] += scale * d.dot(&(e.generator(a) * &t));
                }
                for j in 0..n {
                    g[v * block + 3 + j] += scale * d.dot(&(&rho * e.basis().basis_column(j)));
                }
            }
        }
        g
    }
}
