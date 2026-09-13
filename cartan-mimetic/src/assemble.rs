//! The star on a mesh.
//!
//! A global star is the sum of local ones over the cells, placed by the index
//! each local k-face has in the mesh and signed by whether the cell's ordering
//! of that face agrees with the mesh's. Both matter: a face shared by two cells
//! receives a contribution from each, and a sign error makes the assembled
//! matrix indefinite in a way no single-cell test would show.

use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::combinatorics::k_faces;
use crate::{Simplex, local_star};

/// A simplicial mesh: vertex coordinates, and cells as vertex index lists.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// One row per vertex.
    pub vertices: Vec<Vec<f64>>,
    /// One entry per cell, each listing its vertices.
    pub cells: Vec<Vec<usize>>,
}

impl Mesh {
    /// The `k`-faces of the whole mesh, deduplicated, each sorted ascending.
    ///
    /// Sorting is the orientation convention: a face is oriented by its
    /// increasing vertex order, and a cell that meets it the other way
    /// contributes with a minus sign.
    pub fn k_faces(&self, k: usize) -> (Vec<Vec<usize>>, HashMap<Vec<usize>, usize>) {
        let mut index: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut list = Vec::new();
        for cell in &self.cells {
            for local in k_faces(cell.len(), k) {
                let mut face: Vec<usize> = local.iter().map(|&i| cell[i]).collect();
                face.sort_unstable();
                if !index.contains_key(&face) {
                    index.insert(face.clone(), list.len());
                    list.push(face);
                }
            }
        }
        (list, index)
    }

    /// One cell as a [`Simplex`].
    pub fn simplex(&self, cell: usize) -> Simplex {
        Simplex::new(
            &self.cells[cell]
                .iter()
                .map(|&v| self.vertices[v].clone())
                .collect::<Vec<_>>(),
        )
    }
}

/// The assembled mimetic star on `k`-cochains, dense.
///
/// Dense because the intended callers are eigenvalue studies and moderate
/// meshes; a sparse assembly is the same loop with a different accumulator, and
/// the local stars are what this crate is for.
///
/// Returns `None` when any cell is degenerate, since a mesh containing one has
/// no well-defined inner product and a silent zero row would be worse.
pub fn assemble_star(mesh: &Mesh, k: usize) -> Option<DMatrix<f64>> {
    let (faces, index) = mesh.k_faces(k);
    let mut global = DMatrix::zeros(faces.len(), faces.len());

    for c in 0..mesh.cells.len() {
        let cell = &mesh.cells[c];
        let simplex = mesh.simplex(c);
        let local = local_star(&simplex, k)?;
        let local_faces = k_faces(cell.len(), k);

        // Where each local face sits globally, and whether the orderings agree.
        let mut place = Vec::with_capacity(local_faces.len());
        for lf in &local_faces {
            let mapped: Vec<usize> = lf.iter().map(|&i| cell[i]).collect();
            let mut sorted = mapped.clone();
            sorted.sort_unstable();
            place.push((index[&sorted], permutation_sign(&mapped, &sorted)));
        }

        for (a, &(ga, sa)) in place.iter().enumerate() {
            for (b, &(gb, sb)) in place.iter().enumerate() {
                global[(ga, gb)] += sa * sb * local[(a, b)];
            }
        }
    }
    Some(global)
}

/// The sign of the permutation taking `from` to `to`, which list the same
/// values.
///
/// A k-face has an orientation and the cell may meet it either way round, so
/// this is what keeps the two contributions to a shared face from cancelling
/// when they should add, or adding when they should cancel.
fn permutation_sign(from: &[usize], to: &[usize]) -> f64 {
    let mut perm: Vec<usize> = from
        .iter()
        .map(|v| to.iter().position(|w| w == v).expect("same values"))
        .collect();
    let mut sign = 1.0;
    for i in 0..perm.len() {
        while perm[i] != i {
            let j = perm[i];
            perm.swap(i, j);
            sign = -sign;
        }
    }
    sign
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles sharing an edge.
    fn pair() -> Mesh {
        Mesh {
            vertices: vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
            cells: vec![vec![0, 1, 2], vec![1, 3, 2]],
        }
    }

    #[test]
    fn a_shared_edge_is_one_face_with_two_contributions() {
        let m = pair();
        let (faces, _) = m.k_faces(1);
        // Four boundary edges and one shared diagonal.
        assert_eq!(faces.len(), 5);
    }

    #[test]
    fn the_assembled_star_is_symmetric_and_positive_definite() {
        let m = pair();
        for k in 0..=2 {
            let g = assemble_star(&m, k).expect("both triangles are proper");
            let asym = (&g - g.transpose()).abs().max();
            assert!(asym < 1e-12, "k={k} assembled asymmetrically by {asym:.3e}");
            let min = (0.5 * (&g + g.transpose()))
                .symmetric_eigenvalues()
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            assert!(
                min > 0.0,
                "k={k} assembled with smallest eigenvalue {min:.3e}"
            );
        }
    }

    #[test]
    fn the_top_degree_star_totals_the_area() {
        // At `k = n` there is one face per cell and the star is the reciprocal
        // of its volume, so the trace of the inverse is the total area. This is
        // the one degree with an answer that needs no eigenvalue solve.
        let m = pair();
        let g = assemble_star(&m, 2).unwrap();
        let total: f64 = (0..g.nrows()).map(|i| 1.0 / g[(i, i)]).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "two half-unit triangles gave area {total}"
        );
    }
}
