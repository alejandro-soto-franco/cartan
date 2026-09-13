//! The Galerkin Hodge mass, stored by element.

use exterior::ExteriorGrade;
use formoniq::operators::{ElMatProvider, HodgeMassElmat};
use simplicial::geometry::metric::mesh::MeshLengthsSq;
use simplicial::topology::complex::Complex;

use crate::cg::MassBackend;
use crate::restrict::{CONSTRAINED, Interior};

/// `M_k` on host memory, kept as its element matrices.
///
/// Storage is `nlocal^2` doubles per cell against the assembled sparse matrix's
/// row-compressed entries plus indices. For grade 1 on tetrahedra that is 36
/// doubles per cell, and the assembled `M_1` on the same mesh needs roughly 15
/// entries for each of about 7 edges per cell, so the element form is smaller as
/// well as cheaper to rebuild when the metric moves.
///
/// The element matrices are computed once per metric. A conjugate gradient
/// solve then applies them tens of times, so the cost of computing them is
/// amortised and the application itself is bound by memory traffic.
pub struct HostMass {
    grade: ExteriorGrade,
    ndofs: usize,
    nlocal: usize,
    ncells: usize,
    /// Degree-of-freedom index of every local face, cell-major, `nlocal` per
    /// cell. [`CONSTRAINED`] where the face is constrained away.
    dofs: Vec<u32>,
    /// Element matrices, cell-major, row-major `nlocal * nlocal` per cell.
    elmats: Vec<f64>,
    /// The operator diagonal, for Jacobi preconditioning.
    diag: Vec<f64>,
}

impl HostMass {
    /// Build `M_k` over every degree of freedom at `grade`.
    pub fn new(topology: &Complex, geometry: &MeshLengthsSq, grade: ExteriorGrade) -> Self {
        Self::build(topology, geometry, grade, None)
    }

    /// Build `E^T M_k E` over the unconstrained degrees of freedom alone.
    pub fn restricted(topology: &Complex, geometry: &MeshLengthsSq, interior: &Interior) -> Self {
        Self::build(topology, geometry, interior.grade(), Some(interior))
    }

    fn build(
        topology: &Complex,
        geometry: &MeshLengthsSq,
        grade: ExteriorGrade,
        interior: Option<&Interior>,
    ) -> Self {
        let provider = HodgeMassElmat::new(topology.dim(), grade);
        let ndofs = match interior {
            Some(int) => {
                assert_eq!(int.grade(), grade, "interior grade must match the operator");
                assert_eq!(
                    int.ndofs_full(),
                    topology.skeleton(grade).len(),
                    "interior was built for a different complex"
                );
                int.ndofs()
            }
            None => topology.skeleton(grade).len(),
        };

        let cells = topology.cells();
        let ncells = cells.len();
        let mut nlocal = 0usize;
        let mut dofs: Vec<u32> = Vec::new();
        let mut elmats: Vec<f64> = Vec::new();

        for cell in cells.handle_iter() {
            let metric = geometry.cell_metric(cell);
            let elmat = provider.eval(&metric);

            if nlocal == 0 {
                nlocal = elmat.nrows();
                dofs.reserve(ncells * nlocal);
                elmats.reserve(ncells * nlocal * nlocal);
            }
            debug_assert_eq!(elmat.nrows(), nlocal);
            debug_assert_eq!(elmat.ncols(), nlocal);

            for face in cell.faces(grade) {
                let global = face.kidx();
                let dof = match interior {
                    Some(int) => int.inverse()[global],
                    None => global as u32,
                };
                dofs.push(dof);
            }

            // Row-major, so the inner loop of `apply` walks contiguously.
            for i in 0..nlocal {
                for j in 0..nlocal {
                    elmats.push(elmat[(i, j)]);
                }
            }
        }

        let mut mass = Self {
            grade,
            ndofs,
            nlocal,
            ncells,
            dofs,
            elmats,
            diag: vec![0.0; ndofs],
        };
        mass.build_diagonal();
        mass
    }

    fn build_diagonal(&mut self) {
        let n = self.nlocal;
        for c in 0..self.ncells {
            let d = &self.dofs[c * n..(c + 1) * n];
            let m = &self.elmats[c * n * n..(c + 1) * n * n];
            for i in 0..n {
                if d[i] != CONSTRAINED {
                    self.diag[d[i] as usize] += m[i * n + i];
                }
            }
        }
    }

    /// The grade this operator acts on.
    pub fn grade(&self) -> ExteriorGrade {
        self.grade
    }

    /// Degrees of freedom the operator acts on. Same as the [`MassBackend`]
    /// method, available without the trait in scope.
    pub fn ndofs_total(&self) -> usize {
        self.ndofs
    }

    /// Number of cells contributing element matrices.
    pub fn ncells(&self) -> usize {
        self.ncells
    }

    /// Faces of the operator's grade per cell.
    pub fn nlocal(&self) -> usize {
        self.nlocal
    }

    /// The operator diagonal.
    pub fn diagonal(&self) -> &[f64] {
        &self.diag
    }

    /// The element matrices, cell-major and row-major within a cell. A device
    /// backend uploads this verbatim.
    pub fn elmats(&self) -> &[f64] {
        &self.elmats
    }

    /// The degree-of-freedom index of every local face, cell-major.
    pub fn dof_map(&self) -> &[u32] {
        &self.dofs
    }

    /// The quadratic form `x^T M x`, which is the discrete `L^2` norm squared
    /// of the form `x` represents. Allocates one working vector per call.
    pub fn quadratic_form(&self, x: &[f64]) -> f64 {
        let mut y = vec![0.0; self.ndofs];
        self.apply_slice(x, &mut y);
        x.iter().zip(&y).map(|(a, b)| a * b).sum()
    }

    /// `y <- M x`, element by element.
    pub fn apply_slice(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ndofs);
        assert_eq!(y.len(), self.ndofs);
        y.fill(0.0);
        let n = self.nlocal;
        for c in 0..self.ncells {
            let d = &self.dofs[c * n..(c + 1) * n];
            let m = &self.elmats[c * n * n..(c + 1) * n * n];
            for i in 0..n {
                let row = d[i];
                if row == CONSTRAINED {
                    continue;
                }
                let mut sum = 0.0;
                for j in 0..n {
                    let col = d[j];
                    if col != CONSTRAINED {
                        sum += m[i * n + j] * x[col as usize];
                    }
                }
                y[row as usize] += sum;
            }
        }
    }
}

impl MassBackend for HostMass {
    type Vector = Vec<f64>;

    fn ndofs(&self) -> usize {
        self.ndofs
    }

    fn zeros(&self) -> Vec<f64> {
        vec![0.0; self.ndofs]
    }

    fn from_host(&self, src: &[f64]) -> Vec<f64> {
        assert_eq!(src.len(), self.ndofs);
        src.to_vec()
    }

    fn to_host(&self, src: &Vec<f64>, dst: &mut [f64]) {
        dst.copy_from_slice(src);
    }

    fn copy(&self, src: &Vec<f64>, dst: &mut Vec<f64>) {
        dst.copy_from_slice(src);
    }

    fn apply(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
        self.apply_slice(x, y);
    }

    fn precondition(&self, r: &Vec<f64>, z: &mut Vec<f64>) {
        for i in 0..self.ndofs {
            // A zero diagonal means the degree of freedom sits in no cell, so
            // the operator is singular there and scaling would divide by zero.
            z[i] = if self.diag[i] != 0.0 {
                r[i] / self.diag[i]
            } else {
                r[i]
            };
        }
    }

    fn dot(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn axpy(&self, a: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        for i in 0..self.ndofs {
            y[i] += a * x[i];
        }
    }

    fn aypx(&self, a: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        for i in 0..self.ndofs {
            y[i] = x[i] + a * y[i];
        }
    }
}
