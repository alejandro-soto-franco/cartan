//! Which degrees of freedom a solve runs over.

use exterior::ExteriorGrade;
use simplicial::topology::complex::Complex;

/// Sentinel for a degree of freedom that is constrained away.
pub(crate) const CONSTRAINED: u32 = u32::MAX;

/// A choice of unconstrained degrees of freedom at one grade.
///
/// Restricting an operator to these is the Galerkin projection `E^T M E`, where
/// `E` extends an interior cochain by zero onto the constrained simplices.
/// Applying it element by element needs no projection matrix: a constrained
/// face is skipped in both the row and the column loop, which is the same thing.
#[derive(Debug, Clone)]
pub struct Interior {
    grade: ExteriorGrade,
    /// Global index of every interior degree of freedom, ascending.
    interior: Vec<u32>,
    /// Interior index of every global degree of freedom, or [`CONSTRAINED`].
    inverse: Vec<u32>,
}

impl Interior {
    /// Constrain the topological boundary, which is the perfectly conducting
    /// condition for Maxwell and the homogeneous Dirichlet condition generally.
    pub fn boundary_constrained(topology: &Complex, grade: ExteriorGrade) -> Self {
        let ndofs = topology.skeleton(grade).len();
        let mut constrained = vec![false; ndofs];
        for idx in topology.boundary_simplices(grade) {
            constrained[idx.kidx] = true;
        }
        Self::from_mask(grade, &constrained)
    }

    /// Every degree of freedom at this grade is free.
    pub fn unconstrained(topology: &Complex, grade: ExteriorGrade) -> Self {
        let ndofs = topology.skeleton(grade).len();
        Self::from_mask(grade, &vec![false; ndofs])
    }

    /// Constrain exactly the degrees of freedom the mask marks `true`.
    pub fn from_mask(grade: ExteriorGrade, constrained: &[bool]) -> Self {
        let mut interior = Vec::new();
        let mut inverse = vec![CONSTRAINED; constrained.len()];
        for (global, &is_constrained) in constrained.iter().enumerate() {
            if !is_constrained {
                inverse[global] = interior.len() as u32;
                interior.push(global as u32);
            }
        }
        Self {
            grade,
            interior,
            inverse,
        }
    }

    /// The grade these degrees of freedom sit at.
    pub fn grade(&self) -> ExteriorGrade {
        self.grade
    }

    /// How many degrees of freedom survive.
    pub fn ndofs(&self) -> usize {
        self.interior.len()
    }

    /// How many the full complex has at this grade.
    pub fn ndofs_full(&self) -> usize {
        self.inverse.len()
    }

    /// Global index of an interior degree of freedom.
    pub fn to_global(&self, interior: usize) -> usize {
        self.interior[interior] as usize
    }

    pub(crate) fn inverse(&self) -> &[u32] {
        &self.inverse
    }

    /// Drop the constrained entries of a full-length cochain.
    pub fn restrict(&self, full: &[f64]) -> Vec<f64> {
        assert_eq!(full.len(), self.ndofs_full());
        self.interior.iter().map(|&g| full[g as usize]).collect()
    }

    /// Place interior values back on the full complex, zero elsewhere.
    pub fn extend_by_zero(&self, interior: &[f64]) -> Vec<f64> {
        assert_eq!(interior.len(), self.ndofs());
        let mut full = vec![0.0; self.ndofs_full()];
        for (i, &g) in self.interior.iter().enumerate() {
            full[g as usize] = interior[i];
        }
        full
    }
}
