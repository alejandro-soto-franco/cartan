//! RVE: ordered map of phases with fractions, plus matrix identifier and reference medium.

use crate::{error::HomogError, shapes::UserInclusion, tensor::TensorOrder};
use alloc::{string::String, vec::Vec};

#[derive(Clone, Debug)]
pub struct Phase<O: TensorOrder> {
    pub name: String,
    pub shape: UserInclusion<O>,
    pub property: O::KmMatrix,
    pub fraction: f64,
}

#[derive(Clone, Debug)]
pub enum RefMedium<O: TensorOrder> {
    Matrix,
    Explicit(O::KmMatrix),
    Effective,
}

#[derive(Clone, Debug)]
pub struct Rve<O: TensorOrder> {
    pub phases: Vec<Phase<O>>,
    pub matrix: Option<String>,
    pub ref_medium: RefMedium<O>,
}

impl<O: TensorOrder> Rve<O> {
    pub fn new() -> Self {
        Self { phases: Vec::new(), matrix: None, ref_medium: RefMedium::Matrix }
    }

    pub fn add_phase(&mut self, p: Phase<O>) { self.phases.push(p); }

    pub fn set_matrix(&mut self, name: impl Into<String>) {
        self.matrix = Some(name.into());
    }

    pub fn matrix_property(&self) -> Result<&O::KmMatrix, HomogError> {
        let m_name = self.matrix.as_ref()
            .or_else(|| self.phases.first().map(|p| &p.name))
            .ok_or_else(|| HomogError::UnknownPhase(String::from("(none)")))?;
        let ph = self.phases.iter().find(|p| &p.name == m_name)
            .ok_or_else(|| HomogError::UnknownPhase(m_name.clone()))?;
        Ok(&ph.property)
    }

    pub fn reference_property(&self, effective: Option<&O::KmMatrix>) -> Result<O::KmMatrix, HomogError> {
        match &self.ref_medium {
            RefMedium::Matrix      => Ok(self.matrix_property()?.clone()),
            RefMedium::Explicit(c) => Ok(c.clone()),
            RefMedium::Effective   => effective.cloned().ok_or_else(||
                HomogError::Solver(String::from("RefMedium::Effective used outside iterative scheme"))),
        }
    }

    pub fn sum_fractions(&self) -> f64 {
        self.phases.iter().map(|p| p.fraction).sum()
    }

    pub fn is_matrix_phase(&self, name: &str) -> bool {
        self.matrix.as_deref() == Some(name)
    }
}

impl<O: TensorOrder> Default for Rve<O> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::Sphere, tensor::Order2};
    use alloc::sync::Arc;
    use approx::assert_relative_eq;

    #[test]
    fn two_phase_rve_fractions_sum_to_one() {
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase {
            name: String::from("M"), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0), fraction: 0.7,
        });
        rve.add_phase(Phase {
            name: String::from("I"), shape: Arc::new(Sphere),
            property: Order2::scalar(5.0), fraction: 0.3,
        });
        rve.set_matrix("M");
        assert_relative_eq!(rve.sum_fractions(), 1.0);
        assert_relative_eq!(rve.matrix_property().unwrap()[(0, 0)], 1.0);
    }
}
