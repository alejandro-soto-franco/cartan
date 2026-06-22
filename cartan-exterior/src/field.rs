// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use crate::{Dim, ExteriorElement, ExteriorGrade};

pub trait ExteriorField {
  fn dim_ambient(&self) -> Dim;
  fn dim_intrinsic(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at_point<'a>(&self, coord: impl Into<nalgebra::DVectorView<'a, f64>>) -> ExteriorElement;
}

// Trait aliases.
pub trait MultiVectorField: ExteriorField {}
impl<T: ExteriorField> MultiVectorField for T {}
pub trait MultiFormField: ExteriorField {}
impl<T: ExteriorField> MultiFormField for T {}
pub trait DifferentialMultiForm: MultiFormField {}
impl<T: MultiFormField> DifferentialMultiForm for T {}

pub type DiffFormClosure = ExteriorFieldClosure;

#[allow(clippy::type_complexity)]
pub struct ExteriorFieldClosure {
  closure: Box<dyn Fn(nalgebra::DVectorView<f64>) -> ExteriorElement + Sync>,
  dim: Dim,
  grade: ExteriorGrade,
}

#[allow(clippy::type_complexity)]
impl ExteriorFieldClosure {
  pub fn new(
    closure: Box<dyn Fn(nalgebra::DVectorView<f64>) -> ExteriorElement + Sync>,
    dim: Dim,
    grade: ExteriorGrade,
  ) -> Self {
    Self {
      closure,
      dim,
      grade,
    }
  }
}

// Convenience methods specifically for DiffFormClosure
impl DiffFormClosure {
  /// Create a scalar field (0-form).
  pub fn scalar(
    f: impl Fn(nalgebra::DVectorView<f64>) -> f64 + Sync + 'static,
    dim: Dim,
  ) -> Self {
    let wrapper = move |x: nalgebra::DVectorView<f64>| crate::ExteriorElement::scalar(f(x), dim);
    Self::new(Box::new(wrapper), dim, 0)
  }

  /// Create a 1-form (covector field).
  pub fn one_form(
    f: impl Fn(nalgebra::DVectorView<f64>) -> nalgebra::DVector<f64> + Sync + 'static,
    dim: Dim,
  ) -> Self {
    let wrapper = move |x: nalgebra::DVectorView<f64>| crate::ExteriorElement::line(f(x));
    Self::new(Box::new(wrapper), dim, 1)
  }

  /// Create a constant scalar field.
  pub fn constant_scalar(value: f64, dim: Dim) -> Self {
    Self::scalar(move |_| value, dim)
  }

  /// Create a scalar field that extracts a specific coordinate component.
  pub fn coordinate_component(icomp: usize, dim: Dim) -> Self {
    assert!(icomp < dim, "Component index out of bounds");
    Self::scalar(move |x| x[icomp], dim)
  }

  /// Create a constant differential form of any grade (Plan 4 source terms use this).
  pub fn constant(value: crate::MultiForm, dim: Dim) -> Self {
    let grade = value.grade();
    let wrapper = move |_x: nalgebra::DVectorView<f64>| value.clone();
    Self::new(Box::new(wrapper), dim, grade)
  }
}

impl ExteriorField for ExteriorFieldClosure {
  fn dim_ambient(&self) -> Dim {
    self.dim
  }
  fn dim_intrinsic(&self) -> Dim {
    self.dim
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at_point<'a>(&self, coord: impl Into<nalgebra::DVectorView<'a, f64>>) -> ExteriorElement {
    (self.closure)(coord.into())
  }
}

#[cfg(test)]
mod cartan_tests {
  use super::*;
  use approx::assert_relative_eq;

  #[test]
  fn constant_form_evaluates_to_its_coeffs_everywhere() {
    // A constant 1-form on R^2 with coeffs [1, 2] evaluates to [1, 2]
    // at any point.
    let value = crate::MultiForm::from_grade1(nalgebra::DVector::from_vec(vec![1.0, 2.0]));
    let closure = DiffFormClosure::constant(value.clone(), 2);
    let at = closure.at_point(nalgebra::DVector::from_vec(vec![0.3, 0.7]).as_view());
    assert_eq!(at.grade(), 1);
    assert_relative_eq!(at.coeffs()[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(at.coeffs()[1], 2.0, epsilon = 1e-12);
  }
}
