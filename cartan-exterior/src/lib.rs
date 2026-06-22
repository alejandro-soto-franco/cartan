//! Dimension-generic numerical exterior algebra.
//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.

pub mod combo;
pub mod gramian;
pub mod term;
pub mod list;

pub use gramian::Gramian;

use nalgebra as na;

/// Intrinsic dimension of the linear space underlying the exterior algebra.
pub type Dim = usize;
/// Exterior grade (form degree) k in Λᵏ.
pub type ExteriorGrade = usize;

pub fn exterior_dim(dim: Dim, grade: ExteriorGrade) -> usize {
  combo::binomial(dim, grade)
}

/// An element of an exterior algebra.
#[derive(Debug, Clone)]
pub struct ExteriorElement {
  coeffs: na::DVector<f64>,
  dim: Dim,
  grade: ExteriorGrade,
}

impl ExteriorElement {
  pub fn new(coeffs: na::DVector<f64>, dim: Dim, grade: ExteriorGrade) -> Self {
    assert_eq!(coeffs.len(), exterior_dim(dim, grade));
    Self { coeffs, dim, grade }
  }

  pub fn scalar(v: f64, dim: Dim) -> ExteriorElement {
    Self::new(na::dvector![v], dim, 0)
  }
  pub fn line(coeffs: na::DVector<f64>) -> Self {
    let dim = coeffs.len();
    Self::new(coeffs, dim, 1)
  }

  pub fn zero(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::new(na::DVector::zeros(exterior_dim(dim, grade)), dim, grade)
  }
  pub fn one(dim: Dim) -> Self {
    Self::scalar(1.0, dim)
  }

  pub fn into_grade1(self) -> na::DVector<f64> {
    assert!(self.grade == 1);
    self.coeffs
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  pub fn coeffs(&self) -> &na::DVector<f64> {
    &self.coeffs
  }
  pub fn into_coeffs(self) -> na::DVector<f64> {
    self.coeffs
  }

  pub fn basis_iter(&self) -> impl Iterator<Item = (f64, term::ExteriorTerm)> + '_ {
    let dim = self.dim;
    let grade = self.grade;
    self
      .coeffs
      .iter()
      .copied()
      .enumerate()
      .map(move |(i, coeff)| {
        let basis = term::ExteriorTerm::from_lex_rank(dim, grade, i);
        (coeff, basis)
      })
  }

  pub fn basis_iter_mut(&mut self) -> impl Iterator<Item = (&mut f64, term::ExteriorTerm)> + '_ {
    let dim = self.dim;
    let grade = self.grade;
    self.coeffs.iter_mut().enumerate().map(move |(i, coeff)| {
      let basis = term::ExteriorTerm::from_lex_rank(dim, grade, i);
      (coeff, basis)
    })
  }

  pub fn wedge(&self, other: &Self) -> Self {
    assert_eq!(self.dim, other.dim);
    let dim = self.dim;

    let new_grade = self.grade + other.grade;
    if new_grade > dim {
      return Self::zero(dim, 0);
    }

    let new_basis_size = exterior_dim(dim, new_grade);
    let mut new_coeffs = na::DVector::zeros(new_basis_size);

    for (self_coeff, self_basis) in self.basis_iter() {
      for (other_coeff, other_basis) in other.basis_iter() {
        let self_basis = self_basis.clone();

        let coeff_prod = self_coeff * other_coeff;
        if self_basis == other_basis || coeff_prod == 0.0 {
          continue;
        }
        if let Some((sign, merged_basis)) = self_basis.wedge(other_basis).canonicalized() {
          let merged_basis = merged_basis.lex_rank();
          new_coeffs[merged_basis] += sign.as_f64() * coeff_prod;
        }
      }
    }

    Self::new(new_coeffs, dim, new_grade)
  }

  pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Option<Self> {
    let mut factors = factors.into_iter();
    let first = factors.next()?;
    let prod = factors.fold(first, |acc, factor| acc.wedge(&factor));
    Some(prod)
  }

  pub fn eq_epsilon(&self, other: &Self, eps: f64) -> bool {
    self.dim == other.dim
      && self.grade == other.grade
      && (&self.coeffs - &other.coeffs).norm_squared() <= eps.powi(2)
  }
}

impl std::ops::Add<ExteriorElement> for ExteriorElement {
  type Output = Self;
  fn add(mut self, other: ExteriorElement) -> Self::Output {
    self += other;
    self
  }
}
impl std::ops::AddAssign<ExteriorElement> for ExteriorElement {
  fn add_assign(&mut self, other: ExteriorElement) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs += other.coeffs;
  }
}

impl std::ops::Sub<ExteriorElement> for ExteriorElement {
  type Output = Self;
  fn sub(mut self, other: ExteriorElement) -> Self::Output {
    self -= other;
    self
  }
}
impl std::ops::SubAssign<ExteriorElement> for ExteriorElement {
  fn sub_assign(&mut self, other: ExteriorElement) {
    assert_eq!(self.dim, other.dim);
    assert_eq!(self.grade, other.grade);
    self.coeffs -= other.coeffs;
  }
}

impl std::ops::Mul<f64> for ExteriorElement {
  type Output = Self;
  fn mul(mut self, scalar: f64) -> Self::Output {
    self *= scalar;
    self
  }
}
impl std::ops::MulAssign<f64> for ExteriorElement {
  fn mul_assign(&mut self, scalar: f64) {
    self.coeffs *= scalar;
  }
}
impl std::ops::Mul<ExteriorElement> for f64 {
  type Output = ExteriorElement;
  fn mul(self, rhs: ExteriorElement) -> Self::Output {
    rhs * self
  }
}

impl std::ops::Index<term::ExteriorTerm> for ExteriorElement {
  type Output = f64;
  fn index(&self, term: term::ExteriorTerm) -> &Self::Output {
    assert!(
      term.is_basis(),
      "Can only index exterior element with exterior basis term."
    );
    assert!(term.dim() == self.dim());
    assert!(term.grade() == self.grade());
    &self.coeffs[term.lex_rank()]
  }
}
impl std::ops::IndexMut<term::ExteriorTerm> for ExteriorElement {
  fn index_mut(&mut self, term: term::ExteriorTerm) -> &mut Self::Output {
    assert!(
      term.is_basis(),
      "Can only index exterior element with exterior basis term."
    );
    assert!(term.dim() == self.dim());
    assert!(term.grade() == self.grade());
    &mut self.coeffs[term.lex_rank()]
  }
}
impl std::ops::Index<usize> for ExteriorElement {
  type Output = f64;
  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

impl From<term::ExteriorTerm> for ExteriorElement {
  fn from(term: term::ExteriorTerm) -> Self {
    let mut element = Self::zero(term.dim(), term.grade());
    if let Some((sign, basis)) = term.canonicalized() {
      element[basis] = sign.as_f64();
    }
    element
  }
}

impl std::ops::AddAssign<term::ExteriorTerm> for ExteriorElement {
  fn add_assign(&mut self, term: term::ExteriorTerm) {
    self[term] += 1.0;
  }
}
impl std::ops::Add<term::ExteriorTerm> for ExteriorElement {
  type Output = ExteriorElement;
  fn add(mut self, term: term::ExteriorTerm) -> Self::Output {
    self += term;
    self
  }
}

impl std::iter::FromIterator<term::ExteriorTerm> for ExteriorElement {
  fn from_iter<T: IntoIterator<Item = term::ExteriorTerm>>(iter: T) -> Self {
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    let mut element = Self::from(first);
    iter.for_each(|term| element += term);
    element
  }
}
impl std::iter::Sum for ExteriorElement {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    let mut iter = iter.into_iter();
    let mut sum = iter.next().unwrap();
    for element in iter {
      sum += element;
    }
    sum
  }
}
