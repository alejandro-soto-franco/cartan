// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use crate::combo::{binomial, lex_rank, Sign};
use crate::{Dim, ExteriorGrade};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm {
  indices: Vec<usize>,
  dim: Dim,
}

impl ExteriorTerm {
  pub fn new(indices: Vec<usize>, dim: Dim) -> Self {
    Self { indices, dim }
  }
  pub fn top(dim: Dim) -> Self {
    Self::new((0..dim).collect(), dim)
  }

  pub fn indices(&self) -> &[usize] {
    &self.indices
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.indices.len()
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }

  pub fn is_basis(&self) -> bool {
    self.is_canonical()
  }
  pub fn is_canonical(&self) -> bool {
    let Some((sign, canonical)) = self.clone().canonicalized() else {
      return false;
    };
    sign == Sign::Pos && canonical == *self
  }
  pub fn canonicalized(mut self) -> Option<(Sign, Self)> {
    let sign = sort_signed(&mut self.indices);
    let len = self.indices.len();
    self.indices.dedup();
    if self.indices.len() != len {
      return None;
    }
    Some((sign, self))
  }

  pub fn wedge(mut self, mut other: Self) -> Self {
    self.indices.append(&mut other.indices);
    self
  }

  pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, usize>> {
    self.indices.iter().copied()
  }
}

impl std::ops::Index<usize> for ExteriorTerm {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

impl ExteriorTerm {
  pub fn from_lex_rank(dim: Dim, grade: ExteriorGrade, mut rank: usize) -> Self {
    let mut indices = Vec::with_capacity(grade);
    let mut start = 0;
    for i in 0..grade {
      let remaining = grade - i;
      for x in start..=(dim - remaining) {
        let c = binomial(dim - x - 1, remaining - 1);
        if rank < c {
          indices.push(x);
          start = x + 1;
          break;
        } else {
          rank -= c;
        }
      }
    }
    Self::new(indices, dim)
  }

  pub fn from_graded_lex_rank(dim: Dim, grade: ExteriorGrade, mut rank: usize) -> Self {
    rank -= Self::graded_lex_rank_offset(dim, grade);
    Self::from_lex_rank(dim, grade, rank)
  }

  pub fn lex_rank(&self) -> usize {
    lex_rank(&self.indices, self.dim())
  }

  pub fn graded_lex_rank(&self) -> usize {
    let dim = self.dim();
    let grade = self.grade();
    Self::graded_lex_rank_offset(dim, grade) + self.lex_rank()
  }

  fn graded_lex_rank_offset(dim: usize, grade: usize) -> usize {
    (0..grade).map(|s| binomial(dim, s)).sum()
  }
}

impl ExteriorTerm {
  pub fn basis(dim: Dim, grade: ExteriorGrade) -> impl Iterator<Item = ExteriorTerm> {
    use itertools::Itertools;
    (0..dim)
      .combinations(grade)
      .map(move |indices| ExteriorTerm::new(indices, dim))
  }

  pub fn exterior_bases(dim: Dim, grade: ExteriorGrade) -> impl Iterator<Item = ExteriorTerm> {
    use itertools::Itertools;
    (0..dim)
      .combinations(grade)
      .map(move |indices| ExteriorTerm::new(indices, dim))
  }

  pub fn exterior_terms(dim: Dim, grade: ExteriorGrade) -> impl Iterator<Item = ExteriorTerm> {
    use itertools::Itertools;
    (0..dim)
      .permutations(grade)
      .map(move |indices| ExteriorTerm::new(indices, dim))
  }
}

// Helper function: sort_signed (sign of permutation based on adjacent transpositions)
fn sort_signed(indices: &mut [usize]) -> Sign {
  let mut sign = Sign::Pos;
  let len = indices.len();
  for i in 0..len {
    for j in (i + 1)..len {
      if indices[i] > indices[j] {
        indices.swap(i, j);
        sign = sign.other();
      }
    }
  }
  sign
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::combo::binomial;

  #[test]
  fn exterior_basis_has_binomial_dimension() {
    // dim Λᵏ(ℝⁿ) = C(n,k)
    for n in 0..=4 {
      for k in 0..=n {
        let count = ExteriorTerm::basis(n, k).count();
        assert_eq!(count, binomial(n, k), "n={n}, k={k}");
      }
    }
  }

  #[test]
  fn basis_terms_are_strictly_increasing() {
    for term in ExteriorTerm::basis(4, 2) {
      let idx = term.indices();
      assert!(idx.windows(2).all(|w| w[0] < w[1]), "not strictly increasing: {idx:?}");
    }
  }
}
