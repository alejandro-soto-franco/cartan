// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use crate::{gramian::Gramian, term::ExteriorTerm, ExteriorGrade};
use nalgebra as na;

/// Construct Gramian on lexicographically ordered standard k-element standard
/// basis from Gramian on single elements.
pub fn multi_gramian(single_gramian: &Gramian, grade: ExteriorGrade) -> Gramian {
  let dim = single_gramian.dim();
  let bases: Vec<_> = ExteriorTerm::basis(dim, grade).collect();

  let mut multi_gramian = na::DMatrix::zeros(bases.len(), bases.len());
  let mut multi_basis_mat = na::DMatrix::zeros(grade, grade);

  for icomb in 0..bases.len() {
    let combi = &bases[icomb];
    for jcomb in icomb..bases.len() {
      let combj = &bases[jcomb];

      for iicomb in 0..grade {
        let combii = combi[iicomb];
        for jjcomb in 0..grade {
          let combjj = combj[jjcomb];
          multi_basis_mat[(iicomb, jjcomb)] = single_gramian.basis_inner(combii, combjj);
        }
      }
      let det = multi_basis_mat.determinant();
      multi_gramian[(icomb, jcomb)] = det;
      multi_gramian[(jcomb, icomb)] = det;
    }
  }
  Gramian::new_unchecked(multi_gramian)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::combo::binomial;
  use approx::assert_relative_eq;

  #[test]
  fn multi_gramian_of_standard_metric_is_identity() {
    for n in 1..=4 {
      for k in 0..=n {
        let mg = multi_gramian(&Gramian::standard(n), k);
        let d = binomial(n, k);
        assert_eq!(mg.dim(), d, "n={n} k={k}");
        for i in 0..d {
          for j in 0..d {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(mg.basis_inner(i, j), expected, epsilon = 1e-12);
          }
        }
      }
    }
  }

  #[test]
  fn multi_gramian_scales_with_metric() {
    // Scaling the metric by c scales the grade-k inner product by c^k.
    let n = 3;
    let k = 2;
    let c = 2.0;
    let base = Gramian::standard(n);
    let scaled = Gramian::new(base.matrix().clone() * c);
    let mg_base = multi_gramian(&base, k);
    let mg_scaled = multi_gramian(&scaled, k);
    assert_relative_eq!(
      mg_scaled.basis_inner(0, 0),
      c.powi(k as i32) * mg_base.basis_inner(0, 0),
      epsilon = 1e-12
    );
  }
}
