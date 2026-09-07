//! `H^`-invariant order parameters, built by Reynolds averaging.
//!
//! The order parameter is a harmonic (traceless symmetric) tensor `T(R)` with
//! `T(R h) = T(R)` for every `h` in `H^`. It is built by picking a reference
//! tensor `T0` in the `H^`-invariant subspace of the degree-`m` harmonics and
//! setting `T(R) = rho_m(R) T0`, where `rho_m` is the induced action on
//! degree-`m` polynomials.
//!
//! Nothing here is tabulated per group. The Reynolds projector
//! `P = (1/|H^|) sum_h rho_m(h)` is formed numerically and its image is the
//! invariant subspace, so a new symmetry needs no hand-derived tensor.
//!
//! `H^` acts on `R^3` through `SO(3)`, so `-1` acts trivially and no tensor of
//! any rank distinguishes `R` from `-R`. That is precisely the lift
//! information the rotor keeps and the tensor discards.
//!
//! ## Why the Bombieri inner product
//!
//! Rotations are not orthogonal in the plain monomial basis, so a Reynolds
//! average formed there is not a symmetric projector and its image cannot be
//! read off an eigendecomposition. Under the Bombieri (apolar) inner product,
//! in which `x^a y^b z^c` has squared norm `a! b! c! / m!`, the `SO(3)` action
//! is orthogonal. Every matrix below is written in the Bombieri-orthonormal
//! basis, which makes the harmonic projector and the Reynolds projector both
//! symmetric and lets `SymmetricEigen` do the extraction.

use nalgebra::{DMatrix, DVector};

use cartan_core::rotor::Rotor3;

use crate::error::PaticError;
use crate::group::SymmetryGroup;

/// Exponent triples `(a, b, c)` with `a + b + c = m`, in a fixed order.
#[must_use]
pub fn monomials(m: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    for a in (0..=m).rev() {
        for b in (0..=(m - a)).rev() {
            out.push([a, b, m - a - b]);
        }
    }
    out
}

/// Index of an exponent triple within `monomials(m)`.
fn index_of(basis: &[[usize; 3]], e: &[usize; 3]) -> usize {
    basis
        .iter()
        .position(|b| b == e)
        .expect("exponent not in basis")
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product::<f64>().max(1.0)
}

/// Bombieri scale `sqrt(a! b! c! / m!)` for the monomial `e` of degree `m`.
///
/// Dividing a monomial by this makes the basis orthonormal in the inner
/// product under which `SO(3)` acts orthogonally.
fn bombieri_scale(e: &[usize; 3], m: usize) -> f64 {
    (factorial(e[0]) * factorial(e[1]) * factorial(e[2]) / factorial(m)).sqrt()
}

/// Orthonormal basis of the kernel of a symmetric-positive-semidefinite
/// Gram matrix, taken as the eigenvectors below `tol`.
fn kernel_of_gram(gram: &DMatrix<f64>, tol: f64) -> DMatrix<f64> {
    let eig = gram.clone().symmetric_eigen();
    let mut cols: Vec<DVector<f64>> = Vec::new();
    for i in 0..eig.eigenvalues.len() {
        if eig.eigenvalues[i].abs() <= tol {
            cols.push(eig.eigenvectors.column(i).into_owned());
        }
    }
    if cols.is_empty() {
        DMatrix::zeros(gram.nrows(), 0)
    } else {
        DMatrix::from_columns(&cols)
    }
}

/// The matrix of `p -> p . R^{-1}` on degree-`m` polynomials, in the
/// Bombieri-orthonormal basis, where it is an orthogonal matrix.
fn rep_matrix(m: usize, r: &Rotor3) -> DMatrix<f64> {
    let basis = monomials(m);
    let n = basis.len();
    let rm = r.to_matrix(); // row-major SO(3)
    // Column j of `lin` holds the coefficients of the linear form
    // (R^{-1} v)_j = sum_k R[k][j] v_k.
    let lin = |j: usize| [rm[j], rm[3 + j], rm[6 + j]];

    let mut out = DMatrix::<f64>::zeros(n, n);
    for (col, e) in basis.iter().enumerate() {
        // Accumulate the product of linear forms as a sparse polynomial.
        let mut poly: Vec<(usize, [usize; 3], f64)> = vec![(0, [0, 0, 0], 1.0)];
        for (axis, &power) in e.iter().enumerate() {
            let l = lin(axis);
            for _ in 0..power {
                let mut next: Vec<(usize, [usize; 3], f64)> = Vec::new();
                for (_, ex, c) in &poly {
                    for (k, &lk) in l.iter().enumerate() {
                        if lk == 0.0 {
                            continue;
                        }
                        let mut ex2 = *ex;
                        ex2[k] += 1;
                        next.push((0, ex2, c * lk));
                    }
                }
                poly = next;
            }
        }
        for (_, ex, c) in poly {
            out[(index_of(&basis, &ex), col)] += c;
        }
    }
    // Convert to the Bombieri-orthonormal basis: S rho S^{-1}.
    let s: Vec<f64> = basis.iter().map(|e| bombieri_scale(e, m)).collect();
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] *= s[i] / s[j];
        }
    }
    out
}

/// A basis for the harmonic (traceless) degree-`m` polynomials, orthonormal in
/// the Bombieri inner product, as columns in that basis.
///
/// The harmonic subspace is the kernel of the Laplacian `Sym^m -> Sym^{m-2}`
/// and has dimension `2m + 1`.
#[must_use]
pub fn harmonic_basis(m: usize) -> DMatrix<f64> {
    let basis = monomials(m);
    let n = basis.len();
    if m < 2 {
        return DMatrix::identity(n, n);
    }
    let lower = monomials(m - 2);
    let mut lap = DMatrix::<f64>::zeros(lower.len(), n);
    for (col, e) in basis.iter().enumerate() {
        for axis in 0..3 {
            if e[axis] >= 2 {
                let mut d = *e;
                let coeff = (d[axis] * (d[axis] - 1)) as f64;
                d[axis] -= 2;
                lap[(index_of(&lower, &d), col)] += coeff;
            }
        }
    }
    // Rewrite the Laplacian in the Bombieri-orthonormal bases of both degrees.
    let su: Vec<f64> = basis.iter().map(|e| bombieri_scale(e, m)).collect();
    let sl: Vec<f64> = lower.iter().map(|e| bombieri_scale(e, m - 2)).collect();
    for i in 0..lower.len() {
        for j in 0..n {
            lap[(i, j)] *= sl[i] / su[j];
        }
    }
    // Kernel of a rectangular map as the kernel of its Gram matrix.
    let gram = lap.transpose() * &lap;
    kernel_of_gram(&gram, 1e-8)
}

/// The three `so(3)` generators acting on degree-`m` harmonics, in the
/// Bombieri-orthonormal basis.
///
/// Computed by central difference of the group action, which is exact to
/// `O(eps^2)` and adequate against the tolerances the flow compares at.
#[must_use]
pub fn so3_generators(m: usize) -> [DMatrix<f64>; 3] {
    let eps: f64 = 1e-6;
    core::array::from_fn(|axis| {
        let mut v = [0.0; 3];
        v[axis] = (eps / 2.0).sin();
        let c = (eps / 2.0).cos();
        let fwd = Rotor3 {
            w: c,
            x: v[0],
            y: v[1],
            z: v[2],
        };
        let bwd = Rotor3 {
            w: c,
            x: -v[0],
            y: -v[1],
            z: -v[2],
        };
        // The two rotors are at rotation angles +eps and -eps, so the central
        // difference spans 2 eps.
        (rep_matrix(m, &fwd) - rep_matrix(m, &bwd)) / (2.0 * eps)
    })
}

/// Matrix exponential by scaling and squaring with a Taylor series.
///
/// The matrices here are `(2m+1)` square with `m` at most 6, so a short series
/// after scaling is both accurate and faster than any general routine.
fn expm(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    // The 1-norm is a valid bound on the spectral radius and far tighter than
    // a max-entry estimate scaled by the size, which over-counted squarings.
    let norm = (0..n)
        .map(|j| (0..n).map(|i| a[(i, j)].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };
    let scale = 1.0 / f64::from(1u32 << squarings);
    let b = a * scale;
    let mut term = DMatrix::<f64>::identity(n, n);
    let mut out = DMatrix::<f64>::identity(n, n);
    for k in 1..=12 {
        term = &term * &b / f64::from(k);
        out += &term;
    }
    for _ in 0..squarings {
        out = &out * &out;
    }
    out
}

/// Axis-angle of a rotor: the rotation angle and its unit axis.
fn axis_angle(r: &Rotor3) -> (f64, [f64; 3]) {
    let v = (r.x * r.x + r.y * r.y + r.z * r.z).sqrt();
    if v < 1e-300 {
        return (0.0, [1.0, 0.0, 0.0]);
    }
    let theta = 2.0 * v.atan2(r.w);
    (theta, [r.x / v, r.y / v, r.z / v])
}

/// The `H^`-invariant order parameter of degree `m`.
///
/// `basis` holds an orthonormal basis of the invariant subspace, as columns in
/// the monomial coordinates of degree `m`. Its width is the number of
/// independent amplitudes.
#[derive(Clone, Debug)]
pub struct InvariantBasis {
    degree: usize,
    /// Invariant basis in monomial coordinates.
    basis: DMatrix<f64>,
    /// Orthonormal harmonic frame, monomial coordinates by harmonic index.
    harmonic: DMatrix<f64>,
    /// Invariant basis in harmonic coordinates.
    basis_h: DMatrix<f64>,
    /// `so(3)` generators restricted to the harmonic subspace.
    gens_h: [DMatrix<f64>; 3],
}

impl InvariantBasis {
    /// Build the invariant subspace of degree `m` for `H`.
    ///
    /// Returns an empty basis when the symmetry admits no invariant of that
    /// degree.
    #[must_use]
    pub fn new<H: SymmetryGroup>(m: usize) -> Self {
        let harmonic = harmonic_basis(m);
        let elements: Vec<Rotor3> = H::elements().collect();
        let dim = harmonic.ncols();

        // Reynolds projector restricted to the harmonic subspace.
        // For the continuous families the group is not enumerable, so average
        // over a fine sampling of the axial stabiliser instead.
        let reps: Vec<Rotor3> = if elements.is_empty() {
            axial_samples::<H>()
        } else {
            elements
        };

        let mut p = DMatrix::<f64>::zeros(dim, dim);
        for r in &reps {
            let rho = rep_matrix(m, r);
            // Restrict: harmonic^T * rho * harmonic, valid since the harmonic
            // subspace is SO(3)-invariant and the basis is orthonormal.
            p += harmonic.transpose() * &rho * &harmonic;
        }
        p /= reps.len() as f64;

        // In the Bombieri basis every rho is orthogonal and the harmonic basis
        // is orthonormal, so p is a symmetric projector. Its image is the
        // eigenspace at eigenvalue 1.
        let eig = p.symmetric_eigen();
        let mut cols: Vec<DVector<f64>> = Vec::new();
        for i in 0..eig.eigenvalues.len() {
            if eig.eigenvalues[i] > 0.5 {
                cols.push(&harmonic * eig.eigenvectors.column(i));
            }
        }
        let basis = if cols.is_empty() {
            DMatrix::zeros(harmonic.nrows(), 0)
        } else {
            DMatrix::from_columns(&cols)
        };
        let basis_h = harmonic.transpose() * &basis;
        let full = so3_generators(m);
        let gens_h = core::array::from_fn(|a| harmonic.transpose() * &full[a] * &harmonic);
        Self {
            degree: m,
            basis,
            harmonic,
            basis_h,
            gens_h,
        }
    }

    /// Degree of the harmonics this basis lives in.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Number of independent amplitudes.
    #[must_use]
    pub fn n_amplitudes(&self) -> usize {
        self.basis.ncols()
    }

    /// The action of `r` on degree-`degree` harmonics, in monomial
    /// coordinates. Kept for reference and tests.
    #[must_use]
    pub fn rep(&self, r: &Rotor3) -> DMatrix<f64> {
        rep_matrix(self.degree, r)
    }

    /// The action of `r` restricted to the harmonic subspace.
    ///
    /// Obtained by exponentiating the precomputed generators rather than by
    /// expanding monomials, so the cost is a `(2m+1)` square exponential
    /// rather than a `(m+1)(m+2)/2` square polynomial substitution. At degree
    /// 6 that is 13 against 28.
    #[must_use]
    pub fn rep_harmonic(&self, r: &Rotor3) -> DMatrix<f64> {
        let (theta, axis) = axis_angle(r);
        let mut g = &self.gens_h[0] * (theta * axis[0]);
        g += &self.gens_h[1] * (theta * axis[1]);
        g += &self.gens_h[2] * (theta * axis[2]);
        expm(&g)
    }

    /// The harmonic frame: monomial coordinates by harmonic index.
    #[must_use]
    pub fn harmonic_frame(&self) -> &DMatrix<f64> {
        &self.harmonic
    }

    /// Column `i` of the invariant basis, in harmonic coordinates.
    #[must_use]
    pub fn basis_column_harmonic(&self, i: usize) -> DVector<f64> {
        self.basis_h.column(i).into_owned()
    }

    /// The order parameter as a fully symmetric rank-`m` tensor, flattened in
    /// base 3 so entry `i1 + 3 i2 + 9 i3 + ...` is `T_{i1 i2 i3 ...}`.
    ///
    /// A harmonic polynomial `p(v) = T_{i1..im} v_i1 .. v_im` with `T`
    /// symmetric has `T` equal to the monomial coefficient divided by the
    /// multinomial count of the index tuple.
    #[must_use]
    pub fn as_tensor(&self, t: &DVector<f64>) -> Vec<f64> {
        let m = self.degree;
        let basis = monomials(m);
        let mut coeff = vec![0.0_f64; basis.len()];
        for (i, e) in basis.iter().enumerate() {
            coeff[i] = t[i] / bombieri_scale(e, m);
        }
        let size = 3usize.pow(m as u32);
        let mut out = vec![0.0_f64; size];
        for (flat, o) in out.iter_mut().enumerate() {
            let mut e = [0usize; 3];
            let mut r = flat;
            for _ in 0..m {
                e[r % 3] += 1;
                r /= 3;
            }
            let idx = basis
                .iter()
                .position(|b| *b == e)
                .expect("every exponent triple of degree m is in the basis");
            let multinomial = factorial(m) / (factorial(e[0]) * factorial(e[1]) * factorial(e[2]));
            *o = coeff[idx] / multinomial;
        }
        out
    }

    /// The degree-2 order parameter as a symmetric traceless 3x3 matrix.
    ///
    /// Returns `None` at any other degree, where the order parameter has no
    /// rank-2 representation at all. That absence is the whole reason the
    /// active stress needs derivatives for `p > 2`.
    #[must_use]
    pub fn as_matrix3(&self, t: &DVector<f64>) -> Option<[[f64; 3]; 3]> {
        if self.degree != 2 {
            return None;
        }
        let basis = monomials(2);
        let mut c = [0.0_f64; 6];
        for (i, e) in basis.iter().enumerate() {
            c[i] = t[i] / bombieri_scale(e, 2);
        }
        // monomials(2) is ordered xx, xy, xz, yy, yz, zz, and the quadratic
        // form v^T Q v doubles every off-diagonal monomial.
        Some([
            [c[0], c[1] / 2.0, c[2] / 2.0],
            [c[1] / 2.0, c[3], c[4] / 2.0],
            [c[2] / 2.0, c[4] / 2.0, c[5]],
        ])
    }

    /// The reference tensor in harmonic coordinates.
    #[must_use]
    pub fn reference_harmonic(&self, amplitudes: &[f64]) -> DVector<f64> {
        debug_assert_eq!(amplitudes.len(), self.n_amplitudes());
        let mut t0 = DVector::<f64>::zeros(self.basis_h.nrows());
        for (j, a) in amplitudes.iter().enumerate() {
            t0 += self.basis_h.column(j) * *a;
        }
        t0
    }

    /// The reference tensor `T0`: the amplitude-weighted invariant basis,
    /// before any frame is applied.
    #[must_use]
    pub fn reference(&self, amplitudes: &[f64]) -> DVector<f64> {
        debug_assert_eq!(amplitudes.len(), self.n_amplitudes());
        let mut t0 = DVector::<f64>::zeros(self.basis.nrows());
        for (j, a) in amplitudes.iter().enumerate() {
            t0 += self.basis.column(j) * *a;
        }
        t0
    }

    /// Column `i` of the invariant basis.
    #[must_use]
    pub fn basis_column(&self, i: usize) -> DVector<f64> {
        self.basis.column(i).into_owned()
    }

    /// The order parameter `T(R) = rho_m(R) T0` in monomial coordinates, with
    /// `T0` the amplitude-weighted combination of the invariant basis.
    #[must_use]
    pub fn order_parameter(&self, r: &Rotor3, amplitudes: &[f64]) -> DVector<f64> {
        rep_matrix(self.degree, r) * self.reference(amplitudes)
    }
}

impl InvariantBasis {
    /// Build the invariant basis at the symmetry's declared rank, checking the
    /// declared constants against the computed invariant theory.
    ///
    /// The compile-time amplitude count cannot be derived from a const generic
    /// in general, so it is declared by a closed form and verified here. A
    /// symmetry whose formula is wrong fails at first use rather than
    /// silently indexing past its meaningful amplitudes.
    pub fn for_group<H: SymmetryGroup>() -> Result<Self, PaticError> {
        let computed_rank = separating_degree::<H>(H::INVARIANT_RANK.max(8))?;
        let b = Self::new::<H>(computed_rank);
        if computed_rank != H::INVARIANT_RANK || b.n_amplitudes() != H::N_AMPLITUDES {
            return Err(PaticError::ConstMismatch {
                group: core::any::type_name::<H>(),
                declared_rank: H::INVARIANT_RANK,
                declared_amplitudes: H::N_AMPLITUDES,
                computed_rank,
                computed_amplitudes: b.n_amplitudes(),
            });
        }
        Ok(b)
    }
}

/// The lowest degree whose invariant subspace is non-trivial and separates
/// cosets, searching up to `max_degree`.
///
/// Separation is decided by sampling: two rotors whose order parameters agree
/// must differ by an element of `H^` up to sign, since the tensor cannot see
/// the lift.
pub fn separating_degree<H: SymmetryGroup>(max_degree: usize) -> Result<usize, PaticError> {
    for m in 1..=max_degree {
        let b = InvariantBasis::new::<H>(m);
        if b.n_amplitudes() == 0 {
            continue;
        }
        if separates::<H>(&b) {
            return Ok(m);
        }
    }
    Err(PaticError::NoSeparatingInvariant {
        max_rank: max_degree,
    })
}

/// Dimension of the continuous part of the stabiliser of `t0`.
///
/// Sampling cannot find a rotation axis that is not in the probe set, and the
/// axis of a generic invariant is not a coordinate axis. The infinitesimal
/// test is exact: the stabiliser has positive dimension exactly when some
/// `so(3)` generator annihilates `t0`, so the rank of the three-column map
/// `xi -> d rho(xi) t0` decides it.
fn continuous_stabiliser_dim(m: usize, t0: &DVector<f64>) -> usize {
    let eps: f64 = 1e-5;
    let mut cols: Vec<DVector<f64>> = Vec::new();
    for axis in 0..3 {
        let mut v = [0.0; 3];
        v[axis] = (eps / 2.0).sin();
        let c = (eps / 2.0).cos();
        let fwd = Rotor3 {
            w: c,
            x: v[0],
            y: v[1],
            z: v[2],
        };
        let bwd = Rotor3 {
            w: c,
            x: -v[0],
            y: -v[1],
            z: -v[2],
        };
        let d = (rep_matrix(m, &fwd) * t0 - rep_matrix(m, &bwd) * t0) / eps;
        cols.push(d);
    }
    let mat = DMatrix::from_columns(&cols);
    let gram = mat.transpose() * &mat;
    let scale = t0.norm().max(1e-12);
    3 - kernel_of_gram(&gram, 1e-8 * scale * scale).ncols()
}

/// Value of the harmonic polynomial whose Bombieri coordinates are `t0`.
fn eval_poly(m: usize, t0: &DVector<f64>, v: [f64; 3]) -> f64 {
    let basis = monomials(m);
    let mut acc = 0.0;
    for (i, e) in basis.iter().enumerate() {
        let c = t0[i] / bombieri_scale(e, m);
        acc += c * v[0].powi(e[0] as i32) * v[1].powi(e[1] as i32) * v[2].powi(e[2] as i32);
    }
    acc
}

/// Gradient of the harmonic polynomial at `v`.
fn grad_poly(m: usize, t0: &DVector<f64>, v: [f64; 3]) -> [f64; 3] {
    let basis = monomials(m);
    let mut g = [0.0; 3];
    for (i, e) in basis.iter().enumerate() {
        let c = t0[i] / bombieri_scale(e, m);
        for k in 0..3 {
            if e[k] == 0 {
                continue;
            }
            let mut d = *e;
            d[k] -= 1;
            g[k] += c
                * (e[k] as f64)
                * v[0].powi(d[0] as i32)
                * v[1].powi(d[1] as i32)
                * v[2].powi(d[2] as i32);
        }
    }
    g
}

fn normalise(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Ascend `|p|` to a critical point on the sphere.
///
/// A grid gives an axis only to its own resolution, which is nowhere near the
/// tolerance the stabiliser test compares against, so each candidate is
/// refined by projected gradient ascent before it is used.
fn refine_axis(m: usize, t0: &DVector<f64>, mut v: [f64; 3]) -> [f64; 3] {
    let sign = if eval_poly(m, t0, v) < 0.0 { -1.0 } else { 1.0 };
    let mut step = 0.2;
    for _ in 0..400 {
        let g = grad_poly(m, t0, v);
        let dot = g[0] * v[0] + g[1] * v[1] + g[2] * v[2];
        let tang = [g[0] - dot * v[0], g[1] - dot * v[1], g[2] - dot * v[2]];
        let tn = (tang[0] * tang[0] + tang[1] * tang[1] + tang[2] * tang[2]).sqrt();
        if tn < 1e-14 {
            break;
        }
        let trial = normalise([
            v[0] + sign * step * tang[0],
            v[1] + sign * step * tang[1],
            v[2] + sign * step * tang[2],
        ]);
        if sign * eval_poly(m, t0, trial) > sign * eval_poly(m, t0, v) {
            v = trial;
        } else {
            step *= 0.5;
            if step < 1e-15 {
                break;
            }
        }
    }
    v
}

/// Directions where `|p|` is largest, as candidate symmetry axes.
///
/// A discrete stabiliser is generated by rotations about the invariant's own
/// axes, and those are generically not coordinate axes, so probing fixed axes
/// misses them. The extrema of `|p|` on the sphere recover them.
fn candidate_axes(m: usize, t0: &DVector<f64>, n_keep: usize) -> Vec<[f64; 3]> {
    let n = 4000usize;
    let ga = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut scored: Vec<(f64, [f64; 3])> = Vec::with_capacity(n);
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / (n as f64);
        let r = (1.0 - z * z).max(0.0).sqrt();
        let th = ga * (i as f64);
        let v = [r * th.cos(), r * th.sin(), z];
        scored.push((eval_poly(m, t0, v).abs(), v));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let mut axes: Vec<[f64; 3]> = Vec::new();
    for (_, v) in scored {
        if axes.len() >= n_keep {
            break;
        }
        let dup = axes.iter().any(|a| {
            let d = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
            d.abs() > 0.999
        });
        if !dup {
            axes.push(refine_axis(m, t0, v));
        }
    }
    axes
}

/// Rotations by `2 pi / k` about each candidate axis, for small `k`.
fn axis_probes(axes: &[[f64; 3]]) -> Vec<Rotor3> {
    let mut out = Vec::new();
    for a in axes {
        for k in 2..=12 {
            let half = core::f64::consts::PI / (k as f64);
            let (sn, cs) = half.sin_cos();
            out.push(Rotor3 {
                w: cs,
                x: sn * a[0],
                y: sn * a[1],
                z: sn * a[2],
            });
        }
    }
    out
}

/// Whether the order parameter distinguishes cosets.
///
/// By equivariance it is enough to test at the identity: the stabiliser of
/// `T0` must be no larger than the image of `H^` in `SO(3)`. A rotation that
/// fixes `T0` while lying outside `H^` (up to the sign the tensor cannot see)
/// means the tensor has merged two cosets.
///
/// The probe must be dense along the coordinate axes, because the failure this
/// catches is a uniaxial invariant whose stabiliser is the whole continuous
/// rotation group about its axis.
fn separates<H: SymmetryGroup>(b: &InvariantBasis) -> bool {
    let amps: Vec<f64> = (0..b.n_amplitudes())
        .map(|i| 1.0 + (i as f64) * core::f64::consts::FRAC_1_SQRT_2)
        .collect();
    let t0 = b.order_parameter(&Rotor3::IDENTITY, &amps);
    let scale = t0.norm().max(1e-12);

    // A finite group cannot be separated by an invariant whose stabiliser has
    // positive dimension. The two continuous families are meant to have one.
    if H::ORDER.is_some() && continuous_stabiliser_dim(b.degree(), &t0) < 3 {
        return false;
    }

    let mut probes = probe_rotors();
    probes.extend(axis_probes(&candidate_axes(b.degree(), &t0, 16)));

    for g in probes {
        let t = b.order_parameter(&g, &amps);
        if (t - &t0).norm() / scale > 1e-7 {
            continue;
        }
        let neg = Rotor3 {
            w: -g.w,
            x: -g.x,
            y: -g.y,
            z: -g.z,
        };
        if !H::contains(&g, 1e-6) && !H::contains(&neg, 1e-6) {
            return false;
        }
    }
    true
}

/// A dense spread of rotors for the stabiliser probe: fine rotations about
/// each coordinate axis, plus a quasi-random spread over `S^3`.
fn probe_rotors() -> Vec<Rotor3> {
    let mut out = Vec::new();
    let n = 180;
    for axis in 0..3 {
        for m in 1..n {
            let half = core::f64::consts::PI * (m as f64) / (n as f64);
            let (s, c) = half.sin_cos();
            let mut v = [0.0; 3];
            v[axis] = s;
            out.push(Rotor3 {
                w: c,
                x: v[0],
                y: v[1],
                z: v[2],
            });
        }
    }
    for i in 1..200 {
        let t = i as f64 * 0.6180339887498949;
        let u = i as f64 * 0.4142135623730951;
        let r = Rotor3 {
            w: (t * 3.0).cos(),
            x: (t * 5.0).sin(),
            y: (u * 7.0).cos(),
            z: (u * 11.0).sin(),
        };
        out.push(r.normalized());
    }
    out
}

/// Rotations sampling the continuous axial stabiliser about `z`, plus the
/// perpendicular flip for the apolar case.
fn axial_samples<H: SymmetryGroup>() -> Vec<Rotor3> {
    use crate::group::PointGroupKind;
    let n = 240;
    let mut out = Vec::with_capacity(2 * n);
    for m in 0..n {
        let theta = core::f64::consts::PI * (m as f64) / (n as f64);
        let (s, c) = theta.sin_cos();
        // Rotation about z by 2*theta.
        out.push(Rotor3 {
            w: c,
            x: 0.0,
            y: 0.0,
            z: s,
        });
    }
    if H::kind() == PointGroupKind::AxialApolar {
        let flip = Rotor3 {
            w: 0.0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let half: Vec<Rotor3> = out.clone();
        for r in half {
            out.push(r.compose(&flip));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{
        AxialApolar, AxialPolar, BinaryIcosahedral, BinaryOctahedral, BinaryTetrahedral, Cyclic,
        Dicyclic,
    };

    #[test]
    fn harmonic_dimension_is_two_m_plus_one() {
        for m in 0..=7 {
            assert_eq!(harmonic_basis(m).ncols(), 2 * m + 1, "degree {m}");
        }
    }

    #[test]
    fn rotations_are_orthogonal_in_the_bombieri_basis() {
        let r = Rotor3 {
            w: 0.5,
            x: 0.5,
            y: 0.5,
            z: 0.5,
        };
        for m in 1..=5 {
            let rho = rep_matrix(m, &r);
            let n = rho.ncols();
            let err = (rho.transpose() * &rho - DMatrix::<f64>::identity(n, n)).norm();
            assert!(err < 1e-10, "degree {m}: orthogonality error {err:e}");
        }
    }

    /// The declared constants must equal the computed invariant theory for
    /// every symmetry the crate ships, and across both const-generic families.
    #[test]
    fn declared_constants_match_the_computation() {
        fn check<H: SymmetryGroup>() {
            if let Err(e) = InvariantBasis::for_group::<H>() {
                panic!("{e}");
            }
        }
        check::<AxialPolar>();
        check::<AxialApolar>();
        check::<BinaryTetrahedral>();
        check::<BinaryOctahedral>();
        check::<BinaryIcosahedral>();
        check::<Cyclic<1>>();
        check::<Cyclic<2>>();
        check::<Cyclic<3>>();
        check::<Cyclic<4>>();
        check::<Cyclic<5>>();
        check::<Cyclic<6>>();
        check::<Cyclic<7>>();
        check::<Cyclic<8>>();
        check::<Dicyclic<1>>();
        check::<Dicyclic<2>>();
        check::<Dicyclic<3>>();
        check::<Dicyclic<4>>();
        check::<Dicyclic<5>>();
        check::<Dicyclic<6>>();
        check::<Dicyclic<7>>();
        check::<Dicyclic<8>>();
    }

    /// The uniaxial case must reproduce the Q-tensor: five components, one
    /// amplitude, degree 2.
    #[test]
    fn axial_apolar_is_the_q_tensor() {
        let b = InvariantBasis::for_group::<AxialApolar>().expect("uniaxial consts");
        assert_eq!(b.degree(), 2);
        assert_eq!(b.n_amplitudes(), 1);
        assert_eq!(harmonic_basis(2).ncols(), 5);
    }

    /// The exponential route must reproduce the monomial route restricted to
    /// the harmonic subspace, or the speed-up is buying a different operator.
    #[test]
    fn harmonic_rep_matches_the_monomial_rep() {
        for r in [
            Rotor3 {
                w: 0.5,
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
            Rotor3 {
                w: 0.6,
                x: 0.8,
                y: 0.0,
                z: 0.0,
            },
            Rotor3 {
                w: -0.2,
                x: 0.3,
                y: -0.5,
                z: 0.7893671,
            }
            .normalized(),
            Rotor3::IDENTITY,
        ] {
            for m in 1..=6 {
                let b = InvariantBasis::new::<AxialPolar>(m);
                let h = b.harmonic_frame();
                let direct = h.transpose() * b.rep(&r) * h;
                let viaexp = b.rep_harmonic(&r);
                let err = (&direct - &viaexp).norm() / direct.norm().max(1e-12);
                assert!(err < 1e-8, "degree {m}: relative error {err:e}");
            }
        }
    }

    #[test]
    fn report_invariant_dimensions() {
        fn row<H: SymmetryGroup>(name: &str) {
            let dims: Vec<usize> = (1..=7)
                .map(|m| InvariantBasis::new::<H>(m).n_amplitudes())
                .collect();
            let sep = separating_degree::<H>(8);
            println!("  {name:<20} m=1..7 dims {dims:?}  separating {sep:?}");
        }
        println!();
        row::<AxialPolar>("AxialPolar");
        row::<AxialApolar>("AxialApolar");
        row::<Cyclic<2>>("Cyclic<2>");
        row::<Cyclic<3>>("Cyclic<3>");
        row::<Dicyclic<1>>("Dicyclic<1>");
        row::<Dicyclic<2>>("Dicyclic<2>");
        row::<Dicyclic<3>>("Dicyclic<3>");
        row::<Dicyclic<4>>("Dicyclic<4>");
        row::<BinaryTetrahedral>("BinaryTetrahedral");
        row::<BinaryOctahedral>("BinaryOctahedral");
        row::<BinaryIcosahedral>("BinaryIcosahedral");
    }
}

#[cfg(test)]
mod family_report {
    use super::super::group::{Cyclic, Dicyclic, SymmetryGroup};
    use super::super::invariant::{InvariantBasis, separating_degree};

    fn line<H: SymmetryGroup>(name: &str) {
        match separating_degree::<H>(10) {
            Ok(m) => {
                let a = InvariantBasis::new::<H>(m).n_amplitudes();
                println!(
                    "  {name:<14} separating {m}  amplitudes {a}  declared rank {} amps {}",
                    H::INVARIANT_RANK,
                    H::N_AMPLITUDES
                );
            }
            Err(e) => println!("  {name:<14} none: {e}"),
        }
    }

    #[test]
    fn family_scan() {
        println!();
        line::<Cyclic<1>>("Cyclic<1>");
        line::<Cyclic<2>>("Cyclic<2>");
        line::<Cyclic<3>>("Cyclic<3>");
        line::<Cyclic<4>>("Cyclic<4>");
        line::<Cyclic<5>>("Cyclic<5>");
        line::<Cyclic<6>>("Cyclic<6>");
        line::<Dicyclic<1>>("Dicyclic<1>");
        line::<Dicyclic<2>>("Dicyclic<2>");
        line::<Dicyclic<3>>("Dicyclic<3>");
        line::<Dicyclic<4>>("Dicyclic<4>");
        line::<Dicyclic<5>>("Dicyclic<5>");
        line::<Dicyclic<6>>("Dicyclic<6>");
    }
}
