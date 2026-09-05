//! Preconditioned conjugate gradients over an abstract vector backend.

/// Every vector operation a conjugate gradient iteration needs.
///
/// The point of the trait is that [`pcg`] never touches host memory. A device
/// implementation keeps its vectors resident and the iteration runs entirely
/// where they are. Copying a Krylov iterate across PCIe once per step costs
/// more than the operator application it feeds.
pub trait MassBackend {
    /// The vector representation this backend works in.
    type Vector;

    /// Number of degrees of freedom the operator acts on.
    fn ndofs(&self) -> usize;

    /// A zero vector of the operator's size.
    fn zeros(&self) -> Self::Vector;

    /// Upload a host slice. Its length must equal [`MassBackend::ndofs`].
    ///
    /// Takes `&self` because a device backend allocates against its own
    /// context and stream, which a free function would have no route to.
    #[allow(clippy::wrong_self_convention)]
    fn from_host(&self, src: &[f64]) -> Self::Vector;

    /// Download into a host slice of length [`MassBackend::ndofs`].
    fn to_host(&self, src: &Self::Vector, dst: &mut [f64]);

    /// `dst <- src`.
    fn copy(&self, src: &Self::Vector, dst: &mut Self::Vector);

    /// `y <- M x`, the operator itself.
    fn apply(&self, x: &Self::Vector, y: &mut Self::Vector);

    /// `z <- P^-1 r`, the preconditioner. Jacobi on the operator diagonal.
    fn precondition(&self, r: &Self::Vector, z: &mut Self::Vector);

    /// The Euclidean inner product.
    fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> f64;

    /// `y <- y + a x`.
    fn axpy(&self, a: f64, x: &Self::Vector, y: &mut Self::Vector);

    /// `y <- x + a y`. Distinct from [`MassBackend::axpy`] in which vector is
    /// scaled, which is what the search-direction update needs.
    fn aypx(&self, a: f64, x: &Self::Vector, y: &mut Self::Vector);
}

/// What one solve did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgReport {
    /// Iterations actually taken.
    pub iterations: usize,
    /// Final relative residual `||b - M x|| / ||b||`.
    pub residual: f64,
    /// Whether the relative residual reached the requested tolerance.
    pub converged: bool,
}

/// Solve `M x = b` by Jacobi-preconditioned conjugate gradients.
///
/// `x` is both the initial guess and the result. Convergence is measured on the
/// relative residual, so `tol` is scale-free. A zero right-hand side returns
/// immediately with `x` left as given, since every vector solves it.
///
/// The mass matrix is symmetric positive definite whenever the metric is
/// Riemannian and every cell has positive volume, which is what CG requires. On
/// a Lorentzian metric the pairing is indefinite and this iteration does not
/// apply.
pub fn pcg<B: MassBackend>(
    backend: &B,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iter: usize,
) -> CgReport {
    let n = backend.ndofs();
    assert_eq!(b.len(), n, "right-hand side length must match the operator");
    assert_eq!(x.len(), n, "solution length must match the operator");

    let rhs = backend.from_host(b);
    let rhs_norm = backend.dot(&rhs, &rhs).sqrt();
    if rhs_norm == 0.0 {
        return CgReport {
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    let mut xv = backend.from_host(x);
    let mut r = backend.zeros();
    let mut z = backend.zeros();
    let mut p = backend.zeros();
    let mut ap = backend.zeros();

    // r <- b - M x
    backend.apply(&xv, &mut r);
    backend.aypx(-1.0, &rhs, &mut r);

    backend.precondition(&r, &mut z);
    backend.copy(&z, &mut p);
    let mut rz = backend.dot(&r, &z);

    let mut residual = backend.dot(&r, &r).sqrt() / rhs_norm;
    let mut iterations = 0;

    while residual > tol && iterations < max_iter {
        backend.apply(&p, &mut ap);
        let p_ap = backend.dot(&p, &ap);
        if p_ap <= 0.0 {
            // The operator is not positive definite on this direction, so the
            // step length is meaningless. Stop and report the residual reached.
            break;
        }
        let alpha = rz / p_ap;

        backend.axpy(alpha, &p, &mut xv);
        backend.axpy(-alpha, &ap, &mut r);

        iterations += 1;
        residual = backend.dot(&r, &r).sqrt() / rhs_norm;
        if residual <= tol {
            break;
        }

        backend.precondition(&r, &mut z);
        let rz_next = backend.dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;

        // p <- z + beta p
        backend.aypx(beta, &z, &mut p);
    }

    backend.to_host(&xv, x);
    CgReport {
        iterations,
        residual,
        converged: residual <= tol,
    }
}
