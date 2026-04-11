//! Fiber bundle types and transport representations.
//!
//! A fiber is the "data type" attached to each vertex of a simplicial mesh.
//! The Levi-Civita connection on the base manifold induces parallel transport
//! on all associated bundles. Each fiber type defines how an SO(d) frame
//! rotation acts on its elements via the representation map `transport_by`.
//!
//! ## Trait hierarchy
//!
//! ```text
//! Fiber (abstract fiber type with SO(d) representation)
//!   |
//!   +-- U1Spin2 (nematic on 2-manifold: [q1, q2], spin-2 phase rotation)
//!   +-- TangentFiber<D> (tangent vector in R^D, fundamental representation)
//!   +-- NematicFiber3D (traceless symmetric 3x3, 5 components)
//! ```
//!
//! ## Sign convention
//!
//! The covariant Laplacian built from these fibers follows the DEC convention:
//! **positive-semidefinite** (positive at maxima). Physical equations use
//! `-K * lap` for elastic smoothing.

/// A fiber type for a fiber bundle over a simplicial mesh.
///
/// Each concrete fiber defines:
/// - `Element`: the data stored per vertex (e.g., `[f64; 2]` for a complex scalar)
/// - `FIBER_DIM`: the real dimension of the fiber
/// - `transport_by`: how an SO(d) frame rotation acts on an element
///
/// The `transport_by` method is the representation map rho: SO(d) -> GL(fiber).
/// Given the SO(d) transport matrix from the frame bundle (stored per edge by
/// `DiscreteConnection`), it computes the transported fiber element.
pub trait Fiber: Clone + Send + Sync + 'static {
    /// The element type stored at each vertex.
    type Element: Clone + Send + Sync + Default;

    /// Real dimension of the fiber.
    const FIBER_DIM: usize;

    /// Zero element of the fiber.
    fn zero() -> Self::Element;

    /// Apply an SO(d) frame rotation to a fiber element.
    ///
    /// `rotation` is a d x d orthogonal matrix in row-major flat layout
    /// (length d*d). `d` is the dimension of the base manifold's tangent space.
    fn transport_by(rotation: &[f64], d: usize, element: &Self::Element) -> Self::Element;
}

/// Component-wise operations on fiber elements.
///
/// Needed by `CovLaplacian` to accumulate weighted differences without
/// knowing the concrete array size at compile time.
pub trait FiberOps: Fiber {
    /// Accumulate: `target[i] += scale * (a[i] - b[i])` for all components.
    fn accumulate_diff(target: &mut Self::Element, a: &Self::Element, b: &Self::Element, scale: f64);

    /// Scale: `target[i] *= scale` for all components.
    fn scale_element(target: &mut Self::Element, scale: f64);
}

/// A section of a fiber bundle: one fiber element per mesh vertex.
pub trait Section<F: Fiber> {
    /// Number of vertices.
    fn n_vertices(&self) -> usize;

    /// Reference to the element at vertex `v`.
    fn at(&self, v: usize) -> &F::Element;

    /// Mutable reference to the element at vertex `v`.
    fn at_mut(&mut self, v: usize) -> &mut F::Element;
}

// ─── Vec-backed Section ──────────────────────────────────────────────────────

/// A Vec-backed section: stores one `F::Element` per vertex.
#[derive(Clone, Debug)]
pub struct VecSection<F: Fiber> {
    data: Vec<F::Element>,
    _marker: core::marker::PhantomData<F>,
}

impl<F: Fiber> VecSection<F> {
    /// Create a section with `n` zero elements.
    pub fn zeros(n: usize) -> Self {
        Self {
            data: (0..n).map(|_| F::zero()).collect(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Create from an existing Vec.
    pub fn from_vec(data: Vec<F::Element>) -> Self {
        Self { data, _marker: core::marker::PhantomData }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the section is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<F: Fiber> Section<F> for VecSection<F> {
    fn n_vertices(&self) -> usize { self.data.len() }
    fn at(&self, v: usize) -> &F::Element { &self.data[v] }
    fn at_mut(&mut self, v: usize) -> &mut F::Element { &mut self.data[v] }
}

// ─── U1Spin2 (nematic on 2-manifolds) ────────────────────────────────────────

/// Nematic (spin-2) field on a 2-manifold.
///
/// Element: `[q1, q2]` where `Q = [[q1, q2], [q2, -q1]]` (traceless symmetric 2x2).
/// Represented as a section of the complex line bundle L^2.
///
/// Under an SO(2) rotation by angle theta, the spin-2 representation acts as
/// multiplication by `e^{2i*theta}`:
///   q1' = cos(2*theta) * q1 - sin(2*theta) * q2
///   q2' = sin(2*theta) * q1 + cos(2*theta) * q2
#[derive(Clone, Debug)]
pub struct U1Spin2;

impl Fiber for U1Spin2 {
    type Element = [f64; 2];
    const FIBER_DIM: usize = 2;

    fn zero() -> [f64; 2] { [0.0, 0.0] }

    fn transport_by(rotation: &[f64], d: usize, element: &[f64; 2]) -> [f64; 2] {
        debug_assert_eq!(d, 2, "U1Spin2 requires d=2");
        debug_assert!(rotation.len() >= 4);
        // rotation row-major: [cos(t), -sin(t), sin(t), cos(t)]
        let cos_t = rotation[0];
        let sin_t = rotation[2]; // row 1, col 0
        // Spin-2: rotate by 2*theta via double-angle identities.
        let cos_2t = cos_t * cos_t - sin_t * sin_t;
        let sin_2t = 2.0 * sin_t * cos_t;
        let q1 = element[0];
        let q2 = element[1];
        [cos_2t * q1 - sin_2t * q2, sin_2t * q1 + cos_2t * q2]
    }
}

impl FiberOps for U1Spin2 {
    fn accumulate_diff(target: &mut [f64; 2], a: &[f64; 2], b: &[f64; 2], scale: f64) {
        target[0] += scale * (a[0] - b[0]);
        target[1] += scale * (a[1] - b[1]);
    }
    fn scale_element(target: &mut [f64; 2], scale: f64) {
        target[0] *= scale;
        target[1] *= scale;
    }
}

// ─── TangentFiber<D> (fundamental representation) ────────────────────────────

/// Tangent vector fiber: R^D with the fundamental SO(D) representation.
///
/// Under an SO(D) rotation R, a tangent vector v transforms as v' = R v.
#[derive(Clone, Debug)]
pub struct TangentFiber<const D: usize>;

impl<const D: usize> Fiber for TangentFiber<D>
where
    [f64; D]: Default,
{
    type Element = [f64; D];
    const FIBER_DIM: usize = D;

    fn zero() -> [f64; D] { [0.0; D] }

    fn transport_by(rotation: &[f64], d: usize, element: &[f64; D]) -> [f64; D] {
        debug_assert_eq!(d, D);
        debug_assert!(rotation.len() >= D * D);
        let mut result = [0.0; D];
        for i in 0..D {
            for j in 0..D {
                result[i] += rotation[i * D + j] * element[j];
            }
        }
        result
    }
}

impl<const D: usize> FiberOps for TangentFiber<D>
where
    [f64; D]: Default,
{
    fn accumulate_diff(target: &mut [f64; D], a: &[f64; D], b: &[f64; D], scale: f64) {
        for i in 0..D { target[i] += scale * (a[i] - b[i]); }
    }
    fn scale_element(target: &mut [f64; D], scale: f64) {
        for i in 0..D { target[i] *= scale; }
    }
}

// ─── NematicFiber3D (traceless symmetric 3x3) ────────────────────────────────

/// Nematic Q-tensor on a 3-manifold: 5-component traceless symmetric 3x3.
///
/// Element: `[Q_11, Q_12, Q_13, Q_22, Q_23]` (Q_33 = -Q_11 - Q_22, Q_ji = Q_ij).
///
/// Under SO(3) rotation R, the Q-tensor transforms as Q' = R Q R^T,
/// projected back to the 5-component traceless representation.
#[derive(Clone, Debug)]
pub struct NematicFiber3D;

impl Fiber for NematicFiber3D {
    type Element = [f64; 5];
    const FIBER_DIM: usize = 5;

    fn zero() -> [f64; 5] { [0.0; 5] }

    fn transport_by(rotation: &[f64], d: usize, element: &[f64; 5]) -> [f64; 5] {
        debug_assert_eq!(d, 3, "NematicFiber3D requires d=3");
        debug_assert!(rotation.len() >= 9);

        // Unpack 5 components to full 3x3 symmetric traceless.
        let (q11, q12, q13, q22, q23) = (element[0], element[1], element[2], element[3], element[4]);
        let q33 = -q11 - q22;
        let q = [[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]];

        let r = |i: usize, j: usize| rotation[i * 3 + j];

        // Q' = R Q R^T
        let mut qp = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        sum += r(i, a) * q[a][b] * r(j, b);
                    }
                }
                qp[i][j] = sum;
            }
        }

        // Pack back to 5 components (tracelessness preserved by orthogonal transform).
        [qp[0][0], qp[0][1], qp[0][2], qp[1][1], qp[1][2]]
    }
}

impl FiberOps for NematicFiber3D {
    fn accumulate_diff(target: &mut [f64; 5], a: &[f64; 5], b: &[f64; 5], scale: f64) {
        for i in 0..5 { target[i] += scale * (a[i] - b[i]); }
    }
    fn scale_element(target: &mut [f64; 5], scale: f64) {
        for i in 0..5 { target[i] *= scale; }
    }
}
