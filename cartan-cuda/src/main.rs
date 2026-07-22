//! Batched sphere geometry on the GPU, in double precision, checked against
//! the CPU implementation it is meant to match.
//!
//! The point of this crate is precision. WGSL, which `cartan-gpu` targets, has
//! no `f64`, so anything routed through it is single precision while the rest
//! of cartan is double. CUDA has `f64` natively, so these kernels can be held
//! to the same tolerance the CPU code is: agreement here is checked at 1e-13,
//! not at the 1e-6 an f32 path would force.
//!
//! Kernels are written in ordinary Rust and compiled to PTX by
//! `rustc-codegen-cuda`. Run with:
//!
//! ```text
//! cargo oxide run cartan-cuda
//! ```
//!
//! ## Why each operation is two kernels
//!
//! `DisjointSlice` gives each thread exactly one output element, which is what
//! makes the writes provably non-overlapping. A batched `exp` produces `dim`
//! elements per point, so it does not fit that shape directly.
//!
//! Splitting it does. The first kernel runs one thread per point and performs
//! the O(dim) reduction, writing a single scalar. The second runs one thread
//! per output element and consumes that scalar. The reduction happens once per
//! point rather than once per element, which is the whole reason not to fold
//! them together.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Geodesic length of each tangent vector: one thread per point.
    ///
    /// This is the O(dim) reduction, kept out of the per-element kernel so it
    /// runs once per point instead of once per component.
    #[kernel]
    pub fn sphere_tangent_norm(v: &[f64], dim: u32, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let d = dim as usize;

        let mut sum_sq = 0.0f64;
        for k in 0..d {
            let x = v[i * d + k];
            sum_sq += x * x;
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = sum_sq.sqrt();
        }
    }

    /// `Exp_p(v) = cos(t) p + sin(t)/t v`, one thread per output component.
    ///
    /// `t` comes in precomputed. Below the small-angle cutoff the series
    /// `sin(t)/t -> 1` is used, matching what the CPU path does rather than
    /// dividing by something near zero.
    #[kernel]
    pub fn sphere_exp_apply(
        p: &[f64],
        v: &[f64],
        theta: &[f64],
        dim: u32,
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        let d = dim as usize;
        let t = theta[j / d];

        let (a, b) = if t < 1e-7 {
            (1.0 - 0.5 * t * t, 1.0)
        } else {
            (t.cos(), t.sin() / t)
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = a * p[j] + b * v[j];
        }
    }

    /// Inner product of each pair: one thread per point.
    ///
    /// Clamped into [-1, 1] here rather than at the point of use, so the
    /// per-element kernel never hands `acos` an out-of-range argument.
    #[kernel]
    pub fn sphere_cos_angle(p: &[f64], q: &[f64], dim: u32, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let d = dim as usize;

        let mut c = 0.0f64;
        for k in 0..d {
            c += p[i * d + k] * q[i * d + k];
        }
        if c > 1.0 {
            c = 1.0;
        } else if c < -1.0 {
            c = -1.0;
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = c;
        }
    }

    /// `Log_p(q) = t (q - cos(t) p) / sin(t)`, one thread per output component.
    ///
    /// `t = acos(c)` is recomputed per element rather than stored: it is a
    /// couple of scalar operations against a second global array and a second
    /// launch.
    #[kernel]
    pub fn sphere_log_apply(
        p: &[f64],
        q: &[f64],
        cos_angle: &[f64],
        dim: u32,
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        let d = dim as usize;

        let c = cos_angle[j / d];
        let t = c.acos();

        // At t -> 0 the geodesic degenerates and the projection q - c p is
        // already the answer to first order, so the scale factor is 1.
        let k = if t < 1e-7 { 1.0 } else { t / t.sin() };

        if let Some(slot) = out.get_mut(idx) {
            *slot = (q[j] - c * p[j]) * k;
        }
    }
}

/// A batch of unit vectors and tangents, laid out row-major with stride `dim`.
struct Batch {
    dim: usize,
    n: usize,
    p: Vec<f64>,
    v: Vec<f64>,
    q: Vec<f64>,
}

/// Build a batch whose tangents sit well inside the injectivity radius, so
/// `exp` and `log` are genuinely inverse and the comparison is meaningful.
fn make_batch(dim: usize, n: usize, seed: u64) -> Batch {
    use cartan_core::Manifold;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut p = Vec::with_capacity(n * dim);
    let mut v = Vec::with_capacity(n * dim);
    let mut q = Vec::with_capacity(n * dim);

    macro_rules! fill {
        ($n:literal) => {{
            let m = cartan_manifolds::Sphere::<$n>;
            for _ in 0..n {
                let pi = m.random_point(&mut rng);
                // Scaled to 0.7 radians: comfortably short of the cut locus at
                // pi, where log stops being unique.
                let vi = m.random_tangent(&pi, &mut rng).normalize() * 0.7;
                let qi = m.exp(&pi, &vi);
                p.extend_from_slice(pi.as_slice());
                v.extend_from_slice(vi.as_slice());
                q.extend_from_slice(qi.as_slice());
            }
        }};
    }

    match dim {
        3 => fill!(3),
        10 => fill!(10),
        50 => fill!(50),
        _ => panic!("no CPU reference wired for dim {dim}"),
    }

    Batch { dim, n, p, v, q }
}

/// CPU reference, computed through the same library everything else uses.
fn cpu_reference(b: &Batch) -> (Vec<f64>, Vec<f64>) {
    use cartan_core::Manifold;

    let mut exp_ref = Vec::with_capacity(b.n * b.dim);
    let mut log_ref = Vec::with_capacity(b.n * b.dim);

    macro_rules! run {
        ($n:literal) => {{
            let m = cartan_manifolds::Sphere::<$n>;
            for i in 0..b.n {
                let s = i * b.dim;
                let pi = nalgebra::SVector::<f64, $n>::from_column_slice(&b.p[s..s + b.dim]);
                let vi = nalgebra::SVector::<f64, $n>::from_column_slice(&b.v[s..s + b.dim]);
                let qi = nalgebra::SVector::<f64, $n>::from_column_slice(&b.q[s..s + b.dim]);
                exp_ref.extend_from_slice(m.exp(&pi, &vi).as_slice());
                log_ref.extend_from_slice(m.log(&pi, &qi).unwrap().as_slice());
            }
        }};
    }

    match b.dim {
        3 => run!(3),
        10 => run!(10),
        50 => run!(50),
        _ => unreachable!(),
    }

    (exp_ref, log_ref)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn main() {
    let ctx = CudaContext::new(0).expect("no CUDA device");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("failed to load the compiled module");

    // The CPU exp renormalises its result and the GPU kernel does not, so the
    // two differ by whatever that correction is worth. On an exactly tangent
    // input that is a few ulp, which is why the bound is 1e-13 rather than
    // machine epsilon.
    const TOL: f64 = 1e-13;

    let mut worst_exp = 0.0f64;
    let mut worst_log = 0.0f64;
    let mut failed = false;

    println!("cartan-cuda: batched sphere geometry, double precision\n");
    println!(
        "{:>5}  {:>8}  {:>12}  {:>12}",
        "dim", "points", "exp max err", "log max err"
    );
    println!("{}", "-".repeat(44));

    for &(dim, n) in &[(3usize, 4096usize), (10, 4096), (50, 2048)] {
        let b = make_batch(dim, n, 42);
        let (exp_ref, log_ref) = cpu_reference(&b);

        let p_dev = DeviceBuffer::from_host(&stream, &b.p).unwrap();
        let v_dev = DeviceBuffer::from_host(&stream, &b.v).unwrap();
        let q_dev = DeviceBuffer::from_host(&stream, &b.q).unwrap();

        let elems = b.n * b.dim;

        // exp: reduce per point, then combine per element.
        let mut theta = DeviceBuffer::<f64>::zeroed(&stream, b.n).unwrap();
        // SAFETY: every buffer below is sized from `b.n` and `b.dim`, the same
        // values passed as the kernel's `dim` argument and used to size the
        // launch, so no thread indexes outside its allocation.
        unsafe {
            module
                .sphere_tangent_norm(
                &stream,
                LaunchConfig::for_num_elems(b.n as u32),
                &v_dev,
                b.dim as u32,
                &mut theta,
                )
                .expect("tangent_norm launch");
        }

        let mut exp_dev = DeviceBuffer::<f64>::zeroed(&stream, elems).unwrap();
        // SAFETY: every buffer below is sized from `b.n` and `b.dim`, the same
        // values passed as the kernel's `dim` argument and used to size the
        // launch, so no thread indexes outside its allocation.
        unsafe {
            module
                .sphere_exp_apply(
                &stream,
                LaunchConfig::for_num_elems(elems as u32),
                &p_dev,
                &v_dev,
                &theta,
                b.dim as u32,
                &mut exp_dev,
                )
                .expect("exp_apply launch");
        }

        // log: same shape.
        let mut cosang = DeviceBuffer::<f64>::zeroed(&stream, b.n).unwrap();
        // SAFETY: every buffer below is sized from `b.n` and `b.dim`, the same
        // values passed as the kernel's `dim` argument and used to size the
        // launch, so no thread indexes outside its allocation.
        unsafe {
            module
                .sphere_cos_angle(
                &stream,
                LaunchConfig::for_num_elems(b.n as u32),
                &p_dev,
                &q_dev,
                b.dim as u32,
                &mut cosang,
                )
                .expect("cos_angle launch");
        }

        let mut log_dev = DeviceBuffer::<f64>::zeroed(&stream, elems).unwrap();
        // SAFETY: every buffer below is sized from `b.n` and `b.dim`, the same
        // values passed as the kernel's `dim` argument and used to size the
        // launch, so no thread indexes outside its allocation.
        unsafe {
            module
                .sphere_log_apply(
                &stream,
                LaunchConfig::for_num_elems(elems as u32),
                &p_dev,
                &q_dev,
                &cosang,
                b.dim as u32,
                &mut log_dev,
                )
                .expect("log_apply launch");
        }

        let exp_gpu = exp_dev.to_host_vec(&stream).unwrap();
        let log_gpu = log_dev.to_host_vec(&stream).unwrap();

        let e = max_abs_diff(&exp_gpu, &exp_ref);
        let l = max_abs_diff(&log_gpu, &log_ref);
        worst_exp = worst_exp.max(e);
        worst_log = worst_log.max(l);
        if e > TOL || l > TOL {
            failed = true;
        }

        println!("{dim:>5}  {n:>8}  {e:>12.3e}  {l:>12.3e}");
    }

    println!();
    if failed {
        eprintln!("FAILED: GPU and CPU disagree by more than {TOL:.0e}");
        std::process::exit(1);
    }
    println!(
        "PASSED: exp within {worst_exp:.3e}, log within {worst_log:.3e}, both under {TOL:.0e}"
    );
    println!("Double precision end to end; an f32 path could not be held to this.");
}
