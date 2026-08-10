//! Checks the GPU kernels against the CPU implementation they are meant to
//! match, and reports the worst disagreement.
//!
//! ```text
//! cargo oxide run cartan-cuda
//! ```
//!
//! Exits non-zero when any operation exceeds the tolerance, so it doubles as a
//! hardware smoke test.

use cartan_cuda::Device;

/// A batch of unit vectors and tangents, laid out row-major with stride `dim`.
struct Batch {
    dim: usize,
    n: usize,
    p: Vec<f64>,
    v: Vec<f64>,
    q: Vec<f64>,
}

/// Build a batch whose tangents sit well inside the injectivity radius, so
/// `exp` and `log` invert each other and the comparison carries information.
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

/// A batch of SPD(3) pairs, row-major, nine doubles each.
fn make_spd_batch(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use cartan_core::Manifold;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let m = cartan_manifolds::Spd::<3>;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut p = Vec::with_capacity(n * 9);
    let mut q = Vec::with_capacity(n * 9);

    for _ in 0..n {
        let pi = m.random_point(&mut rng);
        let qi = m.random_point(&mut rng);
        // Row-major, to match the kernel's indexing.
        for r in 0..3 {
            for c in 0..3 {
                p.push(pi[(r, c)]);
            }
        }
        for r in 0..3 {
            for c in 0..3 {
                q.push(qi[(r, c)]);
            }
        }
    }
    (p, q)
}

/// CPU reference through the library's own affine-invariant distance.
fn spd_reference(p: &[f64], q: &[f64], n: usize) -> Vec<f64> {
    use cartan_core::Manifold;

    let m = cartan_manifolds::Spd::<3>;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = i * 9;
        let pi = nalgebra::SMatrix::<f64, 3, 3>::from_row_slice(&p[b..b + 9]);
        let qi = nalgebra::SMatrix::<f64, 3, 3>::from_row_slice(&q[b..b + 9]);
        out.push(m.dist(&pi, &qi).unwrap());
    }
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn main() {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cartan-cuda: {e}");
            std::process::exit(1);
        }
    };

    // The CPU exp renormalises its result and the GPU kernel does not, so the
    // two differ by whatever that correction is worth. On an exactly tangent
    // input that is a few ulp, which is why the bound is 1e-13 rather than
    // machine epsilon.
    const TOL: f64 = 1e-13;

    let mut worst_exp = 0.0f64;
    let mut worst_log = 0.0f64;
    let mut failed = false;

    println!("cartan-cuda: batched manifold geometry, double precision\n");
    println!(
        "{:>5}  {:>8}  {:>12}  {:>12}",
        "dim", "points", "exp max err", "log max err"
    );
    println!("{}", "-".repeat(44));

    for &(dim, n) in &[(3usize, 4096usize), (10, 4096), (50, 2048)] {
        let b = make_batch(dim, n, 42);
        let (exp_ref, log_ref) = cpu_reference(&b);

        let exp_gpu = dev.sphere_exp(&b.p, &b.v, b.dim).expect("sphere_exp");
        let log_gpu = dev.sphere_log(&b.p, &b.q, b.dim).expect("sphere_log");

        let e = max_abs_diff(&exp_gpu, &exp_ref);
        let l = max_abs_diff(&log_gpu, &log_ref);
        worst_exp = worst_exp.max(e);
        worst_log = worst_log.max(l);
        if e > TOL || l > TOL {
            failed = true;
        }

        println!("{dim:>5}  {n:>8}  {e:>12.3e}  {l:>12.3e}");
    }

    // SPD(3): Cholesky then Jacobi eigenvalues, the same route the CPU takes.
    {
        let n = 4096usize;
        let (p_host, q_host) = make_spd_batch(n, 7);
        let reference = spd_reference(&p_host, &q_host, n);

        let gpu = dev.spd3_dist(&p_host, &q_host).expect("spd3_dist");

        // Relative, since SPD distances are not O(1) the way sphere ones are.
        let mut worst_rel = 0.0f64;
        for i in 0..n {
            let denom = reference[i].abs().max(1.0);
            worst_rel = worst_rel.max((gpu[i] - reference[i]).abs() / denom);
        }
        println!("{:>5}  {:>8}  {:>12}  {:>12}", "spd3", n, "-", "-");
        println!("\n  SPD(3) distance, relative error vs CPU: {worst_rel:.3e}");
        if worst_rel > TOL {
            failed = true;
        }
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
