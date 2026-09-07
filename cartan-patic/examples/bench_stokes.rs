//! What the factorisation cache is worth, across mesh sizes.
//!
//! The saddle matrix is fixed for the life of a run, so a frame loop that
//! refactorises each step is doing `O(N^3)` work to get an `O(N^2)` answer.
//!
//! `cargo run -p cartan-patic --release --example bench_stokes`

use cartan_patic::complex3::Complex3;
use cartan_patic::geometry::Geometry3;
use cartan_patic::stokes::Stokes;
use nalgebra::DVector;
use std::time::Instant;

fn main() {
    println!("  n   verts  edges   tets    24 solves refactoring   24 solves cached   speedup");
    for n in [3usize, 4, 5, 6] {
        let c = Complex3::cube_grid(n);
        let g = Geometry3::cube_grid(n);
        let s = Stokes::assemble(&c, &g, 1.0);
        let f = DVector::from_element(c.n_edges(), 1e-3);

        let t0 = Instant::now();
        for _ in 0..24 {
            let _ = s.factor(&[]).solve(&f);
        }
        let naive = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let fac = s.factor(&[]);
        for _ in 0..24 {
            let _ = fac.solve(&f);
        }
        let cached = t1.elapsed().as_secs_f64();

        println!(
            "{n:3} {:7} {:6} {:6}   {naive:16.3} s   {cached:14.4} s   {:6.1}x",
            c.n_vertices(),
            c.n_edges(),
            c.n_tets(),
            naive / cached.max(1e-9)
        );
    }
}
