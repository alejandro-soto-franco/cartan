//! Macroscale slab Darcy solve.
//!
//! Given a slab `[0, Lx] × [0, Ly] × [0, H]` with a piecewise-constant effective
//! permeability tensor `K(x)` (produced by the mean-field homogenisation at each
//! sampled depth), solve the steady Darcy problem
//!
//! ```text
//!     ∇·(K ∇P) = 0   in slab,
//!     P = P_top      at z = H,
//!     P = P_bot      at z = 0,
//!     (K ∇P)·n = 0   on lateral faces (no-flow).
//! ```
//!
//! Compute effective macroscopic permeability components `k_eff^xx` and
//! `k_eff^zz` from `<u> = -(K_eff / mu_f) <∇P>` with `<·>` the volume average
//! over the slab.
//!
//! P1-FEM on a structured tet mesh of the slab (Kuhn triangulation). Uses the
//! fullfield::cell_problem primitives for assembly and fullfield::solver for the
//! linear solve (dense LU fallback for robustness under strong anisotropy).

use crate::{error::HomogError,
            fullfield::{solver,
                        mesh::{PeriodicCubeMeshBuilder, PeriodicCubeMeshBuilderOpts}}};
use alloc::vec::Vec;
use cartan_dec::Mesh;
use cartan_manifolds::Euclidean;
use nalgebra::{Matrix3, Vector3};

pub struct SlabProblem {
    /// Slab dimensions.
    pub l_x: f64,
    pub l_y: f64,
    pub h: f64,
    /// Effective permeability tensor as a function of depth z.
    pub k_of_z: alloc::boxed::Box<dyn Fn(f64) -> Matrix3<f64>>,
    /// Top and bottom pressures (Dirichlet).
    pub p_top: f64,
    pub p_bot: f64,
    /// Mesh resolution per side of the unit cube before scaling.
    pub resolution: usize,
}

pub struct SlabSolution {
    pub mesh: Mesh<Euclidean<3>, 4, 3>,
    pub pressure: nalgebra::DVector<f64>,
    pub k_per_tet: Vec<Matrix3<f64>>,
    /// Volume-averaged effective permeability: `<K ∇P> / <∇P>` component-wise.
    pub k_eff_macro: Matrix3<f64>,
    /// Volume-averaged pressure gradient over the slab.
    pub grad_p_avg: Vector3<f64>,
}

impl SlabProblem {
    /// Solve the slab Darcy problem. Returns the pressure field and the
    /// derived macroscopic effective permeability tensor.
    pub fn solve(&self) -> Result<SlabSolution, HomogError> {
        // Build the unit-cube mesh, then stretch vertices to the slab.
        let builder = PeriodicCubeMeshBuilder::new(&PeriodicCubeMeshBuilderOpts {
            resolution: self.resolution, refine_depth: 0,
        });
        let (unit_mesh, _bary_unit) = builder.build()?;

        // Rescale vertices from unit cube to slab dimensions.
        let vertices: Vec<Vector3<f64>> = unit_mesh.vertices.iter().map(|v| {
            Vector3::new(v.x * self.l_x, v.y * self.l_y, v.z * self.h)
        }).collect();
        let simplices = unit_mesh.simplices.clone();
        let manifold = Euclidean::<3>;
        let mesh = Mesh::<Euclidean<3>, 4, 3>::from_simplices_generic(
            &manifold, vertices, simplices.clone());

        // Recompute barycentres in slab coordinates, tag each tet with K(z_barycentre).
        let barycenters: Vec<Vector3<f64>> = mesh.simplices.iter().map(|tet| {
            (mesh.vertices[tet[0]] + mesh.vertices[tet[1]]
           + mesh.vertices[tet[2]] + mesh.vertices[tet[3]]) / 4.0
        }).collect();
        let k_per_tet: Vec<Matrix3<f64>> = barycenters.iter()
            .map(|b| (self.k_of_z)(b.z))
            .collect();

        // Build tet data with anisotropic K-weight. Our cell_problem primitives
        // take scalar-K; we synthesise three scalar runs (one per x, y, z
        // component direction) won't work for anisotropy. Instead assemble the
        // anisotropic stiffness here directly.
        let nt = mesh.n_simplices();
        let nv = mesh.n_vertices();
        let mut grads: Vec<[Vector3<f64>; 4]> = Vec::with_capacity(nt);
        let mut volumes: Vec<f64> = Vec::with_capacity(nt);
        for s in 0..nt {
            let tet = mesh.simplices[s];
            let v: [Vector3<f64>; 4] = [
                mesh.vertices[tet[0]], mesh.vertices[tet[1]],
                mesh.vertices[tet[2]], mesh.vertices[tet[3]],
            ];
            let jac = Matrix3::from_columns(&[v[1] - v[0], v[2] - v[0], v[3] - v[0]]);
            let det = jac.determinant();
            let vol = det.abs() / 6.0;
            if vol < 1e-18 {
                return Err(HomogError::Mesh(alloc::format!("slab tet {s} degenerate")));
            }
            let jit = jac.try_inverse().ok_or(HomogError::Mesh(alloc::format!("slab tet {s} singular")))?
                         .transpose();
            let g1 = jit.column(0).into_owned();
            let g2 = jit.column(1).into_owned();
            let g3 = jit.column(2).into_owned();
            let g0 = -(g1 + g2 + g3);
            grads.push([g0, g1, g2, g3]);
            volumes.push(vol);
        }

        let mut tri = sprs::TriMat::<f64>::new((nv, nv));
        for s in 0..nt {
            let tet = mesh.simplices[s];
            let g = &grads[s];
            let k = &k_per_tet[s];
            let w = volumes[s];
            for a in 0..4 {
                for b in 0..4 {
                    let kg_b = k * g[b];
                    let gab = g[a].dot(&kg_b);
                    if gab.abs() > 1e-30 || a == b {
                        tri.add_triplet(tet[a], tet[b], w * gab);
                    }
                }
            }
        }
        let mut a_mat = tri.to_csc();
        let mut b_rhs = nalgebra::DVector::<f64>::zeros(nv);

        // Dirichlet BCs: P = p_top at z = H, P = p_bot at z = 0.
        // Lateral faces are natural no-flow (homogeneous Neumann, no row surgery needed).
        let mut prescribed: Vec<(usize, f64)> = Vec::new();
        for (i, v) in mesh.vertices.iter().enumerate() {
            if v.z < 1e-9 { prescribed.push((i, self.p_bot)); }
            else if v.z > self.h - 1e-9 { prescribed.push((i, self.p_top)); }
        }
        apply_dirichlet_values(&mut a_mat, &mut b_rhs, &prescribed);

        let p = solver::solve_dense_lu(&a_mat, &b_rhs)?;

        // Compute macroscopic <K ∇P> and <∇P> by volume-averaging per-tet quantities.
        let mut total_vol = 0.0;
        let mut flux_avg = Vector3::zeros();
        let mut grad_p_avg = Vector3::zeros();
        for s in 0..nt {
            let tet = mesh.simplices[s];
            let g = &grads[s];
            let grad_p = p[tet[0]] * g[0] + p[tet[1]] * g[1]
                       + p[tet[2]] * g[2] + p[tet[3]] * g[3];
            let flux = -(k_per_tet[s] * grad_p);  // u = -K ∇P
            flux_avg += flux * volumes[s];
            grad_p_avg += grad_p * volumes[s];
            total_vol += volumes[s];
        }
        flux_avg /= total_vol;
        grad_p_avg /= total_vol;

        // Effective macroscopic K: diagonal estimate from <K ∇P>_i / (-<∇P>_i)
        // for each direction with a measurable gradient. For a z-only gradient,
        // only K_zz is well-posed; the test driver imposes additional macroscopic
        // gradients by varying BCs or uses three separate runs.
        let mut k_eff_macro = Matrix3::<f64>::zeros();
        for i in 0..3 {
            if grad_p_avg[i].abs() > 1e-20 {
                k_eff_macro[(i, i)] = -flux_avg[i] / grad_p_avg[i];
            }
        }

        Ok(SlabSolution { mesh, pressure: p, k_per_tet, k_eff_macro, grad_p_avg })
    }
}

/// Apply Dirichlet BCs `P[i] = val` by row surgery.
fn apply_dirichlet_values(
    a: &mut sprs::CsMat<f64>, b: &mut nalgebra::DVector<f64>, prescribed: &[(usize, f64)],
) {
    use std::collections::HashMap;
    let prescribed_map: HashMap<usize, f64> =
        prescribed.iter().copied().collect();
    let n = a.rows();
    let mut tri = sprs::TriMat::<f64>::new((n, n));
    // First pass: for each (i, j) entry, if i is prescribed, skip (row surgery).
    // If j is prescribed, fold val into b[i] and skip the column entry.
    for (&val, (i, j)) in a.iter() {
        if prescribed_map.contains_key(&i) { continue; }
        if let Some(&v_j) = prescribed_map.get(&j) {
            b[i] -= val * v_j;
            continue;
        }
        tri.add_triplet(i, j, val);
    }
    for (&idx, &v) in &prescribed_map {
        tri.add_triplet(idx, idx, 1.0);
        b[idx] = v;
    }
    *a = tri.to_csc();
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn homogeneous_slab_produces_constant_pressure_gradient() {
        // Uniform K = I, Lx = Ly = 10, H = 20. P = 0 at top, P = 100 at bot.
        // Solution: linear pressure gradient ∇P = (0, 0, -5). No lateral flux.
        let k_fn = alloc::boxed::Box::new(|_z: f64| Matrix3::<f64>::identity());
        let prob = SlabProblem {
            l_x: 10.0, l_y: 10.0, h: 20.0,
            k_of_z: k_fn,
            p_top: 0.0, p_bot: 100.0,
            resolution: 4,
        };
        let sol = prob.solve().unwrap();
        // grad_p_avg should be approximately (0, 0, -5) (gradient from bottom to top).
        assert_relative_eq!(sol.grad_p_avg.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(sol.grad_p_avg.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(sol.grad_p_avg.z, -5.0, epsilon = 1e-6);
        // k_eff_macro[zz] = 1.0 (homogeneous recovery).
        assert_relative_eq!(sol.k_eff_macro[(2, 2)], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn transversely_isotropic_slab_recovers_vertical_k() {
        // K = diag(10, 10, 0.5). Vertical pressure gradient only; the macroscopic
        // flux-over-gradient recovers K_zz = 0.5 regardless of K_xx.
        let k_fn = alloc::boxed::Box::new(|_z: f64| {
            let mut k = Matrix3::<f64>::zeros();
            k[(0, 0)] = 10.0; k[(1, 1)] = 10.0; k[(2, 2)] = 0.5;
            k
        });
        let prob = SlabProblem {
            l_x: 10.0, l_y: 10.0, h: 20.0,
            k_of_z: k_fn,
            p_top: 0.0, p_bot: 100.0,
            resolution: 4,
        };
        let sol = prob.solve().unwrap();
        assert_relative_eq!(sol.k_eff_macro[(2, 2)], 0.5, epsilon = 1e-6);
    }
}
