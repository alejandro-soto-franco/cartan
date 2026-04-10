use cartan_dec::mesh_gen::icosphere;
use cartan_dec::stokes::StokesSolverAL;
use cartan_manifolds::sphere::Sphere;

#[test]
fn test_stokes_zero_force_gives_zero_velocity() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let nv = mesh.n_vertices();

    let solver = StokesSolverAL::new(&mesh, 1e4, 1e-8, 50, 500);
    let force = vec![0.0; 3 * nv];
    let result = solver.solve(&force);

    let vel_norm: f64 = result.velocity.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        vel_norm < 1e-10,
        "zero force should give zero velocity, got ||u|| = {vel_norm}"
    );
}

#[test]
fn test_stokes_divergence_free() {
    // Apply a tangential force and verify the result is approximately divergence-free.
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let nv = mesh.n_vertices();

    let solver = StokesSolverAL::new(&mesh, 1e4, 1e-6, 100, 1000);

    // Force: tangential to sphere at each vertex.
    // f = n x e_z (cross product of normal with z-axis), giving a "zonal wind" force.
    let mut force = vec![0.0; 3 * nv];
    for v in 0..nv {
        let n = mesh.vertices[v]; // on unit sphere, vertex IS the normal
        // n x e_z = (n_y, -n_x, 0)
        force[v * 3] = n[1];
        force[v * 3 + 1] = -n[0];
        force[v * 3 + 2] = 0.0;
    }

    let result = solver.solve(&force);

    // The result should be approximately divergence-free.
    let force_norm: f64 = force.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        result.div_residual / force_norm < 0.1,
        "divergence residual too large: {} (relative: {})",
        result.div_residual,
        result.div_residual / force_norm,
    );
}

#[test]
fn test_stokes_result_has_zero_rigid_motion() {
    // The velocity should have no net translation or rotation.
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let nv = mesh.n_vertices();

    let solver = StokesSolverAL::new(&mesh, 1e4, 1e-6, 50, 500);

    let mut force = vec![0.0; 3 * nv];
    for v in 0..nv {
        let n = mesh.vertices[v];
        force[v * 3] = n[1];
        force[v * 3 + 1] = -n[0];
    }

    let result = solver.solve(&force);

    // Net translation: sum of all velocities should be near zero.
    let mut sum = [0.0_f64; 3];
    for v in 0..nv {
        sum[0] += result.velocity[v * 3];
        sum[1] += result.velocity[v * 3 + 1];
        sum[2] += result.velocity[v * 3 + 2];
    }
    let translation_norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    assert!(
        translation_norm < 1e-4,
        "net translation should be ~zero, got {translation_norm}"
    );
}
