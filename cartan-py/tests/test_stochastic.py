"""Tests for the L1 stochastic-analysis bindings exposed in `cartan.stochastic_*`."""
import numpy as np
import pytest

import cartan


def test_sphere_bm_shape_and_on_manifold():
    p0 = np.array([0.0, 0.0, 1.0])
    path = cartan.stochastic_bm_on_sphere(
        intrinsic_dim=2, p0=p0, n_steps=50, dt=0.002, seed=42
    )
    assert path.shape == (51, 3)
    # All points on S^2 within float tolerance.
    norms = np.linalg.norm(path, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_sphere_bm_rejects_off_manifold_start():
    bad = np.array([0.5, 0.5, 0.5])  # norm ~ 0.866
    with pytest.raises(ValueError):
        cartan.stochastic_bm_on_sphere(
            intrinsic_dim=2, p0=bad, n_steps=10, dt=0.01, seed=1
        )


def test_sphere_bm_rejects_unsupported_dim():
    p0 = np.zeros(20)
    p0[0] = 1.0
    with pytest.raises(ValueError):
        cartan.stochastic_bm_on_sphere(
            intrinsic_dim=19, p0=p0, n_steps=5, dt=0.01, seed=1
        )


def test_spd_bm_shape_and_spd():
    n = 3
    p0 = np.eye(n)
    path = cartan.stochastic_bm_on_spd(
        n=n, p0=p0, n_steps=30, dt=0.002, seed=7
    )
    assert path.shape == (31, n, n)
    # Every frame must be SPD (symmetric, positive eigenvalues).
    for i, mat in enumerate(path):
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)
        eigs = np.linalg.eigvalsh(mat)
        assert eigs.min() > 0, f"step {i}: min eig {eigs.min()}"


def test_wishart_step_matches_closed_form_mean():
    """E[X_T] = X_0 + n·T·I for Wishart with shape n."""
    N, shape, T, n_steps, n_paths = 3, 3.0, 0.5, 200, 400
    dt = T / n_steps
    X0 = np.eye(N)
    total = np.zeros((N, N))
    for seed in range(n_paths):
        X = X0.copy()
        for k in range(n_steps):
            X = cartan.wishart_step(x=X, shape_param=shape, dt=dt, seed=seed * n_steps + k)
        total += X
    mean = total / n_paths
    expected_diag = 1.0 + shape * T  # = 2.5
    for i in range(N):
        assert abs(mean[i, i] - expected_diag) < 0.25, (
            f"diag[{i}] = {mean[i,i]}, expected {expected_diag}"
        )


def test_determinism_from_seed():
    p0 = np.array([1.0, 0.0, 0.0, 0.0])
    a = cartan.stochastic_bm_on_sphere(intrinsic_dim=3, p0=p0, n_steps=20, dt=0.01, seed=99)
    b = cartan.stochastic_bm_on_sphere(intrinsic_dim=3, p0=p0, n_steps=20, dt=0.01, seed=99)
    np.testing.assert_array_equal(a, b)
