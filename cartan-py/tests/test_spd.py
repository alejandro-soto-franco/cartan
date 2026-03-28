import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL, ATOL, RTOL_RELAXED, ATOL_RELAXED

import cartan


def make_spd(n, rng=None):
    """Create a random SPD matrix of size n x n via A @ A.T + eye(n)."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = rng.standard_normal((n, n))
    return A @ A.T + np.eye(n)


class TestSPDBasic:
    @pytest.mark.parametrize("n,expected_dim,expected_ambient", [
        (2, 3, 4),
        (3, 6, 9),
        (4, 10, 16),
        (5, 15, 25),
        (6, 21, 36),
        (7, 28, 49),
        (8, 36, 64),
    ])
    def test_dim(self, n, expected_dim, expected_ambient):
        m = cartan.SPD(n)
        assert m.dim() == expected_dim
        assert m.ambient_dim() == expected_ambient

    def test_repr(self):
        m = cartan.SPD(3)
        r = repr(m)
        assert "3" in r
        assert "SPD" in r

    def test_unsupported_dim_too_small(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.SPD(1)

    def test_unsupported_dim_zero(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.SPD(0)

    def test_unsupported_dim_too_large(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.SPD(9)


class TestSPDCheckPoint:
    def test_check_point_accepts_valid_spd(self):
        m = cartan.SPD(2)
        p = np.array([[2.0, 0.5], [0.5, 1.0]])
        m.check_point(p)

    def test_check_point_accepts_identity(self):
        for n in [2, 3, 4]:
            m = cartan.SPD(n)
            m.check_point(np.eye(n))

    def test_check_point_rejects_non_symmetric(self):
        m = cartan.SPD(2)
        p = np.array([[2.0, 1.0], [0.0, 2.0]])
        with pytest.raises(cartan.ValidationError):
            m.check_point(p)

    def test_check_point_rejects_non_positive_definite(self):
        m = cartan.SPD(2)
        # Diagonal with a zero eigenvalue (positive semidefinite, not definite)
        p = np.array([[1.0, 0.0], [0.0, 0.0]])
        with pytest.raises(cartan.ValidationError):
            m.check_point(p)

    def test_check_point_rejects_negative_eigenvalue(self):
        m = cartan.SPD(2)
        # Symmetric but indefinite
        p = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        with pytest.raises(cartan.ValidationError):
            m.check_point(p)


class TestSPDExpLog:
    @pytest.mark.parametrize("n", [2, 3, 4, 8])
    def test_exp_log_roundtrip(self, n):
        m = cartan.SPD(n)
        rng = np.random.default_rng(42 + n)
        p = make_spd(n, rng)
        q = make_spd(n, np.random.default_rng(99 + n))
        v = m.log(p, q)
        q_reconstructed = m.exp(p, v)
        assert_allclose(q_reconstructed, q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_exp_zero_tangent_returns_base(self):
        m = cartan.SPD(2)
        p = np.eye(2)
        v = np.zeros((2, 2))
        result = m.exp(p, v)
        assert_allclose(result, p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSPDDist:
    @pytest.mark.parametrize("n", [2, 3, 8])
    def test_dist_nonnegative(self, n):
        m = cartan.SPD(n)
        rng = np.random.default_rng(77 + n)
        p = make_spd(n, rng)
        q = make_spd(n, np.random.default_rng(88 + n))
        d = m.dist(p, q)
        assert d >= 0.0

    @pytest.mark.parametrize("n", [2, 3, 8])
    def test_dist_self_zero(self, n):
        m = cartan.SPD(n)
        p = make_spd(n)
        d = m.dist(p, p)
        assert_allclose(d, 0.0, atol=ATOL_RELAXED)

    def test_dist_symmetry(self):
        m = cartan.SPD(3)
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(6)
        p = make_spd(3, rng1)
        q = make_spd(3, rng2)
        assert_allclose(m.dist(p, q), m.dist(q, p), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSPDCurvature:
    def test_nonpositive_sectional_curvature(self):
        """SPD is a Cartan-Hadamard manifold: sectional curvature <= 0."""
        m = cartan.SPD(2)
        p = np.eye(2)
        # Symmetric tangent vectors (traceless, symmetric)
        u = np.array([[1.0, 0.0], [0.0, -1.0]])
        v = np.array([[0.0, 1.0], [1.0, 0.0]])
        kappa = m.sectional_curvature(p, u, v)
        assert kappa <= 1e-10, f"Expected nonpositive curvature, got {kappa}"

    def test_nonpositive_sectional_curvature_n3(self):
        m = cartan.SPD(3)
        p = np.eye(3)
        u = np.zeros((3, 3))
        u[0, 1] = 1.0
        u[1, 0] = 1.0
        v = np.zeros((3, 3))
        v[0, 2] = 1.0
        v[2, 0] = 1.0
        kappa = m.sectional_curvature(p, u, v)
        assert kappa <= 1e-10, f"Expected nonpositive curvature, got {kappa}"


class TestSPDInjectivityRadius:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_injectivity_radius_infinite(self, n):
        """SPD is simply connected with nonpositive curvature: injectivity radius is infinite."""
        m = cartan.SPD(n)
        p = make_spd(n)
        r = m.injectivity_radius(p)
        assert r == float("inf") or r > 1e10, (
            f"Expected infinite injectivity radius for SPD({n}), got {r}"
        )


class TestSPDParallelTransport:
    @pytest.mark.parametrize("n", [2, 3, 8])
    def test_parallel_transport_preserves_norm(self, n):
        m = cartan.SPD(n)
        rng = np.random.default_rng(13 + n)
        p = make_spd(n, rng)
        q = make_spd(n, np.random.default_rng(37 + n))
        # Random symmetric tangent vector at p
        A = rng.standard_normal((n, n))
        v = (A + A.T) / 2.0
        v_transported = m.parallel_transport(p, q, v)
        norm_before = m.norm(p, v)
        norm_after = m.norm(q, v_transported)
        assert_allclose(norm_after, norm_before, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSPD8:
    """SPD(8) specific tests."""

    def test_random_point_shape(self):
        m = cartan.SPD(8)
        p = m.random_point(seed=42)
        assert p.shape == (8, 8)

    def test_random_point_is_spd(self):
        m = cartan.SPD(8)
        p = m.random_point(seed=42)
        # Symmetry check
        assert_allclose(p, p.T, atol=ATOL_RELAXED)
        # Positive definiteness: all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(p)
        assert np.all(eigvals > 0), f"Random point not SPD, min eigenvalue: {eigvals.min()}"

    def test_exp_preserves_spd(self):
        m = cartan.SPD(8)
        rng = np.random.default_rng(100)
        p = make_spd(8, rng)
        A = rng.standard_normal((8, 8))
        v = (A + A.T) / 2.0
        result = m.exp(p, v)
        # Result should be symmetric
        assert_allclose(result, result.T, atol=ATOL_RELAXED)
        # Result should be positive definite
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0), f"exp result not SPD, min eigenvalue: {eigvals.min()}"

    def test_triangle_inequality(self):
        m = cartan.SPD(8)
        rngs = [np.random.default_rng(200 + i) for i in range(3)]
        pts = [make_spd(8, r) for r in rngs]
        dab = m.dist(pts[0], pts[1])
        dbc = m.dist(pts[1], pts[2])
        dac = m.dist(pts[0], pts[2])
        assert dac <= dab + dbc + 1e-8, (
            f"Triangle inequality failed: {dac} > {dab} + {dbc}"
        )

    def test_geodesic_midpoint_is_spd(self):
        m = cartan.SPD(8)
        rng1 = np.random.default_rng(300)
        rng2 = np.random.default_rng(301)
        p = make_spd(8, rng1)
        q = make_spd(8, rng2)
        mid = m.geodesic(p, q, 0.5)
        # Midpoint must be symmetric
        assert_allclose(mid, mid.T, atol=ATOL_RELAXED)
        # Midpoint must be positive definite
        eigvals = np.linalg.eigvalsh(mid)
        assert np.all(eigvals > 0), (
            f"Geodesic midpoint on SPD(8) not SPD, min eigenvalue: {eigvals.min()}"
        )
