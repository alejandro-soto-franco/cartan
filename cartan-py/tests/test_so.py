import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL, ATOL, RTOL_RELAXED, ATOL_RELAXED

import cartan


def make_random_so(n, rng=None):
    """Return a random element of SO(n) via QR decomposition."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Fix signs so det(Q) = +1
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


class TestSOBasic:
    @pytest.mark.parametrize("n,expected_dim,expected_ambient", [
        (2, 1, 4),
        (3, 3, 9),
        (4, 6, 16),
    ])
    def test_dim(self, n, expected_dim, expected_ambient):
        m = cartan.SO(n)
        assert m.dim() == expected_dim
        assert m.ambient_dim() == expected_ambient

    def test_repr(self):
        m = cartan.SO(3)
        r = repr(m)
        assert "3" in r
        assert "SO" in r

    def test_unsupported_dim_too_small(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.SO(1)

    def test_unsupported_dim_zero(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.SO(0)

    def test_unsupported_dim_too_large(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.SO(5)


class TestSOExpLog:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_exp_log_roundtrip(self, n):
        m = cartan.SO(n)
        p = make_random_so(n, np.random.default_rng(42 + n))
        q = make_random_so(n, np.random.default_rng(99 + n))
        v = m.log(p, q)
        q_reconstructed = m.exp(p, v)
        assert_allclose(q_reconstructed, q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_exp_zero_tangent_returns_base(self):
        m = cartan.SO(3)
        p = np.eye(3)
        v = np.zeros((3, 3))
        result = m.exp(p, v)
        assert_allclose(result, p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSORandomPoint:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_random_point_shape(self, n):
        m = cartan.SO(n)
        p = m.random_point(seed=42)
        assert p.shape == (n, n)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_random_point_is_orthogonal(self, n):
        """R @ R.T should equal I (up to tolerance)."""
        m = cartan.SO(n)
        p = m.random_point(seed=7 + n)
        assert_allclose(p @ p.T, np.eye(n), atol=ATOL_RELAXED)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_random_point_det_one(self, n):
        """det(R) = +1 for elements of SO(n)."""
        m = cartan.SO(n)
        p = m.random_point(seed=13 + n)
        det = np.linalg.det(p)
        assert_allclose(det, 1.0, atol=ATOL_RELAXED)


class TestSOTangent:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_random_tangent_is_skew(self, n):
        """Tangent vectors at p satisfy p.T @ v skew-symmetric."""
        m = cartan.SO(n)
        p = m.random_point(seed=21 + n)
        v = m.random_tangent(p, seed=31 + n)
        # p.T @ v should be skew-symmetric: A + A.T = 0
        A = p.T @ v
        assert_allclose(A + A.T, np.zeros((n, n)), atol=ATOL_RELAXED)


class TestSOCurvature:
    def test_sectional_curvature_so3(self):
        """SO(3) with bi-invariant metric has sectional curvature 1/4.

        At the identity, for any two orthonormal Lie algebra basis elements
        e_i, e_j the sectional curvature equals 1/4.
        See: do Carmo, Riemannian Geometry, Chapter 5.
        """
        m = cartan.SO(3)
        p = np.eye(3)
        # Lie algebra basis for so(3): e_12, e_13 (skew-symmetric matrices)
        e12 = np.zeros((3, 3))
        e12[0, 1] = 1.0
        e12[1, 0] = -1.0
        e13 = np.zeros((3, 3))
        e13[0, 2] = 1.0
        e13[2, 0] = -1.0
        kappa = m.sectional_curvature(p, e12, e13)
        assert_allclose(kappa, 0.25, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
