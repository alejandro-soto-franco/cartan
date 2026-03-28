import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL, ATOL, RTOL_RELAXED, ATOL_RELAXED

import cartan


def make_corr(n, rng=None):
    """Create a random correlation matrix of size n x n.

    Generates a random SPD matrix and normalises to unit diagonal so the
    result is a valid correlation matrix.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    A = rng.standard_normal((n, n))
    S = A @ A.T + np.eye(n)
    # Normalise to correlation matrix: C_ij = S_ij / sqrt(S_ii * S_jj)
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    return C


class TestCorrBasic:
    @pytest.mark.parametrize("n,expected_dim,expected_ambient", [
        (2, 1, 4),
        (3, 3, 9),
        (4, 6, 16),
        (8, 28, 64),
    ])
    def test_dim(self, n, expected_dim, expected_ambient):
        m = cartan.Corr(n)
        assert m.dim() == expected_dim
        assert m.ambient_dim() == expected_ambient

    def test_repr(self):
        m = cartan.Corr(3)
        r = repr(m)
        assert "3" in r
        assert "Corr" in r

    def test_unsupported_dim_too_small(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.Corr(1)

    def test_unsupported_dim_zero(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.Corr(0)

    def test_unsupported_dim_too_large(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.Corr(9)


class TestCorrExpLog:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_exp_log_roundtrip(self, n):
        """Corr is flat: exp and log are exact and invertible."""
        m = cartan.Corr(n)
        p = make_corr(n, np.random.default_rng(42 + n))
        q = make_corr(n, np.random.default_rng(99 + n))
        v = m.log(p, q)
        q_reconstructed = m.exp(p, v)
        assert_allclose(q_reconstructed, q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_exp_zero_tangent_returns_base(self):
        m = cartan.Corr(2)
        p = np.eye(2)
        v = np.zeros((2, 2))
        result = m.exp(p, v)
        assert_allclose(result, p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestCorrRandomPoint:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_random_point_unit_diagonal(self, n):
        """A random Corr point must have ones on the diagonal."""
        m = cartan.Corr(n)
        p = m.random_point(seed=42)
        assert p.shape == (n, n)
        diag = np.diag(p)
        assert_allclose(diag, np.ones(n), atol=ATOL_RELAXED)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_random_point_symmetric(self, n):
        """A random Corr point must be symmetric."""
        m = cartan.Corr(n)
        p = m.random_point(seed=7)
        assert_allclose(p, p.T, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_random_point_positive_semidefinite(self, n):
        """A random Corr point must be positive semidefinite (all eigenvalues >= 0)."""
        m = cartan.Corr(n)
        p = m.random_point(seed=13)
        eigvals = np.linalg.eigvalsh(p)
        assert np.all(eigvals >= -ATOL_RELAXED), (
            f"Random Corr({n}) point not PSD, min eigenvalue: {eigvals.min()}"
        )


class TestCorrCurvature:
    def test_scalar_curvature_zero(self):
        """Corr is a flat manifold: scalar curvature must be zero at every point."""
        m = cartan.Corr(2)
        p = make_corr(2)
        sc = m.scalar_curvature(p)
        assert_allclose(sc, 0.0, atol=ATOL_RELAXED)

    def test_sectional_curvature_zero(self):
        """Sectional curvature of flat Corr must be zero."""
        m = cartan.Corr(3)
        p = make_corr(3)
        # Off-diagonal symmetric tangent vectors (zero diagonal)
        u = np.zeros((3, 3))
        u[0, 1] = 1.0
        u[1, 0] = 1.0
        v = np.zeros((3, 3))
        v[0, 2] = 1.0
        v[2, 0] = 1.0
        kappa = m.sectional_curvature(p, u, v)
        assert_allclose(kappa, 0.0, atol=ATOL_RELAXED)
