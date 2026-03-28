"""Tests for cartan.minimize_rgd (Riemannian Gradient Descent)."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose

import cartan


class TestMinimizeRGD:
    def test_rgd_on_euclidean(self):
        """RGD should find minimum of ||x - target||^2 on R^3."""
        m = cartan.Euclidean(3)
        target = np.array([1.0, 2.0, 3.0])

        def cost(p):
            return float(np.sum((p - target) ** 2))

        def grad(p):
            return 2.0 * (p - target)

        x0 = np.zeros(3)
        result = cartan.minimize_rgd(m, cost, grad, x0)
        assert result.converged
        assert_allclose(result.point, target, rtol=1e-5)

    def test_rgd_on_sphere_rayleigh(self):
        """RGD minimizing Rayleigh quotient on S^2 finds smallest eigenvector."""
        m = cartan.Sphere(2)  # S^2, ambient dim = 3
        A = np.diag([1.0, 2.0, 3.0])

        def cost(p):
            return float(p @ A @ p)

        def grad(p):
            eg = 2.0 * A @ p
            # Riemannian gradient on sphere: project out normal component
            return eg - np.dot(eg, p) * p

        x0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        result = cartan.minimize_rgd(
            m, cost, grad, x0, grad_tol=1e-8, max_iters=5000
        )
        assert result.converged
        # Smallest eigenvector is e_1 = [±1, 0, 0]
        assert_allclose(abs(result.point[0]), 1.0, atol=1e-4)

    def test_rgd_result_fields(self):
        """OptResult exposes all expected attributes with correct types."""
        m = cartan.Euclidean(2)
        result = cartan.minimize_rgd(
            m,
            lambda p: float(np.sum(p ** 2)),
            lambda p: 2.0 * p,
            np.array([5.0, 5.0]),
        )
        assert hasattr(result, "point")
        assert hasattr(result, "value")
        assert hasattr(result, "grad_norm")
        assert hasattr(result, "iterations")
        assert hasattr(result, "converged")
        assert isinstance(result.iterations, int)
        assert isinstance(result.converged, bool)
        assert result.value >= 0.0
        assert result.grad_norm >= 0.0

    def test_rgd_on_spd(self):
        """RGD on SPD(2): minimize squared geodesic distance to a target matrix."""
        m = cartan.SPD(2)
        target = np.array([[2.0, 0.5], [0.5, 1.0]])

        def cost(p):
            return float(m.dist(p, target) ** 2)

        def grad(p):
            # Riemannian gradient of d^2(p, q) at p is -2 * Log_p(q)
            return -2.0 * m.log(p, target)

        x0 = np.eye(2)
        result = cartan.minimize_rgd(m, cost, grad, x0, grad_tol=1e-8)
        assert result.converged
        assert_allclose(result.point, target, rtol=1e-4)

    def test_rgd_repr(self):
        """OptResult has a useful __repr__."""
        m = cartan.Euclidean(1)
        result = cartan.minimize_rgd(
            m,
            lambda p: float(p[0] ** 2),
            lambda p: 2.0 * p,
            np.array([3.0]),
        )
        r = repr(result)
        assert "OptResult" in r
        assert "converged" in r

    def test_rgd_already_converged(self):
        """Starting at the optimum should converge immediately (0 iterations)."""
        m = cartan.Euclidean(2)
        result = cartan.minimize_rgd(
            m,
            lambda p: float(np.sum(p ** 2)),
            lambda p: 2.0 * p,
            np.zeros(2),
            grad_tol=1.0,  # very loose tolerance -- grad norm is 0 at x=0
        )
        assert result.converged
        assert result.iterations == 0

    def test_rgd_custom_params(self):
        """Custom max_iters is forwarded; tight grad_tol prevents early exit."""
        m = cartan.Euclidean(2)
        # Use a cost whose gradient is never zero within 2 steps from this start.
        # f(p) = ||p - [100, 100]||^2, x0 = [0, 0].
        target = np.array([100.0, 100.0])
        result = cartan.minimize_rgd(
            m,
            lambda p: float(np.sum((p - target) ** 2)),
            lambda p: 2.0 * (p - target),
            np.zeros(2),
            max_iters=2,
            grad_tol=1e-12,
        )
        # With max_iters=2 and tight tolerance we should not have fully converged.
        assert result.iterations <= 2

    def test_rgd_unsupported_manifold_raises(self):
        """Passing an unsupported object raises TypeError."""
        with pytest.raises(TypeError):
            cartan.minimize_rgd(
                "not_a_manifold",
                lambda p: 0.0,
                lambda p: p,
                np.zeros(3),
            )

    def test_opt_result_is_class(self):
        """OptResult is accessible as cartan.OptResult."""
        assert hasattr(cartan, "OptResult")
