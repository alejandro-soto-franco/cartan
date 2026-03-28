import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL, ATOL

import cartan


class TestEuclideanBasic:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_dim(self, n):
        m = cartan.Euclidean(n)
        assert m.dim() == n
        assert m.ambient_dim() == n

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_exp_is_addition(self, n):
        m = cartan.Euclidean(n)
        p = np.ones(n)
        v = np.arange(n, dtype=np.float64)
        result = m.exp(p, v)
        assert_allclose(result, p + v)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_log_is_subtraction(self, n):
        m = cartan.Euclidean(n)
        p = np.ones(n)
        q = np.arange(n, dtype=np.float64)
        result = m.log(p, q)
        assert_allclose(result, q - p)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_exp_log_roundtrip(self, n):
        m = cartan.Euclidean(n)
        p = np.random.default_rng(42).standard_normal(n)
        q = np.random.default_rng(43).standard_normal(n)
        result = m.exp(p, m.log(p, q))
        assert_allclose(result, q)

    @pytest.mark.parametrize("n", [2, 3])
    def test_dist(self, n):
        m = cartan.Euclidean(n)
        p = np.zeros(n)
        q = np.ones(n)
        assert_allclose(m.dist(p, q), np.sqrt(n))

    def test_dist_self_is_zero(self):
        m = cartan.Euclidean(3)
        p = np.array([1.0, 2.0, 3.0])
        assert_allclose(m.dist(p, p), 0.0, atol=1e-15)

    @pytest.mark.parametrize("n", [2, 3])
    def test_inner(self, n):
        m = cartan.Euclidean(n)
        p = np.zeros(n)
        u = np.ones(n)
        v = np.arange(n, dtype=np.float64)
        expected = np.dot(u, v)
        assert_allclose(m.inner(p, u, v), expected)

    def test_norm(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([3.0, 4.0, 0.0])
        assert_allclose(m.norm(p, v), 5.0)

    def test_project_point_is_identity(self):
        m = cartan.Euclidean(3)
        p = np.array([1.0, 2.0, 3.0])
        assert_allclose(m.project_point(p), p)

    def test_project_tangent_is_identity(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([1.0, 2.0, 3.0])
        assert_allclose(m.project_tangent(p, v), v)

    def test_zero_tangent(self):
        m = cartan.Euclidean(3)
        p = np.array([1.0, 2.0, 3.0])
        assert_allclose(m.zero_tangent(p), np.zeros(3))

    def test_check_point_accepts_valid(self):
        m = cartan.Euclidean(3)
        m.check_point(np.array([1.0, 2.0, 3.0]))

    def test_check_point_rejects_wrong_shape(self):
        m = cartan.Euclidean(3)
        with pytest.raises(ValueError):
            m.check_point(np.array([1.0, 2.0]))

    def test_random_point_deterministic(self):
        m = cartan.Euclidean(3)
        p1 = m.random_point(seed=42)
        p2 = m.random_point(seed=42)
        assert_allclose(p1, p2)

    def test_random_tangent_deterministic(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        t1 = m.random_tangent(p, seed=99)
        t2 = m.random_tangent(p, seed=99)
        assert_allclose(t1, t2)

    def test_injectivity_radius_infinite(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        assert m.injectivity_radius(p) == float('inf')


class TestEuclideanTransport:
    def test_parallel_transport_is_identity(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        q = np.ones(3)
        v = np.array([1.0, 2.0, 3.0])
        assert_allclose(m.parallel_transport(p, q, v), v)

    def test_retract_equals_exp(self):
        m = cartan.Euclidean(3)
        p = np.ones(3)
        v = np.array([0.1, 0.2, 0.3])
        assert_allclose(m.retract(p, v), m.exp(p, v))


class TestEuclideanCurvature:
    def test_sectional_curvature_zero(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert_allclose(m.sectional_curvature(p, u, v), 0.0, atol=1e-15)

    def test_scalar_curvature_zero(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        assert_allclose(m.scalar_curvature(p), 0.0, atol=1e-15)


class TestEuclideanGeodesic:
    def test_geodesic_interpolation(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        q = np.array([3.0, 0.0, 0.0])
        mid = m.geodesic(p, q, 0.5)
        assert_allclose(mid, np.array([1.5, 0.0, 0.0]))

    def test_geodesic_endpoints(self):
        m = cartan.Euclidean(3)
        p = np.array([1.0, 2.0, 3.0])
        q = np.array([4.0, 5.0, 6.0])
        assert_allclose(m.geodesic(p, q, 0.0), p)
        assert_allclose(m.geodesic(p, q, 1.0), q)


class TestEuclideanRepr:
    def test_repr(self):
        m = cartan.Euclidean(3)
        assert "3" in repr(m)


class TestUnsupportedDimension:
    def test_dim_too_large(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.Euclidean(11)

    def test_dim_zero(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.Euclidean(0)
