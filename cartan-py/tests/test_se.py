import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL, ATOL, RTOL_RELAXED, ATOL_RELAXED

import cartan


class TestSeBasic:
    @pytest.mark.parametrize("n", [2, 3])
    def test_dim(self, n):
        m = cartan.SE(n)
        assert m.dim() == n * (n + 1) // 2

    @pytest.mark.parametrize("n", [2, 3])
    def test_ambient_dim(self, n):
        m = cartan.SE(n)
        assert m.ambient_dim() == n * n + n

    def test_repr(self):
        m = cartan.SE(3)
        r = repr(m)
        assert "SE" in r
        assert "3" in r

    def test_weight_default(self):
        m = cartan.SE(3)
        r = repr(m)
        assert "weight=1" in r

    def test_weight_custom(self):
        m = cartan.SE(3, weight=2.5)
        r = repr(m)
        assert "2.5" in r

    def test_unsupported_dim(self):
        with pytest.raises(ValueError):
            cartan.SE(4)

    def test_unsupported_dim_too_small(self):
        with pytest.raises(ValueError):
            cartan.SE(1)


class TestSeExpLog:
    @pytest.mark.parametrize("n", [2, 3])
    def test_exp_log_roundtrip(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        v = m.log(p, q)
        result = m.exp(p, v)
        R_result, t_result = result
        R_q, t_q = q
        assert_allclose(R_result, R_q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        assert_allclose(t_result, t_q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("n", [2, 3])
    def test_exp_zero_tangent_returns_base(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=7)
        v = m.zero_tangent(p)
        result = m.exp(p, v)
        R_result, t_result = result
        R_p, t_p = p
        assert_allclose(R_result, R_p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        assert_allclose(t_result, t_p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSeRandomPoint:
    def test_random_point_structure(self):
        m = cartan.SE(3)
        p = m.random_point(seed=42)
        assert isinstance(p, tuple)
        R, t = p
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        # R must be orthogonal with det +1
        assert_allclose(R @ R.T, np.eye(3), atol=1e-13)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-13)

    @pytest.mark.parametrize("n", [2, 3])
    def test_random_point_shape(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        R, t = p
        assert R.shape == (n, n)
        assert t.shape == (n,)

    @pytest.mark.parametrize("n", [2, 3])
    def test_random_point_deterministic(self, n):
        m = cartan.SE(n)
        p1 = m.random_point(seed=123)
        p2 = m.random_point(seed=123)
        R1, t1 = p1
        R2, t2 = p2
        assert_allclose(R1, R2)
        assert_allclose(t1, t2)


class TestSeDist:
    def test_dist_self_zero(self):
        m = cartan.SE(3)
        p = m.random_point(seed=42)
        assert_allclose(m.dist(p, p), 0.0, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3])
    def test_dist_symmetric(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        d1 = m.dist(p, q)
        d2 = m.dist(q, p)
        assert_allclose(d1, d2, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("n", [2, 3])
    def test_dist_nonnegative(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        assert m.dist(p, q) >= 0.0


class TestSeInner:
    @pytest.mark.parametrize("n", [2, 3])
    def test_inner_positive_definite(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        v = m.random_tangent(p, seed=43)
        assert m.inner(p, v, v) > 0.0

    @pytest.mark.parametrize("n", [2, 3])
    def test_norm_consistent_with_inner(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        v = m.random_tangent(p, seed=43)
        norm_val = m.norm(p, v)
        inner_val = m.inner(p, v, v)
        assert_allclose(norm_val**2, inner_val, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSeCheckPoint:
    @pytest.mark.parametrize("n", [2, 3])
    def test_check_valid_point(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        m.check_point(p)  # should not raise

    @pytest.mark.parametrize("n", [2, 3])
    def test_check_valid_tangent(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        v = m.random_tangent(p, seed=43)
        m.check_tangent(p, v)  # should not raise


class TestSeRetraction:
    @pytest.mark.parametrize("n", [2, 3])
    def test_retract_inverse_retract_roundtrip(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        v = m.random_tangent(p, seed=43)
        # Scale down to stay near p
        R_v, t_v = v
        small_v = (R_v * 0.01, t_v * 0.01)
        q = m.retract(p, small_v)
        v_back = m.inverse_retract(p, q)
        R_vb, t_vb = v_back
        assert_allclose(R_vb, small_v[0], rtol=1e-4, atol=1e-6)
        assert_allclose(t_vb, small_v[1], rtol=1e-4, atol=1e-6)


class TestSeTransport:
    @pytest.mark.parametrize("n", [2, 3])
    def test_parallel_transport_preserves_norm(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        v = m.random_tangent(p, seed=44)
        transported = m.parallel_transport(p, q, v)
        norm_before = m.norm(p, v)
        norm_after = m.norm(q, transported)
        assert_allclose(norm_before, norm_after, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSeGeodesic:
    @pytest.mark.parametrize("n", [2, 3])
    def test_geodesic_endpoints(self, n):
        m = cartan.SE(n)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        g0 = m.geodesic(p, q, 0.0)
        g1 = m.geodesic(p, q, 1.0)
        R_p, t_p = p
        R_q, t_q = q
        R_g0, t_g0 = g0
        R_g1, t_g1 = g1
        assert_allclose(R_g0, R_p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        assert_allclose(t_g0, t_p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        assert_allclose(R_g1, R_q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        assert_allclose(t_g1, t_q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSeCurvatureNotImplemented:
    def test_sectional_curvature_raises(self):
        m = cartan.SE(3)
        p = m.random_point(seed=42)
        v1 = m.random_tangent(p, seed=43)
        v2 = m.random_tangent(p, seed=44)
        with pytest.raises(NotImplementedError):
            m.sectional_curvature(p, v1, v2)

    def test_ricci_curvature_raises(self):
        m = cartan.SE(3)
        p = m.random_point(seed=42)
        v1 = m.random_tangent(p, seed=43)
        v2 = m.random_tangent(p, seed=44)
        with pytest.raises(NotImplementedError):
            m.ricci_curvature(p, v1, v2)

    def test_scalar_curvature_raises(self):
        m = cartan.SE(3)
        p = m.random_point(seed=42)
        with pytest.raises(NotImplementedError):
            m.scalar_curvature(p)
