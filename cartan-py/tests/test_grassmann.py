import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL_RELAXED, ATOL_RELAXED

import cartan


class TestGrassmannBasic:
    @pytest.mark.parametrize("n,k", [(3, 1), (4, 2), (5, 2), (8, 3)])
    def test_dim(self, n, k):
        m = cartan.Grassmann(n, k)
        assert m.dim() == k * (n - k)

    @pytest.mark.parametrize("n,k", [(3, 1), (4, 2), (5, 2), (8, 3)])
    def test_ambient_dim(self, n, k):
        m = cartan.Grassmann(n, k)
        assert m.ambient_dim() == n * k

    @pytest.mark.parametrize("n,k", [(3, 1), (4, 2), (5, 2)])
    def test_exp_log_roundtrip(self, n, k):
        m = cartan.Grassmann(n, k)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        result = m.exp(p, m.log(p, q))
        # Grassmann points are K-planes (subspaces), not individual matrices.
        # exp/log round-trip recovers the same subspace, not necessarily the
        # same orthonormal representative. Compare projection matrices P = Q Q^T.
        proj_result = result @ result.T
        proj_q = q @ q.T
        assert_allclose(proj_result, proj_q, rtol=1e-5, atol=1e-6)

    def test_point_has_orthonormal_columns(self):
        m = cartan.Grassmann(4, 2)
        p = m.random_point(seed=42)
        assert p.shape == (4, 2)
        assert_allclose(p.T @ p, np.eye(2), atol=1e-13)

    def test_tangent_orthogonal_to_point(self):
        m = cartan.Grassmann(4, 2)
        p = m.random_point(seed=42)
        v = m.random_tangent(p, seed=43)
        assert_allclose(p.T @ v, np.zeros((2, 2)), atol=1e-13)

    def test_dist_nonnegative(self):
        m = cartan.Grassmann(4, 2)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        assert m.dist(p, q) >= 0

    def test_dist_self_zero(self):
        m = cartan.Grassmann(4, 2)
        p = m.random_point(seed=42)
        assert_allclose(m.dist(p, p), 0.0, atol=1e-12)

    def test_random_point_deterministic(self):
        m = cartan.Grassmann(4, 2)
        p1 = m.random_point(seed=42)
        p2 = m.random_point(seed=42)
        assert_allclose(p1, p2)


class TestGrassmannTransport:
    def test_parallel_transport_preserves_norm(self):
        m = cartan.Grassmann(4, 2)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        v = m.random_tangent(p, seed=44)
        vt = m.parallel_transport(p, q, v)
        assert_allclose(m.norm(p, v), m.norm(q, vt), rtol=1e-6)


class TestGrassmannGeodesic:
    def test_geodesic_endpoints(self):
        m = cartan.Grassmann(4, 2)
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        # t=0 endpoint: compare projection matrices (subspace equality)
        geo0 = m.geodesic(p, q, 0.0)
        assert_allclose(geo0 @ geo0.T, p @ p.T, atol=1e-12)
        # t=1 endpoint: same subspace as q
        geo1 = m.geodesic(p, q, 1.0)
        assert_allclose(geo1 @ geo1.T, q @ q.T, rtol=1e-5, atol=1e-6)


class TestGrassmannInvalid:
    def test_k_equals_n(self):
        with pytest.raises(ValueError):
            cartan.Grassmann(3, 3)

    def test_n_too_large(self):
        with pytest.raises(ValueError):
            cartan.Grassmann(9, 1)

    def test_k_zero(self):
        with pytest.raises(ValueError):
            cartan.Grassmann(2, 0)

    def test_n_too_small(self):
        with pytest.raises(ValueError):
            cartan.Grassmann(1, 1)


class TestGrassmannRepr:
    def test_repr(self):
        m = cartan.Grassmann(4, 2)
        r = repr(m)
        assert "4" in r
        assert "2" in r
