import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, floats, sampled_from
from hypothesis.extra.numpy import arrays

import cartan
from conftest import RTOL_RELAXED


class TestManifoldProperties:
    @settings(max_examples=50, deadline=None)
    @given(seed1=integers(0, 10000), seed2=integers(0, 10000), dim=integers(1, 5))
    def test_exp_log_roundtrip_euclidean(self, seed1, seed2, dim):
        m = cartan.Euclidean(dim)
        p = m.random_point(seed=seed1)
        q = m.random_point(seed=seed2)
        result = m.exp(p, m.log(p, q))
        np.testing.assert_allclose(result, q, rtol=1e-10, atol=1e-12)

    @settings(max_examples=50, deadline=None)
    @given(seed1=integers(0, 10000), seed2=integers(0, 10000), dim=integers(1, 5))
    def test_exp_log_roundtrip_sphere(self, seed1, seed2, dim):
        m = cartan.Sphere(dim)
        p = m.random_point(seed=seed1)
        q = m.random_point(seed=seed2)
        # Skip if too close to cut locus
        d = m.dist(p, q)
        assume(d < np.pi - 0.1)
        result = m.exp(p, m.log(p, q))
        np.testing.assert_allclose(result, q, rtol=1e-6, atol=1e-8)

    @settings(max_examples=50, deadline=None)
    @given(seed=integers(0, 10000), dim=integers(1, 5))
    def test_project_point_idempotent_euclidean(self, seed, dim):
        m = cartan.Euclidean(dim)
        p = m.random_point(seed=seed)
        pp = m.project_point(p)
        ppp = m.project_point(pp)
        np.testing.assert_allclose(pp, ppp, atol=1e-14)

    @settings(max_examples=50, deadline=None)
    @given(seed=integers(0, 10000), dim=integers(1, 5))
    def test_project_point_idempotent_sphere(self, seed, dim):
        m = cartan.Sphere(dim)
        rng = np.random.default_rng(seed)
        p_bad = rng.standard_normal(dim + 1)
        pp = m.project_point(p_bad)
        ppp = m.project_point(pp)
        np.testing.assert_allclose(pp, ppp, atol=1e-14)

    @settings(max_examples=50, deadline=None)
    @given(seed=integers(0, 10000), dim=integers(2, 5))
    def test_dist_nonnegative_spd(self, seed, dim):
        m = cartan.SPD(dim)
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((dim, dim))
        p = A @ A.T + np.eye(dim)
        B = rng.standard_normal((dim, dim))
        q = B @ B.T + np.eye(dim)
        assert m.dist(p, q) >= -1e-15

    @settings(max_examples=50, deadline=None)
    @given(seed=integers(0, 10000), dim=integers(1, 5))
    def test_inner_bilinear_euclidean(self, seed, dim):
        """inner(p, au + bv, w) = a*inner(p,u,w) + b*inner(p,v,w)."""
        m = cartan.Euclidean(dim)
        p = m.random_point(seed=seed)
        rng = np.random.default_rng(seed + 1)
        u = rng.standard_normal(dim)
        v = rng.standard_normal(dim)
        w = rng.standard_normal(dim)
        a, b = 2.5, -1.3
        lhs = m.inner(p, a * u + b * v, w)
        rhs = a * m.inner(p, u, w) + b * m.inner(p, v, w)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)
