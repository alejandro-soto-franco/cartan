"""Tests for cartan.Geodesic, cartan.CurvatureQuery, and cartan.integrate_jacobi."""

import numpy as np
import pytest
import sys
import os

# Ensure the package is importable from tests/ directly.
sys.path.insert(0, os.path.dirname(__file__))
from conftest import assert_allclose, RTOL_RELAXED, ATOL_RELAXED

import cartan


class TestGeodesic:
    def test_eval_at_zero_returns_base(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        assert_allclose(geo.eval(0.0), p, atol=1e-14)

    def test_eval_at_one_matches_exp(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        expected = m.exp(p, v)
        assert_allclose(geo.eval(1.0), expected, rtol=1e-10)

    def test_from_two_points_eval_zero(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic.from_two_points(m, p, q)
        assert_allclose(geo.eval(0.0), p, atol=1e-12)

    def test_from_two_points_eval_one(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic.from_two_points(m, p, q)
        assert_allclose(geo.eval(1.0), q, rtol=1e-6, atol=1e-8)

    def test_length_quarter_circle(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic.from_two_points(m, p, q)
        # distance from p to q on S^2 is pi/2
        assert_allclose(geo.length(), np.pi / 2, rtol=1e-10)

    def test_sample_count(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic.from_two_points(m, p, q)
        pts = geo.sample(5)
        assert len(pts) == 5

    def test_sample_on_sphere(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic.from_two_points(m, p, q)
        pts = geo.sample(7)
        for pt in pts:
            assert_allclose(np.linalg.norm(pt), 1.0, atol=1e-13)

    def test_euclidean_geodesic_is_line(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        assert_allclose(geo.eval(0.5), np.array([0.5, 0.0, 0.0]))
        assert_allclose(geo.eval(2.0), np.array([2.0, 0.0, 0.0]))

    def test_midpoint(self):
        m = cartan.Euclidean(3)
        p = np.array([0.0, 0.0, 0.0])
        v = np.array([2.0, 4.0, 6.0])
        geo = cartan.Geodesic(m, p, v)
        assert_allclose(geo.midpoint(), np.array([1.0, 2.0, 3.0]))

    def test_length_euclidean(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([3.0, 4.0, 0.0])  # norm = 5
        geo = cartan.Geodesic(m, p, v)
        assert_allclose(geo.length(), 5.0, rtol=1e-14)

    def test_repr_contains_geodesic(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        assert "Geodesic" in repr(geo)

    def test_sample_n1(self):
        m = cartan.Euclidean(2)
        p = np.array([1.0, 2.0])
        v = np.array([0.0, 1.0])
        geo = cartan.Geodesic(m, p, v)
        pts = geo.sample(1)
        assert len(pts) == 1
        assert_allclose(pts[0], p)

    def test_spd_geodesic_eval(self):
        """Geodesic on SPD(2): eval should stay on SPD manifold."""
        m = cartan.SPD(2)
        p = np.eye(2)
        q = np.array([[2.0, 0.5], [0.5, 1.5]])
        geo = cartan.Geodesic.from_two_points(m, p, q)
        mid = geo.eval(0.5)
        # midpoint should be symmetric
        assert_allclose(mid, mid.T, atol=1e-14)
        # midpoint should be positive definite (all eigenvalues > 0)
        eigs = np.linalg.eigvalsh(mid)
        assert np.all(eigs > 0), f"SPD midpoint not positive definite: {eigs}"


class TestCurvatureQuery:
    def test_sphere_scalar_curvature(self):
        """S^2 (n=2 intrinsic): scalar curvature = n*(n-1) = 2."""
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        curv = cartan.CurvatureQuery(m, p)
        assert_allclose(curv.scalar(), 2.0, rtol=1e-10)

    def test_euclidean_scalar_curvature_zero(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        curv = cartan.CurvatureQuery(m, p)
        assert_allclose(curv.scalar(), 0.0, atol=1e-15)

    def test_sphere_sectional_curvature(self):
        """S^2: sectional curvature of any 2-plane = 1."""
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        # u and v must be tangent to S^2 at p=(1,0,0)
        u = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        curv = cartan.CurvatureQuery(m, p)
        assert_allclose(curv.sectional(u, v), 1.0, rtol=1e-10)

    def test_euclidean_sectional_curvature_zero(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        curv = cartan.CurvatureQuery(m, p)
        assert_allclose(curv.sectional(u, v), 0.0, atol=1e-15)

    def test_euclidean_ricci_zero(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        curv = cartan.CurvatureQuery(m, p)
        assert_allclose(curv.ricci(u, v), 0.0, atol=1e-15)

    def test_euclidean_riemann_zero(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        w = np.array([0.0, 0.0, 1.0])
        curv = cartan.CurvatureQuery(m, p)
        result = curv.riemann(u, v, w)
        assert_allclose(result, np.zeros(3), atol=1e-15)

    def test_repr_contains_curvaturequery(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        curv = cartan.CurvatureQuery(m, p)
        assert "CurvatureQuery" in repr(curv)


class TestJacobi:
    def test_euclidean_linear_field(self):
        """Jacobi field on flat R^3 is linear: J(t) = j0 + t * j0_dot."""
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        j0 = np.array([0.0, 1.0, 0.0])
        j0_dot = np.array([0.0, 0.0, 1.0])
        result = cartan.integrate_jacobi(geo, j0, j0_dot, n_steps=50)

        # n_steps + 1 samples
        assert len(result.params) == 51
        assert len(result.field) == 51
        assert len(result.velocity) == 51

        # At t=0: J(0) = j0
        assert_allclose(result.field[0], j0, atol=1e-14)

        # At each step: J(t) ~ j0 + t * j0_dot (linear on flat space)
        for i, t in enumerate(result.params):
            expected = j0 + t * j0_dot
            assert_allclose(result.field[i], expected, rtol=1e-4, atol=1e-6)

    def test_params_span_zero_to_one(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        j0 = np.array([0.0, 1.0, 0.0])
        j0_dot = np.zeros(3)
        result = cartan.integrate_jacobi(geo, j0, j0_dot, n_steps=10)
        assert_allclose(result.params[0], 0.0, atol=1e-15)
        assert_allclose(result.params[-1], 1.0, atol=1e-15)

    def test_result_repr(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        j0 = np.array([0.0, 1.0, 0.0])
        j0_dot = np.zeros(3)
        result = cartan.integrate_jacobi(geo, j0, j0_dot, n_steps=5)
        assert "JacobiResult" in repr(result)

    def test_sphere_jacobi_initial_value(self):
        """Jacobi field on S^2: at t=0 the field should match j0."""
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, np.pi / 4, 0.0])  # geodesic towards north
        geo = cartan.Geodesic(m, p, v)
        j0 = np.array([0.0, 0.0, 1.0])      # tangent at p
        j0_dot = np.zeros(3)
        result = cartan.integrate_jacobi(geo, j0, j0_dot, n_steps=20)
        assert_allclose(result.field[0], j0, atol=1e-13)

    def test_n_steps_zero_raises(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0])
        geo = cartan.Geodesic(m, p, v)
        j0 = np.zeros(3)
        j0_dot = np.zeros(3)
        with pytest.raises(Exception):
            cartan.integrate_jacobi(geo, j0, j0_dot, n_steps=0)
