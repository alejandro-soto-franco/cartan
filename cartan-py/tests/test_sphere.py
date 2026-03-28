import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL, ATOL, RTOL_RELAXED, ATOL_RELAXED

import cartan


class TestSphereBasic:
    @pytest.mark.parametrize("intrinsic_dim,ambient", [
        (1, 2), (2, 3), (3, 4), (5, 6), (9, 10)
    ])
    def test_dim_and_ambient_dim(self, intrinsic_dim, ambient):
        m = cartan.Sphere(intrinsic_dim)
        assert m.dim() == intrinsic_dim
        assert m.ambient_dim() == ambient

    def test_repr(self):
        m = cartan.Sphere(2)
        r = repr(m)
        assert "2" in r
        assert "3" in r

    def test_unsupported_dim_zero(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.Sphere(0)

    def test_unsupported_dim_too_large(self):
        with pytest.raises(ValueError, match="unsupported"):
            cartan.Sphere(10)


class TestSpherePoints:
    def test_random_point_has_unit_norm(self):
        m = cartan.Sphere(2)
        p = m.random_point(seed=42)
        assert_allclose(np.linalg.norm(p), 1.0)

    @pytest.mark.parametrize("dim", [1, 2, 3, 5])
    def test_random_point_unit_norm_various_dims(self, dim):
        m = cartan.Sphere(dim)
        p = m.random_point(seed=7)
        assert_allclose(np.linalg.norm(p), 1.0)

    def test_random_point_deterministic(self):
        m = cartan.Sphere(2)
        p1 = m.random_point(seed=42)
        p2 = m.random_point(seed=42)
        assert_allclose(p1, p2)

    def test_project_point_normalizes(self):
        m = cartan.Sphere(2)
        v = np.array([3.0, 4.0, 0.0])
        p = m.project_point(v)
        assert_allclose(np.linalg.norm(p), 1.0)
        assert_allclose(p, v / np.linalg.norm(v))

    def test_check_point_accepts_valid(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        m.check_point(p)

    def test_check_point_rejects_wrong_norm(self):
        m = cartan.Sphere(2)
        p = np.array([2.0, 0.0, 0.0])
        with pytest.raises(cartan.ValidationError):
            m.check_point(p)

    def test_check_point_rejects_wrong_shape(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            m.check_point(p)


class TestSphereTangent:
    def test_random_tangent_orthogonal_to_point(self):
        m = cartan.Sphere(2)
        p = m.random_point(seed=13)
        v = m.random_tangent(p, seed=37)
        assert_allclose(np.dot(p, v), 0.0, atol=1e-12)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_random_tangent_orthogonal_various_dims(self, dim):
        m = cartan.Sphere(dim)
        p = m.random_point(seed=5)
        v = m.random_tangent(p, seed=11)
        assert_allclose(np.dot(p, v), 0.0, atol=1e-12)

    def test_zero_tangent(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        assert_allclose(m.zero_tangent(p), np.zeros(3))


class TestSphereExpLog:
    @pytest.mark.parametrize("dim", [1, 2, 3, 5])
    def test_exp_log_roundtrip(self, dim):
        m = cartan.Sphere(dim)
        rng1 = np.random.default_rng(42 + dim)
        rng2 = np.random.default_rng(99 + dim)
        p = m.random_point(seed=42 + dim)
        q = m.random_point(seed=99 + dim)
        v = m.log(p, q)
        q_reconstructed = m.exp(p, v)
        assert_allclose(q_reconstructed, q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_exp_zero_tangent_returns_base(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        v = np.zeros(3)
        result = m.exp(p, v)
        assert_allclose(result, p)


class TestSphereDistance:
    def test_dist_north_east_is_half_pi(self):
        """On S^2, the geodesic distance between the north pole and a point on the equator is pi/2."""
        m = cartan.Sphere(2)
        north = np.array([1.0, 0.0, 0.0])
        east = np.array([0.0, 1.0, 0.0])
        assert_allclose(m.dist(north, east), np.pi / 2, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_dist_antipodal_is_pi(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([-1.0, 0.0, 0.0])
        assert_allclose(m.dist(p, q), np.pi, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_dist_self_zero(self):
        m = cartan.Sphere(2)
        p = np.array([0.0, 1.0, 0.0])
        assert_allclose(m.dist(p, p), 0.0, atol=1e-12)

    def test_dist_symmetry(self):
        m = cartan.Sphere(2)
        p = m.random_point(seed=7)
        q = m.random_point(seed=8)
        assert_allclose(m.dist(p, q), m.dist(q, p), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_dist_triangle_inequality(self):
        m = cartan.Sphere(2)
        for seed in range(10):
            rng = np.random.default_rng(seed * 3)
            pts = [m.random_point(seed=seed * 3 + i) for i in range(3)]
            dab = m.dist(pts[0], pts[1])
            dbc = m.dist(pts[1], pts[2])
            dac = m.dist(pts[0], pts[2])
            assert dac <= dab + dbc + 1e-10


class TestSphereCurvature:
    def test_sectional_curvature_is_one_s2(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        u = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        assert_allclose(m.sectional_curvature(p, u, v), 1.0, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_sectional_curvature_is_one_various_dims(self, dim):
        m = cartan.Sphere(dim)
        ambient = dim + 1
        p = np.zeros(ambient)
        p[0] = 1.0
        u = np.zeros(ambient)
        u[1] = 1.0
        v = np.zeros(ambient)
        if dim >= 2:
            v[2] = 1.0
        else:
            # S^1 in R^2: u and v are both tangent to [1,0]; only one tangent direction
            v[1] = 1.0
        assert_allclose(m.sectional_curvature(p, u, v), 1.0, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSphereParallelTransport:
    def test_parallel_transport_preserves_norm(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.5, 0.5])
        # v must be in the tangent space of p; project it first
        v = v - np.dot(v, p) * p
        transported = m.parallel_transport(p, q, v)
        assert_allclose(np.linalg.norm(transported), np.linalg.norm(v), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_parallel_transport_stays_tangent(self):
        m = cartan.Sphere(2)
        p = m.random_point(seed=17)
        q = m.random_point(seed=23)
        v = m.random_tangent(p, seed=31)
        transported = m.parallel_transport(p, q, v)
        # Transported vector must be orthogonal to q
        assert_allclose(np.dot(transported, q), 0.0, atol=ATOL_RELAXED)


class TestSphereCutLocus:
    def test_log_antipodal_raises_cut_locus_error(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([-1.0, 0.0, 0.0])
        with pytest.raises(cartan.CutLocusError):
            m.log(p, q)

    def test_injectivity_radius_is_pi(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        assert_allclose(m.injectivity_radius(p), np.pi, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_injectivity_radius_is_pi_various_dims(self, dim):
        m = cartan.Sphere(dim)
        p = m.random_point(seed=42)
        assert_allclose(m.injectivity_radius(p), np.pi, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSphereGeodesic:
    def test_geodesic_midpoint_s2(self):
        """Geodesic midpoint of [1,0,0] and [0,1,0] on S^2 is [1,1,0]/sqrt(2)."""
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        mid = m.geodesic(p, q, 0.5)
        expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        assert_allclose(mid, expected, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_geodesic_at_t0_is_p(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        assert_allclose(m.geodesic(p, q, 0.0), p, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_geodesic_at_t1_is_q(self):
        m = cartan.Sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        assert_allclose(m.geodesic(p, q, 1.0), q, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_geodesic_stays_on_sphere(self):
        m = cartan.Sphere(2)
        p = m.random_point(seed=3)
        q = m.random_point(seed=4)
        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            mid = m.geodesic(p, q, t)
            assert_allclose(np.linalg.norm(mid), 1.0, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
