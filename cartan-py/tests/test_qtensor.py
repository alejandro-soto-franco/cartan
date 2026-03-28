import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose, RTOL_RELAXED, ATOL_RELAXED

import cartan


class TestQTensor3:
    def test_dim(self):
        m = cartan.QTensor3()
        assert m.dim() == 5

    def test_ambient_dim(self):
        m = cartan.QTensor3()
        assert m.ambient_dim() == 9

    def test_exp_log_roundtrip(self):
        m = cartan.QTensor3()
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        result = m.exp(p, m.log(p, q))
        assert_allclose(result, q, rtol=1e-6, atol=1e-8)

    def test_random_point_is_symmetric_traceless(self):
        m = cartan.QTensor3()
        Q = m.random_point(seed=42)
        assert Q.shape == (3, 3)
        assert_allclose(Q, Q.T, atol=1e-14)
        assert_allclose(np.trace(Q), 0.0, atol=1e-13)

    def test_dist_self_zero(self):
        m = cartan.QTensor3()
        p = m.random_point(seed=42)
        assert_allclose(m.dist(p, p), 0.0, atol=1e-12)

    def test_curvature_is_zero(self):
        m = cartan.QTensor3()
        p = m.random_point(seed=42)
        assert m.scalar_curvature(p) == 0.0

    def test_parallel_transport_is_identity(self):
        m = cartan.QTensor3()
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        v = m.random_tangent(p, seed=44)
        vt = m.parallel_transport(p, q, v)
        assert_allclose(v, vt, atol=1e-14)

    def test_geodesic_endpoints(self):
        m = cartan.QTensor3()
        p = m.random_point(seed=42)
        q = m.random_point(seed=43)
        assert_allclose(m.geodesic(p, q, 0.0), p, atol=1e-12)
        assert_allclose(m.geodesic(p, q, 1.0), q, rtol=1e-6, atol=1e-8)

    def test_random_point_deterministic(self):
        m = cartan.QTensor3()
        p1 = m.random_point(seed=42)
        p2 = m.random_point(seed=42)
        assert_allclose(p1, p2)

    def test_injectivity_radius_infinite(self):
        m = cartan.QTensor3()
        p = m.random_point(seed=42)
        assert m.injectivity_radius(p) == float('inf')

    def test_repr(self):
        m = cartan.QTensor3()
        assert repr(m) == "QTensor3()"


class TestFrameField3D:
    def test_from_q_field(self):
        q_values = [np.diag([0.3, 0.1, -0.4]).astype(np.float64) for _ in range(5)]
        ff = cartan.FrameField3D(q_values)
        assert ff.len() == 5
        frame = ff.frame_at(0)
        assert frame.shape == (3, 3)

    def test_gauge_fix_chain(self):
        rng = np.random.default_rng(42)
        q_values = []
        for _ in range(10):
            A = rng.standard_normal((3, 3))
            Q = (A + A.T) / 2
            Q -= np.eye(3) * np.trace(Q) / 3
            q_values.append(Q)
        ff = cartan.FrameField3D(q_values)
        ff_fixed = ff.gauge_fix_chain()
        assert ff_fixed.len() == 10

    def test_frame_at_out_of_bounds(self):
        q_values = [np.diag([0.3, 0.1, -0.4]).astype(np.float64)]
        ff = cartan.FrameField3D(q_values)
        with pytest.raises((IndexError, ValueError)):
            ff.frame_at(5)

    def test_repr(self):
        q_values = [np.diag([0.3, 0.1, -0.4]).astype(np.float64) for _ in range(3)]
        ff = cartan.FrameField3D(q_values)
        assert "3" in repr(ff)
