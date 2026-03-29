import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose

import cartan


class TestHolonomyFunctions:
    def test_edge_transition_identity(self):
        R = np.eye(3)
        T = cartan.edge_transition(R, R)
        assert_allclose(T, np.eye(3), atol=1e-14)

    def test_loop_holonomy_identity_frames(self):
        frames = [np.eye(3)] * 4
        H = cartan.loop_holonomy(frames)
        assert_allclose(H, np.eye(3), atol=1e-14)

    def test_holonomy_deviation_identity(self):
        assert_allclose(cartan.holonomy_deviation(np.eye(3)), 0.0, atol=1e-14)

    def test_rotation_angle_identity(self):
        assert_allclose(cartan.rotation_angle(np.eye(3)), 0.0, atol=1e-14)

    def test_rotation_angle_90deg(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        assert_allclose(cartan.rotation_angle(R), np.pi / 2, rtol=1e-10)

    def test_is_half_disclination_false_for_identity(self):
        assert not cartan.is_half_disclination(np.eye(3))

    def test_is_half_disclination_true_for_pi_rotation(self):
        R = np.diag([1.0, -1.0, -1.0])  # 180-degree rotation about x-axis
        assert cartan.is_half_disclination(R)

    def test_loop_holonomy_requires_2_frames(self):
        with pytest.raises(ValueError):
            cartan.loop_holonomy([np.eye(3)])

    def test_edge_transition_returns_3x3(self):
        R1 = np.eye(3)
        c, s = np.cos(0.3), np.sin(0.3)
        R2 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        T = cartan.edge_transition(R1, R2)
        assert T.shape == (3, 3)

    def test_holonomy_deviation_pi_rotation(self):
        # Rotation by pi: ||R - I||_F = 2*sqrt(2)
        R = np.diag([1.0, -1.0, -1.0])
        expected = 2.0 * np.sqrt(2.0)
        assert_allclose(cartan.holonomy_deviation(R), expected, atol=1e-10)

    def test_rotation_angle_pi_rotation(self):
        # Rotation by pi about x-axis
        R = np.diag([1.0, -1.0, -1.0])
        assert_allclose(cartan.rotation_angle(R), np.pi, atol=1e-12)

    def test_is_half_disclination_custom_threshold(self):
        # A 90-degree rotation: not a half-disclination with threshold=pi/2,
        # but is with threshold=pi/4.
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        assert not cartan.is_half_disclination(R, threshold=np.pi / 2)
        assert cartan.is_half_disclination(R, threshold=np.pi / 4)


class TestScanDisclinations:
    def test_uniform_field_no_defects(self):
        nx, ny = 5, 5
        frames = [np.eye(3)] * (nx * ny)
        result = cartan.scan_disclinations(frames, nx, ny)
        assert len(result) == 0

    def test_wrong_frame_count_raises(self):
        with pytest.raises(ValueError):
            cartan.scan_disclinations([np.eye(3)] * 10, 5, 5)

    def test_disclination_has_attributes(self):
        nx, ny = 10, 10
        frames = []
        for j in range(ny):
            for i in range(nx):
                theta = 0.5 * np.arctan2(j - ny / 2, i - nx / 2)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
                frames.append(R)
        result = cartan.scan_disclinations(frames, nx, ny)
        if len(result) > 0:
            d = result[0]
            assert hasattr(d, 'plaquette')
            assert hasattr(d, 'angle')
            assert hasattr(d, 'holonomy')
            assert d.holonomy.shape == (3, 3)
            assert "Disclination" in repr(d)

    def test_planted_defect_detected(self):
        # Construct a +1/2 disclination in a 2x2 grid (one plaquette).
        # frames[i*ny + j]: (0,0)=0, (1,0)=2, (0,1)=1, (1,1)=3
        def rot_z(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

        frames = [np.eye(3)] * 4
        frames[0] = rot_z(0.0)               # (0,0)
        frames[2] = rot_z(np.pi / 4)         # (1,0)
        frames[3] = rot_z(np.pi / 2)         # (1,1)
        frames[1] = rot_z(3 * np.pi / 4)    # (0,1)

        result = cartan.scan_disclinations(frames, 2, 2)
        assert len(result) == 1
        d = result[0]
        assert d.plaquette == (0, 0)
        assert_allclose(d.angle, np.pi, atol=1e-10)
        assert d.holonomy.shape == (3, 3)

    def test_disclination_repr(self):
        def rot_z(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

        frames = [np.eye(3)] * 4
        frames[0] = rot_z(0.0)
        frames[2] = rot_z(np.pi / 4)
        frames[3] = rot_z(np.pi / 2)
        frames[1] = rot_z(3 * np.pi / 4)

        result = cartan.scan_disclinations(frames, 2, 2)
        assert len(result) == 1
        r = repr(result[0])
        assert "Disclination" in r
        assert "plaquette" in r
        assert "angle" in r
