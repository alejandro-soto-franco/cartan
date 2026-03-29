import numpy as np
import pytest
from conftest import assert_allclose

import cartan


class TestDistMatrix:
    def test_dist_matrix_shape(self):
        m = cartan.Sphere(2)
        points = [m.random_point(seed=i) for i in range(5)]
        D = m.dist_matrix(points)
        assert D.shape == (5, 5)

    def test_dist_matrix_symmetric(self):
        m = cartan.Sphere(2)
        points = [m.random_point(seed=i) for i in range(5)]
        D = m.dist_matrix(points)
        assert_allclose(D, D.T, atol=1e-14)

    def test_dist_matrix_zero_diagonal(self):
        m = cartan.Sphere(2)
        points = [m.random_point(seed=i) for i in range(5)]
        D = m.dist_matrix(points)
        assert_allclose(np.diag(D), np.zeros(5), atol=1e-14)

    def test_dist_matrix_spd(self):
        m = cartan.SPD(3)
        points = [m.random_point(seed=i) for i in range(4)]
        D = m.dist_matrix(points)
        assert D.shape == (4, 4)
        assert_allclose(D, D.T, atol=1e-14)
        assert_allclose(np.diag(D), np.zeros(4), atol=1e-14)

    def test_dist_matrix_euclidean(self):
        m = cartan.Euclidean(3)
        points = [np.array([float(i), 0.0, 0.0]) for i in range(3)]
        D = m.dist_matrix(points)
        assert_allclose(D[0, 1], 1.0)
        assert_allclose(D[0, 2], 2.0)
        assert_allclose(D[1, 2], 1.0)

    def test_dist_matrix_so(self):
        m = cartan.SO(3)
        points = [m.random_point(seed=i) for i in range(3)]
        D = m.dist_matrix(points)
        assert D.shape == (3, 3)
        assert_allclose(D, D.T, atol=1e-14)

    def test_dist_matrix_corr(self):
        m = cartan.Corr(3)
        points = [m.random_point(seed=i) for i in range(3)]
        D = m.dist_matrix(points)
        assert D.shape == (3, 3)
        assert_allclose(D, D.T, atol=1e-14)

    def test_dist_matrix_qtensor(self):
        m = cartan.QTensor3()
        points = [m.random_point(seed=i) for i in range(4)]
        D = m.dist_matrix(points)
        assert D.shape == (4, 4)
        assert_allclose(D, D.T, atol=1e-14)
        assert_allclose(np.diag(D), np.zeros(4), atol=1e-14)

    def test_dist_matrix_single_point(self):
        m = cartan.Sphere(2)
        points = [m.random_point(seed=0)]
        D = m.dist_matrix(points)
        assert D.shape == (1, 1)
        assert_allclose(D[0, 0], 0.0, atol=1e-14)


class TestExpBatch:
    def test_exp_batch_euclidean(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        vs = [np.array([float(i), 0.0, 0.0]) for i in range(4)]
        results = m.exp_batch(p, vs)
        assert len(results) == 4
        for i, r in enumerate(results):
            assert_allclose(r, np.array([float(i), 0.0, 0.0]))

    def test_exp_batch_sphere(self):
        m = cartan.Sphere(2)
        p = m.random_point(seed=0)
        vs = [m.random_tangent(p, seed=i) * 0.1 for i in range(3)]
        results = m.exp_batch(p, vs)
        assert len(results) == 3
        for r in results:
            # Points should be on the sphere (unit norm)
            assert_allclose(np.linalg.norm(r), 1.0, atol=1e-12)

    def test_exp_batch_spd(self):
        m = cartan.SPD(3)
        p = m.random_point(seed=0)
        vs = [m.random_tangent(p, seed=i) * 0.1 for i in range(3)]
        results = m.exp_batch(p, vs)
        assert len(results) == 3

    def test_exp_batch_so(self):
        m = cartan.SO(3)
        p = m.random_point(seed=0)
        vs = [m.random_tangent(p, seed=i) * 0.1 for i in range(3)]
        results = m.exp_batch(p, vs)
        assert len(results) == 3

    def test_exp_batch_corr(self):
        m = cartan.Corr(3)
        p = m.random_point(seed=0)
        vs = [m.random_tangent(p, seed=i) * 0.1 for i in range(3)]
        results = m.exp_batch(p, vs)
        assert len(results) == 3

    def test_exp_batch_qtensor(self):
        m = cartan.QTensor3()
        p = m.random_point(seed=0)
        vs = [m.random_tangent(p, seed=i) * 0.1 for i in range(3)]
        results = m.exp_batch(p, vs)
        assert len(results) == 3

    def test_exp_batch_empty(self):
        m = cartan.Euclidean(3)
        p = np.zeros(3)
        results = m.exp_batch(p, [])
        assert len(results) == 0
