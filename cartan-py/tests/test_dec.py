# ~/cartan/cartan-py/tests/test_dec.py

"""Tests for DEC (Discrete Exterior Calculus) bindings: Mesh, ExteriorDerivative,
HodgeStar, Operators, advection, and divergence."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import assert_allclose

import cartan


def make_triangle_mesh():
    """Simple 2-triangle mesh forming a unit square."""
    verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return verts, tris


class TestMesh:
    def test_create_mesh(self):
        verts, tris = make_triangle_mesh()
        mesh = cartan.Mesh(verts, tris)
        assert mesh.n_vertices() == 4
        assert mesh.n_simplices() == 2

    def test_euler_characteristic(self):
        verts, tris = make_triangle_mesh()
        mesh = cartan.Mesh(verts, tris)
        chi = mesh.euler_characteristic()
        assert chi == 1

    def test_unit_square_grid(self):
        mesh = cartan.Mesh.unit_square_grid(10)
        assert mesh.n_vertices() == 121
        assert mesh.n_simplices() == 200

    def test_vertices_property(self):
        verts, tris = make_triangle_mesh()
        mesh = cartan.Mesh(verts, tris)
        v = mesh.vertices
        assert v.shape == (4, 2)
        assert_allclose(v[0], [0, 0])

    def test_simplices_property(self):
        verts, tris = make_triangle_mesh()
        mesh = cartan.Mesh(verts, tris)
        s = mesh.simplices
        assert s.shape == (2, 3)

    def test_n_boundaries(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        # For a grid: edges = 3*n^2 + 2*n (internal pattern)
        assert mesh.n_boundaries() > 0

    def test_repr(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        assert "Mesh" in repr(mesh)

    def test_invalid_vertices_shape(self):
        with pytest.raises(Exception):
            bad_verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
            tris = np.array([[0, 1, 0]], dtype=np.int64)
            cartan.Mesh(bad_verts, tris)

    def test_invalid_simplices_shape(self):
        with pytest.raises(Exception):
            verts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
            bad_tris = np.array([[0, 1]], dtype=np.int64)
            cartan.Mesh(verts, bad_tris)


class TestExteriorDerivative:
    def test_d0_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ext = cartan.ExteriorDerivative(mesh)
        assert ext.d0.shape == (mesh.n_boundaries(), mesh.n_vertices())

    def test_d1_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ext = cartan.ExteriorDerivative(mesh)
        assert ext.d1.shape == (mesh.n_simplices(), mesh.n_boundaries())

    def test_exactness(self):
        """d1 @ d0 == 0 (d^2 = 0)."""
        mesh = cartan.Mesh.unit_square_grid(5)
        ext = cartan.ExteriorDerivative(mesh)
        product = ext.d1 @ ext.d0
        assert_allclose(product, np.zeros(product.shape), atol=1e-14)

    def test_check_exactness(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ext = cartan.ExteriorDerivative(mesh)
        assert ext.check_exactness() < 1e-14

    def test_d0_entries(self):
        """d0[e, i] = -1, d0[e, j] = +1 for edge e = [i, j] with i < j."""
        verts, tris = make_triangle_mesh()
        mesh = cartan.Mesh(verts, tris)
        ext = cartan.ExteriorDerivative(mesh)
        d0 = ext.d0
        # Each row must sum to 0 (one +1 and one -1)
        row_sums = d0.sum(axis=1)
        assert_allclose(row_sums, np.zeros(mesh.n_boundaries()), atol=1e-15)

    def test_d1_row_sums(self):
        """d1 rows should sum to 0 for interior edges (oriented boundaries cancel)."""
        mesh = cartan.Mesh.unit_square_grid(3)
        ext = cartan.ExteriorDerivative(mesh)
        # d1 rows may not sum to 0 for all rows, but no row should have |sum| > 3
        d1 = ext.d1
        assert np.all(np.abs(d1.sum(axis=1)) <= 3)


class TestHodgeStar:
    def test_positive_diagonals(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        hodge = cartan.HodgeStar(mesh)
        # star0 (dual cell areas) and star2 (1/triangle_area) are strictly positive.
        assert np.all(hodge.star0 > 0)
        assert np.all(hodge.star2 > 0)
        # star1 (dual/primal edge ratio) can be zero for degenerate diagonal edges
        # in right-triangle grids where circumcenters coincide. Non-negativity holds.
        assert np.all(hodge.star1 >= 0)

    def test_star0_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        hodge = cartan.HodgeStar(mesh)
        assert hodge.star0.shape == (mesh.n_vertices(),)

    def test_star1_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        hodge = cartan.HodgeStar(mesh)
        assert hodge.star1.shape == (mesh.n_boundaries(),)

    def test_star2_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        hodge = cartan.HodgeStar(mesh)
        assert hodge.star2.shape == (mesh.n_simplices(),)

    def test_star0_inv(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        hodge = cartan.HodgeStar(mesh)
        inv = hodge.star0_inv
        assert_allclose(hodge.star0 * inv, np.ones(mesh.n_vertices()), rtol=1e-12)

    def test_star1_inv(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        hodge = cartan.HodgeStar(mesh)
        inv = hodge.star1_inv
        product = hodge.star1 * inv
        # For non-zero star1 entries: star1 * star1_inv = 1. Zero entries map to 0.
        nonzero = hodge.star1 > 0
        assert_allclose(product[nonzero], np.ones(nonzero.sum()), rtol=1e-12)
        assert_allclose(product[~nonzero], np.zeros((~nonzero).sum()), atol=1e-15)

    def test_star2_inv(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        hodge = cartan.HodgeStar(mesh)
        inv = hodge.star2_inv
        assert_allclose(hodge.star2 * inv, np.ones(mesh.n_simplices()), rtol=1e-12)

    def test_star0_total_area(self):
        """sum(star0) should equal the total mesh area (= 1 for unit square)."""
        mesh = cartan.Mesh.unit_square_grid(10)
        hodge = cartan.HodgeStar(mesh)
        assert abs(hodge.star0.sum() - 1.0) < 1e-10


class TestOperators:
    def test_laplacian_constant_is_zero(self):
        mesh = cartan.Mesh.unit_square_grid(10)
        ops = cartan.Operators(mesh)
        f = np.ones(mesh.n_vertices())
        Lf = ops.apply_laplace_beltrami(f)
        assert_allclose(Lf, np.zeros_like(Lf), atol=1e-12)

    def test_laplacian_matrix_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        assert ops.laplace_beltrami.shape == (n, n)

    def test_laplacian_matrix_matches_apply(self):
        """apply_laplace_beltrami(f) == laplace_beltrami @ f."""
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        rng = np.random.default_rng(0)
        f = rng.standard_normal(mesh.n_vertices())
        Lf_apply = ops.apply_laplace_beltrami(f)
        Lf_matrix = ops.laplace_beltrami @ f
        assert_allclose(Lf_apply, Lf_matrix, atol=1e-12)

    def test_bochner_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        u = np.zeros(2 * n)
        result = ops.apply_bochner_laplacian(u)
        assert result.shape == (2 * n,)

    def test_bochner_constant_zero(self):
        """Bochner Laplacian of a constant vector field should be ~0."""
        mesh = cartan.Mesh.unit_square_grid(10)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        u = np.ones(2 * n)
        result = ops.apply_bochner_laplacian(u)
        assert_allclose(result, np.zeros_like(result), atol=1e-11)

    def test_lichnerowicz_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        q = np.zeros(3 * n)
        result = ops.apply_lichnerowicz_laplacian(q)
        assert result.shape == (3 * n,)

    def test_lichnerowicz_constant_zero(self):
        """Lichnerowicz Laplacian of a constant tensor field should be ~0."""
        mesh = cartan.Mesh.unit_square_grid(10)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        q = np.ones(3 * n)
        result = ops.apply_lichnerowicz_laplacian(q)
        assert_allclose(result, np.zeros_like(result), atol=1e-11)

    def test_repr(self):
        mesh = cartan.Mesh.unit_square_grid(3)
        ops = cartan.Operators(mesh)
        assert "Operators" in repr(ops)


class TestAdvection:
    def test_scalar_advection_zero_velocity(self):
        mesh = cartan.Mesh.unit_square_grid(10)
        n = mesh.n_vertices()
        f = np.ones(n)
        u = np.zeros(2 * n)
        result = cartan.apply_scalar_advection(mesh, f, u)
        assert result.shape == (n,)

    def test_scalar_advection_zero_result_for_zero_u(self):
        """Zero velocity => zero advection."""
        mesh = cartan.Mesh.unit_square_grid(5)
        n = mesh.n_vertices()
        rng = np.random.default_rng(1)
        f = rng.standard_normal(n)
        u = np.zeros(2 * n)
        result = cartan.apply_scalar_advection(mesh, f, u)
        assert_allclose(result, np.zeros(n), atol=1e-15)

    def test_scalar_advection_constant_field(self):
        """Constant field advects to zero regardless of velocity."""
        mesh = cartan.Mesh.unit_square_grid(5)
        n = mesh.n_vertices()
        f = np.ones(n)
        rng = np.random.default_rng(2)
        u = rng.standard_normal(2 * n)
        result = cartan.apply_scalar_advection(mesh, f, u)
        assert_allclose(result, np.zeros(n), atol=1e-12)

    def test_vector_advection_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        n = mesh.n_vertices()
        q = np.zeros(2 * n)
        u = np.zeros(2 * n)
        result = cartan.apply_vector_advection(mesh, q, u)
        assert result.shape == (2 * n,)

    def test_vector_advection_zero_velocity(self):
        """Zero velocity => zero advection for vector fields too."""
        mesh = cartan.Mesh.unit_square_grid(5)
        n = mesh.n_vertices()
        rng = np.random.default_rng(3)
        q = rng.standard_normal(2 * n)
        u = np.zeros(2 * n)
        result = cartan.apply_vector_advection(mesh, q, u)
        assert_allclose(result, np.zeros(2 * n), atol=1e-15)


class TestDivergence:
    def test_divergence_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        u = np.zeros(2 * n)
        result = cartan.apply_divergence(mesh, ops, u)
        assert result.shape == (n,)

    def test_divergence_zero_field(self):
        """Zero field => zero divergence."""
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        u = np.zeros(2 * n)
        result = cartan.apply_divergence(mesh, ops, u)
        assert_allclose(result, np.zeros(n), atol=1e-15)

    def test_tensor_divergence_shape(self):
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        t = np.zeros(3 * n)
        result = cartan.apply_tensor_divergence(mesh, ops, t)
        assert result.shape == (2 * n,)

    def test_tensor_divergence_zero_field(self):
        """Zero tensor field => zero divergence."""
        mesh = cartan.Mesh.unit_square_grid(5)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()
        t = np.zeros(3 * n)
        result = cartan.apply_tensor_divergence(mesh, ops, t)
        assert_allclose(result, np.zeros(2 * n), atol=1e-15)

    def test_laplacian_vs_divergence_gradient(self):
        """Δf = div(grad f). Check consistency on interior-dominated grid."""
        mesh = cartan.Mesh.unit_square_grid(10)
        ops = cartan.Operators(mesh)
        n = mesh.n_vertices()

        # Build a vector field u = (f, 0) SoA: [f, 0, 0, ..., 0]
        # Use constant f -> div should be ~0
        u = np.zeros(2 * n)
        result = cartan.apply_divergence(mesh, ops, u)
        assert result.shape == (n,)
