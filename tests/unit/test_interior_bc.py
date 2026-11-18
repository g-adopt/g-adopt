"""Tests for InteriorBC on scalar, vector, extruded and mixed-subspace spaces.

InteriorBC must constrain exactly the nodes that a DirichletBC on the same
boundary leaves free. A boundary-only mass solve with an InteriorBC must then
give the boundary values of the right-hand side and zero in the interior.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import firedrake as fd

from gadopt.utility import InteriorBC

SOLVER_PARAMETERS = {"ksp_type": "preonly", "pc_type": "lu"}


def check_node_partition(V, boundary_id):
    """Check that InteriorBC and DirichletBC nodes partition the local nodes.

    Args:
      V: Function space on which both boundary conditions are built
      boundary_id: Boundary marker passed to both boundary conditions

    Returns:
      Tuple of InteriorBC nodes and DirichletBC nodes
    """
    interior_nodes = InteriorBC(V, 0, boundary_id)._nodes
    boundary_nodes = fd.DirichletBC(V, 0, boundary_id)._nodes
    # The node indices address rows of the data array, halo rows included.
    all_nodes = np.arange(fd.Function(V).dat.data_ro_with_halos.shape[0])

    assert np.intersect1d(interior_nodes, boundary_nodes).size == 0
    assert_array_equal(np.union1d(interior_nodes, boundary_nodes), all_nodes)

    return interior_nodes, boundary_nodes


def make_case(case):
    """Build the function space, boundary marker, measure and right-hand side.

    Args:
      case: One of "scalar", "vector" or "extruded"

    Returns:
      Tuple of function space, boundary marker, boundary measure and
      right-hand side expression
    """
    if case == "extruded":
        mesh = fd.ExtrudedMesh(fd.UnitIntervalMesh(4), 4)
        boundary_id, measure = "top", fd.ds_t
    else:
        mesh = fd.UnitSquareMesh(4, 4)
        boundary_id, measure = 4, fd.ds(4)
    x = fd.SpatialCoordinate(mesh)[0]

    # The right-hand sides lie in the CG2 trace space on the boundary, so the
    # boundary mass solve reproduces them exactly at the boundary nodes.
    if case == "vector":
        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        rhs = fd.as_vector((x**2, x))
    else:
        V = fd.FunctionSpace(mesh, "CG", 2)
        rhs = x**2

    return V, boundary_id, measure, rhs


@pytest.mark.parametrize("case", ["scalar", "vector", "extruded"])
def test_interior_bc(case):
    """Boundary-only mass solve with an InteriorBC."""
    V, boundary_id, measure, rhs = make_case(case)
    interior_nodes, boundary_nodes = check_node_partition(V, boundary_id)

    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    w = fd.Function(V)
    # The boundary mass matrix is singular in the interior. The InteriorBC
    # replaces those rows with the identity and sets the interior to zero.
    fd.solve(
        fd.inner(u, v) * measure == fd.inner(rhs, v) * measure,
        w,
        bcs=InteriorBC(V, 0, boundary_id),
        solver_parameters=SOLVER_PARAMETERS,
    )

    values = w.dat.data_ro_with_halos
    expected = fd.Function(V).interpolate(rhs).dat.data_ro_with_halos
    assert_allclose(values[interior_nodes], 0.0, atol=1e-12)
    assert_allclose(values[boundary_nodes], expected[boundary_nodes], atol=1e-10)


def test_interior_bc_mixed_subspace():
    """InteriorBC on the scalar subspace of a mixed space."""
    mesh = fd.UnitSquareMesh(4, 4)
    x = fd.SpatialCoordinate(mesh)[0]
    W = fd.VectorFunctionSpace(mesh, "CG", 2) * fd.FunctionSpace(mesh, "CG", 2)

    interior_nodes, boundary_nodes = check_node_partition(W.sub(1), 4)

    (u, eta), (v, q) = fd.TrialFunctions(W), fd.TestFunctions(W)
    z = fd.Function(W)
    # Velocity block: full mass matrix. Scalar block: boundary-only mass
    # matrix, as for a free-surface variable.
    a = fd.inner(u, v) * fd.dx + eta * q * fd.ds(4)
    L = fd.inner(fd.Constant((1.0, 1.0)), v) * fd.dx + x**2 * q * fd.ds(4)
    fd.solve(
        a == L,
        z,
        bcs=InteriorBC(W.sub(1), 0, 4),
        solver_parameters=SOLVER_PARAMETERS,
    )

    u_sol, eta_sol = z.subfunctions
    eta_values = eta_sol.dat.data_ro_with_halos
    expected = fd.Function(eta_sol.function_space()).interpolate(x**2).dat.data_ro_with_halos
    assert_allclose(eta_values[interior_nodes], 0.0, atol=1e-12)
    assert_allclose(eta_values[boundary_nodes], expected[boundary_nodes], atol=1e-10)
    # The velocity block has no boundary condition and is unaffected.
    assert_allclose(u_sol.dat.data_ro_with_halos, 1.0, rtol=1e-10)
