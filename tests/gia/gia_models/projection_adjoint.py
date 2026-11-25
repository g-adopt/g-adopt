"""Helpers of `tests/unit/test_gia_nullspace_projection_adjoint.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_gia_nullspace_projection_adjoint.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    stop_annotating,
)
from gadopt import project_out_kernel


# ---------------------------------------------------------------------------
# Part 1: the block on a small problem without a solver
# ---------------------------------------------------------------------------
def _small_case(comm=fd.COMM_WORLD):
    """A mixed space, a rotation basis on its vector block, and a control.

    The vector block carries the 2-D rotation about the origin. The scalar
    block has no basis, so the projection must leave it unchanged.

    Args:
      comm: the communicator of the mesh.

    Returns:
      `(Z, nullspace, control, direction)`.
    """
    mesh = fd.UnitSquareMesh(6, 6, comm=comm)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    Q = fd.FunctionSpace(mesh, "CG", 1)
    Z = V * Q
    X = fd.SpatialCoordinate(mesh)
    with stop_annotating():
        rotation = fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]]))
        basis = fd.VectorSpaceBasis([rotation], comm=mesh.comm)
        basis.orthonormalize()
        nullspace = fd.MixedVectorSpaceBasis(Z, [basis, Z.sub(1)])
        control = fd.Function(Q).interpolate(1.0 + X[0])
        direction = fd.Function(Q).interpolate(0.3 * fd.sin(3 * X[1]) + 0.1)
    return Z, nullspace, control, direction


def _small_forward(Z, nullspace, control, project=True):
    """Fill a mixed function from the control, project, copy, and integrate.

    The vector block is a nonlinear function of the control with a large
    rotation part, so the projection removes a large component and the
    Hessian of the functional is not zero. The copy after the projection is
    the pattern of the solver (`u.assign(...)` after the projection).

    Args:
      Z: the mixed space.
      nullspace: the nullspace of `Z` from `_small_case`.
      control: the control Function.
      project: whether to apply the projection.

    Returns:
      The functional, an `AdjFloat` when annotation is on.
    """
    X = fd.SpatialCoordinate(Z.mesh())
    c = control
    z = fd.Function(Z)
    z.subfunctions[0].interpolate(
        fd.as_vector([c**2 - 3 * X[1] * c, 3 * X[0] * c + c**3]))
    z.subfunctions[1].interpolate(c**2)
    if project:
        project_out_kernel(nullspace, z)
    copy = fd.Function(Z).assign(z)
    u, p = fd.split(copy)
    # The first term sees the rotation part of `u`, the second is the flux of
    # the first component through the top edge, which a rotation changes.
    return fd.assemble((fd.inner(u, u) + p * fd.inner(u, u)) * fd.dx
                       + u[0] * fd.ds(4))


def _small_taped(Z, nullspace, control):
    """Tape `_small_forward` and return `(J, rf)`."""
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        J = _small_forward(Z, nullspace, control)
        rf = ReducedFunctional(J, Control(control))
    finally:
        pause_annotation()
    return J, rf


def _shifted(control, direction):
    """`control + direction` as a new Function, without annotation."""
    with stop_annotating():
        shifted = control.copy(deepcopy=True)
        shifted += direction
    return shifted
