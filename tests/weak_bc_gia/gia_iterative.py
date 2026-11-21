"""Convergence of the coupled iterative preset with weak "un" boundaries.

The preset preconditions the displacement block with CG inside a fieldsplit. CG
is defined only for a symmetric operator, so an asymmetric displacement block is
a solver-level defect and not only an aesthetic one. A large ratio of the time
step to the Maxwell time makes the asymmetry largest, so that is what runs here.

Usage:
    python3 gia_iterative.py
"""

import firedrake as fd
import gadopt
import numpy as np

from gia_helpers import MAXWELL_TIME, maxwell_approximation

# The ratio of time step to Maxwell time that maximises the asymmetry a wrong
# symmetrising coefficient would produce.
DT_OVER_TAU = 25.0
RESOLUTION = 8


def converged_reasons():
    """Solve the coupled system with the iterative preset and report convergence.

    Returns:
      A single row: the PETSc converged reason of the outer Newton solve, and
      that of the inner CG solve on the displacement block. Both are positive
      when the solve converged, and zero or negative otherwise.
    """
    mesh = fd.UnitSquareMesh(RESOLUTION, RESOLUTION)
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    z = fd.Function(V * S)

    x, _ = fd.SpatialCoordinate(mesh)
    # B_mu = 0 removes the buoyancy term, leaving the surface load as the only
    # forcing, as in the refinement case.
    approximation = maxwell_approximation(mesh, B_mu=0.0)
    bcs = {
        1: {"un": 0.0},
        2: {"un": 0.0},
        3: {"uy": 0.0},
        4: {"normal_stress": fd.cos(2 * fd.pi * x)},
    }
    solver = gadopt.CoupledInternalVariableSolver(
        z, approximation, dt=DT_OVER_TAU * MAXWELL_TIME, bcs=bcs,
        solver_parameters="iterative",
    )
    solver.solve()

    snes = solver.solver.snes
    displacement_ksp = snes.getKSP().getPC().getFieldSplitSubKSP()[0]
    return np.array([[float(snes.getConvergedReason()),
                      float(displacement_ksp.getConvergedReason())]])


if __name__ == "__main__":
    np.savetxt("iterative_preset.dat", converged_reasons())
