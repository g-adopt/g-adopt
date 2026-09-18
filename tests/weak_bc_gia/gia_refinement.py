"""Gap between the coupled and the pointwise formulation under refinement.

The coupled displacement rows carry the elastic tangent in the symmetrising term
and the elastic modulus in the penalty, so their residual differs from the
pointwise one by terms proportional to the normal jump n.u on the weak boundary.
Those terms vanish when the exact solution satisfies the boundary condition, so
both the gap between the two solutions and the normal jump itself must fall at
the discretisation order under refinement.

This case bounds the size of the difference between the two formulations. It
does not detect an asymmetric displacement block; the `block_symmetry` group of
`gia_symmetry.py` does that.

Usage:
    python3 gia_refinement.py --dt-over-tau 0.25
"""

import argparse
from math import sqrt

import firedrake as fd
import gadopt
import numpy as np

from gia_helpers import (
    MAXWELL_TIME,
    REFINEMENT_RESOLUTIONS,
    maxwell_approximation,
)

# Boundary identifiers of a UnitSquareMesh: the two vertical sides carry the
# weak condition, the bottom is held, and the top carries the load.
LEFT, RIGHT, BOTTOM, TOP = 1, 2, 3, 4


def surface_load_bcs(load):
    """Weak no-normal-displacement on the sides, held bottom, loaded top."""
    return {
        LEFT: {"un": 0.0},
        RIGHT: {"un": 0.0},
        BOTTOM: {"uy": 0.0},
        TOP: {"normal_stress": load},
    }


def solve_surface_load(solver_kind, mesh, dt):
    """Solve one viscoelastic step under a surface load and return the displacement.

    A unit square is loaded by a normal stress on the top boundary, held fixed
    at the bottom, and given a weak no-normal-displacement condition on the two
    sides. Cells are affine, so the pointwise and the coupled formulation
    discretise the same continuous problem and differ only through the weak
    boundary terms.

    Args:
      solver_kind: "pointwise" or "coupled".
      mesh: the mesh both formulations are solved on.
      dt: the time step.

    Returns:
      The displacement Function.
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    x, _ = fd.SpatialCoordinate(mesh)
    # A smooth, mean-free surface load. Mean-free so the weak sides are not
    # asked to absorb a net normal force.
    load = fd.cos(2 * fd.pi * x)
    # B_mu = 0 removes the buoyancy term, leaving the load as the only forcing.
    approximation = maxwell_approximation(mesh, B_mu=0.0)
    bcs = surface_load_bcs(load)

    if solver_kind == "pointwise":
        u = fd.Function(V)
        solver = gadopt.InternalVariableSolver(
            u, approximation, dt=dt, internal_variables=fd.Function(S),
            bcs=bcs, solver_parameters="direct",
        )
        solver.solve()
        return u

    z = fd.Function(V * S)
    solver = gadopt.CoupledInternalVariableSolver(
        z, approximation, dt=dt, bcs=bcs, solver_parameters="direct",
    )
    solver.solve()
    return z.subfunctions[0]


def gap_under_refinement(dt_over_tau):
    """Measure the two-formulation gap and the normal jump at two resolutions.

    Returns:
      One row per resolution: the L2 gap between the coupled and the pointwise
      displacement, the L2 normal jump of the coupled solution on the two weak
      boundaries, and the L2 norm of the pointwise displacement. The last is a
      scale for the test to check that the coarse gap is a real difference
      between the formulations and not round-off, which would make the rate
      meaningless.
    """
    dt = dt_over_tau * MAXWELL_TIME
    rows = []
    for resolution in REFINEMENT_RESOLUTIONS:
        mesh = fd.UnitSquareMesh(resolution, resolution)
        mesh.cartesian = True
        u_pointwise = solve_surface_load("pointwise", mesh, dt)
        u_coupled = solve_surface_load("coupled", mesh, dt)

        difference = u_coupled - u_pointwise
        n = fd.FacetNormal(mesh)
        rows.append([
            sqrt(fd.assemble(fd.inner(difference, difference) * fd.dx)),
            sqrt(fd.assemble(
                fd.dot(n, u_coupled) ** 2 * (fd.ds(LEFT) + fd.ds(RIGHT)))),
            sqrt(fd.assemble(fd.inner(u_pointwise, u_pointwise) * fd.dx)),
        ])
    return np.array(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt-over-tau", required=True, type=float,
                        help="ratio of the time step to the Maxwell time")
    args = parser.parse_args()

    np.savetxt(f"refinement-{args.dt_over_tau}.dat",
               gap_under_refinement(args.dt_over_tau))
