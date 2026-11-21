"""Weak free-slip against strong free-slip, on the Tosi viscoplastic rheology.

The manufactured-solution cases in `mms.py` pin the weak boundary branches
against exact fields on smooth problems. This case runs a realistic nonlinear
rheology instead and pins the weak free-slip ("un") solution against the trusted
strong-boundary-condition formulation.

The two formulations are consistent discretisations of the same continuous
problem, so their relative velocity difference decays at the P2 discretisation
order under refinement. A bug that scales both solves identically cancels in the
difference and is not caught here; `mms.py` pins the absolute solution instead.

This is a coarsened, fixed-temperature, single-timestep version of the Tosi
benchmark, not the full time-stepped run.

Usage:
    python3 tosi.py
"""

import firedrake as fd
import gadopt
import numpy as np

from stokes_helpers import TOSI_RESOLUTIONS as RESOLUTIONS


def viscosity(z, T, X):
    """Tosi viscoplastic viscosity, matching `viscoplastic_case_DG.py`.

    The velocity is taken from `split(z)[0]` of the solve's own state, so the
    viscosity depends on the unknown and the boundary terms take their nonlinear
    branch. The harmonic mean of a linear and a plastic branch caps the stress
    at the yield stress.

    Returns:
      (the viscoplastic viscosity, the linear branch on its own). The linear
      branch is returned separately for use as a continuation pre-solve.
    """
    u = fd.split(z)[0]
    gamma_T, gamma_Z = fd.Constant(fd.ln(10**5)), fd.Constant(fd.ln(10))
    mu_star, sigma_y = fd.Constant(0.001), fd.Constant(1.0)
    eps = fd.sym(fd.grad(u))
    # The regularisation keeps the second invariant away from zero where the
    # strain rate vanishes, which would otherwise divide by zero below.
    epsii = fd.sqrt(fd.inner(eps, eps) + 1e-10)
    mu_lin = fd.exp(-gamma_T * T + gamma_Z * (1 - X[1]))
    mu_plast = mu_star + (sigma_y / epsii)
    return (2. * mu_lin * mu_plast) / (mu_lin + mu_plast), mu_lin


def solve(mesh, bcs, use_switch):
    """Run one nonlinear Tosi Stokes solve.

    Args:
      mesh: the mesh to solve on.
      bcs: a function taking the boundary identifiers and returning the boundary
        condition dictionary, so the caller can choose the weak or the strong
        formulation.
      use_switch: run a linear pre-solve first (Spiegelman continuation) as a
        fallback if Newton from zero ever stagnates at higher resolution. Not
        needed at the resolutions used here, where both solves converge
        directly; kept for a future resolution increase.

    Returns:
      The velocity subfunction of the converged solution.
    """
    boundary = gadopt.get_boundary_ids(mesh)
    Z = fd.VectorFunctionSpace(mesh, "CG", 2) * fd.FunctionSpace(mesh, "CG", 1)
    z = fd.Function(Z)

    X = fd.SpatialCoordinate(mesh)
    Q = fd.FunctionSpace(mesh, "CG", 2)
    # Conductive profile plus a single-mode perturbation, held fixed: this case
    # takes one Stokes solve, not a time-stepped run.
    T = fd.Function(Q).interpolate(
        (1.0 - X[1]) + 0.05 * fd.cos(fd.pi * X[0]) * fd.sin(fd.pi * X[1]))

    mu_vp, mu_lin = viscosity(z, T, X)
    if use_switch:
        switch = fd.Constant(1.0)
        mu = fd.conditional(switch > 0.5, mu_vp, mu_lin)
    else:
        mu = mu_vp

    approximation = gadopt.BoussinesqApproximation(fd.Constant(100), mu=mu)
    Z_nullspace = gadopt.create_stokes_nullspace(
        Z, closed=True, rotational=False)
    solver = gadopt.StokesSolver(
        z, approximation, T,
        bcs=bcs(boundary),
        solver_parameters="direct",
        solver_parameters_extra={"snes_rtol": 1e-10, "snes_atol": 1e-12,
                                 "snes_max_it": 50},
        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
    )
    if use_switch:
        switch.assign(0.0)  # linear pre-solve
        solver.solve()
        switch.assign(1.0)
    solver.solve()
    return z.subfunctions[0]


def bcs_strong(b):
    """Free slip as strong conditions on the normal velocity component."""
    return {b.bottom: {"uy": 0}, b.top: {"uy": 0},
            b.left: {"ux": 0}, b.right: {"ux": 0}}


def bcs_weak(b):
    """Free slip as the weak "un" condition, which is the branch under test."""
    return {bid: {"un": 0} for bid in list(b)}


def weak_against_strong():
    """Relative velocity difference between the weak and strong solves.

    Returns:
      One row per refinement, holding
      ||u_weak - u_strong||_L2 / ||u_strong||_L2.
    """
    rows = []
    for N in RESOLUTIONS:
        mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        mesh.cartesian = True
        # Newton from zero converges directly for both formulations at these
        # resolutions, so the continuation pre-solve is not required.
        u_s = solve(mesh, bcs_strong, use_switch=False)
        u_w = solve(mesh, bcs_weak, use_switch=False)

        difference = fd.sqrt(fd.assemble(fd.inner(u_w - u_s, u_w - u_s) * fd.dx))
        magnitude = fd.sqrt(fd.assemble(fd.inner(u_s, u_s) * fd.dx))
        rows.append([difference / magnitude])
    return np.array(rows)


if __name__ == "__main__":
    np.savetxt("tosi.dat", weak_against_strong())
