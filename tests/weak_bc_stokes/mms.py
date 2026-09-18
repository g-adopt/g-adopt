"""Manufactured-solution convergence of the weak boundary branches.

The symmetry and variational-structure cases check two properties of the weak
("u" and "un") SIPG boundary terms in `gadopt.momentum_equation.viscosity_term`:
that the Jacobian is symmetric, and that the residual is the first variation of
the documented boundary functional. This case checks the remaining property,
that the converged solution those terms produce solves the intended continuous
problem. A boundary term inconsistent with that continuous problem converges to
a different solution and degrades the order measured here, while keeping the
Jacobian symmetric, so the symmetry cases alone would pass it.

Both cases here are incompressible; the compressible branches stay covered by
the symmetry cases only.

Usage:
    python3 mms.py --case un      # weak "un" through StokesSolver
    python3 mms.py --case u       # weak "u" at the Equation level
"""

import argparse

import firedrake as fd
import gadopt
import numpy as np
from gadopt.equations import Equation
from gadopt.momentum_equation import viscosity_term

from stokes_helpers import MMS_RESOLUTIONS as RESOLUTIONS


def mu_of(w):
    """Solution-dependent viscosity, incompressible strain invariant.

    The same algebraic form as `nonlinear_mu(., compressible=False)` in
    `stokes_helpers`: 1 + inner(sym(grad(w)), sym(grad(w))).
    """
    return 1 + fd.inner(fd.sym(fd.grad(w)), fd.sym(fd.grad(w)))


def l2(expression, degree=12):
    """The L2 norm of `expression`, integrated at the given quadrature degree.

    The degree is well above the polynomial degree of the fields, so the
    quadrature error stays far below the discretisation error being measured.
    """
    return fd.sqrt(fd.assemble(fd.inner(expression, expression) * fd.dx(degree=degree)))


def weak_un():
    """MMS convergence of the weak "un" branch, driven through `StokesSolver`.

    The manufactured velocity comes from the streamfunction
    psi = sin(pi x) sin(pi y): divergence free, with u.n = 0 on all four sides
    and nonzero tangential slip, which is what a free-slip condition has to
    reproduce. The viscosity is built from `split(z)[0]`, so the nonlinear
    branch of the boundary terms is the one under test.

    Returns:
      One row per refinement: the velocity error, the pressure error, and the
      nodal interpolation error of the exact velocity. The interpolation error
      is a discretisation-only reference, computed with no solve, that the test
      uses to bound the error constant.
    """
    rows = []
    for N in RESOLUTIONS:
        mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        mesh.cartesian = True

        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        W = fd.FunctionSpace(mesh, "CG", 1)
        Z = V * W
        z = fd.Function(Z)
        u, p = fd.split(z)

        x, y = fd.SpatialCoordinate(mesh)
        u_ex = fd.as_vector([
            fd.pi * fd.sin(fd.pi * x) * fd.cos(fd.pi * y),
            -fd.pi * fd.cos(fd.pi * x) * fd.sin(fd.pi * y),
        ])
        p_ex = fd.cos(fd.pi * x) * fd.cos(fd.pi * y)  # zero mean on unit square

        # Ra = 0 removes the buoyancy term, leaving the manufactured forcing as
        # the only source.
        approximation = gadopt.BoussinesqApproximation(0, mu=mu_of(u))

        sigma_ex = 2 * mu_of(u_ex) * fd.sym(fd.grad(u_ex))
        f_mms = -fd.div(sigma_ex) + fd.grad(p_ex)
        v = fd.TestFunctions(Z)[0]
        # The sign matches the convention of momentum_source_term, which
        # subtracts the source from the residual.
        forcing = -fd.dot(v, f_mms) * fd.dx(degree=8)

        n = fd.FacetNormal(mesh)
        traction = fd.dot(sigma_ex, n)
        # Tangential viscous traction: the free-slip condition fixes the normal
        # velocity and leaves this traction as data.
        t_ex = traction - fd.dot(n, traction) * n

        bcs = {bid: {"un": 0, "stress": t_ex}
               for bid in list(gadopt.get_boundary_ids(mesh))}

        Z_nullspace = gadopt.create_stokes_nullspace(
            Z, closed=True, rotational=False)

        solver = gadopt.StokesSolver(
            z, approximation,
            bcs=bcs,
            additional_forcing_term=forcing,
            quad_degree=8,
            solver_parameters="direct",
            # Backtracking line search: the g-adopt default "l2" search diverges
            # from the zero initial guess on this cubic-viscosity residual.
            # Convergence itself is part of what this case asserts, so a failed
            # solve must raise and fail the step.
            solver_parameters_extra={"snes_rtol": 1e-10, "snes_atol": 1e-12,
                                     "snes_linesearch_type": "bt",
                                     "snes_max_it": 100},
            nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
        )
        solver.solve()

        u_h, p_h = z.subfunctions
        # The pressure is fixed only up to a constant by the closed nullspace,
        # so remove its mean before comparing. The domain volume is 1.
        p_mean = fd.assemble(p_h * fd.dx)
        rows.append([
            l2(u_h - u_ex),
            l2(p_h - p_mean - p_ex),
            l2(u_ex - fd.Function(V).interpolate(u_ex)),
        ])
    return np.array(rows)


def weak_u():
    """MMS convergence of the weak "u" branch, driven at the `Equation` level.

    `StokesSolver` converts a "u" boundary condition to a strong `DirichletBC`,
    so the solver never reaches the weak branch. The problem here is velocity
    only, with no pressure, so the manufactured velocity does not need to be
    divergence free; it only needs nonzero strain and nonzero boundary values.

    Returns:
      One row per refinement: the velocity error and the nodal interpolation
      error of the exact velocity.
    """
    rows = []
    for N in RESOLUTIONS:
        mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        mesh.cartesian = True

        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        u = fd.Function(V)  # zero initial guess

        x, y = fd.SpatialCoordinate(mesh)
        u_ex = fd.as_vector([
            fd.sin(fd.pi * x) * fd.cos(fd.pi * y) + 0.3 * y**2,
            fd.cos(fd.pi * x) * fd.sin(fd.pi * y) + 0.2 * x**2,
        ])

        # mu of the unknown Function, which selects the nonlinear branch.
        approximation = gadopt.BoussinesqApproximation(1, mu=mu_of(u))
        f_mms = -fd.div(2 * mu_of(u_ex) * fd.sym(fd.grad(u_ex)))

        bcs = {bid: {"u": u_ex} for bid in list(gadopt.get_boundary_ids(mesh))}
        eq = Equation(fd.TestFunction(V), V, viscosity_term,
                      eq_attrs={"stress": approximation.stress(u)},
                      approximation=approximation, bcs=bcs, quad_degree=8)
        F = eq.residual(u) - fd.dot(fd.TestFunction(V), f_mms) * fd.dx(degree=8)

        problem = fd.NonlinearVariationalProblem(F, u)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters={
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10, "snes_atol": 1e-12,
            # Backtracking takes small steps in the stiff cubic region and can
            # exceed the PETSc default of 50 iterations, so raise the cap.
            "snes_max_it": 200, "ksp_type": "preonly",
            "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
        # The operator is monotone; a failed solve must fail the step.
        solver.solve()

        rows.append([
            l2(u - u_ex),
            l2(u_ex - fd.Function(V).interpolate(u_ex)),
        ])
    return np.array(rows)


CASES = {"un": weak_un, "u": weak_u}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES),
                        help="which weak boundary branch to drive")
    args = parser.parse_args()

    np.savetxt(f"mms-{args.case}.dat", CASES[args.case]())
