"""Solver-output tests for the weak-BC momentum branches with nonlinear mu.

``test_symmetry.py`` checks that the Jacobian of the weak ("u"/"un") SIPG
boundary terms in ``gadopt.momentum_equation.viscosity_term`` is symmetric.
This file checks the complementary property: that the *converged solution*
those terms produce is the solution of the intended continuous problem. A
symmetric residual can still be a symmetric *wrong* residual — a term scaled
by the wrong constant stays the exact first variation of a (wrong)
functional, so its Jacobian is still symmetric — which is why solver output
needs its own coverage.

The file has three tests. ``test_mms_weak_un_convergence`` (1a) drives the weak
"un" branch through ``StokesSolver`` on a manufactured incompressible Stokes
problem and checks optimal convergence. ``test_mms_weak_u_convergence`` (1b)
drives the weak "u" branch at the ``Equation`` level (``StokesSolver`` converts
"u" to a strong ``DirichletBC``, so the solver never reaches that branch) on a
manufactured velocity-only problem. ``test_tosi_weak_freeslip_matches_strong``
(2) runs a single nonlinear Stokes solve with a realistic Tosi viscoplastic
rheology and pins the weak free-slip ("un") solution against the trusted
strong-BC formulation. Both MMS tests are incompressible; the compressible
branches stay covered only by the symmetry tests. Test 2 is a coarsened,
single-solve, fixed-temperature version of the Tosi benchmark, not the full
time-stepped run.
"""

from math import log2

import firedrake as fd
import gadopt
from gadopt.equations import Equation
from gadopt.momentum_equation import viscosity_term


def mu_of(w):
    """Solution-dependent viscosity, incompressible strain invariant.

    Same algebraic form as ``nonlinear_mu(., compressible=False)`` in
    ``test_symmetry.py``: ``1 + inner(sym(grad(w)), sym(grad(w)))``.
    """
    return 1 + fd.inner(fd.sym(fd.grad(w)), fd.sym(fd.grad(w)))


def test_mms_weak_un_convergence():
    """MMS convergence of the weak "un" branch through StokesSolver (1a)."""
    errs_u = []
    errs_p = []
    for N in (8, 16, 32):
        mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        mesh.cartesian = True

        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        W = fd.FunctionSpace(mesh, "CG", 1)
        Z = V * W
        z = fd.Function(Z)
        u, p = fd.split(z)

        X = fd.SpatialCoordinate(mesh)
        x, y = X
        # streamfunction psi = sin(pi x) sin(pi y): divergence-free, u.n = 0 on
        # all four sides, nonzero tangential slip.
        u_ex = fd.as_vector([
            fd.pi * fd.sin(fd.pi * x) * fd.cos(fd.pi * y),
            -fd.pi * fd.cos(fd.pi * x) * fd.sin(fd.pi * y),
        ])
        p_ex = fd.cos(fd.pi * x) * fd.cos(fd.pi * y)  # zero mean on unit square

        # mu built from split(z)[0] so mu_nonlinear is true (exercises new code).
        mu = mu_of(u)
        approximation = gadopt.BoussinesqApproximation(0, mu=mu)  # Ra=0: no buoyancy

        sigma_ex = 2 * mu_of(u_ex) * fd.sym(fd.grad(u_ex))
        f_mms = -fd.div(sigma_ex) + fd.grad(p_ex)
        v = fd.TestFunctions(Z)[0]
        # sign matches the forcing convention of momentum_source_term, which
        # subtracts the source from the residual.
        forcing = -fd.dot(v, f_mms) * fd.dx(degree=8)

        n = fd.FacetNormal(mesh)
        traction = fd.dot(sigma_ex, n)
        t_ex = traction - fd.dot(n, traction) * n  # tangential viscous traction

        bcs = {bid: {'un': 0, 'stress': t_ex}
               for bid in list(gadopt.get_boundary_ids(mesh))}

        Z_nullspace = gadopt.create_stokes_nullspace(
            Z, closed=True, rotational=False)

        solver = gadopt.StokesSolver(
            z, approximation,
            bcs=bcs,
            additional_forcing_term=forcing,
            quad_degree=8,
            solver_parameters="direct",
            # bt line search: the g-adopt default "l2" search diverges from the
            # zero initial guess on this cubic-viscosity residual; backtracking
            # converges. Convergence itself is part of what this test asserts.
            solver_parameters_extra={"snes_rtol": 1e-10, "snes_atol": 1e-12,
                                     "snes_linesearch_type": "bt",
                                     "snes_max_it": 100},
            nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
        )
        solver.solve()  # no try/except: a ConvergenceError must fail the test

        u_h, p_h = z.subfunctions
        err_u = fd.sqrt(fd.assemble(
            fd.inner(u_h - u_ex, u_h - u_ex) * fd.dx(degree=12)))
        p_mean = fd.assemble(p_h * fd.dx)  # domain volume is 1
        err_p = fd.sqrt(fd.assemble(
            (p_h - p_mean - p_ex)**2 * fd.dx(degree=12)))
        errs_u.append(err_u)
        errs_p.append(err_p)

    for k in range(2):
        order_u = log2(errs_u[k] / errs_u[k + 1])
        order_p = log2(errs_p[k] / errs_p[k + 1])
        assert order_u >= 2.5  # expected 3.0
        # observed p orders ~3.7, superconvergent vs the theoretical 2.0; a future
        # drop toward 2.0 is still correct, not a regression.
        assert order_p >= 1.5  # expected 2.0

    # Absolute regression guard on top of the order check above: a correct order
    # with a much worse constant would still pass the order assertions, so pin
    # the N=32 error to a small multiple of its converged value (err_u=1.71e-05,
    # err_p=4.57e-04).
    CAP_U = 3 * 1.71e-05
    CAP_P = 3 * 4.57e-04
    assert errs_u[-1] <= CAP_U
    assert errs_p[-1] <= CAP_P


def test_mms_weak_u_convergence():
    """MMS convergence of the weak "u" branch at the Equation level (1b)."""
    errs_u = []
    for N in (8, 16, 32):
        mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        mesh.cartesian = True

        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        u = fd.Function(V)  # initial guess zero

        X = fd.SpatialCoordinate(mesh)
        x, y = X
        # smooth, generic: nonzero strain and boundary values. No pressure, so
        # u_ex need not be divergence-free.
        u_ex = fd.as_vector([
            fd.sin(fd.pi * x) * fd.cos(fd.pi * y) + 0.3 * y**2,
            fd.cos(fd.pi * x) * fd.sin(fd.pi * y) + 0.2 * x**2,
        ])

        mu = mu_of(u)  # the unknown Function: triggers the nonlinear branch
        approximation = gadopt.BoussinesqApproximation(1, mu=mu)

        f_mms = -fd.div(2 * mu_of(u_ex) * fd.sym(fd.grad(u_ex)))

        bcs = {bid: {'u': u_ex} for bid in list(gadopt.get_boundary_ids(mesh))}
        eq = Equation(fd.TestFunction(V), V, viscosity_term,
                      eq_attrs={"stress": approximation.stress(u)},
                      approximation=approximation, bcs=bcs, quad_degree=8)
        F = eq.residual(u) - fd.dot(fd.TestFunction(V), f_mms) * fd.dx(degree=8)

        problem = fd.NonlinearVariationalProblem(F, u)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters={
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10, "snes_atol": 1e-12,
            # backtracking takes small steps in the stiff cubic region: the
            # N=32 solve needs 51 Newton iterations, above the PETSc default 50.
            "snes_max_it": 200, "ksp_type": "preonly",
            "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
        solver.solve()  # monotone operator; a failed solve fails the test

        err_u = fd.sqrt(fd.assemble(
            fd.inner(u - u_ex, u - u_ex) * fd.dx(degree=12)))
        errs_u.append(err_u)

    for k in range(2):
        order_u = log2(errs_u[k] / errs_u[k + 1])
        assert order_u >= 2.5  # expected 3.0

    # Absolute regression guard, as in test_mms_weak_un_convergence above: pin
    # the N=32 error to a small multiple of its converged value (err_u=5.43e-06).
    CAP_U = 3 * 5.43e-06
    assert errs_u[-1] <= CAP_U


def _tosi_viscosity(z, T, X):
    """Tosi viscoplastic viscosity, matching the formulation in viscoplastic_case_DG.py.

    ``u`` is taken from split(z)[0] of the passed solve's own z.
    """
    u = fd.split(z)[0]
    gamma_T, gamma_Z = fd.Constant(fd.ln(10**5)), fd.Constant(fd.ln(10))
    mu_star, sigma_y = fd.Constant(0.001), fd.Constant(1.0)
    eps = fd.sym(fd.grad(u))
    epsii = fd.sqrt(fd.inner(eps, eps) + 1e-10)
    mu_lin = fd.exp(-gamma_T * T + gamma_Z * (1 - X[1]))
    mu_plast = mu_star + (sigma_y / epsii)
    return (2. * mu_lin * mu_plast) / (mu_lin + mu_plast), mu_lin


def _tosi_solve(mesh, bcs, use_switch):
    """One nonlinear Tosi Stokes solve; returns the velocity subfunction."""
    boundary = gadopt.get_boundary_ids(mesh)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    W = fd.FunctionSpace(mesh, "CG", 1)
    Z = V * W
    z = fd.Function(Z)

    X = fd.SpatialCoordinate(mesh)
    Q = fd.FunctionSpace(mesh, "CG", 2)
    T = fd.Function(Q).interpolate(
        (1.0 - X[1]) + 0.05 * fd.cos(fd.pi * X[0]) * fd.sin(fd.pi * X[1]))

    mu_vp, mu_lin = _tosi_viscosity(z, T, X)
    # use_switch adds a linear pre-solve (Spiegelman continuation) as a fallback
    # if Newton from zero ever stagnates at higher resolution. Not exercised at
    # N<=64 (both solves converge directly); kept for a future resolution bump.
    if use_switch:
        switch = fd.Constant(1.0)
        mu = fd.conditional(switch > 0.5, mu_vp, mu_lin)
    else:
        mu = mu_vp

    approximation = gadopt.BoussinesqApproximation(fd.Constant(100), mu=mu)
    resolved_bcs = bcs(boundary)

    Z_nullspace = gadopt.create_stokes_nullspace(Z, closed=True, rotational=False)
    solver = gadopt.StokesSolver(
        z, approximation, T,
        bcs=resolved_bcs,
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


def test_tosi_weak_freeslip_matches_strong():
    """Weak free-slip ("un") Tosi solve matches the strong-BC formulation (2)."""
    # use_switch=False: Newton from zero converges directly for both the strong
    # and weak solves at these resolutions, so the linear continuation
    # pre-solve in _tosi_solve is not required here.
    def bcs_strong(b):
        return {b.bottom: {'uy': 0}, b.top: {'uy': 0},
                b.left: {'ux': 0}, b.right: {'ux': 0}}

    def bcs_weak(b):
        return {bid: {'un': 0} for bid in list(b)}

    d = {}
    urms_rel = {}
    for N in (32, 64):
        mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        mesh.cartesian = True
        u_s = _tosi_solve(mesh, bcs_strong, use_switch=False)
        u_w = _tosi_solve(mesh, bcs_weak, use_switch=False)

        d[N] = (fd.sqrt(fd.assemble(fd.inner(u_w - u_s, u_w - u_s) * fd.dx))
                / fd.sqrt(fd.assemble(fd.inner(u_s, u_s) * fd.dx)))
        urms_rel[N] = abs(fd.sqrt(fd.assemble(fd.dot(u_w, u_w) * fd.dx))
                          / fd.sqrt(fd.assemble(fd.dot(u_s, u_s) * fd.dx)) - 1)

    # Absolute regression guard on d[32], the relative velocity difference
    # between the weak and strong formulations: pin it to a small multiple of
    # its converged value (7.33e-05) so a solver-output regression is caught
    # even where the refinement check below still passes.
    assert d[32] <= 0.05  # sanity ceiling
    assert d[32] <= 3 * 7.33e-05
    # difference is a discretization effect and must shrink under refinement
    assert d[32] / d[64] >= 1.3
    # urms_rel is the relative difference in bulk kinetic energy between the two
    # formulations, a near-cancellation quantity about 700x smaller than d[64]
    # since it hides in a norm that is far less sensitive to local velocity
    # differences than the pointwise comparison above. Its precise value is not
    # robust across CI's BLAS/MUMPS stack, so the cap is set an order of
    # magnitude above the converged floor (~9e-9) while staying well below the
    # scale a sign error in the tangent-stress term would produce (~1e-6),
    # leaving a clean margin on both sides.
    assert urms_rel[64] <= 1e-2  # ceiling
    assert urms_rel[64] <= 1e-7
