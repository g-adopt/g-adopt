"""Solver-output tests for the weak-BC momentum branches with nonlinear mu.

``test_symmetry.py`` checks two properties of the weak ("u"/"un") SIPG boundary
terms in ``gadopt.momentum_equation.viscosity_term``: that their Jacobian is
symmetric, and that the residual is the first variation of the documented
boundary functional. This file checks the remaining property, that the
*converged solution* those terms produce is the solution of the intended
continuous problem. A boundary term inconsistent with that continuous problem
converges to a different solution and so degrades the convergence measured here,
while keeping the Jacobian symmetric; the symmetry test alone would pass it.

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
fixed-temperature, single-timestep version of the Tosi benchmark (not the full
time-stepped run) compared across three resolutions.
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
    errs_interp_u = []
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

        # Nodal interpolation error of the exact field: no solve, a
        # discretisation-only reference for the constant guard below.
        u_interp = fd.Function(V).interpolate(u_ex)
        errs_interp_u.append(fd.sqrt(fd.assemble(
            fd.inner(u_ex - u_interp, u_ex - u_interp) * fd.dx(degree=12))))

    # Taylor-Hood P2-P1 L2 convergence orders.
    THEORY_ORDER_U, THEORY_ORDER_P = 3.0, 2.0
    # Tolerance band on the measured order, sized to the failure it guards. A
    # boundary term inconsistent with the continuous problem converges to a
    # different solution and collapses the order well below theory, so 0.25
    # separates a passing solve from that failure. A consistent but wrong term
    # (a mis-scaled penalty, or a non-symmetric symmetriser) keeps the optimal
    # order 3; those are caught by the symmetry and variational-structure tests
    # in test_symmetry.py, not by the order here.
    ORDER_TOL = 0.25
    # Quasi-optimality factor: Cea and Aubin-Nitsche bound the Galerkin L2
    # error by a constant times the best-approximation (here interpolation)
    # error. The measured ratio is ~1 at every level, so K=2 is a real
    # stability margin, dimensionless and scaling with h.
    K_STAB = 2

    for k in range(2):
        # One-sided: exceeding the theoretical order is genuine superconvergence
        # on this uniform quad mesh (pressure runs ~3.7), never a regression.
        assert log2(errs_u[k] / errs_u[k + 1]) >= THEORY_ORDER_U - ORDER_TOL
        assert log2(errs_p[k] / errs_p[k + 1]) >= THEORY_ORDER_P - ORDER_TOL

    # Bound the error constant, not just the rate: an inconsistent boundary term
    # inflates the velocity error by orders of magnitude alongside the order
    # collapse, while a correct scheme stays within a small factor of the
    # interpolation error (quasi-optimality). A consistent but mis-scaled penalty
    # leaves this constant essentially unchanged and is pinned by the
    # variational-structure test in test_symmetry.py instead.
    for err, err_interp in zip(errs_u, errs_interp_u):
        assert err <= K_STAB * err_interp


def test_mms_weak_u_convergence():
    """MMS convergence of the weak "u" branch at the Equation level (1b)."""
    errs_u = []
    errs_interp_u = []
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
            # backtracking takes small steps in the stiff cubic region and can
            # exceed the PETSc default of 50 iterations, so raise the cap.
            "snes_max_it": 200, "ksp_type": "preonly",
            "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
        solver.solve()  # monotone operator; a failed solve fails the test

        err_u = fd.sqrt(fd.assemble(
            fd.inner(u - u_ex, u - u_ex) * fd.dx(degree=12)))
        errs_u.append(err_u)

        # Nodal interpolation error of the exact field: the discretisation-only
        # reference for the constant guard below.
        u_interp = fd.Function(V).interpolate(u_ex)
        errs_interp_u.append(fd.sqrt(fd.assemble(
            fd.inner(u_ex - u_interp, u_ex - u_interp) * fd.dx(degree=12))))

    # Taylor-Hood P2 velocity L2 convergence order (see 1a for the rationale
    # of the tolerance band and the quasi-optimality constant guard).
    THEORY_ORDER_U = 3.0
    ORDER_TOL = 0.25
    K_STAB = 2

    for k in range(2):
        assert log2(errs_u[k] / errs_u[k + 1]) >= THEORY_ORDER_U - ORDER_TOL

    for err, err_interp in zip(errs_u, errs_interp_u):
        assert err <= K_STAB * err_interp


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
    """Weak free-slip ("un") Tosi solve matches the strong-BC formulation (2).

    The weak and strong formulations are consistent discretisations of the same
    continuous problem, so their relative velocity difference decays at the P2
    discretisation order under refinement. This tests that trend. A bug that
    scales both solves identically cancels in the difference and is not caught
    here; the MMS tests pin the absolute solution against exact fields instead.
    """
    # use_switch=False: Newton from zero converges directly for both the strong
    # and weak solves at these resolutions, so the linear continuation
    # pre-solve in _tosi_solve is not required here.
    def bcs_strong(b):
        return {b.bottom: {'uy': 0}, b.top: {'uy': 0},
                b.left: {'ux': 0}, b.right: {'ux': 0}}

    def bcs_weak(b):
        return {bid: {'un': 0} for bid in list(b)}

    # Three levels give two refinement pairs, enough to measure a decay rate.
    Ns = (16, 32, 64)
    d = {}
    for N in Ns:
        mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        mesh.cartesian = True
        u_s = _tosi_solve(mesh, bcs_strong, use_switch=False)
        u_w = _tosi_solve(mesh, bcs_weak, use_switch=False)

        d[N] = (fd.sqrt(fd.assemble(fd.inner(u_w - u_s, u_w - u_s) * fd.dx))
                / fd.sqrt(fd.assemble(fd.inner(u_s, u_s) * fd.dx)))

    # d is bounded by the sum of the two P2 velocity L2 errors, so it decays at
    # their order (3) under refinement.
    THEORY_ORDER_D = 3.0
    # The failure this guards is a weak side inconsistent with the continuous
    # problem: d then decays at a visibly degraded rate, or plateaus, well below
    # the threshold rather than at order 3. A consistent but wrong weak term
    # (a mis-scaled penalty, or a non-symmetric symmetriser) keeps the rate and
    # is caught by the symmetry and variational-structure tests, not here. The
    # band is looser than the MMS tests because d is a difference of two errors
    # (partial cancellation) and the Tosi solution's regularity is not certified,
    # so a correct rate can genuinely sit somewhat below 3.
    ORDER_TOL_D = 0.5
    for N, N2 in zip(Ns[:-1], Ns[1:]):
        assert log2(d[N] / d[N2]) >= THEORY_ORDER_D - ORDER_TOL_D

    # The one absolute bound in the file: a specification, not a regression pin.
    # The two formulations must agree to within 1% in relative velocity at the
    # working resolution.
    assert d[Ns[-1]] <= 1e-2
