# Demo for scalar advection-diffusion based on Figure 2.7 in
# Chapter 2 Steady transport problems from Finite element Methods
# for Flow problems - Donea and Huerta, 2003
# This script is also called as part of the CI for testing
# compared with the analytical solution (when grid Peclet < 1)
# and for regression testing.

import sys

import numpy as np

from gadopt import *
from gadopt.time_stepper import DIRK33


def model(n, Pe=0.25, su_advection=True, do_write=False):
    """Demo for scalar advection-diffusion based on Figure 2.7 in
    Chapter 2 Steady transport problems from Finite element Methods
    for Flow problems - Donea and Huerta, 2003

    Args:
        n: number of grid cells in x and y direction
        Pe: grid peclet number
        su_advection: Flag for turning su advection on or off
        do_write: whether to output the scalar/velocity field
    """
    mesh = UnitSquareMesh(n, n, quadrilateral=True)
    mesh.cartesian = True

    # We set up a function space of bilinear elements for :math:`q`, and
    # a vector-valued continuous function space for our velocity field.
    Q = FunctionSpace(mesh, "CG", 1)
    Q2 = FunctionSpace(mesh, "CG", 2)
    V = VectorFunctionSpace(mesh, "CG", 1)

    # We set up the initial velocity field using a simple analytic expression.
    x = SpatialCoordinate(mesh)
    a = Constant(1)
    velocity = as_vector((a, 0))
    u = Function(V).interpolate(velocity)
    if do_write:
        VTKFile("u.pvd").write(u)

    # the diffusivity
    h = 1 / 5  # coarsest grid size
    kappa = Constant(h / (2 * Pe))

    # the tracer function and its initial condition
    q_init = Constant(0.0)
    q = Function(Q).interpolate(q_init)

    # analytical solution from equation 2.23 in Chapter 2 Steady transport problems
    # from Finite element Methods for Flow problems - Donea and Huerta, 2003
    # N.b they have the scalar called 'u' whereas we have 'q'
    gamma = Constant(a / kappa)
    q_anal = Function(Q2)
    q_anal.interpolate((1 / a) * (x[0] - (1 - exp(gamma * x[0])) / (1 - exp(gamma))))

    # We declare the output filename, and write out the initial condition. ::
    if do_write:
        outfile = VTKFile("advdif_DH2.7_CG1_Pe" + str(Pe) + "_SU.pvd")
        outfile.write(q)

    # time period and time step
    T = 10.0
    dt = 0.01

    # Set up boundary conditions
    g_left = 0.0
    g_right = 0.0
    # 'g' sets strong Dirichlet boundary conditions for G-ADOPT's transport solver
    bcs = {1: {"g": g_left}, 2: {"g": g_right}}

    # Use G-ADOPT's GenericTransportSolver to advect the tracer. The system is
    # considered non-dimensional and controlled only by the thermal diffusivity and heat
    # source values. We use the diagonally implicit DIRK33 Runge-Kutta method for
    # timestepping.
    terms = ["advection", "diffusion", "source"]
    eq_attrs = {"diffusivity": kappa, "source": 1, "u": u}
    adv_diff_solver = GenericTransportSolver(
        terms,
        q,
        dt,
        DIRK33,
        eq_attrs=eq_attrs,
        bcs=bcs,
        su_diffusivity=kappa if su_advection else None,
    )

    # this may need tweaking for different length runs/Pe values
    steady_state_tolerance = 1e-7
    t = 0.0
    step = 0
    while t < T - 0.5 * dt:
        adv_diff_solver.solve()

        # Calculate L2-norm of change in temperature:
        maxchange = sqrt(assemble((q - adv_diff_solver.solution_old) ** 2 * dx))
        log("maxchange", maxchange)

        step += 1
        t += dt
        log("t =", t)

        if do_write:
            outfile.write(q)

        if maxchange < steady_state_tolerance:
            log("Steady-state achieved -- exiting time-step loop")
            break

        L2error_q = errornorm(q_anal, q, norm_type="L2")
        L2anal_q = norm(q_anal)
        L2q = norm(q)

        log("L2_anal_q", L2anal_q)
        log("L2_error_q", L2error_q)

    return L2error_q, L2anal_q, L2q


if __name__ == "__main__":
    if len(sys.argv) > 1:
        Pe, SU = sys.argv[1].split("_")
        # parse params
        Pe = float(Pe)
        SU = SU == "True"

        nxs = 5 * 2 ** np.arange(4)

        errors = np.array([model(nx, Pe, SU) for nx in nxs])
        np.savetxt(f"errors-Pe{Pe}_SU{SU}.dat", errors)
    else:
        # default to a simple case if not running a specific
        # case through CI
        model(5, do_write=True)
