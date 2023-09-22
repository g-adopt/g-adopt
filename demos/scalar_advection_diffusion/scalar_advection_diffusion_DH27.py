from gadopt import *
from gadopt.scalar_equation import ScalarAdvectionDiffusionEquation
from gadopt.time_stepper import DIRK33
import numpy as np
import sys


def model(n, Pe=0.25, su_advection=True, do_write=False):
    """ Demo for scalar advection-diffusion based on Figure 2.7 in
    Chapter 2 Steady transport problems from Finite element Methods
    for Flow problems - Donea and Huerta, 2003

    Args:
        n: number of grid cells in x and y direction
        Pe: grid peclet number
        su_advection: Flag for turning su advection on or off
        do_write: whether to output the scalar/velocity field
    """
    mesh = UnitSquareMesh(n, n, quadrilateral=True)

    # We set up a function space of discontinous bilinear elements for :math:`q`, and
    # a vector-valued continuous function space for our velocity field. ::

    V = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)
    W = VectorFunctionSpace(mesh, "CG", 1)

    # We set up the initial velocity field using a simple analytic expression. ::

    x = SpatialCoordinate(mesh)
    a = Constant(1)
    velocity = as_vector((a, 0))
    u = Function(W).interpolate(velocity)
    if do_write:
        File('u.pvd').write(u)

    # the diffusivity
    h = 1/5  # coarsest grid size
    kappa = Constant(1*h/(2*Pe))

    # the tracer function and its initial condition
    q_init = Constant(0.0)
    q = Function(V).interpolate(q_init)

    # We declare the output filename, and write out the initial condition. ::
    if do_write:
        outfile = File("advdif_DH2.7_CG1_Pe"+str(Pe)+"_SU.pvd")
        outfile.write(q)

    # time period and time step
    T = 10.
    dt = 0.01

    eq = ScalarAdvectionDiffusionEquation(V, V, su_advection=su_advection)
    fields = {'velocity': u, 'diffusivity': kappa, 'source': 1.0}
    # weakly applied dirichlet bcs on top and bottom
    strong_bcs = DirichletBC(V, 0, [1, 2])
    timestepper = DIRK33(eq, q, fields, dt, strong_bcs=strong_bcs)

    steady_state_tolerance = 1e-7  # this may need tweaking for different length runs/Pe values
    t = 0.0
    step = 0

    while t < T - 0.5*dt:
        # the solution reaches a steady state and finishes the solve when a  max no. of iterations is reached
        timestepper.advance(t)

        # Calculate L2-norm of change in temperature:
        maxchange = sqrt(assemble((q - timestepper.solution_old)**2 * dx))
        print("maxchange", maxchange)
        step += 1
        t += dt
        print("t=", t)

        if do_write:
            outfile.write(q)

        if maxchange < steady_state_tolerance:
            print("Steady-state acheieved -- exiting time-step loop")
            break

    # analytical solution from equation 2.23 in Chapter 2 Steady transport problems
    # from Finite element Methods for Flow problems - Donea and Huerta, 2003
    # N.b they have the scalar called 'u' whereas we have 'q'
        gamma = Constant(a/kappa)
        q_anal = Function(V2)
        q_anal.interpolate((1/a) * (x[0] - (1 - exp(gamma*x[0]))/(1-exp(gamma))))

        L2error_q = errornorm(q_anal, q, norm_type='L2')
        L2anal_q = norm(q_anal)
        L2q = norm(q)

        print("L2_anal_q", L2anal_q)
        print("L2_error_q", L2error_q)

    return L2error_q, L2anal_q, L2q


if __name__ == "__main__":
    if len(sys.argv) > 1:
        Pe, SU = sys.argv[1].split("_")
        # parse params
        Pe = float(Pe)
        SU = SU == "True"

        nxs = 5 * 2**np.arange(4)

        errors = np.array([model(nx, Pe, SU) for nx in nxs])
        np.savetxt(f"errors-Pe{Pe}_SU{SU}.dat", errors)
    else:
        # default to a simple case if not running a specific
        # case through CI
        model(5, write=True)
