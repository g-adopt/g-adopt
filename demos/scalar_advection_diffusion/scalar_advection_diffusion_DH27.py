
from gadopt import *
from gadopt.scalar_equation import ScalarAdvectionDiffusionEquation
from gadopt.time_stepper import DIRK33

def model(n, do_write=False):
    """ Demo for scalar advection-diffusion based on Figure 2.7 in
    Chapter 2 Steady transport problems from Finite element Methods
    for Flow problems - Donea and Huerta, 2003

    Args: 
        n: number of grid cells in x and y direction
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
    Pe = 0.25  # peclet number for coarsest grid
    h = 1/5 
    kappa = Constant(1*h/(2*Pe))


    # the tracer function and its initial condition
    q_init = Constant(0.0)
    q = Function(V).interpolate(q_init)

    # We declare the output filename, and write out the initial condition. ::

    outfile = File("advdif_DH2.7_CG1_Pe"+str(Pe)+"_SU.pvd")
    outfile.write(q)

    # time period and time step
    T = 10.
    dt = 0.01

    eq = ScalarAdvectionDiffusionEquation(V, V, su_advection=True)
    fields = {'velocity': u, 'diffusivity': kappa, 'source': 1.0}
    # weakly applied dirichlet bcs on top and bottom
    q_left = 0.0
    q_right = 0.0
    strong_bcs = DirichletBC(V, 0, [1, 2])
    bcs = {1: {'q': q_left}, 2: {'q': q_right}}
    timestepper = DIRK33(eq, q, fields, dt, strong_bcs=strong_bcs)
    
    steady_state_tolerance = 1e-7
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
        
        L2anal_q = norm(q)
        L2error_q = errornorm(q_anal, q, norm_type='L2')
        
        print(norm(q_anal))

        print("L2_anal_q", L2anal_q)
        print("L2_error_q", L2error_q)
    
    outfile.write(q)
    ana_outfile = File("advdif_DH2.7_CG1_Pe"+str(Pe)+"_SU_ana.pvd")
    ana_outfile.write(q_anal)

    return L2error_q
import numpy as np
nxs = [5*(2**i) for i in range(4)] #, 0.00625, 0.003125]#, 0.125, 0.0625, 0.03125]
errors = np.array([model(nx) for nx in nxs]) 
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('time surface displacement errors: ', errors[:])
print('time surface displacement conv: ', conv[:])
