# Demo for scalar advection-diffusion based on Figure 2.7 in
# Chapter 2 Steady transport problems from Finite element Methods
# for Flow problems - Donea and Huerta, 2003

from gadopt import *
from gadopt.scalar_equation import ScalarAdvectionDiffusionEquation
from gadopt.time_stepper import DIRK33

mesh = UnitSquareMesh(10, 10, quadrilateral=True)

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

# We set up the initial velocity field using a simple analytic expression. ::

x = SpatialCoordinate(mesh)
velocity = as_vector((1, 0))
u = Function(W).interpolate(velocity)
File('u.pvd').write(u)

# the diffusivity
Pe = 5
h = 0.1
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


t = 0.0
step = 0
while t < T - 0.5*dt:
    # the solution reaches a steady state and finishes the solve when a  max no. of iterations is reached
    timestepper.advance(t)

    step += 1
    t += dt

    if step % 1 == 0:
        outfile.write(q)
        print("t=", t)
