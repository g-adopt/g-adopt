# Demo for scalar advection-diffusion based on Figure 2.19 in
# Chapter 2 Steady transport problems from Finite element Methods
# for Flow problems - Donea and Huerta, 2003
# In this demo the flow direction is 'skewed' relative to the grid 
# so it is a good test of the tensor implementation of SU, compared 
# with the quasi 1D demo based on Figure 2.7.

from gadopt import *
from gadopt.scalar_equation import ScalarAdvectionDiffusionEquation
from gadopt.time_stepper import DIRK33

nx, ny = 10, 10
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)
velocity = as_vector((cos(30*pi/180), sin(30*pi/180)))
u = Function(W).interpolate(velocity)
File('advdif_DH219_u.pvd').write(u)

# the diffusivity based on a chosen grid Peclet number
Pe = 1e4  # This seems very high so presumably diffusion added from SU dominates.
h = 1.0 / nx
kappa = Constant(1*h/(2*Pe))


# the tracer function and its initial condition
q_init = Constant(0.0)
q = Function(V).interpolate(q_init)

# We declare the output filename, and write out the initial condition. ::

outfile = File("advdif_DH219_skew_CG1_Pe"+str(Pe)+"_SU.pvd")
outfile.write(q)

# time period and time step
T = 10.
dt = 0.01

eq = ScalarAdvectionDiffusionEquation(V, V, su_advection=True)
fields = {'velocity': u, 'diffusivity': kappa, 'source': 0.0}
# strongly applied dirichlet bcs on top and bottom
q_left = DirichletBC(V, conditional(y < 0.2, 0.0, 1.0), 1)
q_bottom = DirichletBC(V, 0, 3)
strong_bcs = [q_bottom, q_left]
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