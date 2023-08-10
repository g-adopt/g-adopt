# Demo for scalar advection-diffusion

from gadopt import *
from gadopt.scalar_equation import ScalarAdvectionDiffusionEquation
from gadopt.time_stepper import DIRK33
from math import pi

mesh = UnitSquareMesh(40, 40, quadrilateral=True)

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "CG", 2)
W = VectorFunctionSpace(mesh, "CG", 1)

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)
velocity = as_vector(((0.5 - y)*sin(pi*x), (x - 0.5)*sin(pi*y)))
u = Function(W).interpolate(velocity)
File('u.pvd').write(u)

# the diffusivity
kappa = Constant(1e-5)

bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)
# the tracer function and its initial condition
q_init = Constant(0.0)+slot_cyl
q = Function(V).interpolate(q_init)

# We declare the output filename, and write out the initial condition. ::

outfile = File("advdif_CG2_kappa1e-5_SUPG.pvd")
outfile.write(q)

# time period and time step
T = 10.
dt = T/600.0

eq = ScalarAdvectionDiffusionEquation(V, V)
fields = {'velocity': u, 'diffusivity': kappa, 'SU_advection':None}
# weakly applied dirichlet bcs on top and bottom
q_top = 1.0
q_bottom = 0.0
bcs = {3: {'q': q_bottom}, 4: {'q': q_top}}
#bcs = {}
timestepper = DIRK33(eq, q, fields, dt, bnd_conditions=bcs)


t = 0.0
step = 0
while t < T - 0.5*dt:

    timestepper.advance(t)

    step += 1
    t += dt

    if step % 20 == 0:
        outfile.write(q)
        print("t=", t)
