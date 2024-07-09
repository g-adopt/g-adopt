# Demo for scalar advection-diffusion - this is adapted from the scalar
# advection demo again using G-ADOPT's Energy solver and a CG discretisation
# with Streamline Upwind (SU) stabilisation but here we introduce some explicit diffusion.

from gadopt import *
from gadopt.time_stepper import DIRK33

# We use a 40-by-40 mesh of squares.
mesh = UnitSquareMesh(40, 40, quadrilateral=True)

# We set up a function space of bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

Q = FunctionSpace(mesh, "Q", 1)
V = VectorFunctionSpace(mesh, "Q", 1)

log("Number of scalar DOF:", V.dim())

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)

velocity = as_vector(((0.5 - y)*sin(pi*x), (x - 0.5)*sin(pi*y)))
u = Function(V).interpolate(velocity)

# Specify the explicit diffusivity

kappa = Constant(1e-4)

# Plot velocity
u_outfile = VTKFile("CG_SUadvdiff_u.pvd")
u_outfile.write(u)

# Set up the initial conditions (similar to the scalar advection example)
cyl_r0 = 0.15
cyl_x0 = 0.5
cyl_y0 = 0.75
slot_left = 0.475
slot_right = 0.525
slot_top = 0.85

slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                       conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                       0.0, 1.0), 0.0)

# the tracer function and its initial condition
q_init = Constant(0.0)+slot_cyl
q = Function(Q).interpolate(q_init)

# We declare the output filename, and write out the initial condition. ::

outfile = VTKFile("CG_SUadvdiff_q.pvd")
outfile.write(q)

# time period and time step
T = 10.
dt = T/600.0


# Use G-ADOPT's Energy Solver to advect the tracer. By setting the Rayleigh number to 1
# the choice of units is up to the user. We use the diagonaly implicit DIRK33 Runge-Kutta
# method for timestepping. 'T' means that the boundary conditions will be applied strongly
# by the energy solver.
approximation = BoussinesqApproximation(Ra=1, kappa=kappa)
q_top = 1.0
q_bottom = 0.0
bcs = {3: {'T': q_bottom}, 4: {'T': q_top}}
energy_solver = EnergySolver(q, u, approximation, dt, DIRK33, bcs=bcs, su_advection=True)

# Get nubar (additional SU diffusion) for plotting
nubar = Function(Q).interpolate(energy_solver.fields['su_nubar'])
nubar_outfile = VTKFile("CG_SUadvdiff_nubar.pvd")
nubar_outfile.write(nubar)

# Here is the time stepping loop, with an output every 20 steps.
t = 0.0
step = 0
while t < T - 0.5*dt:

    energy_solver.solve()

    step += 1
    t += dt

    if step % 10 == 0:
        outfile.write(q)
        log("t=", t)
