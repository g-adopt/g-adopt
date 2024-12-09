# Demo for scalar advection-diffusion - this is adapted from the scalar advection demo
# again using G-ADOPT's GenericTransportSolver and a CG discretisation with Streamline
# Upwind (SU) stabilisation, albeit here we introduce some explicit diffusion.

from gadopt import *
from gadopt.time_stepper import DIRK33

# We use a 40-by-40 mesh of squares.
mesh = UnitSquareMesh(40, 40, quadrilateral=True)
mesh.cartesian = True

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

# Use G-ADOPT's GenericTransportSolver to advect the tracer. We use the diagonally
# implicit DIRK33 Runge-Kutta method for timestepping. 'g' means that the boundary
# conditions will be applied strongly by the solver.
terms = ["advection", "diffusion"]
eq_attrs = {"diffusivity": kappa, "u": u}
g_top = 1.0
g_bottom = 0.0
bcs = {3: {"g": g_bottom}, 4: {"g": g_top}}
adv_diff_solver = GenericTransportSolver(
    terms, q, dt, DIRK33, eq_attrs=eq_attrs, bcs=bcs, su_diffusivity=kappa
)

# Get nubar (additional SU diffusion) for plotting
nubar = Function(Q).interpolate(adv_diff_solver.equation.su_nubar)
nubar_outfile = VTKFile("CG_SUadvdiff_nubar.pvd")
nubar_outfile.write(nubar)

# Here is the time stepping loop, with an output every 20 steps.
t = 0.0
step = 0
while t < T - 0.5*dt:
    adv_diff_solver.solve()

    step += 1
    t += dt

    if step % 10 == 0:
        outfile.write(q)
        log("t=", t)
