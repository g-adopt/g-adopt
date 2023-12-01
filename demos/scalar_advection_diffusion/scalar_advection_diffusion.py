# Demo for scalar advection-diffusion - this is mostly copied from the scalar
# advection demo but here we use gadopt's ScalarAdvectionDiffusionEquation and use
# a CG discretisation with Streamline Upwind (SU) stabilisation.

from gadopt import *
from gadopt.scalar_equation import ScalarAdvectionDiffusionEquation
from gadopt.time_stepper import DIRK33
from gadopt.utility import absv, beta, su_nubar

# We use a 40-by-40 mesh of squares.
mesh = UnitSquareMesh(40, 40, quadrilateral=True)

# We set up a function space of bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "Q", 1)
W = VectorFunctionSpace(mesh, "Q", 1)

log("Number of scalar DOF:", V.dim())

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)

velocity = as_vector(((0.5 - y)*sin(pi*x), (x - 0.5)*sin(pi*y)))
u = Function(W).interpolate(velocity)

# Specify the explicit diffusivity

kappa = Constant(1e-4)

# Calculate nu_bar for plotting
J = Function(TensorFunctionSpace(mesh, 'DQ', 1), name='Jacobian').interpolate(Jacobian(mesh))
grid_peclet = absv(dot(u, J)) / (2*kappa)
beta_pe = beta(grid_peclet)
nubar = Function(V).interpolate(su_nubar(u, J, beta_pe))

# Plot nu_bar
nubar_outfile = File("CG_SUadvdiff_nubar.pvd")
nubar_outfile.write(nubar)

# Plot velocity
u_outfile = File("CG_SUadvdiff_u.pvd")
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
q = Function(V).interpolate(q_init)

# We declare the output filename, and write out the initial condition. ::

outfile = File("CG_SUadvdiff_q.pvd")
outfile.write(q)

# time period and time step
T = 10.
dt = T/600.0


# Now we call G-ADOPT's scalar advection diffusion equation to set up the equations
# and set up a timestepper. We use the diagonaly implicit DIRK33 Runge-Kutta method for
# timestepping. We also need to provide the velocity field, u, to the timestepper
# as well as boundary conditions. In this case we apply a boundary value
# for when there is inflow.
eq = ScalarAdvectionDiffusionEquation(V, V, su_advection=True)
fields = {'velocity': u, 'diffusivity': kappa}

# weakly applied dirichlet bcs on top and bottom
q_top = 1.0
q_bottom = 0.0
bcs = {3: {'q': q_bottom}, 4: {'q': q_top}}

# Strongly applied dirichlet bcs on top and bottom
strong_qtop = DirichletBC(V, q_top, 4)
strong_qbottom = DirichletBC(V, q_bottom, 3)
strong_bcs = [strong_qtop, strong_qbottom]
timestepper = DIRK33(eq, q, fields, dt, strong_bcs=strong_bcs)  # uncomment for weak bcs bnd_conditions=bcs)

# Here is the time stepping loop, with an output every 20 steps.
t = 0.0
step = 0
while t < T - 0.5*dt:

    timestepper.advance(t)

    step += 1
    t += dt

    if step % 10 == 0:
        outfile.write(q)
        log("t=", t)
