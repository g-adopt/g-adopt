# Demo for scalar advection-diffusion

from gadopt import *
from gadopt.scalar_equation import ScalarAdvectionDiffusionEquation
from gadopt.time_stepper import BackwardEuler
from gadopt.utility import absv, beta, su_nubar
from math import pi


U = 0.25
Pe = 333
kappa = Constant(1.e-4)
h = Constant(Pe * 2 * kappa / U)

print("h", h.values()[0])

n = 40  # round(1/h)
mesh = UnitSquareMesh(n, n, quadrilateral=True)

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)
velocity = as_vector(((0.5 - y)*sin(pi*x), (x - 0.5)*sin(pi*y)))
u = Function(W).interpolate(velocity)
File('u.pvd').write(u)

# Calculate nu_bar for plotting
J = Function(TensorFunctionSpace(mesh, 'DQ', 1), name='Jacobian').interpolate(Jacobian(mesh))
Pe_field = absv(dot(u, J)) / (2*kappa)
beta_pe = beta(Pe_field)
nubar = Function(V).interpolate(su_nubar(u, J, beta_pe))


log("Number of Velocity DOF:", V.dim())


bell_r0 = 0.15
bell_x0 = 0.25
bell_y0 = 0.5
cone_r0 = 0.15
cone_x0 = 0.5
cone_y0 = 0.25
cyl_r0 = 0.15
cyl_x0 = 0.5
cyl_y0 = 0.75
slot_left = 0.475
slot_right = 0.525
slot_top = 0.85

bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                       conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                       0.0, 1.0), 0.0)

# the tracer function and its initial condition
q_init = Constant(0.0)+slot_cyl
q = Function(V).interpolate(q_init)

# We declare the output filename, and write out the initial condition. ::

outfile = File("advdiff_spiral_CG1_Pe"+str(Pe)+"_SU_nxny"+str(n)+"_U"+str(U)+"_BE_vel_strong_kappa1e-4.pvd")
outfile.write(q)

nubar_outfile = File("nubar.pvd")
nubar_outfile.write(nubar)

# time period and time step
T = 10.
dt = T/600.0

eq = ScalarAdvectionDiffusionEquation(V, V, su_advection=True)
fields = {'velocity': u, 'diffusivity': kappa}
# weakly applied dirichlet bcs on top and bottom
q_top = 1.0
q_bottom = 0.0
bcs = {3: {'q': q_bottom}, 4: {'q': q_top}}

strong_qtop = DirichletBC(V, q_top, 4)
strong_qbottom = DirichletBC(V, q_bottom, 3)
strong_bcs = [strong_qtop, strong_qbottom]
timestepper = BackwardEuler(eq, q, fields, dt, strong_bcs=strong_bcs)  # uncomment for weak bcs bnd_conditions=bcs)


t = 0.0
step = 0
while t < T - 0.5*dt:

    timestepper.advance(t)

    step += 1
    t += dt

    if step % 10 == 0:
        outfile.write(q)
        print("t=", t)
