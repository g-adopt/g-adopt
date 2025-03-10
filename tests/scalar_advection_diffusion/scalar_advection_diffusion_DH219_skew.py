# Demo for scalar advection-diffusion based on Figure 2.19 in
# Chapter 2 Steady transport problems from Finite element Methods
# for Flow problems - Donea and Huerta, 2003
# In this demo the flow direction is 'skewed' relative to the grid
# so it is a good test of the tensor implementation of SU, compared
# with the quasi 1D demo based on Figure 2.7.

from gadopt import *
from gadopt.time_stepper import DIRK33

nx, ny = 10, 10
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)
mesh.cartesian = True

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field.
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)

# We set up the initial velocity field using a simple analytic expression.
x, y = SpatialCoordinate(mesh)
velocity = as_vector((cos(30 * pi / 180), sin(30 * pi / 180)))
u = Function(V).interpolate(velocity)
VTKFile("advdif_DH219_u.pvd").write(u)

# the diffusivity based on a chosen grid Peclet number
Pe = 1e1  # This seems very high so presumably diffusion added from SU dominates.
h = 1.0 / nx
kappa = Constant(1 * h / (2 * Pe))

# the tracer function and its initial condition
q_init = Constant(0.0)
q = Function(Q).interpolate(q_init)

# We declare the output filename, and write out the initial condition.
outfile = VTKFile("advdif_DH219_skew_CG1_Pe" + str(Pe) + "_SU.pvd")
outfile.write(q)

# time period and time step
T = 10.0
dt = 0.01

# Use G-ADOPT's GenericTransportSolver to advect the tracer. We use the diagonally
# implicit DIRK33 Runge-Kutta method for timestepping. 'g' means that the boundary
# conditions will be applied strongly by the solver.
terms = ["advection", "diffusion"]
eq_attrs = {"diffusivity": kappa, "u": u}
# strongly applied Dirichlet bcs on top and bottom
g_left = conditional(y < 0.2, 0.0, 1.0)
g_bottom = 0
bcs = {3: {"g": g_bottom}, 1: {"g": g_left}}
adv_diff_solver = GenericTransportSolver(
    terms, q, dt, DIRK33, eq_attrs=eq_attrs, bcs=bcs, su_advection=True
)

# Get nubar (additional SU diffusion) for plotting
nubar = Function(Q).interpolate(adv_diff_solver.equation.su_nubar)
nubar_outfile = VTKFile("advdof_DH219_skew_CG1_Pe" + str(Pe) + "_SU_nubar.pvd")
nubar_outfile.write(nubar)

t = 0.0
step = 0
while t < T - 0.5 * dt:
    # the solution reaches a steady state and finishes the solve when a  max no. of iterations is reached
    adv_diff_solver.solve()

    step += 1
    t += dt

    if step % 10 == 0:
        outfile.write(q)
        print("t=", t)

# Write out integrated scalar for testing
L2 = sqrt(assemble(q**2 * dx))
np.savetxt("integrated_q_DH219.log", [L2])
