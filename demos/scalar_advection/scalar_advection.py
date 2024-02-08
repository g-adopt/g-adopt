# Demo for pure scalar advection - this is adapted from the Firedrake DG_advection demo,
# but here we use G-ADOPT's Energy solver and use a CG discretisation
# with Streamline Upwind (SU) stabilisation.

from gadopt import *
from gadopt.time_stepper import DIRK33

# We use a 40-by-40 mesh of squares.
mesh = UnitSquareMesh(40, 40, quadrilateral=True)

# We set up a function space of bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

Q = FunctionSpace(mesh, "Q", 1)
V = VectorFunctionSpace(mesh, "Q", 1)

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)

velocity = as_vector((0.5 - y, x - 0.5))
u = Function(V).interpolate(velocity)

# Now, we set up the cosine-bell--cone--slotted-cylinder initial coniditon. The
# first four lines declare various parameters relating to the positions of these
# objects, while the analytic expressions appear in the last three lines. ::

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

# We then declare the inital condition of :math:`q` to be the sum of these fields.
# Furthermore, we add 1 to this, so that the initial field lies between 1 and 2,
# rather than between 0 and 1.  This ensures that we can't get away with
# neglecting the inflow boundary condition.  We also save the initial state so
# that we can check the :math:`L^2`-norm error at the end. ::

q_init = Function(Q).interpolate(1.0 + bell + cone + slot_cyl)
q = Function(Q).assign(q_init)

# We declare the output filename, and write out the initial condition. ::

outfile = File("CG_SUadv_q.pvd")
outfile.write(q)


u_outfile = File("CG_SUadv_u.pvd")
u_outfile.write(u)
# We will run for time :math:`2\pi`, a full rotation.  We take 600 steps, giving
# a timestep close to the CFL limit.  Finally, we define the inflow boundary
# condition, :math:`q_\mathrm{in}`.  In general, this would be a ``Function``, but
# here we just use a ``Constant`` value. ::

T = 2*pi
dt = T/600.0
q_in = Constant(1.0)


# Use G-ADOPT's Energy Solver to advect the tracer. By setting the Rayleigh number to 1
# the choice of units is up to the user. We set the diffusiviy to zero for pure advection.
# We also apply weak boundary conditions on inflow regions.
approximation = BoussinesqApproximation(Ra=1, kappa=Constant(0))
bc_in = {'q': q_in}
bcs = {1: bc_in, 2: bc_in, 3: bc_in, 4: bc_in}
energy_solver = EnergySolver(q, u, approximation, dt, DIRK33, bcs=bcs, su_advection=True)

# Get nubar (additional SU diffusion) for plotting
nubar = Function(Q).interpolate(energy_solver.fields['su_nubar'])
nubar_outfile = File("CG_SUadv_nubar.pvd")
nubar_outfile.write(nubar)

# Here is the time stepping loop, with an output every 20 steps.
t = 0.0
step = 0
while t < T - 0.5*dt:

    energy_solver.solve()

    step += 1
    t += dt

    if step % 20 == 0:
        outfile.write(q)
        print("t=", t)

# Finally, we display the normalised :math:`L^2` error, by comparing to the
# initial condition. ::

L2_err = sqrt(assemble((q - q_init)*(q - q_init)*dx))
L2_init = sqrt(assemble(q_init*q_init*dx))
print(L2_err/L2_init)
