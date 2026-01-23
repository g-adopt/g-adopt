# Demo for pure scalar advection with adaptive timestepping - this is adapted from the Firedrake DG_advection demo.
# Here, however, we use G-ADOPT's GenericTransportSolver with IrksomeRadauIIA adaptive timestepping and use a CG discretisation with
# Streamline Upwind (SU) stabilisation.

from gadopt import *
import numpy as np

# We use a 40-by-40 mesh of squares.
mesh = UnitSquareMesh(40, 40, quadrilateral=True)
mesh.cartesian = True

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

outfile = VTKFile("CG_SUadv_adaptive_q.pvd")
outfile.write(q)

u_outfile = VTKFile("CG_SUadv_adaptive_u.pvd")
u_outfile.write(u)
# We will run for time :math:`2\pi`, a full rotation.  The initial timestep is set
# as a starting point for the adaptive stepper, which will adjust dt based on error
# estimates.  Finally, we define the inflow boundary condition, :math:`q_\mathrm{in}`.
# In general, this would be a ``Function``, but here we just use a ``Constant`` value. ::

T = 2*pi
dt = Constant(T / 600.0)  # Initial timestep - adaptive stepper will adjust this
q_in = Constant(1.0)

# Use G-ADOPT's GenericTransportSolver to advect the tracer with adaptive timestepping.
# We use IrksomeRadauIIA which has an embedded error estimator for adaptive control.
# We only include an advection term and apply weak boundary conditions on inflow regions.
bc_in = {"q": q_in}
bcs = {1: bc_in, 2: bc_in, 3: bc_in, 4: bc_in}
eq_attrs = {"u": u}
adv_solver = GenericTransportSolver(
    "advection",
    q,
    dt,
    RadauIIA,
    eq_attrs=eq_attrs,
    bcs=bcs,
    su_advection=True,
    timestepper_kwargs={
        "tableau_parameter": 3,  # RadauIIA order
        "adaptive_parameters": {
            "tol": 1e-5,  # Error tolerance per step
            "dtmin": 1e-6,  # Minimum allowed dt
            "dtmax": T / 100.0,  # Maximum allowed dt (sensible fraction of total time)
            "KI": 1 / 15,  # Integration gain
            "KP": 0.13,  # Proportional gain
        },
    },
)

# Get nubar (additional SU diffusion) for plotting
nubar = Function(Q).interpolate(adv_solver.equation.su_nubar)
nubar_outfile = VTKFile("CG_SUadv_adaptive_nubar.pvd")
nubar_outfile.write(nubar)

# Here is the time stepping loop with adaptive timestepping, with an output every 20 steps.
# The timestep dt will be automatically adjusted by the adaptive stepper based on error estimates.
t = 0.0
step = 0
dt_values = []  # Store all timestep values for testing
while t < T:
    # Advance with adaptive timestepping; an error estimate and the time step used are
    # returned
    adapt_error, adapt_dt = adv_solver.solve(t=t)

    # Store used time step and increment simulation time and step counter
    dt_values.append(adapt_dt)
    t += adapt_dt
    step += 1

    if T - t < float(dt):  # Set maximum dt to prevent overshooting final time
        adv_solver.ts.stepper.dt_max = T - t

    if step % 20 == 0:
        outfile.write(q)
        print(f"t={t:.6f}, dt={float(dt):.6e}, step={step}")

# Finally, we display the normalised :math:`L^2` error, by comparing to the
# initial condition.

L2_err = sqrt(assemble((q - q_init)*(q - q_init)*dx))
L2_init = sqrt(assemble(q_init*q_init*dx))
final_error = L2_err/L2_init
print(final_error)

# Save results for testing: final error, number of steps, and timestep statistics
np.savetxt("final_error_adaptive.log", [final_error])
np.savetxt("num_steps_adaptive.log", [step])
np.savetxt("dt_stats_adaptive.log", dt_values)
