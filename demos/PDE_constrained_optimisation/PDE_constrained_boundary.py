# PDE Constrained Optimisation with G-ADOPT - Boundary Values
# ===========================================================
#
# In this tutorial, we undertake an inversion for an (unknown) initial condition, to match given
# time-dependent boundary values.
#
# We start with our usual imports:

from gadopt import *
from gadopt.inverse import *

# Create synthetic twin experiment and record solution at all timesteps
# ---------------------------------------------------------------------
#
# Note that the setup is similar to our previous example, except that the velocity is now counter
# clockwise around the origin $(0,0)$ in the corner of the unit square domain. This implies an inflow
# at the bottom boundary and an outflow boundary on the left of the domain.

# +
mesh = UnitSquareMesh(40, 40)
left, right, bottom, top = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
T = Function(Q, name='Temperature')
T0 = Function(Q, name="Initial_Temperature")  # T Initial condition which we will invert for.

x, y = SpatialCoordinate(mesh)
u = interpolate(as_vector((-y, x)), V)
u.rename('Velocity')

approximation = BoussinesqApproximation(Ra=1, kappa=5e-2)

# Unlike the previous tutorial, we specify a zero Dirichlet boundary condition for temperature at
# the bottom inflow boundary:
temp_bcs = {
    bottom: {'T': 0},
}
delta_t = 0.1
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
# -

# The initial condition that we, again, later will invert for, is now centered in the domain.

x0, y0 = 0.5, 0.5
w = .2
r2 = (x-x0)**2 + (y-y0)**2
T0 = interpolate(exp(-r2/w**2), Q)

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(T0, axes=axes, cmap='magma', vmax=0.15)
# fig.colorbar(collection);
# -

# After setting the initial condition for T, we run this simulation for twenty timesteps to ensure
# the entire Gaussian has left the domain. For this example, we checkpoint the solution at every
# timestep, so that we can later use it as the target boundary values.

num_timesteps = 20
T.project(T0)
with CheckpointFile("Model_State.h5", "w") as model_checkpoint:
    model_checkpoint.save_mesh(mesh)
    for timestep in range(num_timesteps):
        model_checkpoint.save_function(T, idx=timestep)
        energy_solver.solve()
    # After saving idx=0, 19 at beginning of each timestep, we include idx=20 for the solution at
    # the end of the final timestep:
    model_checkpoint.save_function(T, idx=timestep)

# As expected the solution has almost completely disappeared (note the different scalebar):

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='magma', vmax=0.05)
# fig.colorbar(collection);
# -

# Advection diffusion model with unknown initial condition
# --------------------------------------------------------
#
# As with our previous example, we again set up the model with the same configuration, albeit where we
# do not know the initial condition. We will try to find the optimal initial condition such that we closely
# match the recorded outflow boundary values.

with CheckpointFile("Model_State.h5", "r") as model_checkpoint:
    mesh = model_checkpoint.load_mesh()

# We now set up the model exactly as before:

# +
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
T = Function(Q, name='Temperature')
T0 = Function(Q, name="Initial_Temperature")
T_wrong = Function(Q, name="Wrong_Initial_Temperature")

x, y = SpatialCoordinate(mesh)
u = interpolate(as_vector((-y, x)), V)
u.rename('Velocity')

approximation = BoussinesqApproximation(Ra=1, kappa=5e-2)
temp_bcs = {
    bottom: {'T': 0},
}
delta_t = 0.1
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

# Make our solver output a little less verbose:
energy_solver.solver_parameters.pop('ksp_converged_reason')
# -

# As a first guess we use a Gaussian that is in the wrong place: centred around $(0.7, 0.7)$
# instead of $(0.5, 0.5)$:

x0, y0 = 0.7, 0.7
w = .2
r2 = (x-x0)**2 + (y-y0)**2
T_wrong = interpolate(exp(-r2/w**2), Q)

# As in our first example, we make sure to clear the tape before our actual model starts and
# specify the control at the right stage. During the model we load back in the solutions from the synthetic twin,
# but only use its values at the boundary to compute a mismatch with the current model as an integral over the left
# boundary. Note that we start calculating the functional already in the first timestep, and we keep on adding terms to it,
# all of which will still be automatically recorded by the pyadjoint tape.

# +
tape = get_working_tape()
tape.clear_tape()

T0.project(T_wrong)

m = Control(T0)

J = AdjFloat(0.0)  # Initialise functional
factor = AdjFloat(0.5)  # First & final boundary integral weighted by 0.5 to implement mid-point rule time-integration.

with CheckpointFile("Model_State.h5", "r") as model_checkpoint:
    for timestep in range(num_timesteps):
        T_target = model_checkpoint.load_function(mesh, 'Temperature', idx=timestep)
        J = J + factor * assemble((T-T_target)**2*ds(left))
        factor = 1.0  # Remaining timesteps weighted by 1
        energy_solver.solve()

    T_target = model_checkpoint.load_function(mesh, 'Temperature', idx=timestep)
    # Add final contribution weighted again by 0.5
    J = J + factor * assemble((T-T_target)**2*ds(left))

print(J)
# -

# We define the reduced functional using the final value of `J` and the specified control. This allows us to rerun
# the model with an arbitrary initial condition. As with our previous example, we first try to simply rerun the
# model with the same "wrong" initial condition, and print the functional.

reduced_functional = ReducedFunctional(J, m)
print(reduced_functional(T_wrong))


# Now we rerun the model with the "correct" initial condition from the twin experiment, ending up with 
# a near-zero misfit.

# +
T0_ref = Function(Q, name="Reference_Initial_Temperature")
x0, y0 = 0.5, 0.5
w = .2
r2 = (x-x0)**2 + (y-y0)**2
T0_ref = interpolate(exp(-r2/w**2), Q)

print(reduced_functional(T0_ref))
# -


# We can again look at the gradient, but this time the gradient is a lot less intuitive. We evaluate the gradient
# around an initial guess of T=0 as the initial condition, noting that when a Function is created its associated
# data values are zero.

T_wrong.assign(0.0)
reduced_functional(T_wrong)

# In unstructured mesh optimisation problems, it is important to work in the L2 Riesz representation
# to ensure a grid-independent result:

gradJ = reduced_functional.derivative(options={"riesz_representation": "L2"})

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(gradJ, axes=axes, cmap='viridis')
# fig.colorbar(collection);
# -

# Invert for optimal initial condition using gradient-based optimisation algorithm
# --------------------------------------------------------------------------------
#
# As in the previous example, we can now use ROL to invert for the inital condition.
# We have last evaluated the reduced functional with a zero initial condition as the control value,
# so this will be our initial guess.

# We first set lower and upper bound values for the control, which we can
# provide as functions in the same function space as the control:

T_lb = Function(Q).assign(0.0)
T_ub = Function(Q).assign(1.0)

# We next specify our minimisation problem using the LinMore algorithm:

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
minimisation_parameters["Status Test"]["Iteration Limit"] = 10

# Define the LinMore Optimiser class:
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)

# And again use our callback function to record convergence:

# +
functional_values = []


def record_value(value, *args):
    if functional_values:
        functional_values.append(min(value, min(functional_values)))
    else:
        functional_values.append(value)


reduced_functional.eval_cb_post = record_value
# -

# We next run the optimisation
optimiser.run()

# Let's see how well we have done. At this point a total number of 10 iterations
# have been performed so lets plot convergence:

# + tags=["active-ipynb"]
# plt.semilogy(functional_values)
# plt.xlabel("Iteration #")
# plt.ylabel("Functional value")
# plt.title("Convergence")
# -

# We next plot the optimal initial condition:

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T_, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# And next plot the reference initial condition:

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T0, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# We can also compare these by calculating the difference and plotting.
