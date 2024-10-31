# PDE Constrained Optimisation with G-ADOPT - Field Values
# ========================================================
#
# In this tutorial, we invert for an (unknown) initial condition, from a given final state.
#
# We will see how we can use the PDE constrained optimisation functionality of G-ADOPT to optimize
# one of the inputs to a PDE for a specified desired outcome. We will use a time-dependent
# advection-diffussion equation as our PDE and see how, for a given final state of the solution,
# we can retrieve what the initial condition should be, via an adjoint approach.
#
# To turn on the adjoint, one simply imports the gadopt.inverse module to enable all taping
# functionality from pyadjoint. The tape automatically registers all operations that form part of
# the forward model, which is used to automatically form the adjoint (backward) model.

from gadopt import *
from gadopt.inverse import *

# Create synthetic twin experiment with final state for a known initial condition.
# --------------------------------------------------------------------------------
#
# We first run a simple advection-diffusion model with a known initial condition. Of that model
# we only store the solution at the final timestep, which we will use in our inversion experiment
# later as the target final state.
#
# We setup the model in a form compatible with our previous examples, with a mesh, function spaces,
# a prescribed velocity field, boundary conditions etc... We utilise the `EnergySolver` of G-ADOPT to
# set up an energy equation under the Boussinesq Approximation, which is just a scalar
# advection-diffusion equation for temperature.

# +
mesh = UnitSquareMesh(40, 40)
left, right, bottom, top = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "CG", 2)  # Function space for velocity
Q = FunctionSpace(mesh, "CG", 1)  # Function space for the scalar (Temperature)
T = Function(Q, name='Temperature')
T0 = Function(Q, name='Initial_Temperature')  # Initial condition for temperature which we will invert for

# Set up prescribed velocity field -- an anti-clockwise rotation around (0.5, 0.5):
x, y = SpatialCoordinate(mesh)
u = interpolate(as_vector((-y+0.5, x-0.5)), V)
u.rename('Velocity')

# The Rayleigh number, Ra, is not actually used here, but we set a value for the diffusivity, kappa.
approximation = BoussinesqApproximation(Ra=1, kappa=1e-2)
temp_bcs = {}  # all closed boundaries by default
delta_t = 0.1  # timestep
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

# Setup the initial condition for Temperature: a Gaussian at $(0.75, 0.5)$.
# This will be the initial condition we will try to invert for later.

x0, y0 = 0.75, 0.5
w = .1
r2 = (x-x0)**2 + (y-y0)**2
T0.interpolate(exp(-r2/w**2))
# -


# We can visualise the initial temperature field using Firedrake's built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(T0, axes=axes, cmap='magma', vmax=0.15)
# fig.colorbar(collection);
# -

# After setting the initial condition for T, we can next run the forward model, for 10 timesteps.
# Pretty simple, huh?

T.project(T0)
for timestep in range(10):
    energy_solver.solve()

# We can plot the final temperature solution, which you will see has been rotated whilst simultaneously diffusing.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='magma', vmax=0.15)
# fig.colorbar(collection);
# -

# We next save this final synthetic model state in a checkpoint file for later use:

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")

# Advection diffusion model with unknown initial condition
# --------------------------------------------------------
#
# We now start from scratch again with an advection-diffusion model with the same configuration,
# except this time we don't know the initial condition. As we want to measure for different
# initial conditions, how well the final state of the model matches the one we just saved,
# we first read back that target final state. We will also use the mesh from the checkpoint file
# to construct the model.

with CheckpointFile("Final_State.h5", "r") as final_checkpoint:
    mesh = final_checkpoint.load_mesh()
    T_target = final_checkpoint.load_function(mesh, name="Temperature")

# With this information stored, we now set up the model exactly as before:

# +
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
T = Function(Q, name='Temperature')
T0 = Function(Q, name='Initial_Temperature')

x, y = SpatialCoordinate(mesh)
u = interpolate(as_vector((-y+0.5, x-0.5)), V)
u.rename('Velocity')

approximation = BoussinesqApproximation(Ra=1, kappa=1e-2)
temp_bcs = {}
delta_t = 0.1
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
# -

# This time, however, we don't want to use the known initial condition. Instead we will start with
# the final state from our synthetic forward model, which will then be further rotated and diffused. After
# ten timesteps we compute the mismatch between the predicted final state, and the checkpointed final state from
# our synthetic twin. This computation, the ten timesteps and the mismatch calculation, forms the forward model
# that we want to invert. Its adjoint will be created automatically from the tape, which registers all
# operations in the model. Since the tape was automatically started at the top when we imported
# `gadopt.inverse`, we must ensure we don't get mixed up with any operations that
# happened in our initial synthetic twin model, so we first clear everything that has already
# been registered from the tape.

tape = get_working_tape()
tape.clear_tape()

# At this stage, we are good to specify our initial guess for temperature, from T_target (i.e. the final
# state of our synthetic forward run). We do this by interpolating T_target to T. Note that in theory, we could
# start from an arbitrary initial condition here, provided it is bounded between 0 and 1.

T0.project(T_target)

# We want to optimise for the initial temperature state, T0, and so we specify this as the control:

m = Control(T0)

# Based on our guess for the initial temperature, we run the model for 10 timesteps:

T.project(T0)
for timestep in range(10):
    energy_solver.solve()

# For good performance of optimisation algorithms, it is important to scale both the control and the
# functional values to be of order 1. Note that mathematically scaling the functional should not
# change the optimal solution.

scaling = 1./assemble(T_target**2*dx)
J = assemble(scaling * (T-T_target)**2*dx)

# We can print the mismatch

print(F"Mismatch functional J={J}")

# And plot the final temperature state (T) for comparison with the true final state (T_target)

# + tags=["active-ipynb"]
# fig, axes = plt.subplots(1,2)
# ax1 = tripcolor(T, axes=axes[0], cmap='magma', vmax=0.15)
# ax2 = tripcolor(T_target, axes=axes[1], cmap='magma', vmax=0.15)
# fig.colorbar(ax2);
# -

# Now we have run the forward model and populated the tape with all operations required for the inverse
# model, we can define the *reduced functional* which combines the functional with the control we
# specified above:

Jhat = ReducedFunctional(J, m)

# The reduced functional allows us to rerun the forward model for different values of the control. It
# can be used as a function that takes in any choice of the control, runs the forward model and
# computes the functional. For instance we can rerun the model again using `T_target` as the initial
# condition, i.e. re-running the exact same model we have just run:

print(Jhat(T_target))

# As expected it produces the exact same functional value. Now we can try to see what happens if we
# use the correct initial condition, exactly matching that used in our twin experiment:

# +
T0_ref = Function(Q, name='Reference_Initial_Temperature')
x0, y0 = 0.75, 0.5
w = .1
r2 = (x-x0)**2 + (y-y0)**2
T0_ref.interpolate(exp(-r2/w**2))

print(Jhat(T0_ref))
# -

# Using the "correct" initial condition, we reach the same final state as in our twin model, and thus
# the functional ends up being almost exactly zero!

# In addition to re-running the model by evaluating the reduced functional, we can also calculate
# its derivative. This computes the sensitivity of the model with respect to its control (the initial
# condition). Here it tells us in what locations a (small) increase in the initial condition will
# lead to an increase in the functional.

# We want to compute the derivative at the "wrong" initial condition T_target,
# so we first rerun the forward model with that value for the control

Jhat(T_target)

# In unstructured mesh optimisation problems, it is important to work in the L2 Riesz representation
# to ensure a grid-independent result:

gradJ = Jhat.derivative(options={"riesz_representation": "L2"})

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(gradJ, axes=axes, cmap='viridis', vmin=-40, vmax=40)
# fig.colorbar(collection);
# -

# Invert for optimal initial condition using gradient-based optimisation algorithm
# --------------------------------------------------------------------------------
#
# We now have all ingredients required to run a gradient-based optimisation algorithm for the initial
# condition:
# - The ability to rerun and re-evaluate the functional for arbitrary input control values.
# - The ability to compute the derivative of the functional with respect to that control.
#
# To keep things simple, we here use the "L-BFGS-B" algorithm as it is implemented in scipy. The
# `minimize()` function that is exported by `gadopt.inverse` provides a wrapper around
# [scipy's minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
# to translate between G-ADOPT/Firedrake objects and numpy arrays. G-ADOPT also provides an interface
# to the [ROL optimisation library](https://trilinos.github.io/), which we generally
# recommend over the scipy interface. We will see this in later tutorials.

# The L-BFGS-B allows for "box constraints", min and max values for the control, which we can
# provide as functions in the same functionspace as the control:

T_lb = Function(Q).assign(0.0)
T_ub = Function(Q).assign(1.0)

# Define the minimisation problem, with the goal to minimise the reduced functional.
# Note: in some scenarios, the goal might be to maximise (rather than minimise) the functional.

minimisation_problem = MinimizationProblem(Jhat, bounds=(T_lb, T_ub))
minimisation_parameters["Status Test"]["Iteration Limit"] = 20

# Define the LinMore Optimiser class:
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)

# For sake of book-keeping the simulation, we have implemented a means to
# record information that might be used to check the optimisation performance. This
# callback function will be executed at the end of each iteration. Here, we write out
# the control field, i.e., the reconstructed intial temperature field, at the end of
# each iteration. We also record the values of the reduced
# functional directly in order to produce a plot of the convergence.

# +
functional_values = []


def record_value(value, *args):
    functional_values.append(value)


Jhat.eval_cb_post = record_value

# If it existed, we could restore the optimisation from last checkpoint:
# optimiser.restore()

# Run the optimisation
optimiser.run()
# -

# At this point a total number of 20 iterations are performed.
# Now we can look at the solution visually and plot convergence.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# plt.plot(functional_values)
# plt.xlabel("Optimisation iteration")
# plt.ylabel("Reduced functional")
# plt.title("Optimisation convergence")
# -


# Let's see how well we have done. We first plot the optimal initial condition and compare to the reference initial condition:

# + tags=["active-ipynb"]
# fig, axes = plt.subplots(1,2)
# ax1 = tripcolor(T0.block_variable.checkpoint, axes=axes[0], cmap='magma', vmax=0.15)
# ax2 = tripcolor(T0_ref, axes=axes[1], cmap='magma', vmax=0.15)
# fig.colorbar(ax2);
# -

# We can also compare these by calculating the difference and plotting. TO DO.
