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
T.interpolate(exp(-r2/w**2))

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# We run this simulation for twenty timesteps to ensure the entire Gaussian has left the domain.
# For this example, we checkpoint the solution at every timestep, so that we can later use it as the
# target boundary values.

with CheckpointFile("Model_State.h5", "w") as model_checkpoint:
    model_checkpoint.save_mesh(mesh)
    for timestep in range(20):
        model_checkpoint.save_function(T, idx=timestep)
        energy_solver.solve()
    # After saving idx=0, 19 at beginning of each timestep, we include idx=20 for the solution at
    # the end of the final timestep:
    model_checkpoint.save_function(T, idx=timestep)

# As expected the solution has almost completely disappeared:

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# Advection diffusion model with unknown initial condition
# --------------------------------------------------------
#
# As with our previous example, we again set up the model with the same configuration, albeit where we
# do not know the initial condition. We will try to find the optimal initial condition such that we closely
# match the recored outflow boundary values.

with CheckpointFile("Model_State.h5", "r") as model_checkpoint:
    mesh = model_checkpoint.load_mesh()

# We now setup the model exactly as before:

# +
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
T = Function(Q, name='Temperature')

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
Twrong = interpolate(exp(-r2/w**2), Q)

# As in our first example, we make sure to clear the tape before our actual model starts and
# specify the control at the right stage. During the model we load back in the solutions from the synthetic twin,
# but only use its values at the boundary to compute a mismatch with the current model as an integral over the left
# boundary. Note that we start calculating the functional already in the first timestep, and we keep on adding terms to it,
# all of which will still be automatically recorded by the pyadjoint tape.

# +
tape = get_working_tape()
tape.clear_tape()

T.interpolate(Twrong)

m = Control(T)

J = AdjFloat(0.0)
factor = AdjFloat(0.5)  # Note that the first and final boundary integral is weighted by 0.5 to implement mid-point rule time-integration.

with CheckpointFile("Model_State.h5", "r") as model_checkpoint:
    for timestep in range(20):
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
# the model with an arbitrary initial condition. Again we first try to simply rerun the model with the same "wrong"
# initial condition.

Jhat = ReducedFunctional(J, m)
Jhat(Twrong)


# Now try to rerun the model with "correct" initial condition from the twin experiment, and we see that indeed we end
# up with a near-zero misfit.

# +
x0, y0 = 0.5, 0.5
w = .2
r2 = (x-x0)**2 + (y-y0)**2
T0 = interpolate(exp(-r2/w**2), Q)

Jhat(T0)
# -


# We can again look at the gradient, but this time the gradient is a lot less intuitive. We evaluate the gradient
# around an initial guess of T=0 as the initial condition

T_init = Function(Q)
Jhat(T_init)

# In unstructured mesh optimisation problems, it is important to work in the L2 Riesz representation
# to ensure a grid-independent result:

gradJ = Jhat.derivative(options={"riesz_representation": "L2"})

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(gradJ, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# Invert for optimal initial condition using gradient-based optimisation algorithm
# --------------------------------------------------------------------------------
#
# As in the previous example, we can now use L-BFGS-B to invert for the inital condition.
# We have last evaluated the reduced functional with a zero initial condition as the control value,
# so this will be our initial guess.

# First specify bounds of 0 and 1 which will be enforced during the optimisation:

Tmin = Function(Q).assign(0.0)
Tmax = Function(Q).assign(1.0)

# And subsequently run the L-BFGS-B algorithm:
T_opt = minimize(Jhat, method='L-BFGS-B', bounds=[Tmin, Tmax], tol=1e-10)

# Let's see how well we have done. We first plot the optimal initial condition:

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T_opt, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# And next plot the reference initial condition:

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T0, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# We can also compare these by calculating the difference and plotting.
