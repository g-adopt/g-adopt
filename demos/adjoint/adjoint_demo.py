# Adjoint Inverse Demo
# ================================
#
# Introduction
# ------------
# In this tutorial, we will demonstrate how to perform an inversion to recover the initial temperature field of the mantle using G-ADOPT.
#
# This demonstration involves a *twin experiment*, where we assess the performance of the inversion scheme by inverting the initial state of a synthetic reference simulation, known as the "*Reference Twin*". To create this reference twin, we run a forward mantle convection simulation and record all relevant fields (velocity and temperature) at each time step.
#
# We have pre-run this simulation and stored the checkpoint file on our servers. These fields serve as benchmarks for evaluating our inverse problem's performance. To download the reference benchmark checkpoint file, click
# <a href="https://data.gadopt.org/demos/adjoint-demo-checkpoint-state.h5" download> here </a>, or alternatively, execute the following command in a terminal:
#
# ```wget https://data.gadopt.org/demos/adjoint-demo-checkpoint-state.h5```

# The fields from the reference simulation are stored under the names "Temperature" and "Velocity". To retrieve the timestepping information, we can use the following code:

# +
# Importing gadopt
from gadopt import *
from gadopt.inverse import *
checkpoint_filename = "adjoint-demo-checkpoint-state.h5"
# Opening the checkpoint file and loading the mesh
checkpoint_file = CheckpointFile(checkpoint_filename, mode="r")
mesh = checkpoint_file.load_mesh("firedrake_default_extruded")
# boundary markers from extruded mesh
bottom_id, top_id, left_id, right_id = "bottom", "top", 1, 2
# Retrieving the timestepping information for the Velocity and Temperature functions
temperature_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Temperature")
velocity_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Velocity")
# -

# We can check the information for each

# + tags=["active-ipynb"]
# print("Timestepping info for Temperature", temperature_timestepping_info)
# print("Timestepping infor for Velocity", velocity_timestepping_info)
# -

# The timestepping information reveals that there are 80 time-steps (from 0 to 79) in the reference simulation, with the temperature field stored only at the initial (index=0) and final (index=79) timesteps, while the velocity field is stored at all timesteps. We can visualise the benchmark fields using Firedrake's built-in VTK functionality. For example, the initial and final temperature fields can be visualised:

# +
# Load the observed final state
Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][-1]))
Tobs.rename("Observed Temperature")
# Load the reference initial state to evaluate performance
Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][0]))
Tic_ref.rename("Reference Initial Temperature")
checkpoint_file.close()
# -

# and subsequently visualising using standard VTK softwares, e.g., Paraview or pyvista:

# + tags=["active-ipynb"]
# import pyvista as pv
# dataset = pv.read('./visualisation_vtk.pvd')
# # Create a plotter object
# plotter = pv.Plotter()
# # Add the dataset to the plotter
# plotter.add_mesh(dataset, scalars='Observed Temperature', cmap='coolwarm')
# # Adjust the camera position
# plotter.camera_position = [(0.5, 0.5, 2.5), (0.5, 0.5, 0), (0, 1, 0)]
# # Show the plot
# plotter.show()
# -

# The inverse code
# ---------------------------------------
#
# The novelty of using the overloading approach provided by pyadjoint is that
# we need minimal changes to our script in order to enable the inverse capabalities of G-ADOPT.
# To turn on adjoint, one simply starts by importing
# the inverse module to enable all taping functionality from pyadjoint.


# Doing so will turn all Firedrake's objects to overloaded types, in a way
# that any UFL operation will be annotated and added to the tape, unless
# specified otherwise. To make sure that the tape is cleared of any
# previous operations:

# +
# get the working tape
tape = get_working_tape()
# clear the tape from any previous annotation
tape.clear_tape()
# -
# + tags=[active-ipynb]
# # print all the blocks
# print(tape.get_blocks())
# -

# From here onwards, every operation will be done with minimal differences with
# respect to our forward code. Under the hood, however, the tape will be populated
# by *blocks* that know the dependencies.
# Knowing the mesh was loaded above, we continue very similar to our base case.

# +
# Set up function spaces for the Q2Q1 pair.
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space

# Test functions and functions to hold solutions:
z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

Ra = Constant(1e6)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# Define time-stepping parameters.
delta_t = Constant(4e-6)  # Constant time step
timesteps = int(temperature_timestepping_info["index"][-1])  # number of timesteps from forward

T = Function(Q, name="Temperature")

# Nullspaces for the problem (constant pressure).
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Stokes and Energy boundary conditions - All free-slip for this problem
stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"uy": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
temp_bcs = {
    bottom_id: {"T": 1.0},
    top_id: {"T": 0.0},
}
# Energy and Stokes solver
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, nullspace=Z_nullspace, transpose_nullspace=Z_nullspace, constant_jacobian=True)
# -

# Define the Control Space
# ========================
#
# In this section, we define the control space, which can be simplified to reduce the risk of encountering an undetermined problem. Here, we select the Q1 space for the initial condition $T_{ic}$. We also provide an initial guess for the control value, which in this synthetic test is the temperature field of the reference simulation at the final time-step (`timesteps - 1`).

# +
# Define the control function space using continuous Galerkin (CG) of degree 1 (Q1)
Q1 = FunctionSpace(mesh, "CG", 1)

# Create a function for the initial temperature with a name for identification
Tic = Function(Q1, name="Initial Temperature")

# Project the temperature field from the reference simulation's final time-step onto the control space
with CheckpointFile(checkpoint_filename, mode="r") as fi:
    Tic.project(fi.load_function(mesh, "Temperature", idx=timesteps - 1))
# -

# We finally introduce our control problem to pyadjoint and continue by integrating the solutions
# each time-step. Notice that we accumulatively compute the misfit term with respect to the
# surface observable of plate velocities.

# +
# control
control = Control(Tic)

u_misfit = 0.0

# Going from Q1 to Q2, and impose the boundary conditions simultaneously.
T.project(Tic, bcs=energy_solver.strong_bcs)

# Populate the tape by running the forward simulation.
# For learning+tweaking purposes, feel free to change e.g., 0 -> timesteps - 3
for time_idx in range(0, timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    # Update the accumulated surface velocity misfit using the observed value.
    with CheckpointFile(checkpoint_filename, mode="r") as fi:
        uobs = fi.load_function(mesh, name="Velocity", idx=time_idx)
    u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)
# -

# Defining the Objective Functional
# =================================
#
# Now that all the calculations are in place, we need to define *the objective functional*.
# The objective functional is our way of expressing what is our goal for this optimisation.
# It is composed of several terms, each representing a different aspect of the model's
# performance and regularisation.

# Regularisation involves imposing constraints on solutions to prevent overfitting, ensuring that the model generalises well to new data. In this context, we use the one-dimensional (1D) temperature profile of the Earth derived from the reference simulation as our regularisation constraint. This profile, referred to as `Taverage`, helps stabilise the inversion process by providing a benchmark that guides the solution towards physically plausible states.
#
# The 1D profile, `Taverage`, is loaded from the checkpoint file and used as the regularisation constraint. This ensures that our inversion is not only fitting the observed data but also adhering to known physical behaviour.

# +
# Load the 1D average temperature profile from the checkpoint file for regularisation
Taverage = Function(Q1, name="Average Temperature")
with CheckpointFile(checkpoint_filename, mode="r") as fi:
    Taverage.project(fi.load_function(mesh, "Average Temperature", idx=0))
# -

# We use `Taverage` for including the damping term, smoothing term.
# Consequently, the complete objective functional is defined mathematically as follows:
#
# Reiterating that:
# - $T_{ic}$ is the initial temperature condition.
# - $T_{\text{average}}$ is the average temperature profile representing mantle's geotherm.
# - $T_{F}$ is the the temperature field at final time-step.
# - $T_{\text{obs}}$ is the observed temperature field at the final time-step.
# - $u_{\text{obs}}$ is the observed velocity field at *each time-step*.
# - $\alpha_u$, $\alpha_d$, $\alpha_s$ are the three different
#   weighting terms for the velocity, damping and smoothing terms.
#
# We define the objective functional as
# $$ \text{Objective Functional}= \int_{\Omega}(T - T_{\text{obs}}) ^ 2 \, dx \\
#                  +\alpha_u\, \frac{D_{T_{obs}}}{N\times D_{u_{obs}}}\sum_{i}\int_{\partial \Omega_{\text{top}}}(u - u_{\text{obs}}) \cdot(u - u_{\text{obs}}) \, ds \\
#                  +\alpha_s\, \frac{D_{T_{obs}}}{D_{\text{smoothing}}}\int_{\Omega} \nabla(T_{ic} - T_{\text{average}}) \cdot \nabla(T_{ic} - T_{\text{average}}) \, dx \\
#                  +\alpha_d\, \frac{D_{T_{obs}}}{D_{\text{damping}}}\int_{\Omega}(T_{ic} - T_{\text{average}}) ^ 2 \, dx $$

# With the three *normlisation terms* of:
# + $D_{\text{damping}} = \int_{\Omega} T_{\text{average}}^2 \, dx$,
# + $D_{\text{smoothing}} = \int_{\Omega} \nabla T_{\text{obs}} \cdot \nabla T_{\text{obs}} \, dx$,
# + $D_{T_{obs}} = \int_{\Omega} T_{\text{obs}} ^ 2 \, dx$, and
# + $D_{\text{damping}} = \int_{\partial \Omega_{\text{top}}} u_{\text{obs}} \cdot u_{\text{obs}} \, ds$
#
# which we define with the following definitions of `objective`

# +
# Define the component terms of the overall objective functional
damping = assemble((Tic - Taverage) ** 2 * dx)
norm_damping = assemble(Taverage**2 * dx)
smoothing = assemble(dot(grad(Tic - Taverage), grad(Tic - Taverage)) * dx)
norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
norm_obs = assemble(Tobs**2 * dx)
norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

# Temperature misfit between solution and observation
t_misfit = assemble((T - Tobs) ** 2 * dx)

# Weighting terms
alpha_u = 1e-1
alpha_d = 1e-2
alpha_s = 1e-1

# Define the overall objective functional
objective = (
    t_misfit +
    alpha_u * (norm_obs * u_misfit / timesteps / norm_u_surface) +
    alpha_d * (norm_obs * damping / norm_damping) +
    alpha_s * (norm_obs * smoothing / norm_smoothing)
)
# -

# Defining the Reduced Functional and Optimisation Method
# ===============================
#
# In optimisation terminology, a reduced functional is a functional that takes a given value for the control and outputs the value of the objective functional defined for it. It does this without explicitly depending on all the intermediary state variables, hence the name "reduced".
#
# To define the reduced functional, we provide the class with an objective (which is an overloaded UFL object) and the control. Both of these are essential for formulating the reduced functional.

# +
# Define the object for pyadjoint
reduced_functional = ReducedFunctional(objective, control)
# -

# Pausing Annotation
# ==================
#
# At this point, we have completed annotating the tape with the necessary information from running the forward simulation. To prevent further annotations during subsequent operations, we stop the annotation process. This ensures that no additional solves are unnecessarily recorded, keeping the tape focused only on the essential steps.
#
# We can then print the contents of the tape to verify what has been recorded.

# +
# Pause the annotation to stop recording further operations
pause_annotation()
# -

# Verification of Gradients: Taylor Remainder Convergence Test
# ============================================================
#
# A fundamental tool for verifying gradients is the Taylor remainder convergence test. This test helps ensure that the gradients computed by our optimisation algorithm are accurate. For the reduced functional, $J(T_{ic})$, and its derivative, $\frac{\mathrm{d} J}{\mathrm{d} T_{ic}}$, the Taylor remainder convergence test can be expressed as:
#
# $$ \left| J(T_{ic} + h \,\delta T_{ic}) - J(T_{ic}) - h\,\frac{\mathrm{d} J}{\mathrm{d} T_{ic}} \cdot \delta T_{ic} \right| \longrightarrow 0 \text{ at } O(h^2). $$
#
# The expression on the left-hand side is termed the second-order Taylor remainder. This term's convergence rate of $O(h^2)$ is a robust indicator for verifying the computational implementation of the gradient calculation. Essentially, if you halve the value of $h$, the magnitude of the second-order Taylor remainder should decrease by a factor of 4.
#
# We employ these so-called *Taylor tests* to confirm the accuracy of the determined gradients. The theoretical convergence rate is $O(2.0)$, and achieving this rate indicates that the gradient information is accurate down to floating-point precision.
#
# ### Performing Taylor Tests
#
# In our implementation, we perform a second-order Taylor remainder test for each term of the objective functional. The test involves computing the functional and the associated gradient when randomly perturbing the initial temperature field, $T_{ic}$, and subsequently halving the perturbations at each level.
#
# Here is how you can perform a Taylor test in the code:

# + tags=["active-ipynb"]
# # Define the perturbation in the initial temperature field
# import numpy as np
# Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
# Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
#
# # Perform the Taylor test to verify the gradients
# minconv = taylor_test(reduced_functional, Tic, Delta_temp)
# -

# The `taylor_test` function computes the Taylor remainder and verifies that the convergence rate is close to the theoretical value of $O(2.0)$. This ensures that our gradients are accurate and reliable for optimisation.

# Running the inversion
# ========
# In the final section, we run the optimisation method. First, we define lower and upper bounds for the optimisation problem to guide the optimisation method towards the solution.
#
# For this simple problem, we perform a bounded nonlinear optimisation where the temperature is only permitted to lie in the range [0, 1]. This means that the optimisation problem should not search for solutions beyond these values.

# +
# Define lower and upper bounds for the temperature
T_lb = Function(Tic.function_space(), name="Lower Bound Temperature")
T_ub = Function(Tic.function_space(), name="Upper Bound Temperature")

# Assign the bounds
T_lb.assign(0.0)
T_ub.assign(1.0)

# Define the minimisation problem, with the goal to minimise the reduced functional
# Note: In other cases, the goal might be to maximise the functional
minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
# -

# Using the LinMore Optimiser
# ===========================
#
# In this study, we employ the trust region method of Lin and Moré (1999) implemented in ROL (Rapid Optimization Library). Lin-Moré is a truncated Newton method, which involves the repeated application of an iterative algorithm to approximately solve Newton’s equations (Dembo and Steihaug, 1983).
#
# Lin-Moré effectively handles provided bound constraints by ensuring that variables remain within their specified bounds. During each iteration, variables are classified into "active" and "inactive" sets. Variables at their bounds that do not allow descent are considered active and are fixed during the iteration. The remaining variables, which can change without violating the bounds, are inactive. These properties make the algorithm a robust and efficient method for solving bound-constrained optimisation problems.
#
# A notable feature of this optimisation approach in ROL is its checkpointing capability. For every iteration, all the information necessary to restart the optimisation from that iteration is saved in the specified `checkpoint_dir`.

# +
# Define the LinMore Optimiser class with checkpointing capability
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir="optimisation_checkpoint",
)

# Restore from last possible checkpoint
# optimiser.restore()
# Run the optimisation
optimiser.run()
