import numpy as np
from gadopt import *
from gadopt.inverse import *

with CheckpointFile("forward_checkpoint.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default")
    psi_obs = forward_check.load_function(mesh, "Level set")
    psi_obs.rename("Level-set observation")

nx, ny = 80, 80
lx, ly = 0.9142, 1
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W])
Q = FunctionSpace(mesh, "CG", 2)
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

z = Function(Z)
u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
T = Function(Q, name="Temperature")
psi = Function(K, name="Level set")

# Define initial condition for psi, which will be our control.
# Also take final state as our initial_guess
with CheckpointFile("forward_checkpoint.h5", "r") as forward_check:
    psi_ic = forward_check.load_function(mesh, "Level set")
    psi_ic.rename("Initial Level-set")

# Take our initial guess and project into psi:
psi.assign(psi_ic)

# Continue in a way consistent with forward run:
min_mesh_edge_length = min(lx / nx, ly / ny)
epsilon = Constant(min_mesh_edge_length / 4)

Ra = 0
RaB = field_interface([psi], [-1, 0], method="arithmetic")

approximation = BoussinesqApproximation(Ra, RaB=RaB)

delta_t = Function(R).assign(0.1)

Z_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

output_file = VTKFile("inverse_output.pvd")

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)

subcycles = 1
level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, subcycles, epsilon)
level_set_solver.reini_params["iterations"] = 0

# Define control:
control = Control(psi_ic)

timesteps = 200

for timestep in range(0, timesteps):

    output_file.write(*z.subfunctions, T, psi, psi_ic)

    stokes_solver.solve()

    level_set_solver.solve(timestep)

# Now specify objective functional and perform Taylor test:
objective = assemble((psi - psi_obs) ** 2 * dx)
reduced_functional = ReducedFunctional(objective, control)

pause_annotation()

log(f"\n\nReduced functional: {reduced_functional(psi_ic)}")
log(f"Objective: {objective}\n\n")

perturbation = Function(control.function_space(), name="Level set perturbation")
perturbation.dat.data[:] = np.random.default_rng().random(perturbation.dat.data.size) * 0.1
log(taylor_test(reduced_functional, psi_ic, perturbation))

# Now optimise:

# Define lower and upper bounds for the temperature
phi_lb = Function(psi_ic.function_space(), name="Lower Bound Phi")
phi_ub = Function(psi_ic.function_space(), name="Upper Bound Phi")

# Assign the bounds
phi_lb.assign(0.0)
phi_ub.assign(1.0)

# Define the minimisation problem, with the goal to minimise the reduced functional
# Note: in some scenarios, the goal might be to maximise (rather than minimise) the functional.
minimisation_problem = MinimizationProblem(reduced_functional, bounds=(phi_lb, phi_ub))
minimisation_parameters["Status Test"] = 20

# Define the LinMore Optimiser class with checkpointing capability
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir="optimisation_checkpoint",
)

solutions_vtk = VTKFile("solutions.pvd")
solution_container = Function(psi_ic.function_space(), name="Solutions")


def callback():
    solution_container.assign(psi_ic.block_variable.checkpoint)
    solutions_vtk.write(solution_container)
#    final_phi_misfit = assemble(
#        (phi.block_variable.checkpoint.restore() - phi_obs) ** 2 * dx
#    )
#    log(f"Terminal Phi Misfit: {final_phi_misfit}")


optimiser.add_callback(callback)
optimiser.run()
