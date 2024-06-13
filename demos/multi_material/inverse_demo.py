from gadopt import *
from gadopt.inverse import *

# def callback():
#     optimisation_file.write(psi_obs.block_variable.checkpoint)


with CheckpointFile("forward_checkpoint.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default")

    z = forward_check.load_function(mesh, "Stokes")
    T = forward_check.load_function(mesh, "Temperature")
    psi_obs = forward_check.load_function(mesh, "Level set")
    psi_obs.rename("Level set observation")

nx, ny = 40, 40
lx, ly = 0.9142, 1

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

R = FunctionSpace(mesh, "R", 0)

u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
psi = Function(psi_obs.function_space(), name="Level set").project(psi_obs)

min_mesh_edge_length = min(lx / nx, ly / ny)
epsilon = Constant(min_mesh_edge_length / 4)

Ra = 0

RaB = field_interface([psi], [-1, 0], method="arithmetic")

approximation = BoussinesqApproximation(Ra, RaB=RaB)

time_now = 1000
delta_t = Function(R).assign(2.5)
output_frequency = 10

Z_nullspace = create_stokes_nullspace(z.function_space())

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

output_file = VTKFile("inverse_output.pvd")
optimisation_file = output.VTKFile("inverse_optimisation.pvd")

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

step = 0
output_counter = 100
time_end = 2000
while True:
    if time_now >= output_counter * output_frequency:
        output_file.write(*z.subfunctions, T, psi)
        output_counter += 1

    time_now += float(delta_t)
    step += 1

    stokes_solver.solve()

    level_set_solver.solve(step)

    if time_now >= time_end:
        log("Reached end of simulation -- exiting time-step loop")
        break

objective = assemble((psi - psi_obs) ** 2 * dx)
log(f"\n\n{objective}\n\n")

reduced_functional = ReducedFunctional(objective, Control(psi_obs))
log(f"\n\n{reduced_functional(psi_obs)}\n\n")

perturbation = [Function(psi.function_space()).interpolate(0.5)]
log(taylor_test(reduced_functional, psi_obs, perturbation))

# psi_lower_bound = Function(psi_obs.function_space()).assign(0)
# psi_upper_bound = Function(psi_obs.function_space()).assign(1)

# minimisation_problem = MinimizationProblem(
#     reduced_functional, bounds=(psi_lower_bound, psi_upper_bound)
# )

# optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
# optimiser.add_callback(callback)
# optimiser.run()

# optimisation_file.write(*optimiser.rol_solver.rolvector.dat)
