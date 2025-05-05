from gadopt import *
from gadopt.inverse import *


def reinitialisation_steady(psi, psi_grad):
    return -psi * (1 - psi) + epsilon * sqrt(inner(psi_grad, psi_grad))


def callback(psi_control, psi_opt, optimisation_file):
    psi_opt.assign(psi_control.block_variable.checkpoint)
    optimisation_file.write(psi_opt)

    psi_check = psi.block_variable.checkpoint
    psi_grad_check = psi_grad.block_variable.checkpoint

    misfit = assemble((psi_check - psi_obs) ** 2 * dx)
    reinitialisation = assemble(
        reinitialisation_steady(psi_check, psi_grad_check) ** 2 * dx
    )

    log(f"Level-set misfit: {misfit}")
    log(f"Level-set reinitialisation: {reinitialisation}")


def simulation(iteration: int) -> None:
    tape.clear_tape()
    if not annotate_tape():
        continue_annotation()

    psi_control = Function(C, name="Level-set control").project(psi_obs)
    psi_opt = Function(C, name="Level-set optimisation")
    psi.project(psi_control)
    level_set_solver.solution_old.assign(psi)

    stokes_solver.solve()

    time_now = 0

    output_file = VTKFile(f"inverse_output_{iteration}.pvd")
    output_file.write(*z.subfunctions, psi, psi_grad, time=time_now)
    optimisation_file = VTKFile(f"optimisation_output_{iteration}.pvd")

    for timestep in tape.timestepper(iter(range(step_count))):
        level_set_solver.solve(disable_reinitialisation=True)
        level_set_solver.update_gradient()
        stokes_solver.solve()

        time_now += float(delta_t)
        output_file.write(*z.subfunctions, psi, psi_grad, time=time_now)

    psi_misfit = assemble((psi - psi_obs) ** 2 * dx)
    psi_reini = assemble(reinitialisation_steady(psi, psi_grad) ** 2 * dx)
    objective = psi_misfit + psi_reini
    reduced_functional = ReducedFunctional(objective, Control(psi_control))

    pause_annotation()

    log(f"Reduced functional: {reduced_functional(psi_control)}")
    log(f"Objective: {objective}")

    perturbation = Function(psi_control, name="Level-set perturbation")
    perturbation.interpolate(0.5 - abs(min_value(max_value(psi_control, 0), 1) - 0.5))
    # random_scale = np.random.default_rng().normal(
    #     5e-2, 1e-3, size=perturbation.dat.data.shape
    # )
    # perturbation.dat.data[:] *= random_scale

    taylor_convergence = taylor_test(reduced_functional, psi_control, perturbation)
    log(f"Taylor test: {taylor_convergence}")

    psi_lb = Function(psi_control, name="Lower bound").assign(0.0)
    psi_ub = Function(psi_control, name="Upper bound").assign(1.0)

    minimisation_problem = MinimizationProblem(
        reduced_functional, bounds=(psi_lb, psi_ub)
    )

    minimisation_parameters["Status Test"]["Gradient Tolerance"] = 1e-4
    minimisation_parameters["Status Test"]["Iteration Limit"] = 50
    optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
    optimiser.add_callback(callback, psi_control, psi_opt, optimisation_file)
    optimiser.run()

    psi_obs.project(psi_opt)


tape = get_working_tape()

with CheckpointFile("forward_checkpoint.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default")
    psi_obs = forward_check.load_function(mesh, "Level set")
    psi_obs.rename("Level set observation")

mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W])
R = FunctionSpace(mesh, "R", 0)
C = FunctionSpace(mesh, "Q", 1)

z = Function(Z)
u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
psi = Function(psi_obs, name="Level-set")

epsilon = interface_thickness(psi)

Ra_c = material_field(psi, [Ra_c_buoyant := 0, Ra_c_dense := 1], interface="arithmetic")
approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

delta_t = Function(R).assign(1.0)

Z_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(
    z,
    approximation,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)
del stokes_solver.solver_parameters["snes_monitor"]

adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
psi_grad = level_set_solver.solution_grad

target_time = 150
time_increment = 50
time_step = 0.85
step_count = int(time_increment / time_step)

for iteration in range(target_time // time_increment):
    simulation(iteration + 1)
