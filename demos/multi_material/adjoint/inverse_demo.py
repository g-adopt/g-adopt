from gadopt import *
from gadopt.inverse import *


def reinitialisation_steady(psi, psi_grad):
    return -psi * (1 - psi) + epsilon * sqrt(inner(psi_grad, psi_grad))


def callback(psi_control, psi_opt, optimisation_file, optimiser, objective):
    psi_opt.assign(psi_control.block_variable.checkpoint)
    optimisation_file.write(psi_opt)

    psi_check = psi.block_variable.checkpoint
    psi_grad_check = psi_grad_proj.block_variable.checkpoint

    misfit = assemble((psi_check - psi_obs) ** 2 * dx)
    reinitialisation = assemble(
        reinitialisation_steady(psi_check, psi_grad_check) ** 2 * dx
    )

    log(f"Level-set misfit: {misfit}")
    log(f"Level-set reinitialisation: {reinitialisation}")

    if misfit + reinitialisation <= 1e-3 * objective:
        plist = ROL.ParameterList({"Status Test": {"Iteration Limit": 0}}, "Parameters")
        optimiser.rol_algorithm.setStatusTest(ROL.StatusTest(plist), False)


def simulation(iteration: int) -> None:
    tape.clear_tape()

    psi_control = Function(C, name="Level-set control").project(psi_obs)
    psi_opt = Function(C, name="Level-set optimisation")
    psi.project(psi_control)
    level_set_solver.solution_old.assign(psi)

    stokes_solver.solve()

    time_now = 0

    output_file = VTKFile(f"inverse_output_{iteration}.pvd")
    output_file.write(*z.subfunctions, psi, psi_grad_proj, time=time_now)
    optimisation_file = VTKFile(f"optimisation_output_{iteration}.pvd")

    step = 0
    while True:
        t_adapt.maximum_timestep = time_increment - time_now
        t_adapt.update_timestep()

        level_set_solver.solve(step)
        level_set_solver.update_level_set_gradient()
        stokes_solver.solve()

        time_now += float(delta_t)
        step += 1

        output_file.write(*z.subfunctions, psi, psi_grad_proj, time=time_now)

        if time_now >= time_increment:
            break

    psi_misfit = assemble((psi - psi_obs) ** 2 * dx)
    psi_reini = assemble(reinitialisation_steady(psi, psi_grad_proj) ** 2 * dx)
    objective = psi_misfit + psi_reini
    reduced_functional = ReducedFunctional(objective, Control(psi_control))

    with stop_annotating():
        log(f"Reduced functional: {reduced_functional(psi_control)}")
        log(f"Objective: {objective}")

        perturbation = Function(psi_control, name="Level set perturbation")
        perturbation.interpolate(
            (0.5 - abs(min_value(max_value(psi_control, 0), 1) - 0.5))
        )
        random_scale = np.random.default_rng().normal(
            5e-2, 1e-3, size=perturbation.dat.data.shape
        )
        perturbation.dat.data[:] *= random_scale

        psi_lb = Function(psi_control, name="Lower bound").assign(0.0)
        psi_ub = Function(psi_control, name="Upper bound").assign(1.0)

        taylor_convergence = taylor_test(reduced_functional, psi_control, perturbation)
        log(f"Taylor test: {taylor_convergence}")

        minimisation_problem = MinimizationProblem(
            reduced_functional, bounds=(psi_lb, psi_ub)
        )

        minimisation_parameters["Status Test"]["Iteration Limit"] = 50
        optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
        optimiser.add_callback(
            callback, psi_control, psi_opt, optimisation_file, optimiser, objective
        )
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
C = FunctionSpace(mesh, FiniteElement("DQ", quadrilateral, 1, variant="equispaced"))

z = Function(Z)
u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
psi = Function(psi_obs, name="Level-set")

epsilon = Constant(mesh.cell_sizes.dat.data.min() / 4)

Ra_c_buoyant = 0
Ra_c_dense = 1
Ra_c = material_field(psi, [Ra_c_buoyant, Ra_c_dense], interface="arithmetic")

approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

Z_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

delta_t = Function(R).assign(1.0)
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6)

stokes_solver = StokesSolver(
    z,
    approximation,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)
del stokes_solver.solver_parameters["snes_monitor"]

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)
psi_grad_proj = level_set_solver.ls_grad_proj
level_set_solver.set_up_solvers()

target_time = 400
time_increment = 80

for iteration in range(target_time // time_increment):
    simulation(iteration + 1)
