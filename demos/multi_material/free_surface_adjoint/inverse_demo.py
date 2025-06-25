from gadopt import *
from gadopt.inverse import *


def reini_steady(psi: Function, psi_grad: Function) -> ufl.algebra.Sum:
    return -psi * (1 - psi) + epsilon * sqrt(inner(psi_grad, psi_grad))


def callback() -> None:
    psi_chk = psi.block_variable.checkpoint
    psi_grad_chk = psi_grad.block_variable.checkpoint
    eta_chk = stokes.block_variable.checkpoint.subfunctions[2]
    eta_chk.rename("Free surface (optimisation)")

    misfit = assemble((psi_chk - psi_obs) ** 2 * dx) / psi_misfit_scale
    reini = assemble(reini_steady(psi_chk, psi_grad_chk) ** 2 * dx) / psi_reini_scale
    gradient = assemble(inner(psi_grad_chk, psi_grad_chk) * dx) / psi_grad_scale
    interface = assemble((0.25 - (psi_chk - 0.5) ** 2) * dx) / psi_interface_scale

    misfit_fs = assemble((eta_chk - eta_obs) ** 2 * ds_top) / eta_misfit_scale

    log(f"Level-set misfit: {misfit / domain_area}")
    log(f"Level-set reinitialisation: {reini / domain_area}")
    log(f"Level-set gradient: {gradient / domain_area}")
    log(f"Level-set interface: {interface / domain_area}")
    log(f"Free-surface misfit: {misfit_fs / boundary_length}")

    psi_opt.assign(psi_control.block_variable.checkpoint)
    optimisation_file.write(psi_opt, eta_chk)


tape = get_working_tape()
tape.clear_tape()
if not annotate_tape():
    continue_annotation()

with CheckpointFile("forward_checkpoint.h5", "r") as forward_chk:
    mesh = forward_chk.load_mesh("firedrake_default")
    stokes_obs = forward_chk.load_function(mesh, "Stokes")
    eta_obs = stokes_obs.subfunctions[2]
    psi_obs = forward_chk.load_function(mesh, "Level set")
    psi_obs.rename("Level set (observation)")

mesh.cartesian = True
boundary = get_boundary_ids(mesh)

R = FunctionSpace(mesh, "R", 0)
C = FunctionSpace(mesh, "Q", 1)

stokes = Function(stokes_obs, name="Stokes")
stokes.subfunctions[0].rename("Velocity")
stokes.subfunctions[1].rename("Pressure")
stokes.subfunctions[2].rename("Free surface")
eta_func = stokes.subfunctions[2]
u = split(stokes)[0]
psi = Function(psi_obs, name="Level set")

epsilon = interface_thickness(psi)

mu = material_field(psi, [mu_mantle := 1e21, mu_slab := 1e23], interface="geometric")
rho_material = material_field(
    psi, [rho_mantle := 3200, rho_slab := 3300], interface="arithmetic"
)

approximation = Approximation(
    "BA",
    dimensional=True,
    parameters={"g": 9.81, "mu": mu, "rho": rho_mantle, "rho_material": rho_material},
)

delta_t = Function(R).assign(1e13)

stokes_bcs = {
    boundary.bottom: {"uy": 0.0},
    boundary.top: {"free_surface": {"eta_index": 2, "rho_ext": 0.0}},
    boundary.left: {"ux": 0.0},
    boundary.right: {"ux": 0.0},
}

stokes_solver = StokesSolver(
    stokes, approximation, coupled_tstep=delta_t, theta=0.5, bcs=stokes_bcs
)
del stokes_solver.solver_parameters["snes_monitor"]

adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
psi_grad = level_set_solver.solution_grad

myr_to_seconds = 1e6 * 365.25 * 8.64e4
time_now = 0.0
target_time = 25 * myr_to_seconds
step_count = int(target_time / float(delta_t))

psi_control = Function(C, name="Level set (control)").project(psi_obs)
psi_opt = Function(C, name="Level set (optimisation)")
psi.project(psi_control)
level_set_solver.solution_old.assign(psi)

output_file = VTKFile("inverse_output.pvd")
output_file.write(*stokes.subfunctions, psi, psi_grad, time=time_now / myr_to_seconds)
optimisation_file = VTKFile("optimisation_output.pvd")

for _ in tape.timestepper(iter(range(step_count))):
    level_set_solver.solve(disable_reinitialisation=True)
    level_set_solver.update_gradient()
    stokes_solver.solve()

    time_now += float(delta_t)
    output_file.write(
        *stokes.subfunctions, psi, psi_grad, time=time_now / myr_to_seconds
    )

ds_top = ds(boundary.top, domain=mesh)
boundary_length = assemble(1.0 * ds_top)
domain_area = assemble(1.0 * dx(domain=mesh))

psi_misfit_scale = max((psi.dat.data_ro - psi_obs.dat.data_ro) ** 2)
psi_grad_scale = np.sum(psi_grad.dat.data_ro**2, axis=1).max()
psi_reini_scale = (-0.25 + epsilon.dat.data_ro.max() * np.sqrt(psi_grad_scale)) ** 2
psi_interface_scale = 1.0

eta_misfit_scale = max((eta_func.dat.data_ro - eta_obs.dat.data_ro) ** 2)

psi_bounded = max_value(min_value(psi, 1.0), 0.0)

psi_misfit = assemble((psi - psi_obs) ** 2 * dx) / psi_misfit_scale
psi_reini = assemble(reini_steady(psi, psi_grad) ** 2 * dx) / psi_reini_scale
psi_grad_inner = assemble(inner(psi_grad, psi_grad) * dx) / psi_grad_scale
psi_interface = assemble((0.25 - (psi_bounded - 0.5) ** 2) * dx) / psi_interface_scale
objective = psi_misfit + psi_reini + psi_grad_inner + psi_interface
objective /= domain_area

eta_misfit = assemble((eta_func - eta_obs) ** 2 * ds_top) / eta_misfit_scale
objective += eta_misfit / boundary_length

pause_annotation()

reduced_functional = ReducedFunctional(objective, Control(psi_control))

log(f"Reduced functional: {reduced_functional(psi_control)}")
log(f"Objective: {objective}")

psi_control_bounded = min_value(max_value(psi_control, 0.0), 1.0)
perturbation = Function(psi_control, name="Level-set perturbation")
perturbation.interpolate(0.5 - abs(psi_control_bounded - 0.5))
# random_scale = np.random.default_rng().normal(
#     1.0, 0.1, size=perturbation.dat.data.shape
# )
# perturbation.dat.data[:] *= random_scale

perturbation_file = VTKFile("taylor_test_perturbation.pvd")
perturbation_file.write(psi_control, perturbation, time=time_now / myr_to_seconds)

taylor_convergence = taylor_test(reduced_functional, psi_control, perturbation)
log(f"Taylor test: {taylor_convergence}")

psi_lb = Function(psi_control, name="Lower bound").assign(0.0)
psi_ub = Function(psi_control, name="Upper bound").assign(1.0)

min_prob = MinimizationProblem(reduced_functional, bounds=(psi_lb, psi_ub))

minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 0.1
minimisation_parameters["Step"]["Trust Region"]["Radius Shrinking Threshold"] = 0.15
minimisation_parameters["Step"]["Trust Region"]["Radius Growing Threshold"] = 0.75
minimisation_parameters["Step"]["Trust Region"][
    "Radius Shrinking Rate (Negative rho)"
] = 0.03125
minimisation_parameters["Step"]["Trust Region"][
    "Radius Shrinking Rate (Positive rho)"
] = 0.125
minimisation_parameters["Step"]["Trust Region"]["Radius Growing Rate"] = 5
minimisation_parameters["Status Test"]["Iteration Limit"] = 100

optimiser = LinMoreOptimiser(min_prob, minimisation_parameters)
optimiser.add_callback(callback)
optimiser.run()
