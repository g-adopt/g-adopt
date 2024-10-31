from gadopt import *
from gadopt.inverse import *


def reinitialisation_steady(psi, psi_grad):
    return -psi * (1 - psi) + epsilon * sqrt(inner(psi_grad, psi_grad))


def callback():
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


with CheckpointFile("forward_checkpoint.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default")

    z = forward_check.load_function(mesh, "Stokes")
    psi = forward_check.load_function(mesh, "Level set")

nx, ny = 128, 32
lx, ly = 3e6, 7e5

mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

R = FunctionSpace(mesh, "R", 0)

u, p, eta = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
z.subfunctions[2].rename("Free surface")
psi_control = Function(psi, name="Level-set control")
psi_obs = Function(psi, name="Level-set observation")
psi_opt = Function(psi, name="Level-set optimisation")

psi.assign(psi_control)

local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

mu_slab = 1e23
mu_mantle = 1e21
mu = material_field(psi, [mu_mantle, mu_slab], interface="geometric")

rho_slab = 3300
rho_mantle = 3200
rho_material = material_field(psi, [rho_mantle, rho_slab], interface="arithmetic")

approximation = Approximation(
    "BA",
    dimensional=True,
    parameters={"g": 9.81, "mu": mu, "rho": rho_mantle, "rho_material": rho_material},
)

delta_t = Function(R).assign(1e13)
t_adapt = TimestepAdaptor(delta_t, u, z.function_space()[0], target_cfl=0.6)

stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"free_surface": {"eta_index": 0, "rho_ext": 0}},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(z, approximation, bcs=stokes_bcs, timestep_fs=delta_t)
del stokes_solver.solver_parameters["snes_monitor"]

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)
psi_grad_proj = level_set_solver.ls_grad_proj

time_now, time_end = 0, 1e7 * 365.25 * 8.64e4

output_file = VTKFile("inverse_output.pvd")
output_file.write(*z.subfunctions, psi, psi_grad_proj, time=time_now)
optimisation_file = VTKFile("optimisation_output.pvd")

step = 0
while True:
    t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve(step, equation="advection")
    level_set_solver.update_level_set_gradient()
    stokes_solver.solve()

    time_now += float(delta_t)
    step += 1

    output_file.write(*z.subfunctions, psi, psi_grad_proj, time=time_now)

    if time_now >= time_end:
        break

psi_misfit = assemble((psi - psi_obs) ** 2 * dx)
psi_reini = assemble(reinitialisation_steady(psi, psi_grad_proj) ** 2 * dx)
objective = psi_misfit + psi_reini
reduced_functional = ReducedFunctional(objective, Control(psi_control))

pause_annotation()

log(f"Reduced functional: {reduced_functional(psi_control)}")
log(f"Objective: {objective}")

perturbation = Function(psi_control, name="Level set perturbation")
perturbation.interpolate((0.5 - abs(min_value(max_value(psi_control, 0), 1) - 0.5)))

random_scale = np.random.default_rng().normal(
    5e-2, 1e-3, size=perturbation.dat.data.shape
)
perturbation.dat.data[:] *= random_scale

taylor_convergence = taylor_test(reduced_functional, psi_control, perturbation)
log(f"Taylor test: {taylor_convergence}")

psi_lb = Function(psi_control, name="Lower bound").assign(0.0)
psi_ub = Function(psi_control, name="Upper bound").assign(1.0)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(psi_lb, psi_ub))

minimisation_parameters["Status Test"]["Step Tolerance"] = 1e-3
optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
optimiser.add_callback(callback)
optimiser.run()
