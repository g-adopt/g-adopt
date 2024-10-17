from gadopt import *
from gadopt.inverse import *


def callback():
    psi_opt.assign(psi_control.block_variable.checkpoint)
    optimisation_file.write(psi_opt)

    psi_check = psi.block_variable.checkpoint
    psi_grad_check = psi_grad_proj.block_variable.checkpoint

    reini_steady = -psi_check * (1 - psi_check) + epsilon * sqrt(
        inner(psi_grad_check, psi_grad_check)
    )

    misfit = assemble((psi_check - psi_obs) ** 2 * dx)
    reinitialisation = assemble(reini_steady**2 * dx)

    log(f"Level-set misfit: {misfit}")
    log(f"Level-set reinitialisation: {reinitialisation}")


with CheckpointFile("forward_checkpoint.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default")

    z = forward_check.load_function(mesh, "Stokes")
    psi = forward_check.load_function(mesh, "Level set")

nx, ny = 80, 80
lx, ly = 0.9142, 1

mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

R = FunctionSpace(mesh, "R", 0)

u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
psi_control = Function(psi, name="Level-set control")
psi_obs = Function(psi, name="Level-set observation")
psi_opt = Function(psi, name="Level-set optimisation")

psi.assign(psi_control)

min_mesh_edge_length = min(lx / nx, ly / ny)
epsilon = Constant(min_mesh_edge_length / 4)

Ra_c_buoyant = 0
Ra_c_dense = 1
Ra_c = material_field(psi, [Ra_c_buoyant, Ra_c_dense], interface="arithmetic")

approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

Z_nullspace = create_stokes_nullspace(z.function_space())

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(
    approximation,
    z,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)

delta_t = Function(R).assign(1.0)
t_adapt = TimestepAdaptor(delta_t, u, z.function_space()[0], target_cfl=0.6)

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)
psi_grad_proj = level_set_solver.ls_grad_proj

time_now, time_end = 0, 150

output_file = VTKFile("inverse_output.pvd")
output_file.write(*z.subfunctions, psi, time=time_now)
optimisation_file = VTKFile("optimisation_17oct.pvd")

step = 0
while True:
    t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve(step, equation="advection")
    level_set_solver.update_level_set_gradient()
    stokes_solver.solve()

    time_now += float(delta_t)
    step += 1

    output_file.write(*z.subfunctions, psi, time=time_now)

    if time_now >= time_end:
        break

reini_steady = -psi * (1 - psi) + epsilon * sqrt(inner(psi_grad_proj, psi_grad_proj))

psi_misfit = assemble((psi - psi_obs) ** 2 * dx)
psi_reini = assemble(reini_steady**2 * dx)
objective = psi_misfit + psi_reini
reduced_functional = ReducedFunctional(objective, Control(psi_control))

pause_annotation()

log(f"\n\nReduced functional: {reduced_functional(psi_control)}")
log(f"Objective: {objective}\n\n")

perturbation = Function(psi_control, name="Level set perturbation")
perturbation.interpolate(0.5 - abs(min_value(max_value(psi_control, 0), 1) - 0.5))

# random_scale = np.random.default_rng().normal(
#     1, 0.001, size=perturbation.dat.data.shape
# )
# perturbation.dat.data[:] *= random_scale

taylor_convergence = taylor_test(reduced_functional, psi_control, perturbation)
log(f"\n\nTaylor test: {taylor_convergence}\n\n")

phi_lb = Function(psi_control, name="Lower Bound Phi").assign(0.0)
phi_ub = Function(psi_control, name="Upper Bound Phi").assign(1.0)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(phi_lb, phi_ub))

optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
optimiser.add_callback(callback)
optimiser.run()

optimisation_file.write(optimiser.rol_solver.rolvector.dat[0])
