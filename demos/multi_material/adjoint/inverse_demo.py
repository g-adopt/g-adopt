from gadopt import *
from gadopt.inverse import *


def callback():
    psi_opt.assign(psi_control.block_variable.checkpoint)
    optimisation_file.write(psi_opt)

    misfit = assemble((psi.block_variable.checkpoint - psi_obs) ** 2 * dx)
    dichotomy = assemble(
        (psi.block_variable.checkpoint * (1 - psi.block_variable.checkpoint)) ** 2 * dx
    )
    gradient = assemble(
        inner(
            psi_grad_proj.block_variable.checkpoint,
            psi_grad_proj.block_variable.checkpoint,
        )
        * dx
    )
    log(f"Level-set misfit: {misfit}")
    log(f"Level-set dicho: {dichotomy}")
    log(f"Level-set gradient: {gradient}")


# enable_disk_checkpointing()

with CheckpointFile("forward_checkpoint.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default")

    z = forward_check.load_function(mesh, "Stokes")
    psi = forward_check.load_function(mesh, "Level set")

nx, ny = 64, 64
lx, ly = 0.9142, 1

mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

tape = get_working_tape()
tape.clear_tape()

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
Ra_c = material_field(psi, [Ra_c_buoyant, Ra_c_dense], interface="sharp")

approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

delta_t = Function(R).assign(1.0)

Z_nullspace = create_stokes_nullspace(z.function_space())

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

output_file = VTKFile("inverse_output.pvd")
output_file.write(*z.subfunctions, psi)
optimisation_file = VTKFile("optimisation_output.pvd")

stokes_solver = StokesSolver(
    approximation,
    z,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)
del stokes_solver.solver_parameters["snes_monitor"]
stokes_solver.solve()

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)
psi_grad_proj = level_set_solver.ls_grad_proj

for step in range(200):
    level_set_solver.solve(step)  # , equation="advection")
    stokes_solver.solve()

    output_file.write(*z.subfunctions, psi)

psi_misfit = assemble((psi - psi_obs) ** 2 * dx)
psi_dicho = assemble((psi * (1 - psi)) ** 2 * dx)
psi_grad = assemble(inner(psi_grad_proj, psi_grad_proj) * dx)
objective = psi_misfit + psi_dicho + 1e-4 * psi_grad
reduced_functional = ReducedFunctional(objective, Control(psi_control))

pause_annotation()

# log(f"\n\nReduced functional: {reduced_functional(psi_control)}")
# log(f"Objective: {objective}\n\n")

# perturbation = Function(psi_control, name="Level set perturbation")
# perturbation.dat.data[:] = (
#     np.random.default_rng().random(perturbation.dat.data.shape) * 0.01
# )
# taylor_convergence = taylor_test(reduced_functional, psi_control, perturbation)
# log(f"\n\nTaylor test: {taylor_convergence}\n\n")

phi_lb = Function(psi_control, name="Lower Bound Phi").assign(-0.5)
phi_ub = Function(psi_control, name="Upper Bound Phi").assign(1.5)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(phi_lb, phi_ub))

optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
optimiser.add_callback(callback)
optimiser.run()

optimisation_file.write(optimiser.rol_solver.rolvector.dat[0])
