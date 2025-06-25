from gadopt import *
from gadopt.inverse import *


def reini_steady(psi: Function, psi_grad: Function) -> ufl.algebra.Sum:
    return -psi * (1 - psi) + epsilon * sqrt(inner(psi_grad, psi_grad))


def callback() -> None:
    psi_chk = psi.block_variable.checkpoint
    psi_grad_chk = psi_grad.block_variable.checkpoint

    misfit = assemble((psi_chk - psi_obs) ** 2 * dx) / psi_misfit_scale
    reini = assemble(reini_steady(psi_chk, psi_grad_chk) ** 2 * dx) / psi_reini_scale
    gradient = assemble(inner(psi_grad_chk, psi_grad_chk) * dx) / psi_grad_scale
    interface = assemble((0.25 - (psi_chk - 0.5) ** 2) * dx) / psi_interface_scale

    log(f"Level-set misfit: {misfit / domain_area}")
    log(f"Level-set reinitialisation: {reini / domain_area}")
    log(f"Level-set gradient: {gradient / domain_area}")
    log(f"Level-set interface: {interface / domain_area}")

    psi_opt.assign(psi_control.block_variable.checkpoint)
    optimisation_file.write(psi_opt)


tape = get_working_tape()
tape.clear_tape()
if not annotate_tape():
    continue_annotation()

with CheckpointFile("forward_checkpoint.h5", "r") as forward_chk:
    mesh = forward_chk.load_mesh("firedrake_default")
    stokes_obs = forward_chk.load_function(mesh, "Stokes")
    psi_obs = forward_chk.load_function(mesh, "Level set")
    psi_obs.rename("Level set (observation)")

mesh.cartesian = True
boundary = get_boundary_ids(mesh)

R = FunctionSpace(mesh, "R", 0)
C = FunctionSpace(mesh, "Q", 1)

stokes = Function(stokes_obs, name="Stokes")
stokes.subfunctions[0].rename("Velocity")
stokes.subfunctions[1].rename("Pressure")
u = split(stokes)[0]
psi = Function(psi_obs, name="Level set")

epsilon = interface_thickness(psi)

Ra_c = material_field(psi, [Ra_c_buoyant := 0, Ra_c_dense := 1], interface="arithmetic")
approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

delta_t = Function(R).assign(0.85)

stokes_nullspace = create_stokes_nullspace(stokes.function_space())

stokes_bcs = {
    boundary.bottom: {"u": 0},
    boundary.top: {"u": 0},
    boundary.left: {"ux": 0},
    boundary.right: {"ux": 0},
}

stokes_solver = StokesSolver(
    stokes,
    approximation,
    bcs=stokes_bcs,
    nullspace={"nullspace": stokes_nullspace, "transpose_nullspace": stokes_nullspace},
)
del stokes_solver.solver_parameters["snes_monitor"]

adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
psi_grad = level_set_solver.solution_grad

time_now = 0.0
target_time = 200.0
step_count = int(target_time / float(delta_t))

psi_control = Function(C, name="Level set (control)").project(psi_obs)
psi_opt = Function(C, name="Level set (optimisation)")
psi.project(psi_control)
level_set_solver.solution_old.assign(psi)

output_file = VTKFile("inverse_output_new_2.pvd")
output_file.write(*stokes.subfunctions, psi, psi_grad, time=time_now)
optimisation_file = VTKFile("optimisation_output_new_2.pvd")

for _ in tape.timestepper(iter(range(step_count))):
    level_set_solver.solve(disable_reinitialisation=True)
    level_set_solver.update_gradient()
    stokes_solver.solve()

    time_now += float(delta_t)
    output_file.write(*stokes.subfunctions, psi, psi_grad, time=time_now)

domain_area = assemble(1.0 * dx(domain=mesh))

psi_misfit_scale = max((psi.dat.data_ro - psi_obs.dat.data_ro) ** 2)
psi_grad_scale = np.sum(psi_grad.dat.data_ro**2, axis=1).max()
psi_reini_scale = (-0.25 + epsilon.dat.data_ro.max() * np.sqrt(psi_grad_scale)) ** 2
psi_interface_scale = 2.0

psi_bounded = max_value(min_value(psi, 1.0), 0.0)

psi_misfit = assemble((psi - psi_obs) ** 2 * dx) / psi_misfit_scale
psi_reini = assemble(reini_steady(psi, psi_grad) ** 2 * dx) / psi_reini_scale
psi_grad_inner = assemble(inner(psi_grad, psi_grad) * dx) / psi_grad_scale
psi_interface = assemble((0.25 - (psi_bounded - 0.5) ** 2) * dx) / psi_interface_scale
objective = psi_misfit + psi_reini + psi_grad_inner + psi_interface
objective /= domain_area

pause_annotation()

reduced_functional = ReducedFunctional(objective, Control(psi_control))

log(f"Reduced functional: {reduced_functional(psi_control)}")
log(f"Objective: {objective}")

psi_control_bounded = min_value(max_value(psi_control, 0.0), 1.0)
perturbation = Function(psi_control, name="Level-set perturbation")
perturbation.interpolate(0.5 - abs(psi_control_bounded - 0.5))
random_scale = np.random.default_rng().normal(
    1.0, 0.1, size=perturbation.dat.data.shape
)
perturbation.dat.data[:] *= random_scale

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
minimisation_parameters["Status Test"]["Iteration Limit"] = 50

optimiser = LinMoreOptimiser(min_prob, minimisation_parameters)
optimiser.add_callback(callback)
optimiser.run()

# Lin-More Trust-Region Method (Type B, Bound Constraints)
#   iter  value          gnorm          snorm          delta          #fval     #grad     #hess     #proj     tr_flag   iterCG    flagCG
#   0     1.893600e-01   3.836773e-01   ---            1.000000e+00   1         1         0         2         ---       ---       ---
# Level-set misfit: 0.15889879372087667
# Level-set reinitialisation: 0.002169700450435749
# Level-set gradient: 0.02043008651033315
# Level-set interface: 0.007861411645702657
#   1     1.893600e-01   3.836773e-01   4.149025e-01   1.037256e-01   2         1         5         8         2         0         0
# Level-set misfit: 0.15889879372087667
# Level-set reinitialisation: 0.011119099392547668
# Level-set gradient: 0.04287612683557787
# Level-set interface: 0.007861411645702657
#   2     1.499347e-01   3.011401e-01   1.037256e-01   1.037256e-01   3         2         16        19        0         5         3
# Level-set misfit: 0.10571290606920238
# Level-set reinitialisation: 0.0025970144963309346
# Level-set gradient: 0.010791012442392861
# Level-set interface: 0.030833815240610865
#   3     1.271463e-01   2.610953e-01   6.363015e-02   1.037256e+00   4         3         27        25        0         8         0
# Level-set misfit: 0.07672820384440117
# Level-set reinitialisation: 0.0036062468965678903
# Level-set gradient: 0.009234687425881807
# Level-set interface: 0.03757713416845924
#   4     1.119894e-01   3.502538e-01   8.022111e-02   1.037256e+01   5         4         41        30        0         10        1
# Level-set misfit: 0.06211032203547218
# Level-set reinitialisation: 0.004316520343648453
# Level-set gradient: 0.007763763433271
# Level-set interface: 0.03779883903895199
#   5     9.919179e-02   2.192541e-01   1.595817e-02   1.037256e+02   6         5         54        35        0         10        1
# Level-set misfit: 0.04894893296226587
# Level-set reinitialisation: 0.004236007281040898
# Level-set gradient: 0.00767817526960309
# Level-set interface: 0.038328670873736745
