from gadopt import *
from gadopt.inverse import *


def reini_steady(psi: Function, psi_grad: Function) -> ufl.algebra.Sum:
    return -psi * (1 - psi) + epsilon * sqrt(dot(psi_grad, psi_grad))


def interface_penalty(psi: Function) -> ufl.algebra.Sum:
    return 0.25 - (max_value(min_value(psi, 1.0), 0.0) - 0.5) ** 2


def callback() -> None:
    psi_chk = psi.block_variable.checkpoint
    psi_grad_chk = psi_grad.block_variable.checkpoint

    misfit = assemble((psi_chk - psi_obs) ** 2 * dx) / psi_misfit_scale
    reini = assemble(reini_steady(psi_chk, psi_grad_chk) ** 2 * dx) / psi_reini_scale
    gradient = assemble(dot(psi_grad_chk, psi_grad_chk) * dx) / psi_grad_scale
    interface = assemble(interface_penalty(psi_chk) * dx) / psi_interface_scale

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

with CheckpointFile("forward_checkpoint.h5", "r") as forward_checkpoint:
    mesh = forward_checkpoint.load_mesh("firedrake_default")
    stokes_obs = forward_checkpoint.load_function(mesh, "Stokes")
    psi_obs = forward_checkpoint.load_function(mesh, "Level set")
    psi_obs.rename("Level set (observation)")

mesh.cartesian = True
boundary = get_boundary_ids(mesh)

R = FunctionSpace(mesh, "R", 0)
C = FunctionSpace(mesh, "Q", 1)

stokes = Function(stokes_obs, name="Stokes")
stokes.subfunctions[0].rename("Velocity")
stokes.subfunctions[1].rename("Pressure")
u = split(stokes)[0]
psi_control = Function(C, name="Level set (control)").project(psi_obs)
psi_opt = Function(C, name="Level set (optimisation)")
psi = Function(psi_obs, name="Level set").project(psi_control)
time_step = Function(R).assign(0.8)

epsilon = interface_thickness(psi.function_space(), min_cell_edge_length=True)

RaB = material_field(
    psi, [RaB_buoyant := 0.0, RaB_dense := 1.0], interface="arithmetic"
)
approximation = BoussinesqApproximation(Ra := 0.0, RaB=RaB)

stokes_nullspace = create_stokes_nullspace(stokes.function_space())

stokes_bcs = {
    boundary.bottom: {"u": 0.0},
    boundary.top: {"u": 0.0},
    boundary.left: {"ux": 0.0},
    boundary.right: {"ux": 0.0},
}

stokes_solver = StokesSolver(
    stokes,
    approximation,
    bcs=stokes_bcs,
    solver_parameters_extra={"snes_monitor": DeleteParam},
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)

adv_kwargs = {"u": u, "timestep": time_step}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
psi_grad = level_set_solver.solution_grad

time_now = 0.0
output_file = VTKFile("output_inverse.pvd")
output_file.write(*stokes.subfunctions, psi, psi_grad, time=time_now)
optimisation_file = VTKFile("output_optimisation.pvd")

target_time = 200.0
step_count = int(target_time / float(time_step))
for _ in tape.timestepper(iter(range(step_count))):
    # level_set_solver.solve()
    level_set_solver.solve(disable_reinitialisation=True)
    level_set_solver.update_gradient()
    stokes_solver.solve()

    time_now += float(time_step)
    output_file.write(*stokes.subfunctions, psi, psi_grad, time=time_now)

# gradient_space = VectorFunctionSpace(mesh, "CG", 1)
# psi_obs_grad = Function(gradient_space, name="Level set (gradient)")

# test = TestFunction(gradient_space)
# trial = TrialFunction(gradient_space)

# bilinear_form = inner(test, trial) * dx
# ibp_element = -psi_obs * div(test) * dx
# ibp_boundary = psi_obs * dot(test, FacetNormal(mesh)) * ds
# linear_form = ibp_element + ibp_boundary

# problem = LinearVariationalProblem(bilinear_form, linear_form, psi_obs_grad)
# gradient_solver = LinearVariationalSolver(problem)
# gradient_solver.solve()

# psi_obs_bounded = max_value(min_value(psi_obs, 1.0), 0.0)
# psi_misfit_scale = assemble(psi_obs**2 * dx)
# psi_reini_scale = assemble(reini_steady(psi_obs, psi_obs_grad) ** 2 * dx)
# psi_grad_scale = assemble(dot(psi_obs_grad, psi_obs_grad) * dx)
# psi_interface_scale = assemble((0.25 - (psi_obs_bounded - 0.5) ** 2) * dx)

domain_area = assemble(1.0 * dx(domain=mesh))

psi_misfit_scale = max((psi.dat.data_ro - psi_obs.dat.data_ro) ** 2)
psi_grad_scale = np.sum(psi_grad.dat.data_ro**2, axis=1).max()
psi_reini_scale = (-0.25 + epsilon.dat.data_ro.max() * np.sqrt(psi_grad_scale)) ** 2
psi_interface_scale = 0.25

psi_misfit = assemble((psi - psi_obs) ** 2 * dx) / psi_misfit_scale
psi_reini = assemble(reini_steady(psi, psi_grad) ** 2 * dx) / psi_reini_scale
psi_grad_dot = assemble(dot(psi_grad, psi_grad) * dx) / psi_grad_scale
psi_interface = assemble(interface_penalty(psi) * dx) / psi_interface_scale
objective = psi_misfit + psi_reini + psi_grad_dot + 0.05 * psi_interface
objective /= domain_area

pause_annotation()

reduced_functional = ReducedFunctional(objective, Control(psi_control))

log(f"Reduced functional: {reduced_functional(psi_control)}")
log(f"Objective: {objective}")

perturbation = Function(psi_control, name="Level set (perturbation)")
perturbation.interpolate(interface_penalty(psi_control))
random_scale = np.random.default_rng().normal(
    1.0, 0.1, size=perturbation.dat.data_ro.shape
)
perturbation.dat.data_wo[:] *= random_scale

taylor_convergence = taylor_test(reduced_functional, psi_control, perturbation)
log(f"Taylor test: {taylor_convergence}")

psi_lb = Function(psi_control, name="Level set (lower bound").assign(0.0)
psi_ub = Function(psi_control, name="Level set (upper bound").assign(1.0)

min_prob = MinimizationProblem(reduced_functional, bounds=(psi_lb, psi_ub))

# minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 0.1
# minimisation_parameters["Step"]["Trust Region"]["Radius Shrinking Threshold"] = 0.15
# minimisation_parameters["Step"]["Trust Region"]["Radius Growing Threshold"] = 0.75
# minimisation_parameters["Step"]["Trust Region"][
#     "Radius Shrinking Rate (Negative rho)"
# ] = 0.03125
# minimisation_parameters["Step"]["Trust Region"][
#     "Radius Shrinking Rate (Positive rho)"
# ] = 0.125
# minimisation_parameters["Step"]["Trust Region"]["Radius Growing Rate"] = 5
# minimisation_parameters["Status Test"]["Iteration Limit"] = 50

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
