"""Initial-topography inversion for Martinec E (paper equations 32--34).

Called by martinec.py --case E. Imports of Firedrake adjoint and ROL are lazy:
the Picard route and the existing B/C/D cases do not need ROL.
"""
import copy
import gc
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

# Scale of the surface-only correction control, not a physical constant.
CONTROL_SCALE_M = 1000.0
# Kept separate from B/C/D smoke truncation: E must reach its terminal time.
SMOKE_STEPS = 6


def add_arguments(parser):
    parser.add_argument("--topography_method", choices=("picard", "rol"), default="picard")
    parser.add_argument("--topography_tolerance_m", type=float, default=0.01,
                        help="E: area-normalised terminal topography RMS tolerance (m)")
    parser.add_argument("--topography_max_iterations", type=int, default=6,
                        help="E: maximum full histories (Picard) or ROL iterations")
    parser.add_argument("--topography_relaxation", type=float, default=1.0,
                        help="E Picard update factor; 1 is paper equation 34")
    parser.add_argument("--topography_one_pass", action="store_true",
                        help="E Picard: run one history, save the next guess, then exit")
    parser.add_argument("--topography_restart", metavar="FILE",
                        help="E: start a NEW history from checkpoint_E_next.h5, same ranks")
    parser.add_argument("--adjoint_checkpoints", type=int, default=5,
                        help="E ROL: RAM snapshots for Revolve (recomputation between snapshots)")
    parser.add_argument("--check_topography_gradient", action="store_true",
                        help="E ROL: compare adjoint derivative with fresh central differences")


def configuration(args, bulk_shear_ratio):
    return {"case": "E", "bulk_shear_ratio": bulk_shear_ratio, "smoke": args.smoke,
            "method": args.topography_method, "tolerance_m": args.topography_tolerance_m,
            "relaxation": args.topography_relaxation,
            "norm": "area-normalised whole-surface RMS", "mask_slope": "fixed terminal target basin",
            "initial_guess": "terminal target basin", "control_scale_m": CONTROL_SCALE_M}


def surface_control(mantle, target_SL, common):
    """A CG2 correction on Re only; no unconstrained interior control DOFs."""
    import firedrake as fd
    surface = common.curve_mesh(fd.Submesh(mantle, 2, common.SURF_RE), name="E_surface")
    surface.cartesian = False
    space = fd.FunctionSpace(surface, "CG", 2)
    control = fd.Function(space, name="initial_topography_correction")
    return control


def lift_control(control, target_SL):
    """Put the surface correction on mantle boundary DOFs; other DOFs are 0."""
    import firedrake as fd
    return fd.Function(target_SL.function_space()).interpolate(control, allow_missing_dofs=True)


def surface_error(solver, target_SL, common):
    """Nondimensional target minus modelled terminal topography (=-SL)."""
    return solver.sea_level() - target_SL


def error_rms(solver, target_SL, common):
    import firedrake as fd
    dss = solver.sea_level_measure()(common.SURF_RE)
    area = float(fd.assemble(fd.Constant(1) * dss))
    error = surface_error(solver, target_SL, common)
    squared = float(fd.assemble(error**2 * dss))
    if not math.isfinite(area) or area <= 0 or not math.isfinite(squared) or squared < 0:
        raise FloatingPointError(f"Invalid E surface error integral: area={area}, squared={squared}")
    return common.D_SCALE * math.sqrt(squared / area)


def project_error(solver, target_SL, common):
    """Surface L2 projection using the same mixed-mesh measure as the solve.

    FacetNormal and the parent's potential trace cannot be interpolated as a
    cell expression. InteriorBC removes the null rows of this trace mass solve.
    """
    import firedrake as fd
    from gadopt import InteriorBC
    space = target_SL.function_space()
    trial, test = fd.TrialFunction(space), fd.TestFunction(space)
    dss = solver.sea_level_measure()(common.SURF_RE)
    # The left hand side contains mantle coefficients only; its measure needs
    # no parent intersection. The right hand side also reads the parent psi.
    ds = fd.ds(domain=space.mesh(), degree=solver.form.quad_degree)(common.SURF_RE)
    value = fd.Function(space, name="terminal_topography_correction")
    fd.solve(trial * test * ds == surface_error(solver, target_SL, common) * test * dss,
             value, bcs=InteriorBC(space, 0, common.SURF_RE),
             solver_parameters={"snes_type": "ksponly", "ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-10})
    return value


def history(args, driver, parent, mantle, case, target_SL, frozen_slope,
            correction, bulk_shear_ratio, directory=None, restart=None,
            record_base=None, tape=None):
    """One full zero-state-to-15-kyr history, or a within-history restart.

    When tape is supplied, only the control lift, solves, time assignments and
    objective are annotated. Diagnostics and file writes stay off the tape.
    """
    import firedrake as fd
    from pyadjoint import stop_annotating
    common = driver.common
    with stop_annotating():
        total_steps = SMOKE_STEPS if args.smoke else int(round(case.terminal_time_kyr * 1000 / driver.T1_DT_YR))
        step_yr = case.terminal_time_kyr * 1000 / total_steps
        dt = fd.Constant(step_yr / common.T_BAR_YR)
        # Firedrake Constants are not overloaded by its adjoint interface.
        # A Real Function records time assignments and restores the correct
        # load during tape replay and Revolve recomputation.
        clock = fd.Function(fd.FunctionSpace(mantle, "R", 0), name="time_kyr").assign(0.0)
        initial_SL = fd.Function(target_SL.function_space(), name="SL_init").assign(target_SL)
        solver, layout, pieces = driver.build_solver(
            case, parent, mantle, dt, clock, bulk_shear_ratio,
            initial_sea_level=initial_SL, mask_slope=frozen_slope)
        solver.benchmark_checkpoint_fields = (("initial_SL", initial_SL),)
        rows = [] if restart is None else list(restart.record["rows"])
        start_step = 0 if restart is None else restart.step
        if restart is not None and abs(restart.t_kyr - start_step * step_yr / 1000) > 1e-9:
            raise ValueError("E within-history restart was written on a different time grid")
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            parameter_log = driver.ParameterLog(str(directory / "params_E.log"), parent)
            parameter_log.log_str(" ".join(driver.ROW_KEYS))
            for row in rows:
                parameter_log.log_str(" ".join(str(row[key]) for key in driver.ROW_KEYS))
            def record():
                return {**record_base, "rows": rows, "wall_s": time.time() - record_base["tic"],
                        "cumulative_wall_s": record_base["cumulative_wall_s"] + time.time() - record_base["tic"]}
            checkpoint = common.RunCheckpoint(
                restart.path if restart is not None else str(directory / "checkpoint_E.h5"),
                parent, solver, args.checkpoint_every, record_base["mesh"], record, restart=restart)
            vtk = (fd.VTKFile(str(directory / "martinec_E_mechanics.pvd")),
                   fd.VTKFile(str(directory / "martinec_E_potential.pvd"))) if args.write_output else None
        else:
            checkpoint = vtk = parameter_log = None
    # This interpolation belongs to the tape: the control affects the initial
    # ocean and grounding masks as well as the terminal mismatch.
    initial_SL.interpolate(target_SL - lift_control(correction, target_SL) * CONTROL_SCALE_M / common.D_SCALE)
    if restart is not None:
        with stop_annotating():
            common.restore_checkpoint(restart, solver, parent)
    if directory is not None and restart is None:
        with stop_annotating():
            row = driver.state_row(solver, layout, case, pieces, 0.0, 0.0, 0, 0.0)
            rows.append(row)
            parameter_log.log_str(" ".join(str(row[key]) for key in driver.ROW_KEYS))
    steps = range(start_step + 1, total_steps + 1)
    if tape is not None:
        steps = tape.timestepper(iter(steps))
    objective = None
    for step in steps:
        t_kyr = step * step_yr / 1000
        clock.assign(t_kyr)
        tick = time.time()
        solver.solve()
        wall = time.time() - tick
        if tape is not None and step == total_steps:
            dss = solver.sea_level_measure()(common.SURF_RE)
            with stop_annotating():
                area = float(fd.assemble(fd.Constant(1) * dss))
            objective = fd.assemble(0.5 * (common.D_SCALE / CONTROL_SCALE_M)**2
                                    * surface_error(solver, target_SL, common)**2 / area * dss)
        with stop_annotating():
            if directory is None:
                driver.log(f"  E {'taped' if tape is not None else 'evaluation'} history "
                           f"step {step:4d}, t {t_kyr:7.3f} kyr, wall {wall:.1f} s")
            if directory is not None:
                row = driver.state_row(solver, layout, case, pieces, t_kyr, step_yr, step, wall)
                rows.append(row)
                parameter_log.log_str(" ".join(str(row[key]) for key in driver.ROW_KEYS))
                driver.log(f"  E step {step:4d}  t {t_kyr:7.3f} kyr  newton {row['newton']} "
                           f"outer(last) {row['outer']}  h_UF {row['h_UF_m']:.6f} m  wall {wall:.1f} s")
                if vtk is not None:
                    vtk[0].write(solver.displacement)
                    vtk[1].write(solver.potential)
                checkpoint(step, t_kyr)
    with stop_annotating():
        if directory is not None:
            parameter_log.close()
            if parent.comm.rank == 0:
                import numpy as np
                np.savez(directory / "martinec-E-timeseries.npz",
                         **{key: np.array([row[key] for row in rows]) for key in driver.ROW_KEYS})
        rms = error_rms(solver, target_SL, common)
    return solver, correction, rows, rms, objective


def gradient_check(reduced, correction, fresh_objective, output):
    """Compare against independent fresh solves; tape self-consistency is insufficient."""
    import firedrake as fd
    from firedrake.adjoint import stop_annotating
    with stop_annotating():
        x = correction.copy(deepcopy=True)
        direction = fd.Function(x.function_space()).assign(1.0)
    base = float(reduced(x))
    gradient = reduced.derivative(apply_riesz=False)
    with gradient.dat.vec_ro as g, direction.dat.vec_ro as h:
        adjoint = g.dot(h)
    epsilon = 1e-3  # one metre in this control scale
    with stop_annotating():
        plus = fd.Function(x.function_space()).assign(x + epsilon * direction)
        minus = fd.Function(x.function_space()).assign(x - epsilon * direction)
        fresh_base = fresh_objective(x)
        difference = (fresh_objective(plus) - fresh_objective(minus)) / (2 * epsilon)
    relative = abs(adjoint - difference) / max(abs(adjoint), abs(difference), 1e-14)
    report = {"adjoint_directional_derivative": float(adjoint), "fresh_central_difference": difference,
              "relative_difference": float(relative), "epsilon": epsilon,
              "taped_objective": base, "fresh_objective": fresh_base}
    output(report)
    if (not all(math.isfinite(v) for v in (base, fresh_base, adjoint, difference, relative))
            or abs(base - fresh_base) > 1e-8 + 1e-4 * abs(fresh_base)
            or relative > 0.01 or abs(adjoint) < 1e-14):
        raise RuntimeError(f"E adjoint/fresh-solve gradient check failed: {report}")
    reduced(x)
    return report


def fit_rol(args, driver, parent, mantle, case, target_SL, frozen_slope,
            correction, bulk_shear_ratio, root):
    from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                                  get_working_tape, pause_annotation, stop_annotating)
    import firedrake as fd
    from checkpoint_schedules import Revolve
    # Import inverse before clearing the tape: this module starts annotation.
    from gadopt.inverse import LinMoreOptimiser, MinimizationProblem, minimisation_parameters
    tape = get_working_tape()
    pause_annotation()
    tape.clear_tape()
    total_steps = SMOKE_STEPS if args.smoke else int(round(case.terminal_time_kyr * 1000 / driver.T1_DT_YR))
    tape.enable_checkpointing(Revolve(total_steps, min(args.adjoint_checkpoints, total_steps - 1)))
    with stop_annotating():
        riesz = fd.RieszMap(correction.function_space(), "L2", constant_jacobian=True,
                            solver_parameters={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-10})
    continue_annotation()
    control = Control(correction, riesz_map=riesz)
    try:
        _, _, _, _, objective = history(args, driver, parent, mantle, case, target_SL, frozen_slope,
                                        correction, bulk_shear_ratio, tape=tape)
    finally:
        pause_annotation()
    evaluations = []
    def evaluation_callback(value, controls):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise FloatingPointError(f"Invalid E objective: {value}")
        evaluations.append({"evaluation": len(evaluations) + 1, "objective": float(value),
                            "rms_m": CONTROL_SCALE_M * math.sqrt(max(0.0, 2 * float(value)))})
        driver.common.write_json(str(root / "topography_rol_evaluations.json"), evaluations)
    reduced = ReducedFunctional(objective, control, eval_cb_post=evaluation_callback)
    derivative_report = None
    if args.check_topography_gradient:
        def fresh(value):
            solver, _, _, rms, _ = history(args, driver, parent, mantle, case, target_SL, frozen_slope,
                                          value, bulk_shear_ratio)
            del solver
            gc.collect()
            return 0.5 * (rms / CONTROL_SCALE_M)**2
        derivative_report = gradient_check(reduced, correction, fresh,
            lambda r: driver.common.write_json(str(root / "topography_gradient.json"), r))
    parameters = copy.deepcopy(minimisation_parameters)
    parameters["Status Test"].update({"Iteration Limit": args.topography_max_iterations,
                                     "Gradient Tolerance": 1e-12, "Step Tolerance": 1e-14})
    # PyROL's LinMoreAlgorithm.run requires a Bounds object. Infinite bounds
    # produce inf*0 in Firedrake vector operations, so use finite bounds far
    # outside any topography correction (1e23 m in the control scale).
    optimiser = LinMoreOptimiser(MinimizationProblem(reduced, bounds=(-1e20, 1e20)), parameters)
    import ROL
    goal = 0.5 * (args.topography_tolerance_m / CONTROL_SCALE_M)**2
    class TopographyStatusTest(ROL.StatusTest):
        def check(self, status):
            optimiser.iteration += 1
            if float(status.value) < goal:
                return False
            return super().check(status)
    optimiser.rol_algorithm.setStatusTest(TopographyStatusTest(optimiser.rol_parameters), False)
    optimiser.run()
    with stop_annotating():
        fitted = optimiser.rol_solver.rolvector.dat[0].copy(deepcopy=True)
    pause_annotation()
    tape.clear_tape()
    return fitted, evaluations, derivative_report


def save_topography(path, initial_SL, parent, mesh_info, record, common):
    """A restartable guess needs only its mesh/topography, not 17 GB of state."""
    holder = SimpleNamespace(solution=SimpleNamespace(subfunctions=()), internal_variables=None,
                             benchmark_checkpoint_fields=(("initial_SL", initial_SL),))
    checkpoint = common.RunCheckpoint(str(path), parent, holder, 1, mesh_info, lambda: record)
    checkpoint.save(0, 0.0)


def run(args, driver):
    import firedrake as fd
    from pyadjoint import stop_annotating
    common = driver.common
    if (not math.isfinite(args.topography_tolerance_m) or args.topography_tolerance_m <= 0
            or args.topography_max_iterations < 1):
        raise ValueError("E tolerance and maximum iterations must be positive")
    if not 0 < args.topography_relaxation <= 1 or args.adjoint_checkpoints < 1:
        raise ValueError("E relaxation must be in (0, 1]; adjoint snapshots must be positive")
    if args.restart and args.topography_restart:
        raise ValueError("Use --restart within a history OR --topography_restart between histories")
    if args.topography_method == "rol" and (args.restart or args.topography_one_pass):
        raise ValueError("ROL restarts from a topography warm start, not a partially taped history")
    if args.topography_method == "picard" and args.check_topography_gradient:
        raise ValueError("--check_topography_gradient requires --topography_method rol")
    root = Path(args.output_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    bulk = args.bulk_shear_ratio if args.bulk_shear_ratio is not None else driver.BULK_SHEAR_RATIO["E"]
    config = configuration(args, bulk)
    mesh_kind = "smoke" if args.smoke else "martinec"
    restart_path = args.restart or args.topography_restart
    with stop_annotating():
        if restart_path:
            parent, mantle, mesh_info, state = common.load_checkpoint(restart_path, mesh_kind)
            written = state.record["configuration"]
            if written != config:
                raise ValueError(f"E restart settings differ: stored {written}, requested {config}")
            if args.restart and state.record["stage"] != "history":
                raise ValueError("This is a between-history checkpoint; use --topography_restart")
            if args.topography_restart and state.record["stage"] != "next_topography":
                raise ValueError("This is a within-history checkpoint; use --restart")
            pass_number = state.record["pass"]
            passes = state.record["passes"]
            previous_wall_s = state.record.get("cumulative_wall_s", 0.0)
        else:
            parent, mantle, mesh_info = common.build_meshes(mesh_kind, str(root), "martinec_E")
            state, pass_number, passes, previous_wall_s = None, 1, [], 0.0
        case = driver.read_case("E")
        target_SL = driver.basin_sea_level(mantle, case)
        # Mask width is numerical regularisation. Keep it fixed for both fits:
        # surface_slope deliberately has no adjoint dependency on topography.
        frozen_slope = driver.surface_slope(target_SL)
        correction = surface_control(mantle, target_SL, common)
        if state is not None:
            initial_SL = fd.Function(target_SL.function_space()).interpolate(state.fields["initial_SL"])
            correction.interpolate((target_SL - initial_SL) * common.D_SCALE / CONTROL_SCALE_M)
    if args.topography_method == "picard" and pass_number > args.topography_max_iterations:
        raise ValueError(f"Restart requests pass {pass_number}; increase --topography_max_iterations "
                         f"above {args.topography_max_iterations}")
    driver.log(driver.describe(case))
    driver.log(f"  E topography method {args.topography_method}, tolerance {args.topography_tolerance_m:g} m RMS, "
               f"fixed target-basin mask slope; start pass {pass_number}")
    evaluations = gradient_report = None
    if args.topography_method == "rol":
        correction, evaluations, gradient_report = fit_rol(args, driver, parent, mantle, case, target_SL,
                                                          frozen_slope, correction, bulk, root)
        pass_number = 1
    final_solver = None
    for number in range(pass_number, args.topography_max_iterations + 1):
        directory = root / f"topography_{number:02d}"
        base = {"configuration": config, "stage": "history", "pass": number,
                "passes": passes, "mesh": mesh_info, "tic": tic,
                "cumulative_wall_s": previous_wall_s}
        within_restart = state if args.restart and number == pass_number else None
        with stop_annotating():
            solver, _, rows, rms, _ = history(args, driver, parent, mantle, case, target_SL, frozen_slope,
                                             correction, bulk, directory, within_restart, base)
            passes.append({"pass": number, "terminal_topography_rms_m": rms,
                           "steps": rows[-1]["step"], "h_UF_m_last": rows[-1]["h_UF_m"],
                           "ocean_area_last": rows[-1]["ocean_area"], "directory": directory.name})
            converged = rms < args.topography_tolerance_m
            driver.log(f"  E topography pass {number}: terminal RMS {rms:.8g} m; converged={converged}")
            # Retain latest history as a diagnostic, clearly labelled by the
            # convergence flag. The acceptance test requires a converged fit.
            if parent.comm.rank == 0:
                shutil.copyfile(directory / "martinec-E-timeseries.npz", root / "martinec-E-timeseries.npz")
                shutil.copyfile(directory / "params_E.log", root / "params_E.log")
            summary = {"benchmark": "Martinec et al. (2018)", "case": "E", "smoke": args.smoke,
                       "bulk_shear_ratio": bulk, "dtn_degree": common.DTN_DEGREE,
                       "ranks": parent.comm.size, "mesh": mesh_info, "steps": rows[-1]["step"],
                       "epochs_kyr": sorted(set(row["t_kyr"] for row in rows)),
                       "terminal_time_kyr": case.terminal_time_kyr, "giamip_version": driver.giamip.__version__,
                       "time_step_yr": case.terminal_time_kyr * 1000 / rows[-1]["step"],
                       "reference_time_scale_yr": common.T_BAR_YR,
                       "timeseries_file": "martinec-E-timeseries.npz", "profile_files": [],
                       "h_UF_m_last": rows[-1]["h_UF_m"], "wall_s": previous_wall_s + time.time() - tic,
                       "topography": {**config, "converged": converged, "passes": passes,
                                      "terminal_rms_m": rms, "rol_evaluations": evaluations,
                                      "gradient_check": gradient_report}}
            analytic_mass = (driver.cap_thickness_integral(
                common.RE, float(case.cap["height_m"]) / common.D_SCALE,
                math.radians(float(case.cap["radius_deg"])))
                * common.D_SCALE**3 * float(case.ice_density))
            ice_mass = rows[-1]["ice_mass_grounded_kg"] + rows[-1]["ice_mass_floating_kg"]
            summary["cap_volume_error"] = abs(ice_mass / analytic_mass - 1)
            common.write_json(str(root / "summary_E.json"), summary)
            fitted_record = {**base, "stage": "next_topography", "pass": number,
                             "passes": passes[:-1], "cumulative_wall_s": summary["wall_s"], "rows": []}
            save_topography(root / "initial_topography_E.h5", solver.sea_level_parameters.SL_init,
                            parent, mesh_info, fitted_record, common)
            if args.write_output:
                initial_topography = fd.Function(target_SL.function_space(), name="initial_topography_m").interpolate(-solver.sea_level_parameters.SL_init * common.D_SCALE)
                error = project_error(solver, target_SL, common)
                terminal_topography = fd.Function(target_SL.function_space(), name="terminal_topography_m").interpolate((-target_SL - error) * common.D_SCALE)
                fd.VTKFile(str(directory / "topography.pvd")).write(initial_topography, terminal_topography)
            if converged or args.topography_method == "rol":
                final_solver = solver
                break
            update = project_error(solver, target_SL, common)
            update_on_surface = fd.Function(correction.function_space()).interpolate(update)
            correction.assign(correction + args.topography_relaxation * common.D_SCALE / CONTROL_SCALE_M * update_on_surface)
            solver.sea_level_parameters.SL_init.interpolate(target_SL - lift_control(correction, target_SL) * CONTROL_SCALE_M / common.D_SCALE)
            next_record = {**base, "stage": "next_topography", "pass": number + 1, "passes": passes,
                           "cumulative_wall_s": previous_wall_s + time.time() - tic, "rows": []}
            save_topography(root / "checkpoint_E_next.h5", solver.sea_level_parameters.SL_init,
                            parent, mesh_info, next_record, common)
            if args.topography_one_pass:
                driver.log("  E next initial topography saved; continue with --topography_restart checkpoint_E_next.h5")
                return
            # Do not retain a whole previous solver, including its assembled
            # preconditioners, while the next production history runs.
            del solver, update, update_on_surface
            gc.collect()
    # ROL always performs an independent final history; the fit is accepted
    # on physical RMS, not merely on the optimiser's stopping criterion.
    if not summary["topography"]["converged"] and not args.smoke:
        raise RuntimeError(f"E topography did not converge: RMS {summary['topography']['terminal_rms_m']:.6g} m. "
                           "Latest diagnostic history and initial_topography_E.h5 are saved.")
    return final_solver
