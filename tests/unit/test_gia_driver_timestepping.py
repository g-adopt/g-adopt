"""Driver checkpoint continuation, startup reset and prescribed-step changes."""
from pathlib import Path
from types import SimpleNamespace
import sys

import firedrake as fd
import numpy as np
import pytest

from gadopt.viscoelastic_timestepping import (
    assign_timestep, time_history_fields, time_history_metadata,
    restore_time_history_metadata,
)
import test_gia_bdf2 as base
import test_gia_direct_bdf2 as direct

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/gia_selfgrav_benchmarks'))
import selfgrav_common as benchmarks

meshes = direct.meshes
clean_tape = base.clean_tape


def field_norm(f):
    if f.function_space().ufl_element().family() == "Quadrature":
        return float(fd.assemble(fd.inner(f, f)*fd.dx(degree=4)))**.5
    return fd.norm(f)


def assert_state_equal(actual, expected):
    for a, b in zip(actual.solution.subfunctions, expected.solution.subfunctions):
        np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize('scheme', ['backward_euler', 'bdf2'])
@pytest.mark.parametrize('material_solver', ['coupled', 'eliminated'])
@pytest.mark.parametrize('storage', ['benchmark', 'giamip'])
def test_checkpoint_continues_without_spurious_startup(meshes, tmp_path, storage, material_solver, scheme):
    reference, dt = build_driver_solver(meshes, material_solver, scheme)
    for h in base.STEPS[:2]:
        assign_timestep(reference, dt, h)
        reference.solve()
    parent, mantle = meshes
    path = parent.comm.bcast(str(tmp_path/f'{storage}.h5'), root=0)
    if storage == 'benchmark':
        writer = benchmarks.RunCheckpoint(path, parent, reference, 1,
                                          {'kind': 'test'}, lambda: {'rows': []})
        writer.save(2, .21)
        with fd.CheckpointFile(path, 'r') as chk:
            loaded_parent = chk.load_mesh(name=parent.name)
            fields = {name: chk.load_function(loaded_parent, name, idx=0)
                      for name, _ in benchmarks._stored_fields(reference)}
            record = benchmarks._read_json_dataset(chk, '/gadopt_restart/0/record')
        state = SimpleNamespace(fields=fields, record=record, path=path)
        resumed, rdt = reloaded_solver(loaded_parent, material_solver, scheme)
        benchmarks.restore_checkpoint(state, resumed, loaded_parent)
    else:
        sys.path.insert(0, str(ROOT/'demos/glacial_isostatic_adjustment/giamip/exp01'))
        from forward_exp01 import Checkpoints
        def output():
            return SimpleNamespace(S0_m=fd.Function(fd.FunctionSpace(mantle, 'CG', 1)),
                                   set_reference_from_mantle=lambda: None)
        with fd.CheckpointFile(path, 'w') as chk:
            chk.save_mesh(parent)
        writer = Checkpoints(path, parent, reference, reference.layout, output())
        writer.save(0, {'step': 2, 'year': .21, 'index': 0, 'out_index': 2, 'leg': 0})
        info = writer.latest()
        with fd.CheckpointFile(path, 'r') as chk:
            loaded_parent = chk.load_mesh(name=parent.name)
        resumed, rdt = reloaded_solver(loaded_parent, material_solver, scheme)
        loaded_mantle = resumed.layout.mechanics_mesh
        loaded_output = SimpleNamespace(S0_m=fd.Function(fd.FunctionSpace(loaded_mantle, 'CG', 1)),
                                        set_reference_from_mantle=lambda: None)
        reader = Checkpoints(path, loaded_parent, resumed, resumed.layout, loaded_output)
        reader.load(info)
        reader.save(1, {"step": 2, "year": .21, "index": 1, "out_index": 2, "leg": 1})
        assert int(reader.latest()["index"]) == 1
    if resumed.time_integrator is not None:
        assert resumed.time_integrator.previous_dt == pytest.approx(base.STEPS[1])
    for (_, a), (_, b) in zip(time_history_fields(resumed), time_history_fields(reference)):
        assert field_norm(a) == pytest.approx(field_norm(b), rel=1e-12, abs=1e-14)
    for h in base.STEPS[2:]:
        assign_timestep(reference, dt, h)
        assign_timestep(resumed, rdt, h)
        reference.solve()
        resumed.solve()
    if resumed.time_integrator is not None:
        assert (float(resumed.time_integrator.alpha) != 1.) == (scheme == "bdf2")
    for a, b in zip(resumed.solution.subfunctions, reference.solution.subfunctions):
        assert fd.norm(a) == pytest.approx(fd.norm(b), rel=2e-7, abs=2e-9)


def build_driver_solver(meshes, material_solver, scheme="bdf2"):
    if material_solver == "coupled":
        return base.build(meshes, scheme=scheme)
    control = fd.Function(fd.FunctionSpace(meshes[1], "CG", 1)).assign(.1)
    solver, dt, ice = direct.build(meshes, control, direct=True, exponent="spatial", scheme=scheme,
                                   representation="lowrank", iterative=True)
    direct_load(ice)
    return solver, dt


def direct_load(ice):
    from test_gia_eliminated_lowrank import load
    load(ice, 0)


def reloaded_solver(parent, material_solver, scheme):
    from validate_selfgrav_annulus import curve_mesh
    parent.cartesian = False
    mantle = curve_mesh(fd.Submesh(parent, 2, 101))
    mantle.cartesian = False
    return build_driver_solver((parent, mantle), material_solver, scheme)


@pytest.mark.parametrize("material_solver", ["coupled", "eliminated"])
def test_elastic_diagnostic_leaves_no_bdf_history(meshes, material_solver):
    solver, dt = build_driver_solver(meshes, material_solver)
    fresh, fdt = build_driver_solver(meshes, material_solver)
    # A real solver, including the diagnostic solve and its accepted history.
    steps = [(0., .08*benchmarks.T_BAR_YR, .08*benchmarks.T_BAR_YR, 1, True)]
    benchmarks.march(solver, dt, steps, elastic=True,
                     on_step=lambda *args: None, on_epoch=lambda *args: None)
    assign_timestep(fresh, fdt, .08)
    fresh.solve()
    assert float(solver.time_integrator.alpha) == 1.
    assert_state_equal(solver, fresh)


def test_large_driver_step_change_uses_explicit_be_restart(meshes):
    solver, dt = base.build(meshes)
    solver.solve()
    assert assign_timestep(solver, dt, .4)
    solver.time_integrator.prepare()
    assert float(solver.time_integrator.alpha) == 1.
    solver.solve()
    assert not assign_timestep(solver, dt, .4)
    solver.time_integrator.prepare()
    assert float(solver.time_integrator.alpha) == 1.5


def test_scheme_mismatch_and_invalid_history_metadata(meshes):
    solver, _ = base.build(meshes)
    with pytest.raises(ValueError, match='time order'):
        restore_time_history_metadata(solver, {})
    metadata = time_history_metadata(solver)
    with pytest.raises(ValueError, match='positive'):
        restore_time_history_metadata(solver, {**metadata, 'time_previous_dt': float('nan')})


@pytest.mark.parametrize('driver', ['spada', 'martinec'])
def test_driver_material_selection_and_restart_settings(driver):
    from importlib import import_module
    from gadopt.viscoelastic_timestepping import material_solver_options
    module = import_module(driver)
    args = module.parse_args(['--time_scheme', 'bdf2', '--material_solver', 'eliminated',
                              '--material_quad_degree', '6'])
    space, equation = material_solver_options(args.material_solver, args.material_quad_degree, 2)
    assert space['condense_internal_variables']
    assert space['internal_variable_family'] == 'Quadrature'
    assert space['internal_variable_degree'] == equation['quad_degree'] == 6
    assert equation['constitutive_method'] == 'direct'
    if driver == 'spada':
        settings = module.run_settings(args)
        upgraded = module.checkpoint_run_settings({'run': settings})
    else:
        scenario = "T0" if args.case == "B" else "T1"
        ladder, _ = module.time_schedule(args, module.EPOCHS_KYR[scenario])
        settings = module.run_settings(args, 1000., ladder)
        upgraded = module.checkpoint_run_settings(settings)
    assert upgraded == settings
    assert settings['material_solver'] == 'eliminated'
    assert settings['material_quad_degree'] == 6


def test_material_restart_cannot_change_formulation_or_quadrature(meshes):
    solver, _ = build_driver_solver(meshes, 'eliminated')
    metadata = time_history_metadata(solver)
    with pytest.raises(ValueError, match='material solver'):
        restore_time_history_metadata(solver, {k: v for k, v in metadata.items()
                                               if not k.startswith('material_')})
    with pytest.raises(ValueError, match='material solver'):
        restore_time_history_metadata(solver, {**metadata, 'material_quad_degree': 6.})
    coupled, _ = build_driver_solver(meshes, 'coupled')
    with pytest.raises(ValueError, match='material solver'):
        restore_time_history_metadata(coupled, metadata)


def test_martinec_e_eliminated_history_replay_and_adjoint(meshes):
    """Run the actual E history/control loop on the small two-mesh geometry."""
    import martinec_e as case_e
    import test_gia_gravity as gravity
    from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                                  get_working_tape, pause_annotation, stop_annotating)
    from test_gravity_adjoint import assert_taylor_with_guards
    parent, mantle = meshes
    with stop_annotating():
        space = fd.FunctionSpace(mantle, 'CG', 2)
        theta = fd.atan2(fd.SpatialCoordinate(mantle)[1], fd.SpatialCoordinate(mantle)[0])
        target = fd.Function(space).interpolate(.005*fd.cos(theta))
        correction = fd.Function(space).assign(.001)
    common = SimpleNamespace(T_BAR_YR=1., D_SCALE=case_e.CONTROL_SCALE_M,
                             SURF_RE=gravity.CURVE_RE)
    args = SimpleNamespace(smoke=False, time_step_yr=.08, time_scheme='bdf2',
                           material_solver='eliminated', material_quad_degree=4)
    case = SimpleNamespace(terminal_time_kyr=.00032)

    def build(case, parent, mantle, dt, clock, bulk, *, initial_sea_level,
              time_scheme, material_solver, material_quad_degree):
        from gadopt.viscoelastic_timestepping import material_solver_options
        _, equation = material_solver_options(material_solver, material_quad_degree, 1)
        control = fd.Function(fd.FunctionSpace(mantle, 'CG', 1)).assign(.1)
        solver, _, ice = direct.build((parent, mantle), control, direct=True,
            exponent='spatial', representation='lowrank', iterative=True,
            constitutive_method=equation['constitutive_method'], scheme=time_scheme,
            timestep=dt, initial_sea_level=initial_sea_level, moving_coastline=True,
            options_extra={'snes_atol': 1e-11})
        direct_load(ice)
        return solver, solver.layout, {}

    driver = SimpleNamespace(common=common, T1_DT_YR=.08, build_solver=build,
                             log=lambda *args: None)
    continue_annotation()
    control = Control(correction)
    solver, _, _, _, objective = case_e.history(
        args, driver, parent, mantle, case, target, correction, 1000.,
        tape=get_working_tape())
    reduced = ReducedFunctional(objective, control)
    pause_annotation()
    with stop_annotating():
        direction = fd.Function(space).interpolate(.0001*(1+.2*fd.cos(theta)))
        perturbed = fd.Function(space).assign(correction+direction)
        fresh, *_ = case_e.history(args, driver, parent, mantle, case, target, perturbed, 1000.)
        dss = fresh.sea_level_measure()(gravity.CURVE_RE)
        fresh_objective = float(fd.assemble(.5*(fresh.sea_level()-target)**2*dss))
    assert float(reduced(perturbed)) == pytest.approx(fresh_objective, rel=2e-7, abs=1e-13)
    assert_taylor_with_guards(reduced, correction, direction, objective, min_rate=1.9)


def test_martinec_e_topography_only_checkpoint(meshes, tmp_path):
    import martinec_e as case_e
    parent, mantle = meshes
    initial = fd.Function(fd.FunctionSpace(mantle, 'CG', 2)).assign(.003)
    path = parent.comm.bcast(str(tmp_path/'initial_topography.h5'), root=0)
    case_e.save_topography(path, initial, parent, {'kind': 'test'}, {}, benchmarks)
    with fd.CheckpointFile(path, 'r') as chk:
        mesh = chk.load_mesh(name=parent.name)
        info = benchmarks._read_json_dataset(chk, '/gadopt_restart/mesh_info')
        assert info['fields'] == ['initial_SL']
        stored = chk.load_function(mesh, 'initial_SL', idx=0)
    assert fd.norm(stored) == pytest.approx(fd.norm(initial), rel=1e-12)
