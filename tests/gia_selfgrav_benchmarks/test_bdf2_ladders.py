"""Actual benchmark schedules: endpoint preservation and no BDF2 restarts."""
from types import SimpleNamespace

import firedrake as fd
import numpy as np
import pytest

import martinec
import spada
import selfgrav_common as common
from gadopt.viscoelastic_timestepping import BDF2History, assign_timestep


def nodes(segments):
    return np.array([segments[0][0]] + [a+k*h for a, b, h, n, epoch in segments
                                      for k in range(1, n+1)])


def assert_safe(original, refined):
    old = nodes(original)
    new = nodes(refined)
    steps = np.array([h for a, b, h, n, epoch in refined for _ in range(n)])
    assert np.all(steps > 0)
    assert np.all(steps[1:] <= 2*steps[:-1]*(1+1e-12))
    for point in old:
        assert np.min(abs(new-point)) < 1e-8
    assert [(b, epoch) for a, b, h, n, epoch in original if epoch] == [
        (b, epoch) for a, b, h, n, epoch in refined if epoch]
    for a, b, h, n, epoch in refined:
        assert a+n*h == pytest.approx(b, abs=1e-8)


@pytest.mark.parametrize('scale', [.5, 1., 1.3, 2., 5., 10., 100.])
def test_spada_scaled_safe_ladder_preserves_every_old_endpoint(scale):
    arguments = ['--time_scheme', 'bdf2', '--timestep_scale', str(scale)]
    original = spada.time_segments(spada.parse_args(arguments), spada.EPOCHS_KYR)
    args = spada.parse_args(arguments+['--bdf2_safe_ladder'])
    refined = spada.time_segments(args, spada.EPOCHS_KYR)
    assert_safe(original, refined)
    if scale == 1:
        assert len(nodes(original))-1 == 138
        assert len(nodes(refined))-1 == 140
    assert spada.checkpoint_run_settings({'run': spada.run_settings(args)}) == spada.run_settings(args)


def test_spada_extra_output_marks_do_not_replan_safe_steps():
    args = spada.parse_args(['--time_scheme', 'bdf2', '--bdf2_safe_ladder'])
    planned = spada.time_segments(args, spada.output_epochs(args))
    # The 120-year transition point is newly inserted by the safe planner.
    args = spada.parse_args(['--time_scheme', 'bdf2', '--bdf2_safe_ladder',
                             '--extra_epochs_kyr', '.12'])
    marked = spada.time_segments(args, spada.output_epochs(args))
    np.testing.assert_allclose(nodes(marked), nodes(planned), rtol=0, atol=1e-9)
    assert any(epoch and b == pytest.approx(120.) for a, b, h, n, epoch in marked)
    with pytest.raises(SystemExit):
        spada.parse_args(['--bdf2_safe_ladder', '--extra_epochs_kyr', '.121'])


@pytest.mark.parametrize('ladder', [
    ['100:10', '1000:50', '10000:100'],
    ['100:17', '1100:63', '20000:131'],
    ['100:1', '10000:1000'],
])
def test_martinec_graded_safe_ladder(ladder):
    arguments = ['--case', 'B', '--time_scheme', 'bdf2', '--time_step_ladder', *ladder]
    epochs = martinec.EPOCHS_KYR['T0']
    _, original = martinec.time_schedule(martinec.parse_args(arguments), epochs)
    args = martinec.parse_args(arguments+['--bdf2_safe_ladder'])
    effective, refined = martinec.time_schedule(args, epochs)
    assert_safe(original, refined)
    settings = martinec.run_settings(args, 1000., effective)
    assert settings['bdf2_safe_ladder']
    assert martinec.checkpoint_run_settings(settings) == settings


@pytest.mark.parametrize('case', ['B', 'C', 'D'])
def test_martinec_uniform_schedule_is_unchanged(case):
    args = martinec.parse_args(['--case', case])
    epochs = martinec.EPOCHS_KYR['T0' if case == 'B' else 'T1']
    original = martinec.time_schedule(args, epochs)
    args.bdf2_safe_ladder = True
    assert martinec.time_schedule(args, epochs) == original


def test_safe_benchmark_steps_pass_actual_bdf2_guard_without_restarts():
    args = spada.parse_args(['--bdf2_safe_ladder', '--time_scheme', 'bdf2'])
    segments = spada.time_segments(args, spada.EPOCHS_KYR)
    mesh = fd.UnitIntervalMesh(1)
    value = fd.Function(fd.FunctionSpace(mesh, 'DG', 0))
    dt = fd.Function(fd.FunctionSpace(mesh, 'R', 0))
    history = BDF2History([value], dt)
    solver = SimpleNamespace(time_integrator=history, restart_time_integration=history.restart)
    count = 0
    for a, b, h, n, epoch in segments:
        for _ in range(n):
            assert not assign_timestep(solver, dt, h/common.T_BAR_YR)
            history.prepare()
            assert (float(history.alpha) == 1.) == (count == 0)
            history.accept([value])
            count += 1
    assert count == 140


def test_safe_ladder_is_part_of_restart_configuration():
    args = spada.parse_args(['--time_scheme', 'bdf2'])
    legacy = spada.run_settings(args)
    args.bdf2_safe_ladder = True
    safe = spada.run_settings(args)
    assert safe != legacy
    old_record = {key: value for key, value in legacy.items() if key != 'bdf2_safe_ladder'}
    assert spada.checkpoint_run_settings({'run': old_record}) == legacy
