"""Exercise replay and Taylor checks with a real Firedrake tape, without ROL."""
import argparse

import pytest

from martinec_e import add_arguments


def test_check_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--check_topography_replay', '--check_topography_taylor'])
    assert args.check_topography_replay and args.check_topography_taylor
    assert args.adjoint_checkpointing == 'revolve'
    args = parser.parse_args(['--adjoint_checkpointing', 'single_disk', '--adjoint_checkpoint_dir', '/tmp/checkpoints'])
    assert args.adjoint_checkpoint_dir == '/tmp/checkpoints'


@pytest.fixture(params=['plain', 'revolve', 'single_memory', 'single_disk'])
def quadratic_tape(request, tmp_path):
    fd = pytest.importorskip('firedrake')
    from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                                  get_working_tape, pause_annotation, stop_annotating)
    from pyadjoint import Tape, set_working_tape
    pause_annotation()
    previous_tape = get_working_tape()
    tape = Tape()
    set_working_tape(tape)
    with stop_annotating():
        mesh = fd.UnitSquareMesh(2, 2)
        mesh.coordinates.dat.data[:] += 1.0
        x = fd.Function(fd.FunctionSpace(mesh, 'CG', 2))
        state = fd.Function(x.function_space())
    if request.param != 'plain':
        from types import SimpleNamespace
        from martinec_e import configure_adjoint_checkpointing
        args = SimpleNamespace(adjoint_checkpointing=request.param, adjoint_checkpoints=2,
                               adjoint_checkpoint_dir=str(tmp_path / 'disk') if request.param == 'single_disk' else None)
        configure_adjoint_checkpointing(args, tape, 4, mesh.comm, tmp_path)
    continue_annotation()
    control = Control(x)
    if request.param != 'plain':
        for step in tape.timestepper(iter(range(4))):
            state.assign(x)
            if step == 3:
                objective = fd.assemble(0.5 * (state - 1)**2 * fd.dx)
    else:
        objective = fd.assemble(0.5 * (x - 1)**2 * fd.dx)
    pause_annotation()
    try:
        yield fd, x, float(objective), ReducedFunctional(objective, control)
    finally:
        pause_annotation()
        tape.clear_tape()
        set_working_tape(previous_tape)


def test_replay_and_taylor(quadratic_tape):
    from martinec_e import replay_check, taylor_check
    fd, x, recorded, reduced = quadratic_tape
    reports = []
    assert replay_check(reduced, x, recorded, reports.append)['passed']
    report = taylor_check(reduced, x, reports.append)
    assert report['passed']
    assert report['minimum_convergence_rate'] == pytest.approx(2, abs=1e-7)
    assert fd.norm(x) < 1e-12  # restored after perturbations


def test_replay_rejects_wrong_recording(quadratic_tape):
    from martinec_e import replay_check
    _, x, recorded, reduced = quadratic_tape
    reports = []
    with pytest.raises(RuntimeError, match='replay check failed'):
        replay_check(reduced, x, recorded * 2, reports.append)
    assert not reports[-1]['passed']


def test_taylor_rejects_wrong_gradient(quadratic_tape):
    from martinec_e import taylor_check
    _, x, _, reduced = quadratic_tape

    class WrongGradient:
        def __call__(self, value):
            return reduced(value)

        def derivative(self, **kwargs):
            derivative = reduced.derivative(**kwargs)
            with derivative.dat.vec as vec:
                vec.scale(1.1)
            return derivative

    reports = []
    with pytest.raises(RuntimeError, match='Taylor check failed'):
        taylor_check(WrongGradient(), x, reports.append)
    assert not reports[-1]['passed']
    assert reports[-1]['minimum_convergence_rate'] < 1.1


@pytest.mark.parametrize('dimension', [2, 3])
def test_disk_surface_control(tmp_path, dimension):
    """Disk replay must retain the surface-to-volume map and original spaces."""
    fd = pytest.importorskip('firedrake')
    from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                                  get_working_tape, pause_annotation, stop_annotating)
    from pyadjoint import Tape, set_working_tape
    from types import SimpleNamespace
    from martinec_e import configure_adjoint_checkpointing, replay_check, taylor_check
    pause_annotation()
    previous_tape = get_working_tape()
    tape = Tape()
    set_working_tape(tape)
    try:
        with stop_annotating():
            if dimension == 2:
                parent = mesh = fd.UnitSquareMesh(2, 2)
                boundary = 4
            else:
                parent = fd.UnitCubeMesh(2, 2, 2)
                coord = fd.SpatialCoordinate(parent)
                marker = fd.Function(fd.FunctionSpace(parent, 'DG', 0))
                marker.interpolate(fd.conditional(coord[0] < 0.5, 1, 0))
                parent.mark_entities(marker, 101)
                mesh = fd.Submesh(parent, 3, 101)
                boundary = 6
            surface = fd.Submesh(mesh, dimension - 1, boundary)
            x = fd.Function(fd.FunctionSpace(surface, 'CG', 2))
            state = fd.Function(fd.FunctionSpace(mesh, 'CG', 2))
            lifted = fd.Function(state.function_space())
            clock = fd.Function(fd.FunctionSpace(parent, 'CG', 2))
            volume_clock = fd.Function(state.function_space())
        args = SimpleNamespace(adjoint_checkpointing='single_disk', adjoint_checkpoints=0,
                               adjoint_checkpoint_dir=str(tmp_path / 'functions'))
        configure_adjoint_checkpointing(args, tape, 4, mesh.comm, tmp_path)
        continue_annotation()
        control = Control(x)
        for step in tape.timestepper(iter(range(4))):
            clock.assign(float(step + 1))
            volume_clock.interpolate(clock)
            lifted.interpolate(x, allow_missing_dofs=True)
            state.assign(0.5 * state + lifted)
            if step == 3:
                objective = fd.assemble(0.5 * (state - volume_clock / 4)**2 * fd.ds(boundary))
        pause_annotation()
        recorded = float(objective)
        reduced = ReducedFunctional(objective, control)
        assert replay_check(reduced, x, recorded, lambda r: None)['passed']
        assert taylor_check(reduced, x, lambda r: None)['minimum_convergence_rate'] == pytest.approx(2, abs=1e-7)
    finally:
        pause_annotation()
        tape.clear_tape()
        set_working_tape(previous_tape)
