"""Case B schedules and restart compatibility; no full GIA integration."""

import json
from types import SimpleNamespace

import martinec
import pytest


def schedule(arguments):
    args = martinec.parse_args(arguments)
    scenario = "T0" if args.case == "B" else "T1"
    ladder, segments = martinec.time_schedule(args, martinec.EPOCHS_KYR[scenario])
    settings = martinec.run_settings(args, martinec.BULK_SHEAR_RATIO[args.case], ladder)
    return args, ladder, segments, settings


@pytest.mark.parametrize("case,steps,dt", [("B", 1000, 10.), ("C", 300, 50.), ("D", 300, 50.)])
def test_uniform_defaults(case, steps, dt):
    args, ladder, segments, settings = schedule(["--case", case])
    assert sum(segment[3] for segment in segments) == steps
    assert all(segment[2] == dt for segment in segments)
    assert settings["time_step_yr"] == dt
    assert settings["time_step_ladder_yr"] is None


def test_spada_ladder_and_output_epochs():
    _, ladder, segments, settings = schedule(
        ["--time_step_ladder", "100:10", "1000:50", "10000:100"])
    assert sum(segment[3] for segment in segments) == 118
    assert [s[1]/1000 for s in segments if s[4]] == [1., 2., 5., 10.]
    assert settings["time_step_yr"] is None
    assert settings["time_step_ladder_yr"] == [[100., 10.], [1000., 50.], [10000., 100.]]
    assert martinec.checkpoint_run_settings(json.loads(json.dumps(settings))) == settings


def test_nondivisible_steps_land_on_epochs_and_terminal_time():
    _, ladder, segments, _ = schedule(
        ["--time_step_ladder", "100:17", "1100:63", "20000:131"])
    assert ladder[-1] == (10000., 131.)
    assert [s[1]/1000 for s in segments if s[4]] == [1., 2., 5., 10.]
    previous = 0.
    for start, end, dt, steps, _ in segments:
        assert start == previous
        assert start + steps*dt == pytest.approx(end)
        previous = end
    assert previous == 10000.


@pytest.mark.parametrize("arguments", [
    ["--time_step_ladder"],
    ["--time_step_ladder", "10000"],
    ["--time_step_ladder", "10000:10:1"],
    ["--time_step_ladder", "nan:10"],
    ["--time_step_ladder", "10000:inf"],
    ["--time_step_ladder", "0:10"],
    ["--time_step_ladder", "10000:0"],
    ["--time_step_ladder", "10000:-10"],
    ["--time_step_ladder", "1000:10", "100:50"],
    ["--time_step_ladder", "1000:10", "1000:50"],
    ["--time_step_ladder", "1000:50"],
    ["--time_step_yr", "10", "--time_step_ladder", "10000:50"],
    *[["--case", case, "--time_step_ladder", "15000:50"] for case in ("C", "D", "E")],
])
def test_invalid_ladders(arguments):
    with pytest.raises(SystemExit) as error:
        martinec.parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("dt", [None, 25.])
def test_legacy_uniform_checkpoints(dt):
    arguments = [] if dt is None else ["--time_step_yr", str(dt)]
    _, _, _, expected = schedule(arguments)
    legacy = {"case": "B", "bulk_shear_ratio": 1000., "smoke": False}
    if dt is not None:
        legacy["time_step_yr"] = dt
    assert martinec.checkpoint_run_settings(legacy) == expected


@pytest.mark.parametrize("written_arguments", [
    [],
    ["--time_step_ladder", "100:10", "1000:25", "10000:100"],
    ["--time_step_ladder", "100:10", "2000:50", "10000:100"],
])
def test_restart_rejects_changed_schedule(tmp_path, monkeypatch, written_arguments):
    _, _, _, written = schedule(written_arguments)
    arguments = ["--time_step_ladder", "100:10", "1000:50", "10000:100",
                 "--restart", "checkpoint_B.h5", "--output_path", str(tmp_path)]
    args = martinec.parse_args(arguments)
    monkeypatch.setattr(martinec, "parse_args", lambda: args)
    restart = SimpleNamespace(record={"run": json.loads(json.dumps(written))})
    monkeypatch.setattr(martinec.common, "load_checkpoint", lambda *args: (None, None, {}, restart))
    with pytest.raises(RuntimeError, match="was written by the run"):
        martinec.main()


def test_ladder_march_resumes_inside_segment():
    # Exercise the actual shared loop across step-size changes with a scalar
    # backward-Euler problem, retaining both the elastic diagnostic and reset.
    _, _, segments, _ = schedule(
        ["--time_step_ladder", "100:10", "1000:50", "10000:100"])

    class Scalar:
        def __init__(self, value=0.):
            self.value = value

        def assign(self, value):
            self.value = value

        def __float__(self):
            return float(self.value)

    class Solver:
        def __init__(self, dt, value=0.):
            self.dt = dt
            self.solution = Scalar(value)
            self.solution_old = Scalar(value)

        def solve(self):
            dt = float(self.dt)
            value = (float(self.solution_old) + dt)/(1 + dt)
            self.solution.assign(value)
            self.solution_old.assign(value)

    def integrate(start_step=0, value=0.):
        dt = Scalar()
        solver = Solver(dt, value)
        rows, epochs = [], []
        martinec.common.march(
            solver, dt, segments, elastic=True, start_step=start_step,
            on_step=lambda t, d, step, wall: rows.append((t, d, step, float(solver.solution))),
            on_epoch=epochs.append)
        return rows, epochs

    rows, epochs = integrate()
    assert len(rows) == 119  # elastic diagnostic plus 118 marched steps
    assert epochs == [0., 1., 2., 5., 10.]
    assert rows[1][3] == pytest.approx(
        (10./martinec.common.T_BAR_YR)/(1 + 10./martinec.common.T_BAR_YR))
    # Step 20 lies inside the 50-year segment, away from output epochs.
    resumed, resumed_epochs = integrate(start_step=20, value=rows[20][3])
    assert resumed == rows[21:]
    assert resumed_epochs == [1., 2., 5., 10.]
