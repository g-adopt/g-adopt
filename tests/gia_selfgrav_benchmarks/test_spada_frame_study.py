"""Paired frame-study matrix and shared submission safeguards (stdlib study)."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import spada_frame_study as frame


def test_core_pairs_change_only_the_frame_flag():
    plan = {"runs": frame.run_matrix()}
    core = frame.selected_runs(plan, "core")
    assert len(core) == 16
    assert len(frame.selected_runs(plan, "smoke")) == 8
    assert len(frame.selected_runs(plan, "production")) == 16
    assert len({run["output"] for run in plan["runs"]}) == 24
    for off, on in zip(core[::2], core[1::2]):
        assert not off["centre_of_mass"] and on["centre_of_mass"]
        assert [arg for arg in on["arguments"] if arg != "--centre_of_mass"] == off["arguments"]
        assert off["settings"] == on["settings"]
        assert off["load_centre_deg"] == on["load_centre_deg"] == ([25., 75.] if on["rotation"] else [0., 0.])
        assert off["settings"]["mesh_h_km"] in (125., 250.)
        assert off["settings"]["litho_layers"] == 2
        assert "--write_frame_diagnostics" in on["arguments"]
        assert "--write_output" not in on["arguments"]
        assert on["surface_output"]
        assert on["epochs_kyr"] == [0., .1, 1., 2., 4.9, 5., 9.8, 9.9, 10., 10.5, 20.]
    pilot = frame.selected_runs(plan, "pilot")
    assert len(pilot) == 2 and all(run in core for run in pilot)


def test_optional_checks_and_matched_location_controls():
    plan = {"runs": frame.run_matrix(checks=frame.CHECKS)}
    assert len(plan["runs"]) == 36
    assert len(frame.selected_runs(plan, "production")) == 28
    for stage in ("tolerance", "timestep"):
        runs = frame.selected_runs(plan, stage)
        assert len(runs) == 2
        assert all(run["scenario"] == "cap-off" and run["settings"]["mesh_h_km"] == 125. for run in runs)
    assert frame.selected_runs(plan, "tolerance")[0]["settings"]["outer_rtol"] == 1e-8
    assert frame.selected_runs(plan, "timestep")[0]["settings"]["timestep_scale"] == .5
    controls = frame.selected_runs(plan, "location")
    assert len(controls) == 8
    for off in controls:
        on = next(run for run in frame.selected_runs(plan, "core") if run["rotation"]
                  and run["shape"] == off["shape"] and run["settings"] == off["settings"]
                  and run["centre_of_mass"] == off["centre_of_mass"])
        assert not off["rotation"]
        assert on["load_centre_deg"] == off["load_centre_deg"] == [25., 75.]
        assert on["arguments"][2:] == off["arguments"][2:]
    with pytest.raises(ValueError, match="Unknown checks"):
        frame.run_matrix(checks=["unrecognised"])


def test_prepared_frame_jobs_are_inspectable_and_portable(tmp_path, monkeypatch):
    monkeypatch.setattr(frame.study, "source_record", lambda directory: {"sha256": {}})
    plan = frame.prepare("plan", directory=tmp_path, checks=frame.CHECKS)
    assert len(list((tmp_path / "plan/jobs").glob("*.pbs"))) == 36
    assert json.loads((tmp_path / "plan/manifest.json").read_text()) == plan
    for run in plan["runs"]:
        script = tmp_path / run["script"]
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        content = script.read_text()
        assert "expected = '2026.10.0'" in content
        assert "spada_study.py verify-source" in content
        assert ".run-claimed" in content
        assert str(tmp_path) not in content
        assert "--write_frame_diagnostics" in content


def test_pilot_and_core_share_submission_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(frame.study, "verify_source", lambda *args: None)
    plan = {"runs": frame.run_matrix()}
    (tmp_path / "plan").mkdir()
    for run in plan["runs"]:
        run["script"] = "plan/job.pbs"
    calls = []

    def qsub(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{len(calls)}.gadi-pbs\n")

    monkeypatch.setattr(frame.study.subprocess, "run", qsub)
    frame.submit(plan, "plan", "pilot", directory=tmp_path)
    assert len(calls) == 2
    frame.submit(plan, "plan", "core", directory=tmp_path)
    assert len(calls) == 16
    records = [json.loads(line) for line in (tmp_path / "plan/submissions.jsonl").read_text().splitlines()]
    assert len({record["id"] for record in records}) == 16
