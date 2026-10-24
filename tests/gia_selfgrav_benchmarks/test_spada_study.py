"""Study manifest, submission safeguards and accounting interpretation."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
import spada_study as study
from estimate_spada_cost import estimate, parse_job


def test_matrix_covers_independent_variants_and_outputs():
    runs = study.run_matrix()
    production = study.selected_runs({"runs": runs}, "production")
    assert len(runs) == 31 and len(production) == 27
    assert len({run["output"] for run in runs}) == len(runs)
    assert len(study.selected_runs({"runs": runs}, "baselines")) == 6
    assert len(study.selected_runs({"runs": runs}, "variants")) == 21
    assert sum(run["surface_output"] for run in production) == 6
    for run in production:
        changed = {key for key in study.BASELINE if run["settings"][key] != study.BASELINE[key]}
        expected = ({"outer_rtol", "block0_rtol"} if run["variant"] == "loose" else
                    set(study.VARIANTS[run["variant"]]))
        assert changed == expected
        assert "--write_profiles" in run["arguments"]
        assert "--write_output" not in run["arguments"]
        assert run["epochs_kyr"] == [0., .1, 1., 2., 5., 10., 20.]
    for shape in ("cap", "disc"):
        on = next(run for run in production if run["id"] == f"{shape}-on/baseline")
        off = next(run for run in production if run["id"] == f"{shape}-offaxis-off/baseline")
        assert on["load_centre_deg"] == off["load_centre_deg"] == [25., 75.]
        assert on["settings"] == off["settings"]
        assert on["rotation"] and not off["rotation"]


def test_source_guard_checks_contents_not_only_commit(tmp_path):
    directory = tmp_path / "tests" / "bench"
    directory.mkdir(parents=True)
    source = tmp_path / "solver.py"
    source.write_text("original")
    plan = {"source": {"sha256": {"solver.py": hashlib.sha256(b"original").hexdigest()}}}
    study.verify_source(plan, directory)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Source changed"):
        study.verify_source(plan, directory)


def test_submit_records_job_id_and_prevents_duplicate_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "verify_source", lambda *args: None)
    run = study.selected_runs({"runs": study.run_matrix()}, "smoke")[0]
    run["script"] = "plan/job.pbs"
    plan = {"runs": [run]}
    (tmp_path / "plan").mkdir()
    calls = []

    def qsub(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="123.gadi-pbs\n")

    monkeypatch.setattr(study.subprocess, "run", qsub)
    study.submit(plan, "plan", "smoke", directory=tmp_path)
    study.submit(plan, "plan", "smoke", directory=tmp_path)
    assert len(calls) == 1
    assert calls[0][0] == "qsub"
    record = json.loads((tmp_path / "plan/submissions.jsonl").read_text())
    assert record["job_id"] == "123.gadi-pbs"


def test_submit_refuses_existing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "verify_source", lambda *args: None)
    run = study.selected_runs({"runs": study.run_matrix()}, "smoke")[0]
    (tmp_path / "plan").mkdir()
    output = tmp_path / run["output"]
    output.mkdir(parents=True)
    (output / "summary_cap.json").write_text("{}")
    with pytest.raises(ValueError, match="nonempty output"):
        study.submit({"runs": [run]}, "plan", "smoke", directory=tmp_path)


def test_prepared_scripts_are_portable_and_valid_bash(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "source_record", lambda directory: {"sha256": {}})
    plan = study.prepare("plan", directory=tmp_path)
    assert len(list((tmp_path / "plan/jobs").glob("*.pbs"))) == 31
    for run in plan["runs"]:
        script = tmp_path / run["script"]
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        assert "#PBS -P vo05" in script.read_text()
        assert str(tmp_path) not in script.read_text()
        assert f"SMOKE={int(run['smoke'])}" in script.read_text()
    with pytest.raises(ValueError, match="not empty"):
        study.prepare("plan", directory=tmp_path)


def test_cost_uses_elapsed_walltime_and_excludes_failed_or_high_degree_jobs(tmp_path):
    path = tmp_path / "spada_test.o123"
    path.write_text("""=== module: firedrake/main-20261005 ===
K/mu 100
projected diagnostic degrees 2..10
step 0 t 0.0000 kyr outer 2 wall 100.0 s
step 138 t 20.0000 kyr outer 2 wall 2600.0 s
Exit Status: 0
Service Units: 3120.00
NCPUs Requested: 1560
Walltime Requested: 06:00:00 Walltime Used: 01:00:00
""")
    job = parse_job(path)
    assert job["calibration"]
    assert job["su_per_core_hour"] == 2.
    result = estimate({"runs": study.run_matrix()}, [job])
    assert result["baseline_hours"] == 1.
    assert result["totals"]["walltime_su_cap"] == 253552.
    assert result["totals"]["su_low"] < result["totals"]["su_estimate"] < result["totals"]["su_high"]
    path.write_text(path.read_text().replace("Exit Status: 0", "Exit Status: -29"))
    assert not parse_job(path)["calibration"]
    path.write_text(path.read_text().replace("Exit Status: -29", "Exit Status: 0").replace("2..10", "2..40"))
    assert not parse_job(path)["calibration"]
