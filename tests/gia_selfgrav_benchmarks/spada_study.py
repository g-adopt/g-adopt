"""Prepare, inspect and submit the Spada paper study on Gadi (stdlib only).

Preparation never submits jobs. Run this from the Gadi checkout, then submit
smoke checks and baselines before the sensitivity sweep. See SPADA_STUDY.md.
"""

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = "outputs/spada-study-plan"
BASELINE = {"timestep_scale": 1., "mesh_h_km": 250., "litho_layers": 2,
            "bulk_shear_ratio": 1000., "dtn_degree": 5,
            "outer_rtol": 1e-6, "block0_rtol": 1e-4}
VARIANTS = {"baseline": {}, "dt2": {"timestep_scale": 2.},
            "mesh500": {"mesh_h_km": 500.}, "litho1": {"litho_layers": 1},
            "bulk100": {"bulk_shear_ratio": 100.}, "dtn2": {"dtn_degree": 2}}
EPOCHS = [0., .1, 1., 2., 5., 10., 20.]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(directory=HERE):
    root = directory.parent.parent
    files = sorted(set(root.glob("gadopt/**/*.py")) |
                   set(directory.glob("spada*.py")) |
                   {directory / "selfgrav_common.py", directory / "run_benchmark.pbs",
                    directory / "convert_spada_surface.py", directory / "reference.npz"})
    tracked = {str(path.relative_to(root)): sha256(path) for path in files if path.is_file()}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=root, text=True).strip())
    return {"revision": revision, "tracked_files_dirty": dirty, "sha256": tracked}


def verify_source(plan, directory=HERE):
    root = directory.parent.parent
    changed = [name for name, digest in plan["source"]["sha256"].items()
               if not (root / name).is_file() or sha256(root / name) != digest]
    if changed:
        raise ValueError("Source changed since preparation; prepare a new plan: " + ", ".join(changed))


def run_matrix(output_root="outputs/spada-study"):
    runs = []

    def add(scenario, variant, *, shape, rotating, centre, smoke=False, changes=None):
        settings = {**BASELINE, **(changes or {})}
        if smoke:
            settings.update(mesh_h_km=1000., litho_layers=1)
        case = "polar-motion" if rotating else "cap"
        stage = "smoke" if smoke else ("baselines" if variant == "baseline" else "variants")
        identity = f"smoke/{scenario}" if smoke else f"{scenario}/{variant}"
        surface = smoke or variant == "baseline"
        arguments = ["--case", case, "--load_shape", shape, "--load_centre", *map(str, centre),
                     "--load_representation", "spatial", "--diagnostic_nmax", "10"]
        for key, value in settings.items():
            arguments += ["--" + key, str(value)]
        arguments += ["--write_profiles", "--profile_spacing_deg", "0.1",
                      "--profile_depth_fraction", "1e-6", "--profile_extent_deg", "180",
                      "--checkpoint_every", "50"]
        arguments += (["--profile_bearings", "0", "90"] if centre[0] != 0 else
                      ["--profile_longitudes", "0", "90", "180"])
        if surface:
            arguments += ["--write_surface_output", "--surface_output_format", "hdf5"]
        if smoke:
            arguments += ["--smoke"]
        epochs = [0., .1] if smoke else EPOCHS
        expected = [f"summary_{case}.json", f"params_{case}.log", f"spada_{case}.msh",
                    *[f"profiles_{case}_{epoch:g}kyr.npz" for epoch in epochs]]
        if surface:
            expected.append(f"spada_{case}_surface.h5")
        if not smoke:
            expected.append(f"checkpoint_{case}.h5")
        runs.append({"id": identity, "stage": stage, "scenario": scenario, "variant": variant,
                     "case": case, "shape": shape, "rotation": rotating, "load_centre_deg": centre,
                     "settings": settings, "smoke": smoke, "surface_output": surface,
                     "epochs_kyr": epochs, "expected_outputs": expected,
                     "output": str(Path(output_root) / identity), "arguments": arguments,
                     "resources": {"ncpus": 104 if smoke else 1560,
                                   "mem_gb": 500 if smoke else 7500, "jobfs_gb": 100,
                                   "walltime": "01:00:00" if smoke else "03:00:00"}})

    for shape in ("cap", "disc"):
        for rotating in (False, True):
            scenario = f"{shape}-{'on' if rotating else 'off'}"
            centre = [25., 75.] if rotating else [0., 0.]
            add(scenario, "smoke", shape=shape, rotating=rotating, centre=centre, smoke=True)
            for variant, changes in VARIANTS.items():
                add(scenario, variant, shape=shape, rotating=rotating, centre=centre, changes=changes)
    add("cap-on", "loose", shape="cap", rotating=True, centre=[25., 75.],
        changes={"outer_rtol": 1e-5, "block0_rtol": 1e-3})
    for shape in ("cap", "disc"):
        add(f"{shape}-offaxis-off", "baseline", shape=shape, rotating=False, centre=[25., 75.])
    return runs


def _job_path(value):
    path = Path(value)
    return shlex.quote(str(path)) if path.is_absolute() else '"$PBS_O_WORKDIR"/' + shlex.quote(str(path))


def job_script(run, config, manifest_path):
    resource = run["resources"]
    name = "sp_" + run["id"].replace("/", "_").replace("-", "_")
    # Keep CASE and the output flags explicit in each inspectable job file.
    extra = [arg for arg in run["arguments"][2:] if arg != "--smoke"]
    return f'''#!/bin/bash
#PBS -P {config['project']}
#PBS -q {config['queue']}
#PBS -N {name}
#PBS -l ncpus={resource['ncpus']}
#PBS -l mem={resource['mem_gb']}GB
#PBS -l jobfs={resource['jobfs_gb']}GB
#PBS -l walltime={resource['walltime']}
#PBS -l storage={config['storage']}
#PBS -l wd

set -e
source /etc/profile
module use /g/data/fp50/modules
module load {shlex.quote(config['module'])}
cd "$PBS_O_WORKDIR"
python3 spada_study.py verify-source --manifest {_job_path(manifest_path)}
export DRIVER=spada CASE={shlex.quote(run['case'])} SMOKE={int(run['smoke'])}
export FIREDRAKE_MODULE={shlex.quote(config['module'])}
export OUTPUT={_job_path(run['output'])}
export EXTRA={shlex.quote(shlex.join(extra))}
mkdir -p "$OUTPUT"
mkdir "$OUTPUT/.run-claimed" || {{ echo "Run directory already claimed: $OUTPUT"; exit 2; }}
set +e
source "$PBS_O_WORKDIR/run_benchmark.pbs"
'''


def prepare(plan_dir, *, output_root="outputs/spada-study", project="vo05", queue="normalsr",
            module="firedrake/main-20261005",
            storage="scratch/xd2+scratch/vo05+gdata/vo05+gdata/fp50", directory=HERE):
    plan_path = directory / plan_dir
    if plan_path.exists() and any(plan_path.iterdir()):
        raise ValueError(f"Plan directory is not empty: {plan_path}; use a new --plan-dir")
    config = {"project": project, "queue": queue, "module": module, "storage": storage}
    # PBS directives must be one literal line; environment values are shell-quoted.
    if any(not value or any(c.isspace() for c in value) for value in config.values()):
        raise ValueError("Project, queue, module and storage must be nonempty and contain no whitespace")
    manifest = {"version": 1, "created_utc": datetime.now(UTC).isoformat(),
                "source": source_record(directory), "config": config, "runs": run_matrix(output_root)}
    plan_path.mkdir(parents=True, exist_ok=True)
    (plan_path / "jobs").mkdir()
    manifest_name = Path(plan_dir) / "manifest.json"
    for run in manifest["runs"]:
        run["script"] = str(Path(plan_dir) / "jobs" / (run["id"].replace("/", "_") + ".pbs"))
        (directory / run["script"]).write_text(job_script(run, config, manifest_name))
    (plan_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def selected_runs(plan, stage):
    return [run for run in plan["runs"]
            if run["stage"] == stage or (stage == "production" and not run["smoke"])]


def submit(plan, plan_dir, stage, *, directory=HERE):
    verify_source(plan, directory)
    with (directory / plan_dir / ".submission.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another submission process is using this plan") from error
        _submit_jobs(plan, plan_dir, stage, directory=directory)


def _submit_jobs(plan, plan_dir, stage, *, directory):
    state_path = directory / plan_dir / "submissions.jsonl"
    submitted = {entry["id"] for entry in
                 (json.loads(line) for line in state_path.read_text().splitlines())} if state_path.exists() else set()
    for run in selected_runs(plan, stage):
        if run["id"] in submitted:
            print(f"Already submitted: {run['id']}")
            continue
        output = directory / run["output"]
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Refusing nonempty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        command = ["qsub", "-d", str(directory), "-o", str(output / "pbs.stdout"),
                   "-e", str(output / "pbs.stderr"), str(directory / run["script"])]
        result = subprocess.run(command, cwd=directory, text=True, capture_output=True, check=True)
        job_id = result.stdout.strip()
        entry = {"id": run["id"], "job_id": job_id, "qsub": command,
                 "submitted_utc": datetime.now(UTC).isoformat()}
        with state_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(f"{run['id']}: {job_id}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="write manifest and 31 PBS files; submit nothing")
    create.add_argument("--plan-dir", default=DEFAULT_PLAN)
    create.add_argument("--output-root", default="outputs/spada-study")
    create.add_argument("--project", default="vo05")
    create.add_argument("--queue", default="normalsr")
    create.add_argument("--module", default="firedrake/main-20261005")
    create.add_argument("--storage", default="scratch/xd2+scratch/vo05+gdata/vo05+gdata/fp50")
    launch = commands.add_parser("submit", help="explicitly submit one stage with qsub")
    launch.add_argument("--plan-dir", default=DEFAULT_PLAN)
    launch.add_argument("--stage", choices=("smoke", "baselines", "variants", "production"), required=True)
    inspect = commands.add_parser("show", help="list planned runs and their exact driver commands")
    inspect.add_argument("--plan-dir", default=DEFAULT_PLAN)
    inspect.add_argument("--stage", choices=("smoke", "baselines", "variants", "production"), default="production")
    verify = commands.add_parser("verify-source", help="check prepared source hashes before execution")
    verify.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            plan = prepare(args.plan_dir, output_root=args.output_root, project=args.project,
                           queue=args.queue, module=args.module, storage=args.storage)
            print(f"Prepared {len(plan['runs'])} jobs (4 smoke + 27 production) in {HERE / args.plan_dir}")
            print("No jobs submitted. Inspect with: python3 spada_study.py show")
            return
        manifest_path = Path(args.manifest) if args.command == "verify-source" else HERE / args.plan_dir / "manifest.json"
        plan = json.loads(manifest_path.read_text())
        if args.command == "verify-source":
            verify_source(plan)
        elif args.command == "submit":
            submit(plan, args.plan_dir, args.stage)
        else:
            for run in selected_runs(plan, args.stage):
                command = ["python3", "spada.py", *run["arguments"], "--output_path", run["output"]]
                print(f"{run['id']} ({run['resources']['ncpus']} cores, {run['resources']['walltime']}):\n  {shlex.join(command)}")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
