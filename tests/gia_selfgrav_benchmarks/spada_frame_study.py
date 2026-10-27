"""Prepare a paired Spada centre-of-mass study; preparation submits nothing.

Run from the benchmark directory on Gadi. See SPADA_FRAME_STUDY.md.
This module needs only the Python standard library.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import spada_study as study

DEFAULT_PLAN = "outputs/spada-frame-study-plan"
DEFAULT_OUTPUT = "outputs/spada-frame-study"
EXTRA_EPOCHS = [4.9, 9.8, 9.9, 10.5]
CHECKS = ("tolerance", "timestep", "location")
STAGES = ("smoke", "pilot", "core", *CHECKS, "production")


def run_matrix(output_root=DEFAULT_OUTPUT, checks=()):
    """Eight short construction checks, sixteen core runs, optional controls."""
    unknown = set(checks) - set(CHECKS)
    if unknown:
        raise ValueError(f"Unknown checks: {sorted(unknown)}")
    runs = []

    def pair(shape, rotating, mesh, stage, *, centre=None, changes=None):
        smoke = stage == "smoke"
        centre = centre if centre is not None else ([25., 75.] if rotating else [0., 0.])
        scenario = f"{shape}-{'on' if rotating else 'off'}"
        case = "polar-motion" if rotating else "cap"
        settings = {**study.BASELINE, "mesh_h_km": float(mesh), **(changes or {})}
        if smoke:
            settings.update(litho_layers=1, timestep_scale=10.)
        for constrained in (False, True):
            frame = "com-on" if constrained else "com-off"
            identity = f"{stage}/{scenario}/mesh{mesh:g}/{frame}"
            arguments = ["--case", case, "--load_shape", shape,
                         "--load_centre", *map(str, centre),
                         "--load_representation", "spatial", "--diagnostic_nmax", "10"]
            for key, value in settings.items():
                arguments += ["--" + key, str(value)]
            arguments += ["--write_frame_diagnostics", "--write_profiles",
                          "--profile_spacing_deg", "0.1", "--profile_depth_fraction", "1e-6",
                          "--profile_extent_deg", "180", "--checkpoint_every", "50",
                          "--write_surface_output", "--surface_output_format", "hdf5"]
            arguments += (["--profile_bearings", "0", "90"] if centre[0] != 0 else
                          ["--profile_longitudes", "0", "90", "180"])
            if constrained:
                arguments += ["--centre_of_mass"]
            if smoke:
                arguments += ["--smoke"]
            else:
                arguments += ["--extra_epochs_kyr", *map(str, EXTRA_EPOCHS)]
            epochs = [0., .1] if smoke else sorted(study.EPOCHS + EXTRA_EPOCHS)
            expected = [f"summary_{case}.json", f"params_{case}.log", f"spada_{case}.msh",
                        f"spada_{case}_surface.h5",
                        *[f"profiles_{case}_{epoch:g}kyr.npz" for epoch in epochs]]
            if not smoke:
                expected.append(f"checkpoint_{case}.h5")
            walltime = ("01:00:00" if smoke else "18:00:00" if stage in ("tolerance", "timestep")
                        else "12:00:00" if mesh == 125 else "04:00:00")
            runs.append({"id": identity, "stage": stage, "scenario": scenario,
                         "variant": frame, "case": case, "shape": shape,
                         "rotation": rotating, "centre_of_mass": constrained,
                         "load_centre_deg": list(centre), "settings": settings,
                         "smoke": smoke, "surface_output": True, "epochs_kyr": epochs,
                         "expected_outputs": expected, "arguments": arguments,
                         "output": str(Path(output_root) / identity),
                         "resources": {"ncpus": 104 if smoke else 1560,
                                       "mem_gb": 500 if smoke else 7500,
                                       "jobfs_gb": 100, "walltime": walltime}})

    for shape in ("cap", "disc"):
        for rotating in (False, True):
            pair(shape, rotating, 1000, "smoke")
            for mesh in (250, 125):
                pair(shape, rotating, mesh, "core")
    if "tolerance" in checks:
        pair("cap", False, 125, "tolerance", changes={"outer_rtol": 1e-8, "block0_rtol": 1e-6})
    if "timestep" in checks:
        pair("cap", False, 125, "timestep", changes={"timestep_scale": .5})
    if "location" in checks:
        for shape in ("cap", "disc"):
            for mesh in (250, 125):
                pair(shape, False, mesh, "location", centre=[25., 75.])
    return runs


def selected_runs(plan, stage):
    if stage == "pilot":
        return [run for run in plan["runs"] if run["stage"] == "core"
                and run["scenario"] == "cap-off" and run["settings"]["mesh_h_km"] == 125.]
    return study.selected_runs(plan, stage)


def prepare(plan_dir=DEFAULT_PLAN, *, output_root=DEFAULT_OUTPUT, checks=(), **kwargs):
    return study.prepare(plan_dir, output_root=output_root,
                         runs=run_matrix(output_root, checks), **kwargs)


def submit(plan, plan_dir, stage, **kwargs):
    # Reuse the same journal and output/source guards across pilot/core selections.
    selected = {**plan, "runs": selected_runs(plan, stage)}
    study.submit(selected, plan_dir, "production" if stage == "pilot" else stage, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="write manifest/PBS files; submit nothing")
    create.add_argument("--plan-dir", default=DEFAULT_PLAN)
    create.add_argument("--output-root", default=DEFAULT_OUTPUT)
    create.add_argument("--checks", nargs="+", choices=CHECKS, default=[])
    create.add_argument("--project", default="vo05")
    create.add_argument("--queue", default="normalsr")
    create.add_argument("--module", default=study.DEFAULT_MODULE)
    create.add_argument("--firedrake-version", default=study.DEFAULT_FIREDRAKE_VERSION)
    create.add_argument("--storage", default="scratch/xd2+scratch/vo05+gdata/vo05+gdata/fp50")
    for command in ("show", "submit"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--plan-dir", default=DEFAULT_PLAN)
        subparser.add_argument("--stage", choices=STAGES, default="core", required=command == "submit")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            plan = prepare(**{key: value for key, value in vars(args).items() if key != "command"})
            print(f"Prepared {len(plan['runs'])} jobs in {study.HERE / args.plan_dir}; no jobs submitted.")
            print("Inspect with: python3 spada_frame_study.py show --stage core")
            return
        plan = json.loads((study.HERE / args.plan_dir / "manifest.json").read_text())
        runs = selected_runs(plan, args.stage)
        if not runs:
            raise ValueError(f"No {args.stage} runs in this plan; enable the corresponding --checks at preparation")
        if args.command == "submit":
            submit(plan, args.plan_dir, args.stage)
        else:
            for run in runs:
                command = ["python3", "spada.py", *run["arguments"], "--output_path", run["output"]]
                print(f"{run['id']} ({run['resources']['ncpus']} cores, {run['resources']['walltime']}):\n"
                      f"  {shlex.join(command)}")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
