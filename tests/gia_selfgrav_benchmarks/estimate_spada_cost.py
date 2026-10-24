"""Estimate the prepared Spada study's SU cost from Gadi PBS accounting logs.

Uses completed 20 kyr, diagnostic-degree-10 runs for calibration. Mesh and
K/mu multipliers are explicit planning assumptions, not measured speedups.
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from spada_study import DEFAULT_PLAN, HERE, sha256


def hours(value):
    h, m, s = map(float, value.split(":"))
    return h + m/60 + s/3600


def parse_job(path):
    text = path.read_text(errors="replace")
    patterns = {"exit": r"Exit Status:\s+(-?\d+)", "su": r"Service Units:\s+([\d.]+)",
                "ncpus": r"NCPUs Requested:\s+(\d+)", "walltime": r"Walltime Used:\s+([\d:]+)",
                "module": r"=== module: (.*?) ===", "bulk": r"K/mu\s+([\d.]+)",
                "diagnostic_degree": r"projected diagnostic degrees 2\.\.(\d+)",
                "mesh_md5": r"md5 (\w+), cells mantle"}
    job = {"file": path.name, "sha256": sha256(path)}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            job[key] = match.group(1)
    if not {"exit", "su", "ncpus", "walltime"} <= job.keys():
        return None
    for key in ("exit", "ncpus"):
        job[key] = int(job[key])
    job["su"] = float(job["su"])
    job["wall_hours"] = hours(job["walltime"])
    job["su_per_core_hour"] = job["su"] / (job["ncpus"] * job["wall_hours"])
    if "diagnostic_degree" not in job:
        match = re.search(r"degrees 2\.\.(\d+), SphericalDtN", text)
        if match:
            job["diagnostic_degree"] = match.group(1)
    if "diagnostic_degree" in job:
        job["diagnostic_degree"] = int(job["diagnostic_degree"])
    steps = re.findall(r"step\s+(\d+)\s+t\s+([\d.]+) kyr\s+outer\s+\d+\s+wall\s+([\d.]+)", text)
    job["last_time_kyr"] = float(steps[-1][1]) if steps else None
    job["steps"] = len(steps)
    job["march_hours"] = sum(float(wall) for step, _, wall in steps if int(step) > 0)/3600
    job["calibration"] = (job["exit"] == 0 and job["last_time_kyr"] == 20.
                          and job.get("diagnostic_degree") == 10 and job["ncpus"] == 1560
                          and float(job.get("bulk", "nan")) == 100.)
    return job


def estimate(plan, jobs):
    calibration = [job for job in jobs if job["calibration"]]
    if not calibration:
        raise ValueError("No completed 20 kyr degree-10, 1560-core calibration jobs found")
    rate = statistics.median(job["su_per_core_hour"] for job in calibration)
    baseline_hours = statistics.median(job["wall_hours"] for job in calibration)
    dt_ratio = statistics.median((job["wall_hours"] - .5*job["march_hours"])/job["wall_hours"]
                                 for job in calibration)
    smoke = [job for job in jobs if job["exit"] == 0 and job["ncpus"] == 104
             and job["last_time_kyr"] == .1]
    smoke_hours = statistics.median(job["wall_hours"] for job in smoke) if smoke else .25
    rows = []
    for run in plan["runs"]:
        base = smoke_hours if run["smoke"] else baseline_hours
        variant = run["variant"]
        central, low, high = {"dt2": (dt_ratio, dt_ratio, .75),
                              "mesh500": (.5, .3, 1.),
                              "litho1": (.85, .75, 1.)}.get(variant, (1., 1., 1.))
        # Historical calibration is K/mu=100. No production timings for the
        # new K/mu=1000, disc or updated rate/HDF5 exports are available yet.
        if run["settings"]["bulk_shear_ratio"] == 1000:
            central *= 1.25
            high *= 2.
        else:
            high *= 1.25
        cpus = run["resources"]["ncpus"]
        rows.append({"id": run["id"], "stage": run["stage"], "variant": variant,
                     "ncpus": cpus, "hours_estimate": base*central,
                     "su_estimate": cpus*rate*base*central,
                     "su_low": cpus*rate*base*low, "su_high": cpus*rate*base*high,
                     "walltime_su_cap": cpus*rate*hours(run["resources"]["walltime"])})
    return {"su_per_core_hour": rate, "baseline_hours": baseline_hours,
            "dt2_ratio": dt_ratio, "calibration_files": [job["file"] for job in calibration],
            "jobs": rows, "totals": {key: sum(row[key] for row in rows)
                                        for key in ("su_estimate", "su_low", "su_high", "walltime_su_cap")}}


def report(result, jobs):
    totals = result["totals"]
    lines = ["# Spada Gadi cost estimate", "",
             "31 prepared jobs: 27 production runs and four one-node smoke checks.", "",
             (f"Central estimate: **{totals['su_estimate']:,.0f} SU**; planning range "
             f"**{totals['su_low']:,.0f}–{totals['su_high']:,.0f} SU**."),
             f"A 25% margin on the central estimate is {1.25*totals['su_estimate']:,.0f} SU.", "",
             (f"The accounting logs imply {result['su_per_core_hour']:.4f} SU/core-hour. "
             "Charges follow elapsed walltime, not the requested limit. "
             f"If all jobs reach their requested limits, the corresponding charge is "
             f"{totals['walltime_su_cap']:,.0f} SU (a resource limit, not the expected cost)."), "",
             "## Measured calibration", "",
             "| Log | Walltime | SU | Notes |", "|---|---|---:|---|"]
    for job in jobs:
        if job["calibration"]:
            lines.append(f"| `{job['file']}` | {job['walltime']} | {job['su']:,.2f} | "
                         f"K/μ={job.get('bulk', '?')}; {job.get('module', '?')} |")
    lines += ["", f"Median full-run walltime: {60*result['baseline_hours']:.1f} min on 1,560 cores.",
              ("The spatial cap run 181040847 used 3,438.07 SU and 1.63 TB of memory. "
              "All calibration runs have 139 solved states (elastic plus 138 marched steps), "
              "diagnostic degree 10 and the same recorded mesh MD5."), "",
              ("Failed/partial jobs, smoke checks and degree-20/40 diagnostics are excluded from "
              "full-run calibration. Their accounting rows are retained in historical_jobs.csv."), "",
              "## Planning assumptions", "",
              ("- All production runs retain 1,560 cores / 7,500 GB and a three-hour limit, "
              "so walltime and SU comparisons use the same hardware allocation."),
              ("- K/μ=1000: central 1.25×, lower 1×, upper 2× the historical K/μ=100 cost. "
              "This allowance also covers the unmeasured disc/rotation and new output overhead; "
              "it is not a measured compressibility scaling. K/μ=100 uses 1× centrally, 1.25× above."),
              (f"- Doubled timesteps: {result['dt2_ratio']:.3f}× centrally, derived by halving "
              "the measured marched-solve time while retaining elastic solve and non-solve overhead. "
              "The upper factor is 0.75× before the K/μ allowance."),
              ("- 500 km mesh: 0.5× centrally, 0.3–1× range. One lithosphere subdivision: "
              "0.85× centrally, 0.75–1× range. These are deliberately broad assumptions; "
              "no coarse-mesh production timing is available."),
              ("- No assumed saving for DtN degree 2 or looser tolerances. "
              "Historical polar-motion cost in the README (~3,162 SU) is comparable to the cap; "
              "there is no local completed rotating/disc log for the new settings."),
              "- Smoke checks use the successful 104-core smoke timing with the same K/μ allowance.",
              ("- This is a planning envelope, not a statistical confidence interval. "
              "Run the six baselines first and replace these assumptions with measured costs "
              "before submitting the 21 variants. It excludes extra retries and offline conversion jobs."),
              "", "## Estimated cost by variant", "",
              "| Group | Jobs | Central SU | Range SU |", "|---|---:|---:|---:|"]
    groups = defaultdict(list)
    for row in result["jobs"]:
        groups["smoke" if row["stage"] == "smoke" else row["variant"]].append(row)
    for group, rows in groups.items():
        sums = {key: sum(row[key] for row in rows) for key in ("su_estimate", "su_low", "su_high")}
        lines.append(f"| {group} | {len(rows)} | {sums['su_estimate']:,.0f} | "
                     f"{sums['su_low']:,.0f}–{sums['su_high']:,.0f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", default=DEFAULT_PLAN)
    parser.add_argument("--logs-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    path = HERE / args.plan_dir
    plan = json.loads((path / "manifest.json").read_text())
    jobs = [job for source in sorted(args.logs_dir.glob("spada_*.o*"))
            if (job := parse_job(source)) is not None]
    result = estimate(plan, jobs)
    (path / "cost-estimate.md").write_text(report(result, jobs))
    (path / "cost-estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in (("historical_jobs.csv", jobs), ("job_estimates.csv", result["jobs"])):
        with (path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, sorted(set().union(*(row.keys() for row in rows))))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Estimated {result['totals']['su_estimate']:,.0f} SU "
          f"({result['totals']['su_low']:,.0f}–{result['totals']['su_high']:,.0f}); see {path / 'cost-estimate.md'}")


if __name__ == "__main__":
    main()
