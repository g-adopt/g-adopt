"""Audit and compare completed Spada study outputs without importing Firedrake.

Run: python3 analyse_spada_study.py outputs/spada-study
Outputs: CSV metrics, input hashes, PDF/PNG figures and a Markdown report.
"""

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from estimate_spada_cost import parse_job  # noqa: E402
from plot_spada_profiles import reference_profiles  # noqa: E402
from spada_study import BASELINE, EPOCHS, run_matrix  # noqa: E402

QUANTITIES = ("displacement", "U", "along", "across", "N")
SCENARIOS = ("cap-off", "cap-on", "disc-off", "disc-on")
VARIANTS = ("dt2", "mesh500", "litho1", "bulk100", "dtn2", "loose", "mesh125")


@dataclass
class Run:
    path: Path
    identity: str
    summary: dict
    profiles: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)

    @property
    def scenario(self):
        return self.identity.rsplit("/", 1)[0]

    @property
    def variant(self):
        return self.identity.rsplit("/", 1)[-1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wrapped_degrees(value):
    return (value + 180) % 360 - 180


def norms(model, reference):
    """Scalar/vector norms across sampled profile points, never pointwise ratios.

    Arrays have shape (points, components). RMS is over points and uses
    Euclidean vector lengths. A vanishing reference has no relative score.
    """
    model, reference = np.asarray(model), np.asarray(reference)
    difference = np.linalg.norm(model - reference, axis=-1)
    scale = np.linalg.norm(reference, axis=-1)
    maximum, rms = float(difference.max()), float(np.sqrt(np.mean(difference**2)))
    peak, rms_scale = float(scale.max()), float(np.sqrt(np.mean(scale**2)))
    return {"max_abs": maximum, "rms_abs": rms, "reference_peak": peak,
            "reference_rms": rms_scale,
            "max_percent": 100 * maximum / peak if peak > 1e-12 else None,
            "rms_percent": 100 * rms / rms_scale if rms_scale > 1e-12 else None}


def fields(profile, *, rates=False):
    suffix = "_m_per_yr" if rates else "_m"
    result = {q: profile[q + suffix] for q in ("U", "along", "across", "N")}
    result["displacement"] = np.stack([result[q] for q in ("U", "along", "across")], axis=-1)
    return result


def reference_fields(reference, t, profile, summary, *, rates=False):
    if rates:
        start = float(profile["rate_interval_start_kyr"])
        end = float(profile["rate_interval_end_kyr"])
        dt = float(profile["rate_dt_yr"])
        first = reference_profiles(reference, start, profile, summary)
        last = reference_profiles(reference, end, profile, summary)
        result = {q: (last[q] - first[q]) / dt for q in first}
    else:
        result = reference_profiles(reference, t, profile, summary)
    result["displacement"] = np.stack([result[q] for q in ("U", "along", "across")], axis=-1)
    return result


def masks(profile, summary):
    theta, phi = np.deg2rad(summary["load_centre_deg"])
    axis = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    distance = np.rad2deg(np.arccos(np.clip(profile["unit_vectors"] @ axis, -1, 1)))
    return {"near": distance <= 25 + 1e-9,
            "global": np.ones(distance.shape, dtype=bool),
            "edge-excluded": np.abs(distance - 10) >= 1}


def check_geometry(left, right):
    for key in ("profile_kind", "profile_labels"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Profile {key} differs")
    for key in ("coordinate_deg", "unit_vectors", "tangent_vectors", "right_vectors", "depth_fraction"):
        if left[key].shape != right[key].shape or not np.allclose(left[key], right[key], atol=1e-12, rtol=0):
            raise ValueError(f"Profile {key} differs; comparisons require identical sampling")


def same_physics(left, right):
    for key in ("case", "rotation", "load_shape", "load_centre_deg", "load_representation",
                "load_nmax", "diagnostic_nmax"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Cannot match runs with different {key}")


def validate_controls(run, expected):
    """Directory labels are checked against settings saved by the solver."""
    required = {**expected["settings"], "case": expected["case"], "rotation": expected["rotation"],
                "load_shape": expected["shape"], "load_centre_deg": expected["load_centre_deg"],
                "load_representation": "spatial", "diagnostic_nmax": 10}
    for key, value in required.items():
        if run.summary.get(key) != value:
            run.errors.append(f"Manifest control mismatch: {key}: {run.summary.get(key)} != {value}")


def read_run(path, root, provenance):
    summary = json.loads(path.read_text())
    run = Run(path.parent, path.parent.relative_to(root).as_posix(), summary)
    inputs = [path]
    times = [row["t_kyr"] for row in summary["epochs"]]
    if times != EPOCHS:
        run.errors.append(f"Expected epochs {EPOCHS}, found {times}")
    for key in ("flat_cells", "folded_cells_parent", "folded_cells_mantle"):
        if summary["mesh"].get(key) != 0:
            run.errors.append(f"Invalid/missing mesh check: {key}")
    for t in times:
        profile_path = path.parent / f"profiles_{summary['case']}_{t:g}kyr.npz"
        if not profile_path.exists():
            run.errors.append(f"Missing {profile_path.name}")
            continue
        inputs.append(profile_path)
        with np.load(profile_path, allow_pickle=False) as archive:
            profile = dict(archive)
        run.profiles[t] = profile
        if not profile["found"].size or not profile["found"].all():
            run.errors.append(f"Missing profile points at {t} kyr")
        if not np.isclose(float(profile["t_kyr"]), t):
            run.errors.append(f"Wrong profile epoch at {t} kyr")
        if bool(profile["rate_valid"]) != (t > 0):
            run.errors.append(f"Wrong rate validity at {t} kyr")
        for q in ("U", "V", "W", "along", "across", "N"):
            if not np.isfinite(profile[q + "_m"]).all():
                run.errors.append(f"Nonfinite {q} at {t} kyr")
            rate = profile[q + "_m_per_yr"]
            valid = np.isnan(rate).all() if t == 0 else np.isfinite(rate).all()
            if not valid:
                run.errors.append(f"Invalid {q} rate at {t} kyr")
        if t > 0:
            a, b = float(profile["rate_interval_start_kyr"]), float(profile["rate_interval_end_kyr"])
            if not (a < b and np.isclose(b, t) and np.isclose((b-a)*1000, profile["rate_dt_yr"])):
                run.errors.append(f"Invalid rate interval at {t} kyr")
    log = path.parent / "pbs.stdout"
    if log.exists():
        inputs.append(log)
        text = log.read_text(errors="replace")
        exits = re.findall(r"=== finished .*?, exit (-?\d+) ===", text)
        if not exits or int(exits[-1]) != 0:
            run.errors.append("No successful launcher exit in pbs.stdout")
        if "Traceback (most recent call last)" in text:
            run.errors.append("Python traceback in pbs.stdout")
        outer = re.findall(r"Linear SelfGravitatingGIA_ solve (?:converged|did not converge) due to (\w+) iterations (\d+)", text)
        failures = re.findall(r"Linear (\S+) solve did not converge due to (\w+) iterations (\d+)", text)
        run.solver = {"outer_solves": len(outer), "outer_max_iterations": max((int(n) for _, n in outer), default=0),
                      "inner_caps": sum(name != "SelfGravitatingGIA_" and reason == "DIVERGED_ITS"
                                        for name, reason, _ in failures)}
        if any(not reason.startswith("CONVERGED_") for reason, _ in outer):
            run.errors.append("Outer solver failure")
        if any(name == "SelfGravitatingGIA_" or reason != "DIVERGED_ITS" for name, reason, _ in failures):
            run.errors.append("Solver failure beyond inner iteration caps")
        if not outer:
            run.warnings.append("No outer convergence records")
        run.cost = parse_job(log) or {}
        for key, pattern in (("git", r"=== git:\s+(\S+)"), ("module", r"=== module: (.*?) ===")):
            match = re.search(pattern, text)
            run.solver[key] = match.group(1) if match else None
    else:
        run.warnings.append("No pbs.stdout; completion/provenance not independently checked")
    parameters = path.parent / f"params_{summary['case']}.log"
    if parameters.exists():
        inputs.append(parameters)
        series = np.genfromtxt(parameters, names=True)
        if not np.isclose(series["t_kyr"][-1], 20):
            run.errors.append("Timestep log does not reach 20 kyr")
        if not all(np.isfinite(series[key]).all() for key in series.dtype.names):
            run.errors.append("Nonfinite timestep diagnostics")
    else:
        run.errors.append("Missing timestep log")
    for filename in inputs:
        provenance[str(filename.relative_to(root))] = digest(filename)
    return run


def profile_rows(run, baseline, reference):
    rows = []
    if baseline is not None:
        same_physics(run.summary, baseline.summary)
    for t, profile in run.profiles.items():
        if baseline is not None:
            check_geometry(profile, baseline.profiles[t])
        for rate in (False, True):
            if rate and t == 0:
                continue
            model = fields(profile, rates=rate)
            comparisons = {}
            if baseline is not None and run is not baseline:
                comparisons["baseline"] = fields(baseline.profiles[t], rates=rate)
            if not run.summary["rotation"]:
                comparisons["TABOO"] = reference_fields(reference, t, profile, run.summary, rates=rate)
            for comparison, target in comparisons.items():
                for region, keep in masks(profile, run.summary).items():
                    for q in QUANTITIES:
                        a, b = model[q][keep], target[q][keep]
                        if q != "displacement":
                            a, b = a[:, None], b[:, None]
                        rows.append({"run": run.identity, "scenario": run.scenario, "variant": run.variant,
                                     "comparison": comparison, "t_kyr": t, "quantity": q,
                                     "kind": "rate" if rate else "value", "region": region,
                                     "units": "m/yr" if rate else "m",
                                     "rate_dt_yr": float(profile["rate_dt_yr"]) if rate else None,
                                     **norms(a, b)})
    return rows


def scalar_rows(run, baseline):
    rows = []
    for i, epoch in enumerate(run.summary["epochs"]):
        quantities = ("polar_vector", "absm", "phase", "lod_ms") if run.summary["rotation"] else ("U0", "N0", "Vmax")
        for comparison in ("TABOO", "baseline"):
            if comparison == "baseline" and (baseline is None or run is baseline):
                continue
            target = baseline.summary["epochs"][i] if comparison == "baseline" else epoch
            for q in quantities:
                if comparison == "TABOO" and q == "lod_ms":
                    continue  # Eq.33 and physical LOD currently have different signed conventions.
                suffix = "_ref" if comparison == "TABOO" else ""
                if q == "polar_vector":
                    model = np.array([[epoch["mx"], epoch["my"]]])
                    ref = np.array([[target["mx" + suffix], target["my" + suffix]]])
                else:
                    if target.get(q + suffix) is None:
                        continue
                    model, ref = np.array([[epoch[q]]]), np.array([[target[q + suffix]]])
                    if q == "phase":
                        model = ref + wrapped_degrees(model - ref)
                row = {"run": run.identity, "scenario": run.scenario, "variant": run.variant,
                       "comparison": comparison, "t_kyr": epoch["t_kyr"], "quantity": q,
                       "units": "ms" if q == "lod_ms" else ("deg" if run.summary["rotation"] else "m"),
                       **norms(model, ref)}
                if q == "phase":
                    row["max_percent"] = row["rms_percent"] = None
                rows.append(row)
    return rows


def spectral_rows(run):
    if run.summary["rotation"]:
        return []
    rows = []
    for epoch in run.summary["epochs"]:
        for q in ("U", "V", "N"):
            for degree, model, ref in zip(epoch["degrees"], epoch[q+"_n"], epoch[q+"_n_ref"]):
                rows.append({"run": run.identity, "t_kyr": epoch["t_kyr"], "quantity": q,
                             "degree": degree, "model_m": model, "TABOO_m": ref,
                             "signed_error_m": model-ref,
                             "signed_error_percent": 100*(model/ref-1) if abs(ref) > 1e-12 else None})
    return rows


def mesh_rows(runs):
    result = []
    for scenario in SCENARIOS:
        levels = [runs.get(scenario + "/" + name) for name in ("mesh500", "baseline", "mesh125")]
        if any(run is None or run.errors for run in levels):
            continue
        coarse, base, fine = levels
        for other in (coarse, fine):
            same_physics(base.summary, other.summary)
            changes = [key for key in BASELINE if base.summary[key] != other.summary[key]]
            if changes != ["mesh_h_km"]:
                raise ValueError(f"Mesh sequence changes other controls: {other.identity}: {changes}")
        for t in EPOCHS:
            check_geometry(coarse.profiles[t], base.profiles[t])
            check_geometry(base.profiles[t], fine.profiles[t])
            for rate in (False, True):
                if rate and t == 0:
                    continue
                data = [fields(run.profiles[t], rates=rate) for run in levels]
                for q in ("displacement", "N"):
                    keep = masks(base.profiles[t], base.summary)["near"]
                    values = [d[q][keep] for d in data]
                    if q != "displacement":
                        values = [v[:, None] for v in values]
                    first, second = values[0]-values[1], values[1]-values[2]
                    n1, n2 = np.linalg.norm(first), np.linalg.norm(second)
                    alignment = float(np.sum(first*second)/(n1*n2)) if n1*n2 > 0 else None
                    order = float(np.log2(n1/n2)) if n1 > n2 > 0 and alignment is not None and alignment > .9 else None
                    result.append({"scenario": scenario, "t_kyr": t, "quantity": q,
                                   "kind": "rate" if rate else "value", "difference_alignment": alignment,
                                   "apparent_order_if_aligned": order,
                                   "coarse_to_base_rms": float(n1/np.sqrt(len(first))),
                                   "base_to_fine_rms": float(n2/np.sqrt(len(second))),
                                   **{"base_to_fine_"+key: value for key, value in norms(values[1], values[2]).items()}})
    return result


def low_mode_fit(values, unit_vectors):
    """Diagnose a monopole/dipole-shaped difference on the saved lines.

    This is a least-squares pattern fit on profile points, not an area-weighted
    spherical harmonic projection. Nothing is subtracted from scored errors.
    """
    y = np.asarray(values).ravel()
    design = np.column_stack([np.ones(y.size), np.asarray(unit_vectors).reshape(-1, 3)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    remainder = y - design @ coefficients
    energy = float(y @ y)
    return {**dict(zip(("constant", "dipole_x", "dipole_y", "dipole_z"), map(float, coefficients))),
            "fraction_squared_norm_explained": float(1 - remainder @ remainder / energy) if energy > 0 else None,
            "raw_rms": float(np.sqrt(np.mean(y**2))),
            "unfitted_remainder_rms": float(np.sqrt(np.mean(remainder**2)))}


def low_mode_rows(runs):
    rows = []
    for scenario in ("cap-off", "cap-on"):
        base, fine = runs.get(scenario+"/baseline"), runs.get(scenario+"/mesh125")
        if base is None or fine is None or base.errors or fine.errors:
            continue
        for t in EPOCHS:
            a, b = fine.profiles[t], base.profiles[t]
            check_geometry(a, b)
            for rate in (False, True):
                if t == 0 and rate:
                    continue
                key = "N_m_per_yr" if rate else "N_m"
                rows.append({"scenario": scenario, "t_kyr": t, "kind": "rate" if rate else "value",
                             "units": "m/yr" if rate else "m", **low_mode_fit(a[key]-b[key], b["unit_vectors"])})
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output, name):
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def plots(output, runs, profiles, reference):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), layout="constrained")
    matrices = []
    panels = [(q, kind) for kind in ("value", "rate") for q in ("displacement", "N")]
    for q, kind in panels:
        matrix = np.full((len(SCENARIOS), len(VARIANTS)), np.nan)
        for i, scenario in enumerate(SCENARIOS):
            for j, variant in enumerate(VARIANTS):
                values = [r["max_percent"] for r in profiles if r["scenario"] == scenario and r["variant"] == variant
                          and r["comparison"] == "baseline" and r["region"] == "near"
                          and r["quantity"] == q and r["kind"] == kind and r["max_percent"] is not None]
                if values:
                    matrix[i, j] = max(values)
        matrices.append(matrix)
    positive = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in matrices])
    if positive.size:
        norm = LogNorm(vmin=max(float(positive.min()), 1e-8), vmax=max(float(positive.max()), 1e-7))
        for ax, matrix, (q, kind) in zip(axes.flat, matrices, panels):
            im = ax.imshow(np.ma.masked_invalid(matrix), norm=norm, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=30, ha="right")
            ax.set_yticks(range(len(SCENARIOS)), SCENARIOS)
            ax.set_title(f"{'Gravitational geoid' if q == 'N' else 'Displacement vector'}: {kind}")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:.3g}" if np.isfinite(value) else "—", ha="center", va="center",
                        color="white" if np.isfinite(value) and value < np.sqrt(norm.vmin*norm.vmax) else "black", fontsize=8)
        fig.colorbar(im, ax=list(axes.flat), label="Largest peak-normalized difference (%) across seven epochs; rates omit t=0")
    fig.suptitle("Sensitivity to one parameter: profiles within 25° of load centre")
    save(fig, output, "parameter_sensitivity")

    for scenario in ("cap-on", "disc-on"):
        available = [run for run in runs.values() if run.scenario == scenario and not run.errors]
        if not available:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for run in available:
            epochs = run.summary["epochs"]
            t = np.array([e["t_kyr"] for e in epochs])
            amplitude = np.array([e["absm"] for e in epochs])
            ref = np.array([e["absm_ref"] for e in epochs])
            phase = np.array([wrapped_degrees(e["phase"]-e["phase_ref"]) for e in epochs])
            axes[0].plot(t, amplitude, ".-", label=run.variant)
            axes[1].plot(t, 100*(amplitude/ref-1), ".-")
            axes[2].plot(t, phase, ".-")
        epochs = available[0].summary["epochs"]
        axes[0].plot([e["t_kyr"] for e in epochs], [e["absm_ref"] for e in epochs], "k--", label="TABOO")
        for ax, title in zip(axes, ("Amplitude (degrees)", "Signed amplitude error (%)", "Phase error (degrees)")):
            ax.set_title(title)
            ax.set_xlabel("Time (kyr)")
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{scenario}: raw TABOO comparison, without amplitude correction")
        save(fig, output, scenario + "_polar_motion")

    for scenario in SCENARIOS:
        sequence = [runs.get(scenario + "/" + name) for name in ("mesh500", "baseline", "mesh125")]
        if any(run is None or run.errors for run in sequence):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for run in sequence:
            for ax, q, kind in zip(axes.flat, ("displacement", "N", "displacement", "N"), ("value", "value", "rate", "rate")):
                rows = [r for r in profiles if r["run"] == run.identity and r["comparison"] == "baseline"
                        and r["quantity"] == q and r["kind"] == kind and r["region"] == "near"]
                if rows:
                    ax.plot([r["t_kyr"] for r in rows], [r["max_percent"] for r in rows], ".-", label=f"{run.summary['mesh_h_km']:g} km − 250 km")
                ax.set_title(f"{q}, {kind}")
                ax.set_xlabel("Time (kyr)")
                ax.set_ylabel("Peak-normalized difference (%)")
                ax.grid(alpha=.25)
        axes[0, 0].legend()
        fig.suptitle(f"{scenario}: differences from 250 km baseline, within 25°")
        save(fig, output, scenario + "_mesh_sensitivity")

    for shape in ("cap", "disc"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for i, rotation in enumerate(("off", "on")):
            scenario = f"{shape}-{rotation}"
            base = runs.get(scenario + "/baseline")
            if base is None or base.errors:
                continue
            for name in ("mesh500", "baseline", "mesh125"):
                run = runs.get(scenario + "/" + name)
                if run is None or run.errors:
                    continue
                profile = run.profiles[20.]
                x = profile["coordinate_deg"]
                keep = np.abs(x) <= 25 + 1e-9
                for ax, q in zip(axes[i], ("U", "N")):
                    ax.plot(x[keep], profile[q+"_m"][0, keep], label=f"{run.summary['mesh_h_km']:g} km")
            if not base.summary["rotation"]:
                profile = base.profiles[20.]
                x = profile["coordinate_deg"]
                keep = np.abs(x) <= 25 + 1e-9
                ref = reference_fields(reference, 20., profile, base.summary)
                for ax, q in zip(axes[i], ("U", "N")):
                    ax.plot(x[keep], ref[q][0, keep], "k--", label="TABOO 2–256")
            for ax, q in zip(axes[i], ("U", "N")):
                ax.set_title(f"{scenario}: {'radial displacement' if q == 'U' else 'gravitational geoid'}")
                ax.set_xlabel("Profile coordinate (degrees)")
                ax.set_ylabel("m")
                ax.axvline(10, color="grey", ls=":", lw=.8)
                if rotation == "on":
                    ax.axvline(-10, color="grey", ls=":", lw=.8)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        fig.suptitle(f"{shape}: first saved profile at 20 kyr; all-profile errors are in the CSVs")
        save(fig, output, shape + "_mesh_profiles")

    base, fine = runs.get("cap-off/baseline"), runs.get("cap-off/mesh125")
    if base is not None and fine is not None and not base.errors and not fine.errors:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for ax, t in zip(axes, (10., 20.)):
            for run in (base, fine):
                profile = run.profiles[t]
                x = profile["coordinate_deg"]
                keep = x <= 25 + 1e-9
                ax.plot(x[keep], 1000*profile["N_m_per_yr"][0, keep], label=f"{run.summary['mesh_h_km']:g} km")
            profile = base.profiles[t]
            ref = reference_fields(reference, t, profile, base.summary, rates=True)
            ax.plot(x[keep], 1000*ref["N"][0, keep], "k--", label="TABOO, matched interval")
            ax.set_title(f"{t:g} kyr")
            ax.set_xlabel("Colatitude (degrees)")
            ax.set_ylabel("Gravitational geoid rate (mm/yr)")
            ax.legend()
            ax.grid(alpha=.2)
        fig.suptitle("Cap-off geoid rates: small differences of saved potentials need separate accuracy checks")
        save(fig, output, "cap_off_geoid_rates")


def format_number(value):
    return "—" if value is None else f"{value:.4g}"


def report(output, runs, profiles, scalars, coverage):
    lines = ["# Spada study analysis", "", "## Coverage and checks", "",
             f"{len(runs)} non-smoke summaries; {sum(not r.errors for r in runs.values())} pass the local audit.", ""]
    for row in coverage:
        if row["status"] != "complete":
            lines.append(f"- {row['run']}: {row['status']}; {row.get('detail', '')}")
    for run in runs.values():
        for message in run.errors + run.warnings:
            lines.append(f"- {run.identity}: {message}")
    total = sum(run.cost.get("su", 0) for run in runs.values())
    lines += ["", f"Completed summary-bearing runs used **{total:,.2f} SU** (PBS accounting where present).",
              "Failed runs are listed separately in coverage.csv; their cost is excluded from this total.", "",
              "## Individual parameter effects", "",
              "Largest change across saved epochs and all sampled profiles within 25° of the load centre. "
              "Percentages divide the maximum absolute scalar/vector difference by the matching baseline's peak "
              "at that epoch, avoiding division by local zero crossings. These are line-profile norms, not surface-area norms.", "",
              "| Run | Displacement (%) | Geoid (%) | Displacement rate (%) | Geoid rate (%) | SU |",
              "|---|---:|---:|---:|---:|---:|"]
    for run in runs.values():
        if run.variant == "baseline":
            continue
        values = []
        for kind in ("value", "rate"):
            for q in ("displacement", "N"):
                matching = [r["max_percent"] for r in profiles if r["run"] == run.identity and r["comparison"] == "baseline"
                            and r["quantity"] == q and r["kind"] == kind and r["region"] == "near" and r["max_percent"] is not None]
                values.append(max(matching) if matching else None)
        lines.append(f"| {run.identity} | " + " | ".join(format_number(v) for v in [*values, run.cost.get("su")]) + " |")
    lines += ["", "## Agreement with TABOO", "",
              "| Run | Near-load displacement max error (%) | Near-load geoid max error (%) | Polar-vector max error (%) | Max phase error (deg) |",
              "|---|---:|---:|---:|---:|"]
    for run in runs.values():
        if run.variant not in ("baseline", "mesh125"):
            continue
        values = []
        for q in ("displacement", "N"):
            matching = [r["max_percent"] for r in profiles if r["run"] == run.identity and r["comparison"] == "TABOO"
                        and r["quantity"] == q and r["kind"] == "value" and r["region"] == "near" and r["max_percent"] is not None]
            values.append(max(matching) if matching else None)
        for q, key in (("polar_vector", "max_percent"), ("phase", "max_abs")):
            matching = [r[key] for r in scalars if r["run"] == run.identity and r["comparison"] == "TABOO" and r["quantity"] == q]
            values.append(max(matching) if matching else None)
        lines.append(f"| {run.identity} | " + " | ".join(format_number(v) for v in values) + " |")
    lines += ["", "## Low-degree diagnostics", "",
              "These are reconstructed from degrees 2–10 and therefore describe large-scale response. "
              "They do not resolve the local load-edge errors in the direct profile table above.", "",
              "| Run | Max U(0) error (%) | Max N(0) error (%) | Max Vmax error (%) | Max polar-vector change from baseline (%) |",
              "|---|---:|---:|---:|---:|"]
    for run in runs.values():
        if run.variant not in ("baseline", "mesh125"):
            continue
        values = []
        for q in ("U0", "N0", "Vmax", "polar_vector"):
            comparison = "baseline" if q == "polar_vector" else "TABOO"
            matching = [r["max_percent"] for r in scalars if r["run"] == run.identity
                        and r["comparison"] == comparison and r["quantity"] == q]
            values.append(max(matching) if matching else None)
        lines.append(f"| {run.identity} | " + " | ".join(format_number(v) for v in values) + " |")
    lines += ["", "## Interpretation limits", "",
              "- The audit covers transferred summaries, profiles, timestep logs and PBS logs. Surface HDF5 "
              "and full checkpoint contents have not been transferred or audited by this script.",
              "- The runs vary one parameter at a time. They measure individual sensitivities, not parameter interactions.",
              "- Lithosphere subdivision regenerates the mesh, so its differences include remeshing effects "
              "as well as layer count. Mesh hashes and seeds are retained in runs.csv.",
              "- Available 125 km runs provide observed refinement differences, not exact solutions. "
              "Missing 125 km scenarios are omitted. mesh_convergence.csv records changes over 500–250–125 km; "
              "an apparent order is reported only for decreasing, strongly aligned differences (cosine > 0.9). "
              "Three meshes alone do not prove asymptotic convergence.",
              "- Spatial TABOO comparisons use degrees 2–256; sharp-disc edge errors include finite reference cutoff effects. "
              "edge-excluded metrics omit points within 1° of the 10° load edge. Profiles are sampled slightly below the surface.",
              "- Reference rates are averaged over each saved backward-difference interval using the analytic "
              "reference displacement/geoid at its endpoints. Variant-minus-baseline rates can cover different intervals; "
              "dt2 differences include this averaging effect. Initial jump rates are excluded.",
              "- Polar-motion errors are raw. The known C−A convention difference is not silently corrected. "
              "There is no independent rotating spatial reference in this archive.",
              "- Physical signed LOD is compared between numerical runs only. No signed Eq.33 error is scored "
              "while its convention difference remains unresolved.",
              "- Inner DIVERGED_ITS caps are recorded as preconditioner limits, separately from outer convergence. "
              "Requested inner tolerance is not necessarily attained.",
              "- low_mode_diagnostics.csv fits monopole/dipole patterns to the 125-minus-250 km geoid differences "
              "on saved profiles. This diagnoses frame/low-degree contamination; no fitted component is removed "
              "from any reported error. In particular, cap-off geoid rates at 10 kyr need investigation. "
              "The Spada setup does not enable the library's centre-of-mass constraints; whether enforcing that "
              "frame resolves the observed dipole requires a controlled follow-up run.",
              "- Individual changes do not form a rigorous combined error bound. Coarser timestep/layer/DtN checks "
              "alone do not prove the baseline has reached a limiting solution. Absolute and RMS errors are retained "
              "in the CSVs for judging an application-specific accuracy target.", "",
              "Reproduce with `python3 analyse_spada_study.py outputs/spada-study`. Input SHA256 hashes, "
              "source/module strings from PBS logs and all metrics are written beside this report."]
    (output / "analysis.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("outputs/spada-study"))
    parser.add_argument("--output", type=Path, default=Path("outputs/spada-study-analysis"))
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference.npz"))
    args = parser.parse_args(argv)
    paths = sorted(args.root.rglob("summary_*.json"))
    if not paths:
        parser.error(f"No summaries under {args.root}")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"reference.npz": digest(args.reference), "analysis_script": digest(Path(__file__)),
                  "analysis_environment": {"python": sys.version, "numpy": np.__version__,
                                           "matplotlib": matplotlib.__version__},
                  "analysis_dependencies": {name: digest(Path(__file__).with_name(name)) for name in
                                            ("estimate_spada_cost.py", "plot_spada_profiles.py",
                                             "spada_loads.py", "spada_study.py")}}
    runs = {}
    for path in paths:
        if json.loads(path.read_text()).get("smoke"):
            continue
        run = read_run(path, args.root, provenance)
        runs[run.identity] = run
    coverage = []
    for expected in (r for r in run_matrix() if not r["smoke"]):
        identity = expected["id"]
        run = runs.get(identity)
        if run is not None:
            validate_controls(run, expected)
        row = {"run": identity, "status": "complete" if run and not run.errors else "invalid" if run else "missing summary"}
        if run is None:
            log = args.root / identity / "pbs.stdout"
            if log.exists():
                provenance[str(log.relative_to(args.root))] = digest(log)
                cost = parse_job(log)
                row.update({"detail": "PBS exit " + str(cost.get("exit")) if cost else "See PBS log", "su": cost.get("su") if cost else None})
        coverage.append(row)
    profiles, scalars, spectra = [], [], []
    with np.load(args.reference, allow_pickle=False) as reference:
        for run in runs.values():
            if run.errors:
                continue
            base = runs.get(run.scenario + "/baseline")
            if base is None or base.errors:
                run.warnings.append("No valid matching baseline")
                base = None
            if base is not None:
                changes = [key for key in BASELINE if run.summary[key] != base.summary[key]]
                if run.variant != "baseline" and not changes:
                    run.warnings.append("Settings identical to baseline")
            profiles.extend(profile_rows(run, base, reference))
            scalars.extend(scalar_rows(run, base))
            spectra.extend(spectral_rows(run))
    run_rows = [{"run": r.identity, **{key: r.summary[key] for key in BASELINE},
                 "errors": "; ".join(r.errors), "warnings": "; ".join(r.warnings),
                 "su": r.cost.get("su"), "wall_hours": r.cost.get("wall_hours"),
                 "mantle_cells": r.summary["mesh"]["cells_mantle"],
                 "mesh_md5": r.summary["mesh"].get("md5"),
                 "mesh_seed": r.summary["mesh"].get("random_seed"), **r.solver} for r in runs.values()]
    for name, rows in (("runs", run_rows), ("coverage", coverage), ("profiles", profiles),
                       ("scalars", scalars), ("spectra", spectra), ("mesh_convergence", mesh_rows(runs)),
                       ("low_mode_diagnostics", low_mode_rows(runs))):
        write_csv(args.output / f"{name}.csv", rows)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with np.load(args.reference, allow_pickle=False) as reference:
        plots(args.output, runs, profiles, reference)
    report(args.output, runs, profiles, scalars, coverage)
    print(f"Analysed {len(runs)} production summaries; {sum(bool(r.errors) for r in runs.values())} invalid; output: {args.output}")


if __name__ == "__main__":
    main()
