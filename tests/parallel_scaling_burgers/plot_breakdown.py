"""Stacked cost breakdown per solve for the Burgers weak-scaling jobs.

One panel per solver configuration, x is the cubed-sphere level, y is the
wall time of one time step's solve, stacked by where the time goes. The
dashed line is the ``SNESSolve`` time per solve. Each x position also carries
the GAMG V-cycle count per step, because a cheap solve bought by fewer
V-cycles reads differently from a cheap solve bought by cheaper V-cycles.

The input is PETSc's nested log (``-log_view :file.xml:ascii_xml``), which
``pbs_job.sh`` writes when submitted with ``LOG_FORMAT=ascii_xml``. The flat
text log is unusable here for two reasons. Its events nest inside each other
(the Schur complement's ``MatMult`` contains the internal-variable block
solves, ``PCApply`` contains the GAMG ``PCSetUp``), so summing events double
counts. And it reports each event's maximum over ranks, and different ranks
are slowest in different events, so the maxima do not add up to the total
either. The nested log gives a call tree with every node's own time averaged
over ranks, which is what a stacked figure needs.

Usage::

    python plot_breakdown.py [--results DIR] [--tag nested] [--out FILE]

A second figure, ``iterations.pdf`` next to ``--out``, puts V-cycles per
step and time per solve side by side, one line per solver.
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "3d_sphere_burgers"))
from solver_configs import displacement_ksp_prefix  # noqa: E402

LEVELS = (5, 6, 7)

# Panel order and titles. The substituted solver is the reference and is
# measured inside every job; its panel averages the four measurements.
SOLVERS = [
    ("substituted", "Substituted (reference)"),
    ("static-condensation", "Static condensation"),
    ("schur-substituted", "Schur, GAMG on $\\eta_{\\mathrm{eff}}$"),
    ("schur-a11", "Schur, GAMG on $A_{uu}$"),
    ("multiplicative", "Multiplicative (shipped)"),
]

# Line colour and marker per solver for the iterations-against-time figure.
STYLE = {
    "substituted": ("#000000", "o"),
    "static-condensation": ("#2ca02c", "s"),
    "schur-substituted": ("#1f77b4", "^"),
    "schur-a11": ("#9467bd", "v"),
    "multiplicative": ("#d62728", "X"),
}

# Cost bands, drawn from the bottom of each stack in this order. The key is
# matched by ``classify`` below; the label goes in the legend.
BANDS = [
    ("assembly", "Operator assembly (PC matrix, Slate condensation)", "#9ecae1"),
    ("gamg_setup", "GAMG setup", "#3182bd"),
    ("vcycle", "GAMG V-cycles (smooth, residual, transfer)", "#08519c"),
    ("matvec", "Krylov: operator action", "#e6550d"),
    ("vecwork", "Krylov: vector work", "#fdae6b"),
    ("blocksolve", "Internal-variable block solves", "#756bb1"),
    ("sc_elim", "Static condensation: eliminate, back-substitute", "#74c476"),
    ("residual", "Residual evaluation", "#c7c7c7"),
    ("other", "Other", "#f0f0f0"),
]

# Times everywhere, sized for a paper figure.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


# ── Nested log parsing ──────────────────────────────────────────────────────
def _time(node):
    """Rank-averaged time of one node of the nested log, as a percentage.

    PETSc's nested XML writes every timer as a percentage of the run's
    total time (the ``totaltime`` field of the timer tree, in seconds).
    ``stage_breakdown`` converts to seconds once, after summing.
    """
    t = node.find("time")
    if t is None:
        return 0.0
    return float(t.findtext("value") or t.findtext("avgvalue") or 0.0)


def _children(node):
    ev = node.find("events")
    return [] if ev is None else ev.findall("event")


def _name(node):
    return node.findtext("name") or ""


def classify(node, parent_name):
    """Return the band of a node, or ``None`` to keep descending.

    A node that gets a band contributes its whole subtree time to it, so the
    rules must name events that do not contain events of another band.
    The one exception is handled by the caller: an assembled ``MatMult`` has
    no children of interest and is operator action, while the Schur
    complement's ``MatMult`` contains block solves and form actions and is
    descended into.
    """
    name = _name(node)
    if name == "SNESFunctionEval":
        return "residual"
    if name == "PCSetUp":
        # Directly under SNESSolve this is Firedrake assembling the matrix
        # the preconditioner runs on (and, for static condensation, the Slate
        # condensation itself). Anywhere deeper it is the GAMG hierarchy
        # build, which the python PC triggers from its first apply.
        return "assembly" if parent_name == "SNESSolve" else "gamg_setup"
    if name in ("MatSOR", "MatResidual", "MatMultAdd", "MatMultTranspose"):
        return "vcycle"
    if name == "MatSolve" or name == "PCApplyOnBlocks":
        return "blocksolve"
    if name.startswith("firedrake.matrix_free.operators.ImplicitMatrixContext.mult"):
        return "matvec"
    if name.startswith("Vec"):
        return "vecwork"
    if name in ("SCForwardElim", "SCBackSub"):
        return "sc_elim"
    return None


def _accumulate(node, parent_name, bands):
    """Walk one subtree, adding classified time into ``bands``."""
    band = classify(node, parent_name)
    name = _name(node)
    if band is None and name == "MatMult":
        # Assembled operator: a leaf apart from vector scatters. The Schur
        # complement's MatMult holds block solves and form actions, so it is
        # descended into and only its own time is operator action.
        kids = [k for k in _children(node) if _name(k) != "self"]
        if not any(classify(k, name) in ("blocksolve", "matvec") or _name(k).startswith("KSPSolve")
                   for k in kids):
            band = "matvec"
    if band is not None:
        bands[band] += _time(node)
        return
    if name == "self":
        bands["other"] += _time(node)
        return
    kids = _children(node)
    if not kids:
        bands["other"] += _time(node)
        return
    for kid in kids:
        _accumulate(kid, name, bands)
    if name == "MatMult":
        # Own time of the Schur complement's MatMult is operator action, not other.
        selfnode = [k for k in kids if _name(k) == "self"]
        for s in selfnode:
            bands["other"] -= _time(s)
            bands["matvec"] += _time(s)


def stage_breakdown(xml_path, stage_name):
    """Per-solve seconds in each band for one PETSc stage of one nested log.

    Returns ``None`` when the stage is missing. The bands are normalised by
    the number of ``SNESSolve`` calls; the nested log prunes nodes below
    one percent of the run time, and the pruned remainder lands in "other"
    so that the bands always add up to the SNESSolve time.
    """
    root = ET.parse(xml_path).getroot()
    tree = root.find(".//timertree")
    # Percent of the run to seconds.
    total_seconds = float(tree.findtext("totaltime"))
    stage = None
    for ev in tree.iter("event"):
        if _name(ev) == stage_name:
            stage = ev
            break
    if stage is None:
        return None
    snes = None
    for ev in stage.iter("event"):
        if _name(ev) == "SNESSolve":
            snes = ev
            break
    if snes is None:
        return None
    # The stage is entered once per time step, and each entry holds one
    # SNESSolve, so the stage's call count is the number of solves.
    ncalls = stage.find("ncalls")
    n = float(ncalls.findtext("value")) if ncalls is not None else 1.0
    bands = {key: 0.0 for key, _, _ in BANDS}
    for kid in _children(snes):
        _accumulate(kid, "SNESSolve", bands)
    total = _time(snes)
    bands["other"] = total - sum(v for k, v in bands.items() if k != "other")
    scale = total_seconds / 100.0 / n
    out = {k: v * scale for k, v in bands.items()}
    out["total"] = total * scale
    return out


# ── Iteration counts and mesh sizes from the driver output ──────────────────
def read_output(out_path, config):
    """V-cycles per step for both solvers and the mesh sizes of one job.

    For the multiplicative preset the displacement KSP runs once per outer
    sweep, so its count per step is the sum over the sweeps of that step.
    Sweeps are attributed to steps by the outer ``CoupledInternalVariable_``
    convergence line that closes each step.
    """
    prefixes = {"substituted": displacement_ksp_prefix("substituted"),
                config: displacement_ksp_prefix(config)}
    per_step = {name: [] for name in prefixes}
    pending = {name: 0 for name in prefixes}
    meta = {}
    with open(out_path) as f:
        for line in f:
            m = re.match(r"\s+Linear (\S+) solve (?:converged|did not converge) due to \S+ iterations (\d+)", line)
            if m:
                for name, prefix in prefixes.items():
                    if m.group(1) == prefix:
                        pending[name] += int(m.group(2))
                continue
            m = re.match(r"^step (\d+) (\S+): ", line)
            if m:
                name = m.group(2)
                per_step[name].append(pending[name])
                pending[name] = 0
                continue
            m = re.match(r"level (\d+) config \S+: displacement dofs (\d+), internal variable dofs (\d+), ranks (\d+)", line)
            if m:
                meta = {"level": int(m.group(1)), "u_dofs": int(m.group(2)),
                        "m_dofs": int(m.group(3)), "ranks": int(m.group(4))}
    return per_step, meta


def warm_mean(values):
    """Mean over the warm steps (every step after the first)."""
    v = values[1:] if len(values) > 1 else values
    return float(np.mean(v)) if v else np.nan


# ── Data assembly ───────────────────────────────────────────────────────────
def collect(results, tag):
    """Breakdown, V-cycles and mesh sizes for every (solver, level).

    The substituted solver's entry at each level averages the four jobs that
    measured it, so its panel is one record of the reference.
    """
    suffix = f"_{tag}" if tag else ""
    data = {}
    for level in LEVELS:
        sub_breaks, sub_cycles = [], []
        for config, _ in SOLVERS[1:]:
            xml = results / f"profile_{level}_{config}{suffix}.xml"
            out = results / f"level_{level}_{config}{suffix}_full.out"
            if not (xml.exists() and out.exists()):
                continue
            cycles, meta = read_output(out, config)
            bd = stage_breakdown(xml, f"burgers_{config}_solve")
            if bd is not None:
                data[(config, level)] = {"bands": bd, "cycles": warm_mean(cycles[config]), **meta}
            bd_sub = stage_breakdown(xml, "burgers_substituted_solve")
            if bd_sub is not None:
                sub_breaks.append(bd_sub)
                sub_cycles.append(warm_mean(cycles["substituted"]))
                data.setdefault(("substituted", level), {**meta})
        if sub_breaks:
            keys = sub_breaks[0].keys()
            data[("substituted", level)]["bands"] = {
                k: float(np.mean([b[k] for b in sub_breaks])) for k in keys}
            data[("substituted", level)]["cycles"] = float(np.mean(sub_cycles))
    return data


def level_label(rec):
    """Three-line tick label: level, ranks, displacement dofs."""
    u = rec["u_dofs"] / 1e6
    dofs = f"{u:.1f}M" if u < 10 else f"{u:.0f}M"
    return f"level {rec['level']}\n{rec['ranks']} ranks\n{dofs} u-dofs"


# ── Drawing ─────────────────────────────────────────────────────────────────
def panel_letter(ax, letter):
    ax.text(0.06, 0.93, letter, transform=ax.transAxes, ha="center", va="center",
            fontsize=14, zorder=8,
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="lightgrey", edgecolor="black"))


def draw(data, out_path):
    """Five panels. The first four share one y axis; the multiplicative
    preset is an order of magnitude more expensive and gets its own scale,
    with its axis on the right and the difference stated in its title."""
    xs = np.arange(len(LEVELS))
    fig, axes = plt.subplots(1, len(SOLVERS), figsize=(4.4 * len(SOLVERS), 5.6))
    shared = axes[:-1]
    title_box = dict(boxstyle="round,pad=0.4", facecolor="lightblue", edgecolor="black")

    tallest_shared = 0.0
    tallest_own = 0.0
    for si, ((config, title), ax) in enumerate(zip(SOLVERS, axes)):
        recs = [data.get((config, lv)) for lv in LEVELS]
        series = [np.array([(r["bands"][key] if r else 0.0) for r in recs]) for key, _, _ in BANDS]
        ax.stackplot(xs, *series, colors=[c for _, _, c in BANDS],
                     edgecolor="white", linewidth=0.6, zorder=3)
        total = np.array([r["bands"]["total"] if r else np.nan for r in recs])
        ax.plot(xs, total, color="black", lw=1.2, ls="--", zorder=6)
        if ax in shared:
            tallest_shared = max(tallest_shared, np.nanmax(total))
        else:
            tallest_own = max(tallest_own, np.nanmax(total))

        own_scale = "\n(own scale)" if ax not in shared else ""
        ax.set_title(title + own_scale, bbox=title_box, pad=14)
        panel_letter(ax, "ABCDE"[si])
        ax.set_xticks(xs)
        labels = [level_label(r) if r else f"level {lv}" for r, lv in zip(recs, LEVELS)]
        ax.set_xticklabels(labels)
        ax.set_xlim(xs[0], xs[-1])
        ax.margins(x=0)
        ax.get_xticklabels()[0].set_horizontalalignment("left")
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        if si == 0:
            ax.set_ylabel("time per solve (s)")
        if si == len(SOLVERS) // 2:
            ax.set_xlabel("cubed-sphere level / ranks / displacement dofs / GAMG V-cycles per step",
                          labelpad=44)

    top_shared = tallest_shared * 1.12
    top_own = tallest_own * 1.12
    for (config, _), ax in zip(SOLVERS, axes):
        top = top_shared if ax in shared else top_own
        ax.set_ylim(0, top)
        if ax in shared and ax is not axes[0]:
            ax.tick_params(labelleft=False)
        if ax not in shared:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("time per solve (s)", rotation=270, labelpad=18)
        # V-cycle row under the tick labels, at the tick in x and a fixed
        # distance below the axis in y.
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for x, lv in zip(xs, LEVELS):
            r = data.get((config, lv))
            if r is None or np.isnan(r["cycles"]):
                continue
            ha = "left" if x == xs[0] else "right" if x == xs[-1] else "center"
            ax.text(x, -0.20, f"{r['cycles']:.0f} V-cycles", transform=trans, ha=ha,
                    va="top", fontsize=11.5, color="0.3", clip_on=False, zorder=7)

    # Grid as explicit lines over the stacks: ax.grid's zorder is not
    # honoured against a stackplot PolyCollection.
    fig.canvas.draw()
    for ax in axes:
        top = ax.get_ylim()[1]
        for yt in ax.get_yticks():
            if 0 <= yt <= top:
                ax.axhline(yt, color="0.4", lw=0.7, alpha=0.6, zorder=5)
        for xt in xs:
            ax.axvline(xt, color="0.4", lw=0.7, alpha=0.6, zorder=5)

    handles = [Patch(facecolor=c, edgecolor="white", label=lab) for _, lab, c in BANDS]
    handles.append(Line2D([0], [0], color="black", lw=1.2, ls="--", label="SNESSolve total"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.24),
               columnspacing=1.4, handletextpad=0.5, frameon=True, fancybox=False,
               edgecolor="black", framealpha=1.0)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2,
                metadata={"CreationDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    print(f"wrote {out_path}")


def draw_iterations(data, out_path):
    """V-cycles per step against time per solve, one line per solver.

    The two panels invert the story. Every route with the internal
    variables eliminated needs the reference's V-cycle count within a few
    cycles, and the count is flat with level for all of them. The time per
    solve still spans a factor of 2.6 between those routes. Iteration counts
    alone say nothing about which route is cheapest; the breakdown figure
    says where the time goes.

    Both axes are linear and scaled to the four eliminated routes. The
    multiplicative preset is an order of magnitude off in both quantities;
    drawing it would squash the others, so its values are written at the top
    of each panel instead.
    """
    xs = np.arange(len(LEVELS))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    offscale = {}
    for config, title in SOLVERS:
        recs = [data.get((config, lv)) for lv in LEVELS]
        if not any(r and "bands" in r for r in recs):
            continue
        cycles = [r["cycles"] if r else np.nan for r in recs]
        total = [r["bands"]["total"] if r and "bands" in r else np.nan for r in recs]
        if config == "multiplicative":
            offscale[title] = (cycles, total)
            continue
        colour, marker = STYLE[config]
        for ax, ys in zip(axes, (cycles, total)):
            ax.plot(xs, ys, color=colour, marker=marker, markersize=8, lw=1.9,
                    label=title, zorder=4)
    labels = [level_label(r) if (r := data.get(("substituted", lv))) else f"level {lv}" for lv in LEVELS]
    for ax, ylabel, letter in zip(axes, ("GAMG V-cycles per step", "time per solve (s)"), "AB"):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.3, len(LEVELS) - 0.7)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=0.3, zorder=0)
        panel_letter(ax, letter)
    # Headroom for the off-scale note.
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    for title, (cycles, total) in offscale.items():
        colour, _ = STYLE["multiplicative"]
        fmt_c = ", ".join(f"{c:.0f}" for c in cycles)
        fmt_t = ", ".join(f"{t:.0f}" for t in total)
        axes[0].text(0.97, 0.86, f"{title}: {fmt_c} V-cycles, off scale",
                     transform=axes[0].transAxes, ha="right", va="top",
                     color=colour, fontsize=12)
        axes[1].text(0.97, 0.86, f"{title}: {fmt_t} s, off scale",
                     transform=axes[1].transAxes, ha="right", va="top",
                     color=colour, fontsize=12)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel("time per solve (s)", rotation=270, labelpad=20)
    fig.supxlabel("cubed-sphere level / ranks / displacement dofs", y=-0.10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.24),
               frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2,
                metadata={"CreationDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    print(f"wrote {out_path}")


def print_table(data):
    """Per-solve band seconds for every (solver, level), for the notes."""
    keys = [k for k, _, _ in BANDS] + ["total"]
    print("solver, level, V-cycles, " + ", ".join(keys))
    for config, _ in SOLVERS:
        for lv in LEVELS:
            r = data.get((config, lv))
            if r is None or "bands" not in r:
                continue
            print(f"{config:20s} L{lv} {r['cycles']:5.1f} " +
                  " ".join(f"{r['bands'][k]:6.2f}" for k in keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--tag", default="nested", help="RUN_TAG of the nested-log jobs")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "breakdown.pdf")
    args = ap.parse_args()
    data = collect(args.results, args.tag)
    if not data:
        raise SystemExit(f"no nested logs with tag {args.tag!r} under {args.results}")
    print_table(data)
    draw(data, args.out)
    draw_iterations(data, args.out.with_name("iterations.pdf"))


if __name__ == "__main__":
    main()
