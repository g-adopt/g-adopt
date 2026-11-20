"""The scenarios of the GIA cases, the march loop and the diagnostics of one step.

The drivers of `tests/gia` (`gia_forward.py` and the others) build the
coupled self-gravitating GIA solver through `gia_cases.build_new`, march it
over a fixed list of time steps, and write one row of diagnostics per step.
This module holds the parts they share, so that every driver runs the same
scenario in the same way:

- `SEA` and `LOAD`, the two base configurations, and `ice_at`, the ice
  history of the forward scenario;
- `Run`, `build` and `advance`, the build and the march;
- `diagnostics`, the row of one step;
- `write_log` and `write_values`, the two output formats.

Every diagnostic is an integral, a norm of a coefficient vector, a `Real`
value or a count. None is a nodal minimum, a nodal maximum or a point value,
so a row does not depend on the partition of the mesh or on the order of the
degrees of freedom.

The module imports Firedrake. The test files of `tests/gia` must not import
it; they read the files that the drivers write.
"""

import math
from dataclasses import dataclass, replace

import gadopt  # noqa: F401  (gadopt is imported before firedrake)
import firedrake as fd
from firedrake.adjoint import stop_annotating

import gia_cases as C
from gadopt import gia_couplings as cp

#: The main configuration: the Quadrature history with BDF2, the low-rank
#: DtN, two Maxwell elements, a fluid core, rotation, the centre-of-mass
#: constraint and a sea level with a moving coastline. Every coupling of the
#: solver is active in it.
SEA = C.Case(family="Quadrature", scheme="bdf2", representation="lowrank",
             model="maxwell2", core="fluid", rotation=True, centre=True,
             load="sea_moving")

#: `SEA` with the prescribed load `SIGMA_HAT (0.5 + cos 2 theta)` in place of
#: the sea level. The ocean of `SEA` has no degree-0 part and almost no
#: second moment, so the core pressure and the rotation unknown `m3` stay at
#: round-off there. The mean of this load gives both a value well above
#: round-off (core pressure about -1.2e-4 and `m3` about -5.8e-4 after two
#: steps).
LOAD = replace(SEA, load="surface", load_mean=0.5)

#: The steps of the forward scenario, in units of the Maxwell time of the
#: approximation. The change from 0.08 to 0.13 at step 5 (1-based) tests the
#: variable-step BDF2 coefficients and the rebuild of the cached operators.
FORWARD_STEPS = (.08, .08, .08, .08, .13, .13, .13, .13)

#: The nullspace that every scenario declares: the rigid rotation of the
#: mantle about the axis normal to the plane, which no row of the 2-D system
#: fixes.
NULLSPACE = "rigid_rotation"


def ice_at(sea, k):
    """Set the ice thickness before step `k` (0-based) of the forward scenario.

    `.002 (1 + .2 min(k, 3)) (1 - cos theta) / 2`: a cap on the side
    `theta = pi` with a grounding line, growing by 20 percent per step up to
    step 3 and constant afterwards. With growth through step 4 together with
    the longer steps, Newton from the previous state fails at step 5, so the
    growth stops before the change of the step length.

    Args:
      sea: the `gia_cases.SeaFields` of the run.
      k: the 0-based index of the step about to be solved.
    """
    X = fd.SpatialCoordinate(sea.I.function_space().mesh())
    theta = fd.atan2(X[1], X[0])
    sea.I.interpolate(.002 * (1 + .2 * min(k, 3)) * (1 - fd.cos(theta)) / 2)


@dataclass
class Run:
    """One built solver and the fields a march changes.

    Attributes:
      case: the `gia_cases.Case`.
      solver: the `SelfGravitatingGIASolver`.
      u, psi: the displacement and the potential that the solver writes.
      sea: the `SeaFields` of a sea-level case, else `None`.
      dt: the time step, a `Real` Function on the mantle submesh.
      time: the model time after the last step.
    """

    case: C.Case
    solver: object
    u: fd.Function
    psi: fd.Function
    sea: object
    dt: fd.Function
    time: float = 0.0


def build(meshes, case, *, preset="direct", default_options=True, **kwargs):
    """Build the solver of `case` on `meshes`, untaped.

    Args:
      meshes: the parent annulus and its mantle submesh
        (`gia_cases.annulus_meshes`).
      case: the configuration.
      preset: the solver preset name, `"direct"` or `"iterative"`.
      default_options: `True` adds the tight tolerances of `build_new`, so
        that the direct presets solve to round-off; `False` keeps the
        tolerances of the preset.
      kwargs: further arguments of `gia_cases.build_new`.

    Returns:
      A `Run`.
    """
    sub = meshes[1]
    dt = C.dt_function(sub)
    sea = C.SeaFields(sub) if case.load in ("sea_fixed", "sea_moving") else None
    with stop_annotating():
        solver, u, psi = C.build_new(
            meshes, case, dt=dt, sea=sea, nullspace=NULLSPACE,
            solver_parameters=preset, default_options=default_options, **kwargs)
    return Run(case=case, solver=solver, u=u, psi=psi, sea=sea, dt=dt)


def advance(run, steps, *, first=0, on_step=None, ice=ice_at):
    """March `run` over `steps`, calling `on_step(k, run)` after each solve.

    Args:
      run: a `Run`.
      steps: the step lengths.
      first: the 0-based index of the first step, for the ice history and
        the row index of a restarted run.
      on_step: a callable `(k, run)` called after the solve of step `k`.
      ice: the ice history, `ice(sea, k)`, applied before step `k` of a
        sea-level case.
    """
    for k, h in enumerate(steps, start=first):
        run.dt.assign(h)
        if run.sea is not None:
            ice(run.sea, k)
        run.solver.solve()
        run.time += h
        if on_step is not None:
            on_step(k, run)


# -- Diagnostics ---------------------------------------------------------------------

#: The counters of the preconditioner contexts, read from each Python context
#: in the tree of the outer PC. Each counts the builds of one cached
#: operator; a changed count means a cache rebuilds at another step.
COUNTERS = ("build_count", "column_builds", "assembly_count", "elimination_count")


def _python_contexts(pc, path="pc"):
    """Yield `(path, context)` for every Python PC context below `pc`.

    The walk descends into the sub-KSPs of a fieldsplit, into the inner PC of
    a Python context (`context.pc`), and into every KSP that a Python context
    owns (for example the condensed solve of `ReducedGIABlockPC`). It only
    reads objects that the solve has set up, so it is called after a solve.
    """
    from petsc4py import PETSc
    if pc.getType() == "python":
        context = pc.getPythonContext()
        yield path, context
        for name, value in sorted(vars(context).items()):
            if isinstance(value, PETSc.PC) and name == "pc":
                yield from _python_contexts(value, f"{path}_inner")
            elif isinstance(value, PETSc.KSP):
                yield from _python_contexts(value.getPC(), f"{path}_{name.strip('_')}")
    elif pc.getType() == "fieldsplit":
        for i, ksp in enumerate(pc.getFieldSplitSubKSP()):
            yield from _python_contexts(ksp.getPC(), f"{path}_split{i}")


def preconditioner_counters(solver):
    """`{name: count}` of every cache counter in the solver's PC tree.

    The name is the path of the context in the tree, its class name and the
    counter, for example `pc_DtNTwoBlockSchurPC_ainvb_build_count`. A
    `_SchurAinvBCache` hangs on its PC as `ainvb` and is read there.
    """
    out = {}
    for path, context in _python_contexts(solver.solver.snes.ksp.pc):
        owners = [(type(context).__name__, context)]
        if getattr(context, "ainvb", None) is not None:
            owners.append((type(context).__name__ + "_ainvb", context.ainvb))
        for label, owner in owners:
            for counter in COUNTERS:
                if hasattr(owner, counter):
                    out[f"{path}_{label}_{counter}"] = int(getattr(owner, counter))
    return out


def _coupling(solver, kind):
    """The coupling of class `kind`, or `None`."""
    found = [c for c in solver.couplings if isinstance(c, kind)]
    return found[0] if found else None


def _real(solver, name):
    """The value of the `Real` field `name`, or NaN if the solver has none.

    `float` gathers the value from the owning rank, so every rank gets it.
    """
    if name not in solver.field_index:
        return math.nan
    return float(solver.real_field(name))


def diagnostics(run, k):
    """The row of diagnostics after step `k` (0-based) of `run`.

    The columns, all integrals, norms, `Real` values or counts:

    - `step time dt`;
    - `u_l2 psi_l2 history_l2`: L2 norms on the mantle, the parent and the
      mantle at the quadrature points of the history;
    - `u_norm u_probe u_probe_norm` and the same for `psi` and `history`:
      `gia_cases.probe_pairing` of the coefficient vectors;
    - `ur_mean ur_sq geoid_mean geoid_sq`: the integrals of the radial
      displacement and of the geoid height (with the rotational potential)
      and of their squares over the outer surface of the mantle;
    - sea level, NaN without a `SeaLevel` coupling: `sl_mean sl_sq` (the
      integrals of `SL` and `SL^2`), `shift`, `ocean_area` (the integral of
      the water fraction), `sheet_mass` and `sheet_abs_mass` (the integrals
      of `sigma` and `|sigma|`), and `mass_identity mass_newton mass_bound`,
      the value of the net sheet mass that the converged `Shift` row implies
      and its bounds (`_mass_terms`);
    - frame: `com_0 com_1` (the centre-of-mass multipliers), `dipole_0
      dipole_1` (the first mass moment `D_i`, mantle and every sheet), and
      `sheet_moment_0 sheet_moment_1`, the integral of `|x_i sigma|` summed
      over every sheet, the scale against which `D_i = 0` is checked;
    - core: `core_pressure` and `cmb_flux`, the integral of `u . n` over the
      core-mantle boundary, which the fluid core constrains to zero;
    - rotation: `m3`;
    - counts: `newton linear`, the cache counters of the preconditioners
      (`preconditioner_counters`) and `operator_version`.

    Args:
      run: the `Run` after the solve of step `k`.
      k: the 0-based index of the step.

    Returns:
      `{column: float}`.
    """
    solver, sub = run.solver, run.u.function_space().mesh()
    ctx = solver.coupling_context
    row = {"step": k + 1, "time": run.time, "dt": float(run.dt)}
    history = solver.internal_variables
    dx_h = fd.dx(domain=sub, metadata={"quadrature_degree": int(solver.history.degree)})
    row["u_l2"] = fd.norm(run.u)
    row["psi_l2"] = fd.norm(run.psi)
    row["history_l2"] = math.sqrt(fd.assemble(fd.inner(history, history) * dx_h))
    for name, f in (("u", run.u), ("psi", run.psi), ("history", history)):
        label = {"u": "displacement", "psi": "potential", "history": "history"}[name]
        norm, probe, probe_norm = C.probe_pairing(f, label)
        row.update({f"{name}_norm": norm, f"{name}_probe": probe,
                    f"{name}_probe_norm": probe_norm})
    # The outer surface of the mantle, with the parent paired facet to facet,
    # so that the potential is read at the right points.
    ds_re = ctx.surface_measure(C.CURVE_RE)
    n = fd.FacetNormal(sub)
    u_r = fd.dot(solver.field("displacement"), n)
    geoid = ctx.parent_trace(solver.geoid())
    row["ur_mean"] = fd.assemble(u_r * ds_re)
    row["ur_sq"] = fd.assemble(u_r ** 2 * ds_re)
    row["geoid_mean"] = fd.assemble(geoid * ds_re)
    row["geoid_sq"] = fd.assemble(geoid ** 2 * ds_re)
    row.update(_sea_level_columns(solver))
    row.update(_frame_columns(solver))
    row["core_pressure"] = _real(solver, "core_pressure")
    row["cmb_flux"] = fd.assemble(fd.dot(solver.field("displacement"), n)
                                  * ctx.surface_measure(C.CURVE_RC))
    row["m3"] = _real(solver, "m3")
    row["newton"] = solver.appctx["gia_last_solve_newton_its"]
    row["linear"] = solver.appctx["gia_last_solve_linear_its"]
    row.update(preconditioner_counters(solver))
    version = solver.appctx.get("operator_version")
    row["operator_version"] = math.nan if version is None else version
    return row


SEA_COLUMNS = ("sl_mean", "sl_sq", "shift", "ocean_area", "sheet_mass",
               "sheet_abs_mass", "mass_identity", "mass_newton", "mass_bound")


def _sea_level_columns(solver):
    """The sea-level columns of `diagnostics`; NaN without a `SeaLevel` coupling."""
    sea = _coupling(solver, cp.SeaLevel)
    if sea is None:
        return {name: math.nan for name in SEA_COLUMNS}
    dss = sea.measure()
    SL, sigma = sea.sea_level(), sea.sheet()
    return {"sl_mean": fd.assemble(SL * dss),
            "sl_sq": fd.assemble(SL ** 2 * dss),
            "shift": float(sea.shift()),
            "ocean_area": fd.assemble(sea.water_fraction() * dss),
            "sheet_mass": fd.assemble(sigma * dss),
            "sheet_abs_mass": fd.assemble(abs(sigma) * dss),
            **_mass_terms(solver, sea)}


def _mass_terms(solver, sea):
    """The identity and the bound of the net sheet mass that the `Shift` row sets.

    At convergence the `Shift` row gives the exact identity

        int sigma dS = sum_i lambda_i int x_i dsigma/dShift dS / g_s,

    because the centre-of-mass rows keep the column `lambda . dD/dShift`, so
    the net sheet mass is of the order of the multipliers `lambda_i`, which
    is discretisation error. Newton leaves at most

        mass_newton = 10 max(snes_rtol ||F_0||, snes_atol) / (c B_mu g_s)

    of the row, with `||F_0||` the first residual norm of the last solve
    (the SNES convergence history) and the factor 10 for the difference
    between that norm and the one SNES tests. Returned:

    - `mass_identity`, the right-hand side of the identity;
    - `mass_newton`, the Newton allowance;
    - `mass_bound`, the bound of
      `test_sea_level.py::TestLiveCoastlineSolve::
      test_mass_is_conserved_to_order_lambda`: the triangle inequality on
      the identity with a factor 2, plus `mass_newton`.

    All three are NaN without a `CentreOfMass` coupling.
    """
    nan = {"mass_identity": math.nan, "mass_newton": math.nan, "mass_bound": math.nan}
    com = _coupling(solver, cp.CentreOfMass)
    if com is None:
        return nan
    import ufl
    from ufl.algorithms import expand_derivatives
    from gadopt.utility import scalar_value
    sigma, dss = sea.sheet(), sea.measure()
    # `dsigma/dShift`: the Gateaux derivative of the sheet in the direction
    # of the unit `Shift`, with the coastline at the solution.
    direction = fd.Function(solver.solution.function_space())
    direction.subfunctions[solver.field_index["sea_level_shift"]].assign(1.0)
    dsigma_dshift = expand_derivatives(ufl.derivative(sigma, solver.solution, direction))
    X = fd.SpatialCoordinate(solver.mesh)
    g_s = scalar_value(sea.g_surface)
    lam = [float(f) for f in com.multipliers()]
    identity = sum(lam_i * fd.assemble(X[i] * dsigma_dshift * dss)
                   for i, lam_i in enumerate(lam)) / g_s
    frame = sum(abs(lam_i) * fd.assemble(abs(X[i] * dsigma_dshift) * dss)
                for i, lam_i in enumerate(lam)) / g_s
    snes = solver.solver.snes
    history, _ = snes.getConvergenceHistory()
    first = float(history[0]) if len(history) else math.nan
    rtol, atol = snes.getTolerances()[:2]
    row_scale = (scalar_value(solver.scaling_factor)
                 * scalar_value(solver.approximation.B_mu) * g_s)
    newton = 10 * max(rtol * first, atol) / row_scale
    return {"mass_identity": identity, "mass_newton": newton,
            "mass_bound": 2.0 * frame + newton}


def _frame_columns(solver):
    """The centre-of-mass columns of `diagnostics`; NaN without the coupling."""
    com = _coupling(solver, cp.CentreOfMass)
    out = {}
    X = fd.SpatialCoordinate(solver.mesh)
    sheets = solver.coupling_context.sheets
    for i in range(2):
        if com is None:
            out[f"com_{i}"] = out[f"dipole_{i}"] = math.nan
        else:
            out[f"com_{i}"] = float(com.multipliers()[i])
            out[f"dipole_{i}"] = fd.assemble(com.dipole()[i])
        out[f"sheet_moment_{i}"] = sum(
            fd.assemble(abs(X[i] * sheet.density) * sheet.measure) for sheet in sheets)
    return out


def enable_residual_history(solver):
    """Make SNES keep the residual norms of each solve, for `_mass_terms`.

    `reset=True` starts the history again at every solve, so entry 0 is the
    first residual norm of the last solve.
    """
    solver.solver.snes.setConvergenceHistory(reset=True)


# -- Output -----------------------------------------------------------------------

def _format(value):
    """A number with 17 significant digits, so a float survives a round trip."""
    return f"{float(value):.17g}"


def write_log(path, rows):
    """Write `rows` as a table: a header line, then one line per row.

    The columns are the union of the keys of all rows, in the order of first
    appearance; a column that a row lacks is written as NaN. Rank 0 writes.
    Read with `pandas.read_csv(path, sep=r"\\s+")`.
    """
    if fd.COMM_WORLD.rank != 0:
        return
    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(" ".join(columns) + "\n")
        for row in rows:
            f.write(" ".join(_format(row.get(c, math.nan)) for c in columns) + "\n")


def write_values(path, mapping):
    """Write `{name: value}` as one line `name value` per entry. Rank 0 writes."""
    if fd.COMM_WORLD.rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name, value in mapping.items():
            f.write(f"{name} {_format(value)}\n")
