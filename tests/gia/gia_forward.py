"""Driver of the `forward` case: forward marches of the coupled GIA solver.

Usage: `python gia_forward.py --entry <name>` (under `mpiexec` for an entry
with more than one core), with the entry names of `gia_entries.FORWARD` and
`gia_entries.FORWARD_VALUES`.

A march entry (`FORWARD`) builds its configuration, marches it over
`gia_march.FORWARD_STEPS` with the ice history `gia_march.ice_at`, and
writes one row of `gia_march.diagnostics` per step to
`out/forward-<entry>.log`. With `save_state` it also writes the parent mesh
and the solver state after `RESTART_STEP` steps to `out/forward-<entry>.h5`
(`gadopt.gia_state.write_state`). With `restart_from` it reads that file of
another entry, restores the state, and marches the remaining steps.

The entry `SPHERE` marches a small 3-D configuration (`run_sphere`) and
writes its table in the same format.

The entries of `FORWARD_VALUES` compare the solver with an independent
system or with the stored results of the earlier solver, and write
`name value` lines to `out/forward-<entry>.dat`.

The test `test_gia_forward.py` reads the files; this script asserts nothing.
"""

import argparse
from dataclasses import replace

import gadopt  # imported before firedrake
import firedrake as fd
import numpy as np
from firedrake.adjoint import stop_annotating
from gadopt import gia_couplings as cp
from gadopt.gia_state import read_state, state_from_checkpoint, write_state
from gadopt.viscoelastic_checkpointing import written_rank_count

import gia_cases as C
import gia_march as M
from gia_entries import (CASE_DIR, FORWARD, FORWARD_VALUES, REFERENCE_LABEL,
                         RESTART_STEP, SPHERE, SPHERE_STEP_COUNT, forward_log,
                         forward_state, forward_values)

BASES = {"SEA": M.SEA, "LOAD": M.LOAD}

#: The group and the attribute under which `save` stores the name of the
#: parent mesh, so that a restart loads that mesh by name.
MESH_GROUP, MESH_NAME_KEY = "/gia_cases", "parent_name"


def configuration(entry):
    """The `Case` of a march entry: its base with the entry's changes."""
    return replace(BASES[entry["base"]], **entry.get("changes", {}))


# -- March entries -----------------------------------------------------------------

def run_march(name):
    """March the entry `name` of `FORWARD` and write its table (and its state)."""
    entry = FORWARD[name]
    case = configuration(entry)
    options = dict(preset=entry["preset"],
                   default_options=entry.get("default_options", True),
                   parameters=entry.get("parameters"))
    steps = M.FORWARD_STEPS
    if "restart_from" in entry:
        path = CASE_DIR / forward_state(entry["restart_from"])
        meshes = load_meshes(path)
        run = M.build(meshes, case, **options)
        with fd.CheckpointFile(str(path), "r") as chk:
            run.solver.restore(read_state(chk, run.solver, idx=0))
        run.time = sum(steps[:RESTART_STEP])
        steps, first = steps[RESTART_STEP:], RESTART_STEP
    else:
        run = M.build(C.annulus_meshes(), case, **options)
        first = 0
    M.enable_residual_history(run.solver)
    rows = []

    def on_step(k, run):
        rows.append(M.diagnostics(run, k))
        if entry.get("save_state") and k + 1 == RESTART_STEP:
            save(run, CASE_DIR / forward_state(name))

    M.advance(run, steps, first=first, on_step=on_step)
    M.write_log(CASE_DIR / forward_log(name), rows)


def save(run, path):
    """Write the parent mesh and the solver state to `path`, collectively.

    The restart reads the parent from the file (`CheckpointFile.load_mesh`)
    and builds its solver on it, as `gadopt.gia_state.read_state` requires.
    """
    if fd.COMM_WORLD.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd.COMM_WORLD.barrier()
    parent = run.psi.function_space().mesh()
    with fd.CheckpointFile(str(path), "w") as chk:
        chk.save_mesh(parent)
        chk.require_group(MESH_GROUP)
        chk.set_attr(MESH_GROUP, MESH_NAME_KEY, parent.name)
        write_state(chk, run.solver, idx=0)


def load_meshes(path, name=None):
    """The parent stored in `path` and its curved mantle submesh.

    The file stores the parent with its P2 coordinates. `Submesh` does not
    inherit them, so the submesh is curved again, as in
    `gia_cases.annulus_meshes`. The numbering of the loaded mesh can differ
    from the one of the mesh that was saved, which the diagnostics do not
    see: they are integrals and norms.

    Args:
      path: the checkpoint file.
      name: the name of the parent; `None` reads it from the attribute that
        `save` wrote.
    """
    with fd.CheckpointFile(str(path), "r") as chk:
        if name is None:
            name = chk.get_attr(MESH_GROUP, MESH_NAME_KEY)
            name = name.decode() if isinstance(name, bytes) else str(name)
        parent = chk.load_mesh(name)
    parent.cartesian = False
    sub = C.curve_mesh(fd.Submesh(parent, 2, C.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


# -- unreduced -----------------------------------------------------------------------

#: The steps of the comparison with the unreduced system, as in
#: `tests/unit/test_gia_direct_bdf2.py`.
UNREDUCED_STEPS = (.08, .13, .07, .12)

#: The agreement of the unreduced system and the Quadrature solver at one
#: node is `|a - b| <= UNREDUCED_RTOL |b| + UNREDUCED_ATOL`, the bound of
#: `test_direct_matches_coupled_quadrature`. The driver writes the largest
#: ratio of the difference to this bound; the test asserts it is at most 1.
UNREDUCED_RTOL, UNREDUCED_ATOL = 5e-7, 2e-9


def _bound_ratio(a, b):
    """`max |a - b| / (UNREDUCED_RTOL |b| + UNREDUCED_ATOL)` over the nodes of one rank."""
    a, b = np.ravel(a), np.ravel(b)
    return float(np.max(np.abs(a - b) / (UNREDUCED_RTOL * np.abs(b) + UNREDUCED_ATOL),
                        initial=0.0))


def run_unreduced():
    """The Quadrature solver against the unreduced mixed system, four steps.

    The configuration of `test_direct_matches_coupled_quadrature[spatial-
    maxwell]`: two Maxwell elements, the power-law exponent 3 on one half of
    the mantle and 1 on the other, a viscosity scaled by `exp(.1 x)`, fluid
    core, rotation, centre of mass and a fixed-coastline sea level under a
    growing ice load. Both solvers are built on the same meshes in one
    serial process, so their coefficient arrays have the same numbering.
    Per step and per field the driver writes the largest ratio of the
    difference to the bound (`_bound_ratio`) and the norm of the
    Quadrature solution, which shows that the compared field is not zero.
    """
    from gia_reference_mixed import build
    meshes = C.annulus_meshes()
    sub = meshes[1]
    X = fd.SpatialCoordinate(sub)
    control = fd.Function(fd.FunctionSpace(sub, "CG", 1)).interpolate(.1 * X[0])
    direct, dt, ice = build(meshes, control, direct=True, exponent="spatial",
                            model="maxwell")
    reference, rdt, rice = build(meshes, control, direct=False, exponent="spatial",
                                 model="maxwell")
    out = {}
    for k, h in enumerate(UNREDUCED_STEPS):
        dt.assign(h)
        rdt.assign(h)
        Xi = fd.SpatialCoordinate(ice.function_space().mesh())
        ice.interpolate(.001 * (1 + .2 * k) * (1 + .5 * fd.cos(2 * fd.atan2(Xi[1], Xi[0]))))
        rice.assign(ice)
        direct.solve()
        # The reference Newton is not a taped solve.
        with stop_annotating():
            reference.solve()
        pairs = {
            name: (direct.solution.subfunctions[direct.field_index[name]],
                   reference.solution.subfunctions[reference.field_index[name]])
            for name in ("displacement", "potential")}
        pairs["history"] = (
            direct.internal_variables,
            reference.solution.subfunctions[reference.field_index["history"]])
        for i in direct.real_indices:
            name = direct.field_names[i]
            pairs[name] = (direct.real_field(name), reference.real_field(name))
        for name, (a, b) in pairs.items():
            out[f"step{k + 1}_{name}_ratio"] = _bound_ratio(a.dat.data_ro, b.dat.data_ro)
            out[f"step{k + 1}_{name}_size"] = float(np.max(np.abs(a.dat.data_ro)))
        out[f"step{k + 1}_newton"] = direct.appctx["gia_last_solve_newton_its"]
    M.write_values(CASE_DIR / forward_values("unreduced"), out)


# -- reference-quadrature ------------------------------------------------------------

#: The Quadrature configuration of `FORWARD` in `tests/unit/test_gia_solver.py`
#: and its steps; the stored states of the earlier solver are in
#: `tests/unit/data/gia_forward_reference.json` under its label.
REFERENCE_CASE = C.Case(family="Quadrature", scheme="bdf2", representation="lowrank",
                        model="maxwell2", core="fluid", centre=True)
REFERENCE_STEPS = (.08, .13, .07)
assert REFERENCE_CASE.label == REFERENCE_LABEL


def run_reference():
    """Three steps of `REFERENCE_CASE`; writes the fingerprints and `Real` values.

    Per step, `step<k>_<field>_<part>` for the displacement, the potential
    and the accepted history, with the three numbers of
    `gia_cases.probe_pairing` (`norm`, `probe`, `probe_norm`), and
    `step<k>_real_<name>` for every `Real` field. The test compares them
    with the stored values of the earlier solver.
    """
    meshes = C.annulus_meshes()
    sub = meshes[1]
    dt = C.dt_function(sub)
    solver, u, psi = C.build_new(meshes, REFERENCE_CASE, dt=dt,
                                 nullspace="rigid_rotation")
    out = {}
    for k, h in enumerate(REFERENCE_STEPS):
        dt.assign(h)
        solver.solve()
        fields = {"displacement": u, "potential": psi, "history": solver.internal_variables}
        for name, f in fields.items():
            for part, value in zip(("norm", "probe", "probe_norm"),
                                   C.probe_pairing(f, name)):
                out[f"step{k + 1}_{name}_{part}"] = value
        for i in solver.real_indices:
            name = solver.field_names[i]
            out[f"step{k + 1}_real_{name}"] = float(solver.real_field(name))
    M.write_values(CASE_DIR / forward_values("reference-quadrature"), out)


# -- convert-quadrature-bdf2 -----------------------------------------------------------

#: The checkpoint of the earlier solver after two steps of `REFERENCE_CASE`,
#: in the `selfgrav_common` layout, and the name of its mesh.
OLD_CHECKPOINT = C.DATA_DIR / "gia_old_checkpoint_quadrature_bdf2.h5"
OLD_MESH_NAME = "firedrake_default"
CONVERT_STEPS = (.08, .13, .07)


def _relative(a, b):
    """`|a - b| / |b|` of the coefficient arrays of two Functions on one mesh."""
    x, y = np.ravel(a.dat.data_ro), np.ravel(b.dat.data_ro)
    return float(np.sqrt(np.sum((x - y) ** 2) / np.sum(y ** 2)))


def run_convert():
    """Restore this solver from a file of the earlier solver and continue.

    The file holds the earlier solver's state after two steps of
    `REFERENCE_CASE`. A solver on the mesh loaded from the file restores the
    converted state (`gadopt.gia_state.state_from_checkpoint`). A second
    solver on the same loaded mesh marches the same two steps from rest. The
    driver writes the relative differences of the displacement, the
    potential and the history after the restore and after one more step of
    both, the restored `previous_dt`, and 1 or 0 for each refusal: a state
    with the wrong time order and a conversion without the number of ranks
    of the writer. The file was written in serial, so this entry is serial.
    """
    meshes = load_meshes(OLD_CHECKPOINT, OLD_MESH_NAME)
    sub = meshes[1]
    new, u, psi = C.build_new(meshes, REFERENCE_CASE, dt=C.dt_function(sub),
                              nullspace="rigid_rotation")
    names = [f"solution_{i}" for i in range(len(new.solution.subfunctions))]
    names += ["internal_variables_0", "time_current_0", "time_previous_0"]
    with fd.CheckpointFile(str(OLD_CHECKPOINT), "r") as chk:
        loaded = {n: chk.load_function(meshes[0], n, idx=0) for n in names}
        history = chk.get_timestepping_history(meshes[0], names[0])
        writer_ranks = written_rank_count(chk, OLD_MESH_NAME)
    metadata = {k: float(v[-1]) for k, v in history.items() if k != "index"}
    new.restore(state_from_checkpoint(new, loaded, metadata, writer_ranks=writer_ranks))
    out = {"previous_dt": new.time_integrator.previous_dt}
    out["refuses_time_order"] = _raises(
        lambda: state_from_checkpoint(new, loaded, {**metadata, "time_scheme_order": 1.0},
                                      writer_ranks=1), "time order")
    out["refuses_missing_writer_ranks"] = _raises(
        lambda: state_from_checkpoint(new, loaded, metadata), "writer_ranks")
    reference, u_ref, psi_ref = C.build_new(meshes, REFERENCE_CASE, dt=C.dt_function(sub),
                                            nullspace="rigid_rotation")
    for h in CONVERT_STEPS[:2]:
        reference.dt.assign(h)
        reference.solve()

    def differences(when):
        for name, a, b in [("displacement", u, u_ref), ("potential", psi, psi_ref),
                           ("history", new.internal_variables,
                            reference.internal_variables)]:
            out[f"{when}_{name}_relative"] = _relative(a, b)
            out[f"{when}_{name}_norm"] = float(np.linalg.norm(np.ravel(b.dat.data_ro)))

    differences("restored")
    for solver in (new, reference):
        solver.dt.assign(CONVERT_STEPS[2])
        solver.solve()
    differences("continued")
    M.write_values(CASE_DIR / forward_values("convert-quadrature-bdf2"), out)


def _raises(call, match):
    """1.0 if `call()` raises a `ValueError` whose message contains `match`, else 0.0."""
    try:
        call()
    except ValueError as error:
        return 1.0 if match in str(error) else 0.0
    return 0.0


# -- 3-D entry ---------------------------------------------------------------------

#: The rotation and the outer radius of the extruded sphere, as in
#: `tests/unit/test_gia_solver_gate_3d.py`: the Earth's `Omega^2` in the
#: units of the cases, the secular Love number that closes the polar-wander
#: pair (`C - A = Q k_s`), and the radius `1 + 2 x 0.5` of the top surface.
SPHERE_OMEGA_SQ = 1.566e-3
SPHERE_K_S = 0.9
SPHERE_RADIUS = 2.0


def sphere_mesh():
    """The 24-column radially extruded cubed sphere, inner radius 1, outer 2.

    One mesh carries the displacement and the potential; the CMB is its
    `"bottom"` surface and the loaded surface its `"top"`.
    """
    base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
    mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5, extrusion_type="radial")
    mesh.cartesian = False
    return mesh


def build_sphere(mesh, dt):
    """The solver of the 3-D entry on `mesh`, with the iterative preset.

    The prescribed surface density `SIGMA_HAT (0.3 + z + x z + y z)` has
    every moment that a row reads: a mean (the core pressure), `z` (the
    third centre-of-mass row) and `x z`, `y z` (the polar-wander rows). No
    nullspace is declared: the three rotation rows fix the rigid rotations.

    Args:
      mesh: `sphere_mesh()`.
      dt: the time step, a `Real` Function on `mesh`.

    Returns:
      `(solver, u, psi)`.
    """
    X = fd.SpatialCoordinate(mesh)
    polar_moment = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
    sigma = C.SIGMA_HAT * (0.3 + X[2] + X[0] * X[2] + X[1] * X[2])
    u = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2), name="potential")
    dtn = {"top": {"dtn": gadopt.SphericalDtN(1)},
           "bottom": {"dtn": gadopt.SphericalDtN(1)}}
    with stop_annotating():
        solver = gadopt.SelfGravitatingGIASolver(
            u, psi, C.approximation(C.Case(model="maxwell2")), dt=dt,
            gravity_bcs=dtn, time_scheme="bdf2", internal_variable_family="Quadrature",
            quad_degree=C.QUAD, dtn_representation="lowrank",
            couplings=[cp.FluidCore(boundary="bottom", rho_core=2.0),
                       cp.Rotation(C=polar_moment, k_s=SPHERE_K_S,
                                   surface_radius=SPHERE_RADIUS,
                                   Omega_sq=SPHERE_OMEGA_SQ),
                       cp.CentreOfMass(),
                       cp.SurfaceLoad(boundary="top", sigma=sigma, g_surface=1.0)],
            solver_parameters="iterative")
    return solver, u, psi


def sphere_diagnostics(solver, u, psi, k, time, dt):
    """The row of the 3-D entry after step `k` (0-based).

    The columns: `step time dt`, the L2 norms `u_l2 psi_l2 history_l2`, the
    three rotation unknowns `m1 m2 m3`, the frame columns of
    `gia_march._frame_columns` (three directions), `core_pressure`,
    `cmb_flux` (the integral of `u . n` over the CMB), and the counts
    `newton linear newton_reduction`. All are integrals, norms, `Real`
    values or counts, so the row does not depend on the partition.
    """
    history = solver.internal_variables
    dx_h = fd.dx(domain=history.function_space().mesh(),
                 metadata={"quadrature_degree": int(solver.history.degree)})
    row = {"step": k + 1, "time": time, "dt": float(dt),
           "u_l2": fd.norm(u), "psi_l2": fd.norm(psi),
           "history_l2": float(np.sqrt(fd.assemble(fd.inner(history, history) * dx_h)))}
    for name in ("m1", "m2", "m3"):
        row[name] = M._real(solver, name)
    row.update(M._frame_columns(solver))
    row["core_pressure"] = M._real(solver, "core_pressure")
    n = fd.FacetNormal(solver.mesh)
    row["cmb_flux"] = fd.assemble(
        fd.dot(solver.field("displacement"), n)
        * solver.coupling_context.surface_measure("bottom"))
    row["newton"] = solver.appctx["gia_last_solve_newton_its"]
    row["linear"] = solver.appctx["gia_last_solve_linear_its"]
    history_norms = M.residual_history(solver)
    row["newton_reduction"] = (float(history_norms[-1] / history_norms[0])
                               if len(history_norms) > 1 else float("nan"))
    return row


def run_sphere():
    """March the 3-D entry over the first `SPHERE_STEP_COUNT` forward steps."""
    mesh = sphere_mesh()
    dt = fd.Function(fd.FunctionSpace(mesh, "R", 0))
    solver, u, psi = build_sphere(mesh, dt)
    M.enable_residual_history(solver)
    rows, time = [], 0.0
    for k, h in enumerate(M.FORWARD_STEPS[:SPHERE_STEP_COUNT]):
        dt.assign(h)
        solver.solve()
        time += h
        rows.append(sphere_diagnostics(solver, u, psi, k, time, dt))
    M.write_log(CASE_DIR / forward_log(SPHERE), rows)


VALUE_ENTRIES = {"unreduced": run_unreduced, "reference-quadrature": run_reference,
                 "convert-quadrature-bdf2": run_convert}
assert set(VALUE_ENTRIES) == set(FORWARD_VALUES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True,
                        choices=sorted([*FORWARD, *FORWARD_VALUES, SPHERE]))
    name = parser.parse_args().entry
    if name == SPHERE:
        run_sphere()
    elif name in VALUE_ENTRIES:
        VALUE_ENTRIES[name]()
    else:
        run_march(name)
