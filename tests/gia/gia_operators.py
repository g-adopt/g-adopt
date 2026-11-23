"""Driver of the `operators` case: properties of the coupled operator, without a solve.

Usage: `python gia_operators.py --entry <name>` (under `mpiexec -n 2` for
`schur-np2`), with the entry names of `gia_entries.OPERATORS`.

Each entry builds its solvers, writes a coordinate-defined state into them
(`gia_cases.coordinate_state`) and measures a property of the Jacobian of
the step through its actions on coordinate-defined vectors
(`gia_cases.residual_and_actions`). No entry solves a time step. On the
Quadrature history family the Jacobian is the one of the step backend, with
the history eliminated at the quadrature points, plus the low-rank DtN
update on the low-rank representation.

The instrument is the bilinear pairing. For two fields `a` and `b` with
directions `x_a` and `x_b` (each zero outside its own field), the action
`J x_b` restricted to field `a` paired with `x_a` is `x_a . J_ab x_b`. Two
blocks are transposes when `x_a . J_ab x_b = x_b . J_ba x_a`. A `Real` row
or column is read with the unit `Real` function as its direction, so no
matrix block with a `Real` row or column is ever assembled: pyop3 Firedrake
cannot assemble such a block on an interior facet of the parent
(firedrakeproject/firedrake#5665). Every number is a reduction over all ranks, so it does
not depend on the order of the degrees of freedom.

The driver writes, for each row, the measured number and the number it is
checked against (an exact identity, a derived constant or an independent
elimination) as `name value` lines to `out/operators-<entry>.dat`. The test
`test_gia_operators.py` holds the tolerances and asserts; this script
asserts nothing. No row uses a recorded value. Every solver runs on the
Quadrature history family.

The configurations come from the helper modules of `gia_models`, which the
unit tests also use, and from two sibling drivers:

- `gia_models.sea_level`: `build`, `surface_fields`, `SHIFT`, `RHO_W`, `G_RE`;
- `gia_models.annulus`: `approximation`, `gravity_bcs`, `new_fields`,
  `load_density`, `polar_moment`, `coupling`, `real_block`,
  `predicted_real_diagonal`, `real_block_index_sets`, `read_real_rows`,
  `FluidCoreBuilder.build_fluid`, `CentreOfMassFrame` (its `build`), and
  the constants `B_MU`, `SIGMA_HAT`, `CURVE_RE`, `CURVE_RC`;
- `gia_models.schur_ainvb`: `direct_params`, `ainvb_cache`;
- `gia_models.dense_schur`: `_dense_schur_params`, `_dense_ctx`,
  `_apply_pc` and the constants of its annulus;
- `gia_limits.py`: `unit_meshes` (the gmsh annulus of the unit fixture) and
  `QUADRATURE`;
- `gia_forward.py`: `sphere_mesh` (the 24-column cubed sphere).
"""

import argparse
from contextlib import ExitStack, contextmanager

import gadopt
import firedrake as fd
import numpy as np
from firedrake import dmhooks
from firedrake.adjoint import stop_annotating
from firedrake.formmanipulation import ExtractSubBlock
from gadopt import CylindricalDtN, SphericalDtN
from gadopt.gia_couplings import CentreOfMass, FluidCore, Rotation, SurfaceLoad
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.gia_solver import SelfGravitatingGIASolver

import gia_cases as C
import gia_march as M
from gia_entries import (CASE_DIR, FRAME_CONFIGURATIONS, OPERATORS, ROTATION_PAIRS,
                         SEA_PAIRS, SEA_STATES, BLOCK1_TRUNCATIONS, operators_values)
from gia_forward import sphere_mesh
from gia_limits import QUADRATURE, unit_meshes

from gia_models import dense_schur as TD
from gia_models import schur_ainvb as TA
from gia_models import annulus as G
from gia_models import sea_level as S

#: The short names of the fields that the rows pair, mapped to the solver's
#: field names. The first six are those of `test_sea_level._PAIR_FIELDS`.
FIELDS = {"u": "displacement", "psi": "potential", "Shift": S.SHIFT,
          "lambda_x": "centre_of_mass_0", "lambda_y": "centre_of_mass_1",
          "lambda_z": "centre_of_mass_2", "rot": "m3",
          "mult0": "dtn_multiplier_0"}


# -- The instrument -----------------------------------------------------------------

def set_state(solver, amplitude=1e-3):
    """Write the coordinate-defined state of amplitude `amplitude` into `solver`.

    Every field, the accepted history levels and the Newton iterate of the
    history get the values of `gia_cases.coordinate_state`; the time
    coefficients are those of a backward-Euler step (no previous step). An
    amplitude of 1e-3 keeps the coastline of the shoreline state inside its
    facets and the saturated states saturated, as the perturbation of
    `test_sea_level.jacobian` did.

    Args:
      solver: a `SelfGravitatingGIASolver`.
      amplitude: the scale of the state.
    """
    with stop_annotating():
        C.write_state(solver, C.coordinate_state(solver, amplitude), previous_dt=None)


def block_dot(a, b, index):
    """The Euclidean inner product of sub-vector `index` of `a` and of `b`.

    A reduction over all ranks, so every rank gets the same number.

    Args:
      a, b: mixed Functions or Cofunctions on the same mixed space.
      index: the field index of the sub-vector.
    """
    with a.subfunctions[index].dat.vec_ro as av, b.subfunctions[index].dat.vec_ro as bv:
        return float(av.dot(bv))


def block_norm(a, index):
    """The Euclidean norm of sub-vector `index` of `a`, over all ranks."""
    with a.subfunctions[index].dat.vec_ro as av:
        return float(av.norm())


def direction(solver, name):
    """The direction of field `name`: coordinate-defined values, zero elsewhere.

    A `Real` field gets the unit value, so that its pairing is the matrix
    entry itself. Any other field gets `gia_cases.coordinate_function` of its
    space with the label `operators:<name>`, so the direction is the same on
    every rank count.

    Args:
      solver: a `SelfGravitatingGIASolver`.
      name: a short name of `FIELDS`.

    Returns:
      A Function of the solver's mixed space.
    """
    index = solver.field_index[FIELDS[name]]
    out = fd.Function(solver.solution_space)
    sub = out.subfunctions[index]
    if index in solver.real_indices:
        sub.assign(1.0)
    else:
        C.fill(sub, C.coordinate_function(sub.function_space(), f"operators:{name}", 1.0))
    return out


def pairings(solver, names):
    """The pairings of the Jacobian between every two of the fields `names`.

    For the directions `x_a` of `direction`, `pairing[a, b] = x_a . (J x_b)_a`
    is the bilinear value of block `(a, b)`, and `action_norm[a, b] =
    |(J x_b)_a|` the norm of its action. With a unit `Real` direction on
    both sides the pairing is the matrix entry.

    Args:
      solver: a `SelfGravitatingGIASolver` with its state set.
      names: short names of `FIELDS`.

    Returns:
      `(pairing, action_norm, direction_norm)`: two dictionaries keyed by
      `(a, b)` and one keyed by `a`.
    """
    index = {name: solver.field_index[FIELDS[name]] for name in names}
    directions = {name: direction(solver, name) for name in names}
    _, actions = C.residual_and_actions(solver, [directions[n] for n in names])
    action = dict(zip(names, actions))
    pairing, action_norm = {}, {}
    for a in names:
        for b in names:
            pairing[a, b] = block_dot(directions[a], action[b], index[a])
            action_norm[a, b] = block_norm(action[b], index[a])
    direction_norm = {a: block_norm(directions[a], index[a]) for a in names}
    return pairing, action_norm, direction_norm


def pair_values(prefix, measured, a, b):
    """The three numbers of the transpose row of the pair `(a, b)`.

    `<prefix>_ab = x_a . J_ab x_b`, `<prefix>_ba = x_b . J_ba x_a` and
    `<prefix>_scale = max(|x_a| |J_ab x_b|, |x_b| |J_ba x_a|)`, the
    Cauchy-Schwarz bound of either pairing.

    Args:
      prefix: the name of the row.
      measured: the result of `pairings`.
      a, b: the two short names.
    """
    pairing, action_norm, direction_norm = measured
    scale = max(direction_norm[a] * action_norm[a, b], direction_norm[b] * action_norm[b, a])
    return {f"{prefix}_ab": pairing[a, b], f"{prefix}_ba": pairing[b, a],
            f"{prefix}_scale": scale}


def symmetry_of_block_zero(solver, label, count=3):
    """The bilinear asymmetry of block 0 for `count` coordinate-defined directions.

    The instrument of `TestEliminatedBlockSymmetry`: the matrix of the
    pairings `x_i . J x_j` for directions `x_k` with zero `Real` components.
    The caller reads `max |P - P^T|` against `max |P|`. On the Quadrature
    family block 0 is `(u, psi)`.

    Args:
      solver: a `SelfGravitatingGIASolver` with its state set.
      label: the label of the directions.
      count: the number of directions.

    Returns:
      `(directions, pairing matrix)`.
    """
    directions = []
    for k in range(count):
        x = C.coordinate_mixed_function(solver, f"{label}{k}", 1.0)
        for index in solver.real_indices:
            x.subfunctions[index].assign(0.0)
        directions.append(x)
    _, actions = C.residual_and_actions(solver, directions)
    pairing = np.zeros((count, count))
    for i, x in enumerate(directions):
        for j, action in enumerate(actions):
            with action.dat.vec_ro as av, x.dat.vec_ro as xv:
                pairing[i, j] = av.dot(xv)
    return directions, pairing


# -- Entry `sea-transposes` ---------------------------------------------------------

def run_sea_transposes(meshes):
    """The transpose pairs that the sea-level sheet creates, and its mechanics row.

    The configuration of `test_sea_level.jacobian`: the centre-of-mass frame
    with a fluid core, the multiplier representation, the surface state
    `cap` or `shoreline` with an ice change of 0.5, on the Quadrature
    family. Rows:

    - `TestTransposes`: the nine pairs of `SEA_PAIRS` on both states;
    - `TestRotationTransposes`: the three pairs of `ROTATION_PAIRS` on the
      shoreline state with rotation, and the two entries `(Shift, m3)` and
      `(m3, Shift)` against the `(Shift, Shift)` diagonal. Without the sheet
      in `Rotation.inertia_form` the entry `(m3, Shift)` is zero and the
      pairs `(u, rot)`, `(psi, rot)` lose their transposes;
    - `TestSheetSigns::test_the_mechanics_row_matches_a_normal_stress`: at
      `u = psi = 0` and `Shift = S` over the deep ocean, the displacement
      row of the sea level against a plain `normal_stress` of
      `B_mu g(Re) rho_w S`, in the max norm.
    """
    _, sub = meshes
    out = {}
    for state, rotation in [(s, False) for s in SEA_STATES] + [("shoreline", True)]:
        with stop_annotating():
            solver, _, _ = S.build(meshes, S.surface_fields(sub, state, d_ice=0.5),
                                   rotation=rotation, **QUADRATURE)
        set_state(solver)
        names = ["u", "psi", "Shift", "lambda_x", "lambda_y"] + (["rot"] if rotation else [])
        measured = pairings(solver, names)
        if rotation:
            for a, b in ROTATION_PAIRS:
                out.update(pair_values(f"rotation_{a}_{b}", measured, a, b))
            pairing = measured[0]
            out["shift_rot_entry"] = pairing["Shift", "rot"]
            out["rot_shift_entry"] = pairing["rot", "Shift"]
            out["shift_shift_entry"] = pairing["Shift", "Shift"]
        else:
            for a, b in SEA_PAIRS:
                out.update(pair_values(f"{state}_{a}_{b}", measured, a, b))

    # The mechanics row of the sea level against a normal stress of the same
    # density: the deep uniform ocean with `Shift = S` and every other
    # unknown zero, so the sheet is `rho_w S` and the row is the load alone.
    shift = 1e-2
    with stop_annotating():
        solver, _, sea = S.build(meshes, S.surface_fields(sub, "ocean"), **QUADRATURE)
        sea.shift().assign(shift)
        row_sea = fd.assemble(sea.residual(sea.context)).subfunctions[
            solver.field_index["displacement"]]
        # The solver warns that a normal stress in `bcs` has no mass; a
        # traction without mass is what the comparison wants.
        plain, _, _ = S.build(
            meshes, sea_level=False,
            surface_bcs={G.CURVE_RE: {"normal_stress": G.B_MU * S.G_RE * S.RHO_W * shift}},
            **QUADRATURE)
        row_plain = fd.assemble(plain.F).subfunctions[plain.field_index["displacement"]]
    with row_sea.dat.vec_ro as a, row_plain.dat.vec_ro as b:
        difference = a.copy()
        difference.axpy(-1.0, b)
        out["normal_stress_difference"] = float(difference.norm(fd.PETSc.NormType.INFINITY))
        out["normal_stress_scale"] = float(b.norm(fd.PETSc.NormType.INFINITY))
    return out


# -- Entry `frame` ------------------------------------------------------------------

def frame_rows(prefix, solver, directions):
    """The saddle-pair rows of the centre-of-mass multipliers `directions`.

    For each multiplier `l`: the pair `(u, l)` (`pair_values`). For each two
    multipliers: their entry, which the form does not contain.

    Args:
      prefix: the name of the configuration.
      solver: a solver with its state set.
      directions: the short names of the multipliers.
    """
    measured = pairings(solver, ["u", *directions])
    out = {}
    for name in directions:
        out.update(pair_values(f"{prefix}_u_{name}", measured, "u", name))
    out[f"{prefix}_multiplier_block_max"] = max(
        abs(measured[0][a, b]) for a in directions for b in directions)
    return out


def sphere_frame_solver():
    """The solver of `test_extruded_sphere_assembles_the_saddle_pair` on the Quadrature family.

    The 24-column cubed sphere with one mesh for both roles, a fluid core on
    `"bottom"`, the centre-of-mass multipliers, the multiplier DtN at `L = 1`
    and a top traction without mass (the solver warns).
    """
    mesh = sphere_mesh()
    u = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2), name="potential")
    with stop_annotating():
        solver = SelfGravitatingGIASolver(
            u, psi, G.approximation(), dt=1.0,
            gravity_bcs={"top": {"dtn": SphericalDtN(1)},
                         "bottom": {"dtn": SphericalDtN(1)}},
            bcs={"top": {"normal_stress": G.B_MU * G.SIGMA_HAT}},
            couplings=[FluidCore(boundary="bottom",
                                 rho_core=G.CentreOfMassFrame.RHO_CORE),
                       CentreOfMass()],
            dtn_representation="multiplier", **QUADRATURE)
    return solver


def sphere_translation_moment(solver):
    """`D[u = e_i]` of the sphere against the whole mass, per direction.

    The body of the second half of `test_extruded_sphere_assembles_the_saddle_pair`:
    the unit-density shell plus the core inside the faceted bottom sphere,
    whose volume is `A r / 3` from its area `A` and mean radius `r`. The
    only sheet is the core's, because the top traction has no mass.

    Returns:
      `{sphere_mass, sphere_moment_<i>, sphere_moment_cross_<i>}`: the mass,
      `D_i[e_i]` and the largest `|D_j[e_i]|`, `j != i`.
    """
    mesh = solver.mesh
    centre = G.coupling(solver, CentreOfMass)
    ctx = solver.coupling_context
    shell = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
    ds_b = G.coupling(solver, FluidCore).measure()
    X = fd.SpatialCoordinate(mesh)
    area = fd.assemble(fd.Constant(1.0) * ds_b)
    radius = fd.assemble(fd.sqrt(fd.dot(X, X)) * ds_b) / area
    mass = shell + G.CentreOfMassFrame.RHO_CORE * area * radius / 3
    out = {"sphere_mass": mass}
    displacement = solver.solution.subfunctions[solver.field_index["displacement"]]
    for i in range(3):
        e = [0.0, 0.0, 0.0]
        e[i] = 1.0
        displacement.interpolate(fd.as_vector(e))
        D = [fd.assemble(centre.dipole_form(ctx, j)) for j in range(3)]
        out[f"sphere_moment_{i}"] = D[i]
        out[f"sphere_moment_cross_{i}"] = max(abs(D[j]) for j in range(3) if j != i)
    displacement.assign(0.0)
    return out


def run_frame(meshes):
    """The centre-of-mass saddle pair (2-D and the sphere) and the inertia of a `cos 2 phi` sheet.

    - `TestCentreOfMassFrame::test_blocks_form_an_exact_saddle_pair` on the
      four configurations of `FRAME_CONFIGURATIONS` (both representations,
      rotation off and on): the frame configuration with a degree-1 sheet.
    - `test_extruded_sphere_assembles_the_saddle_pair`: the three
      multipliers of the 24-column sphere (assembly only) and its
      translation moment, and the pair `(u, psi)` of the sphere.
    - `TestInertiaTranspose::test_inertia_of_a_cos2_sheet_vanishes`: the
      sheet half of `dI_33` of the `cos 2 phi` load, read at `u = 0` so that
      the whole value is the load's sheet (the original read it at the
      solved state, where the mantle half integrates a degree-2 field to
      zero as well), against the integral of `|sigma| r^2` as its scale.
    """
    out = {}
    frame = G.CentreOfMassFrame()
    for representation, rotation in FRAME_CONFIGURATIONS:
        with stop_annotating():
            solver, _, _ = frame.build(meshes, rotation=rotation,
                                       dtn_representation=representation)
        set_state(solver)
        prefix = f"frame_{representation}_{'rot' if rotation else 'norot'}"
        out.update(frame_rows(prefix, solver, ["lambda_x", "lambda_y"]))

    solver = sphere_frame_solver()
    set_state(solver)
    out.update(frame_rows("frame_sphere", solver, ["lambda_x", "lambda_y", "lambda_z"]))
    out.update(sphere_translation_moment(solver))
    # The pair (u, psi) of the sphere: the self-gravity body force of the
    # momentum row against the displacement source of the potential row.
    # Both are integrated at the mechanics quadrature degree; with the
    # source at UFL's estimate the two pairings differ by 2.4e-4 of their
    # value here (6.2e-6 of the Cauchy-Schwarz scale), while on the annulus
    # the two rules agree.
    out.update(pair_values("frame_sphere_u_psi", pairings(solver, ["u", "psi"]), "u", "psi"))

    with stop_annotating():
        solver, _, _ = G.FluidCoreBuilder.build_fluid(meshes, rotation=True, **QUADRATURE)
    rotation = G.coupling(solver, Rotation)
    load = G.coupling(solver, SurfaceLoad)
    _, sub = meshes
    X = fd.SpatialCoordinate(sub)
    out["inertia_cos2"] = fd.assemble(rotation.inertia_perturbation()["dI_33"])
    out["inertia_cos2_scale"] = fd.assemble(
        abs(G.load_density(sub)) * fd.dot(X, X) * load.measure())
    return out


# -- Entry `block0` -----------------------------------------------------------------

#: The two Maxwell elements of `TestEliminatedBlockSymmetry`.
MAXWELL2 = {"viscosity": [1.0, 3.0], "shear_modulus": [1.0, 0.5]}


def run_block0(meshes):
    """The symmetry of block 0: eliminated history, and with a live coastline.

    - `TestEliminatedBlockSymmetry::test_the_eliminated_block_is_symmetric`
      (Quadrature half): a fluid core with rotation, two Maxwell elements,
      the low-rank DtN; the operator is the backend's Jacobian plus the
      low-rank update. The DG half of that test (the uncondensed block is
      not symmetric) is not carried: the DG family has no case rows.
    - A live coastline leaves block 0 symmetric
      (`TestLiveCoastlineRebuildRules::test_block_zero_symmetry_is_unchanged`,
      measured here on the operator and not through the CG solve): the
      shoreline state with rotation on the low-rank path. The sheet depends
      on the unknowns only through `Delta`, so its second variation is
      symmetric. The sea-level part of block 0 is written beside it, so the
      row is known to measure a block that contains the live sheet.
    """
    out = {}
    with stop_annotating():
        solver, _, _ = G.FluidCoreBuilder.build_fluid(
            meshes, rotation=True, approximation_kwargs=MAXWELL2, **QUADRATURE)
    set_state(solver)
    _, pairing = symmetry_of_block_zero(solver, "symmetry")
    out["eliminated_asymmetry"] = float(np.abs(pairing - pairing.T).max())
    out["eliminated_pairing_max"] = float(np.abs(pairing).max())

    _, sub = meshes
    with stop_annotating():
        solver, _, sea = S.build(meshes, S.surface_fields(sub, "shoreline"),
                                 rotation=True, representation="lowrank", **QUADRATURE)
    set_state(solver)
    directions, pairing = symmetry_of_block_zero(solver, "coastline")
    out["coastline_asymmetry"] = float(np.abs(pairing - pairing.T).max())
    out["coastline_pairing_max"] = float(np.abs(pairing).max())
    # The sea-level rows alone, at the same state: the variation of
    # `SeaLevel.residual` in the same directions.
    J_sea = fd.derivative(sea.residual(sea.context), solver.solution)
    sea_pairing = np.zeros_like(pairing)
    for j, x in enumerate(directions):
        action = fd.assemble(fd.action(J_sea, x))
        for i, y in enumerate(directions):
            with action.dat.vec_ro as av, y.dat.vec_ro as yv:
                sea_pairing[i, j] = av.dot(yv)
    out["coastline_sea_pairing_max"] = float(np.abs(sea_pairing).max())
    out["coastline_sea_asymmetry"] = float(np.abs(sea_pairing - sea_pairing.T).max())
    return out


# -- Entry `block1` -----------------------------------------------------------------

def fluid_solver(meshes, *, truncation=3, rotation=False, approximation_kwargs=None):
    """`TestFluidCore.build_fluid` with the DtN truncation as an argument, multiplier path.

    The annulus with a fluid core on Rc, the `cos 2 phi` load as a
    `SurfaceLoad` and optionally the rotation row, on the multiplier
    representation (one `Real` row per DtN mode) and the Quadrature family.
    The rigid core of `test_gia_gravity.build`, which the block-1 tests
    used, is refused by the Quadrature family, so the core pressure is one
    more `Real` row here, a Lagrange multiplier with a zero diagonal.

    Args:
      meshes: the parent annulus and its mantle submesh.
      truncation: the `CylindricalDtN` truncation on both DtN circles.
      rotation: add the `Rotation` coupling.
      approximation_kwargs: settings merged into `test_gia_gravity.approximation`.
    """
    parent, sub = meshes
    u, psi = G.new_fields(meshes)
    couplings = [FluidCore(boundary=G.CURVE_RC, rho_core=2.0),
                 SurfaceLoad(boundary=G.CURVE_RE, sigma=G.load_density(sub), g_surface=1.0)]
    if rotation:
        couplings.append(Rotation(C=G.polar_moment(meshes), Omega_sq=OMEGA_SQ_EARTH))
    with stop_annotating():
        return SelfGravitatingGIASolver(
            u, psi, G.approximation(**(approximation_kwargs or {})), dt=1.0,
            gravity_bcs=G.gravity_bcs(parent, truncation=truncation, sheet=False),
            bcs={}, couplings=couplings, dtn_representation="multiplier", **QUADRATURE)


def real_block_rows(prefix, solver):
    """The `Real` block against its derived diagonal, entry by entry.

    The block is recovered by `test_gia_gravity.real_block`: column `j` is
    the residual at the unit `Real` function `e_j` minus the residual at
    zero, read on the `Real` rows, which is the action of the Jacobian on
    `e_j` because the residual is linear in every `Real` unknown. The
    derived diagonal is `test_gia_gravity.predicted_real_diagonal`: the DtN
    multiplier entries from the constants of the form, `theta_rot_i K_i` for
    a rotation row, zero for the core pressure and the frame multipliers.

    Args:
      prefix: the name of the configuration.
      solver: a solver on the multiplier representation; its solution is
        left at zero.

    Returns:
      `(values, block)`: the rows `<prefix>_count`, `_diag_rel` (the largest
      relative error of a nonzero derived entry), `_diag_min` (the smallest
      such entry, which must not be zero), `_zero_rows`, `_zero_diag` (the
      largest assembled entry where the derived one is zero, over the
      largest diagonal), `_offdiag` (the largest per-row ratio of an
      off-diagonal entry to the diagonal) and `_zero_offdiag` (the same for
      the zero rows, over the largest diagonal); and the block.
    """
    block = G.real_block(solver)
    claimed = G.predicted_real_diagonal(solver)
    diag = np.diag(block)
    scale = float(np.abs(diag).max())
    nonzero = claimed != 0.0
    off = block - np.diag(diag)
    row_off = np.abs(off).max(axis=1)
    values = {
        f"{prefix}_count": len(claimed),
        f"{prefix}_diag_rel": float(np.max(np.abs(claimed - diag)[nonzero]
                                           / np.abs(diag[nonzero]))),
        f"{prefix}_diag_min": float(np.abs(diag[nonzero]).min()),
        f"{prefix}_zero_rows": int((~nonzero).sum()),
        f"{prefix}_zero_diag": float(np.abs(diag[~nonzero]).max() / scale
                                     if (~nonzero).any() else 0.0),
        f"{prefix}_offdiag": float(np.max(row_off[nonzero] / np.abs(diag[nonzero]))),
        f"{prefix}_zero_offdiag": float(row_off[~nonzero].max() / scale
                                        if (~nonzero).any() else 0.0),
    }
    return values, block


def multiplier_alignment(prefix, solver):
    """Drive `psi` with each mode expression and read the multiplier rows it moves.

    The body of `test_gia_gravity.check_multiplier_alignment`, with the
    worst values returned and not asserted. Each constraint row reads
    `theta_psi int psi_h e_k ds` over its own boundary when the
    coefficients are zero, so the mode expression of key `k` must move
    exactly the rows whose key is `k`, on both boundaries, and the fired
    row must equal `theta_psi int psi_h e_k ds` through the form's own
    boundary measure (the discrete prediction). The other `Real` rows (the
    core pressure, the rotation rows) must not see the potential at all.

    Args:
      prefix: the name of the configuration.
      solver: a solver on the multiplier representation; its solution is
        left at zero.

    Returns:
      `<prefix>_discrete`, `_analytic`, `_silent`, `_other_rows` (the
      worst relative deviations of `check_multiplier_alignment` and of the
      rows that are not multipliers, relative to the weakest fired row),
      `_fired_min` (the weakest fired row), and `_order` (1 when the
      multiplier fields are the first `Real` fields, the order the keys
      assume).
    """
    z = solver.solution
    comm = solver.solution_space.mesh().comm
    ises = G.real_block_index_sets(solver)
    form = solver.form
    theta = float(solver.theta_psi)
    psi_h = z.subfunctions[solver.field_index["potential"]]
    keys = form.multiplier_keys
    n_mult = len(keys)
    order = tuple(solver.multiplier_indices) == tuple(solver.real_indices[:n_mult])

    modes = {}
    for bc_id, dtn in form.dtn_boundaries:
        side, radius = form.boundary_geometry[bc_id]
        for mode in dtn.modes(side, radius, form.X):
            modes[(bc_id, mode.key)] = mode

    z.assign(0.0)
    base = G.read_real_rows(fd.assemble(solver.F), ises, comm)
    worst = {"discrete": 0.0, "analytic": 0.0, "silent": 0.0, "other_rows": 0.0}
    fired_min = np.inf
    for driven in sorted({key for _, key in keys}):
        expression = next(mode.expr for (_, key), mode in modes.items() if key == driven)
        z.assign(0.0)
        psi_h.interpolate(fd.Constant(expression) if isinstance(expression, float)
                          else expression)
        row = G.read_real_rows(fd.assemble(solver.F), ises, comm) - base
        fired = [i for i, (_, key) in enumerate(keys) if key == driven]
        silent = [i for i, (_, key) in enumerate(keys) if key != driven]
        for i in fired:
            bc_id, key = keys[i]
            mode = modes[(bc_id, key)]
            discrete = theta * fd.assemble(psi_h * mode.expr * form.ds(bc_id))
            worst["discrete"] = max(worst["discrete"], abs(row[i] - discrete) / abs(discrete))
            analytic = theta * float(mode.norm)
            worst["analytic"] = max(worst["analytic"], abs(row[i] - analytic) / abs(analytic))
        weakest = min(abs(row[i]) for i in fired)
        fired_min = min(fired_min, weakest)
        for i in silent:
            worst["silent"] = max(worst["silent"], abs(row[i]) / weakest)
        for i in range(n_mult, len(solver.real_indices)):
            worst["other_rows"] = max(worst["other_rows"], abs(row[i]) / weakest)
    z.assign(0.0)
    out = {f"{prefix}_{name}": value for name, value in worst.items()}
    out[f"{prefix}_fired_min"] = fired_min
    out[f"{prefix}_order"] = 1.0 if order else 0.0
    return out


def sphere_block1_solver():
    """The shell solver of `TestBlockOneInThreeDimensions.build`, on the Quadrature family.

    The 24-column cubed sphere with one mesh for both roles, the multiplier
    DtN at `L = 1` on both surfaces, the rotation rows with `C - A = 0.1 C`
    (so that the two closure constants differ by a factor 10), a fluid core
    on `"bottom"` (the rigid core of the original is refused by the
    Quadrature family) and a top traction without mass.
    """
    mesh = sphere_mesh()
    u = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2), name="potential")
    X = fd.SpatialCoordinate(mesh)
    polar = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
    with stop_annotating():
        return SelfGravitatingGIASolver(
            u, psi, G.approximation(), dt=1.0,
            gravity_bcs={"top": {"dtn": SphericalDtN(1)},
                         "bottom": {"dtn": SphericalDtN(1)}},
            bcs={"top": {"normal_stress": G.SIGMA_HAT * X[2]}},
            couplings=[Rotation(C=polar, C_minus_A=0.1 * polar, Omega_sq=OMEGA_SQ_EARTH),
                       FluidCore(boundary="bottom", rho_core=2.0)],
            dtn_representation="multiplier", **QUADRATURE)


def polar_pair_rows(solver, block):
    """The rotation corner of the sphere's `Real` block.

    `test_the_polar_pair_is_uncoupled_and_carries_C_minus_A`: `m1` and `m2`
    carry the same entry, `m3` carries `-C / (C - A)` times it, and no
    rotation row couples to a multiplier or to another rotation row.

    Returns:
      `sphere_m1`, `sphere_m2`, `sphere_m3` (the three diagonal entries),
      `sphere_rotation_multiplier_max`, `sphere_multiplier_rotation_max`,
      `sphere_rotation_offdiag_max`, and the floor scale
      `sphere_rotation_min` (the smallest rotation entry).
    """
    real = list(solver.real_indices)
    rotation = [real.index(solver.field_index[n]) for n in ("m1", "m2", "m3")]
    multipliers = [real.index(i) for i in solver.multiplier_indices]
    entries = np.array([block[k, k] for k in rotation])
    corner = block[np.ix_(rotation, rotation)] - np.diag(entries)
    return {"sphere_m1": entries[0], "sphere_m2": entries[1], "sphere_m3": entries[2],
            "sphere_rotation_multiplier_max": float(np.abs(block[np.ix_(rotation, multipliers)]).max()),
            "sphere_multiplier_rotation_max": float(np.abs(block[np.ix_(multipliers, rotation)]).max()),
            "sphere_rotation_offdiag_max": float(np.abs(corner).max()),
            "sphere_rotation_min": float(np.abs(entries).min())}


def run_block1(meshes):
    """The `Real` block: its diagonal, its alignment, the polar pair, and `B_mu = 0`.

    - `TestBlockOneIsTheDiagonalItClaims::test_the_block_is_the_claimed_diagonal`
      at truncation 3 and 5, rotation on and off, with the expected number
      of `Real` rows (`2T + (2T + 1)` multipliers, the core pressure, the
      rotation row);
    - `test_each_multiplier_row_is_the_mode_its_key_names` at truncation 3
      and 5 (rotation on);
    - `TestBlockOneInThreeDimensions`: the claimed diagonal, the alignment at
      `L = 1` and the polar pair on the 24-column sphere (assembly only);
    - `TestNullCoupling::test_the_potential_row_is_not_annihilated`: at
      `B_mu = 0` the potential block and the potential column of a
      multiplier row are not zero, the potential column of the displacement
      row (the body force) is exactly zero, and the displacement column of
      the potential row is not zero.

    `TestFamiliesAgree::test_the_real_block_does_not_depend_on_the_family`
    is not carried: it compared the DG and the Quadrature family, and the DG
    family has no case rows. The derived diagonal above is the independent
    check of the Quadrature block.
    """
    out = {}
    for truncation in BLOCK1_TRUNCATIONS:
        for rotation in (False, True):
            solver = fluid_solver(meshes, truncation=truncation, rotation=rotation)
            prefix = f"block1_t{truncation}_{'rot' if rotation else 'norot'}"
            values, _ = real_block_rows(prefix, solver)
            out.update(values)
            out[f"{prefix}_expected_count"] = (2 * truncation) + (2 * truncation + 1) + 1 + int(rotation)
            if rotation:
                out.update(multiplier_alignment(f"alignment_t{truncation}", solver))

    solver = sphere_block1_solver()
    values, block = real_block_rows("block1_sphere", solver)
    out.update(values)
    out["block1_sphere_expected_count"] = 2 * (1 + 1) ** 2 + 3 + 1
    out.update(polar_pair_rows(solver, block))
    out.update(multiplier_alignment("alignment_sphere", solver))

    solver = fluid_solver(meshes, rotation=True, approximation_kwargs={"B_mu": 0.0})
    set_state(solver)
    _, action_norm, _ = pairings(solver, ["u", "psi", "mult0"])
    out["null_psi_psi"] = action_norm["psi", "psi"]
    out["null_multiplier_psi"] = action_norm["mult0", "psi"]
    out["null_u_psi"] = action_norm["u", "psi"]
    out["null_psi_u"] = action_norm["psi", "u"]
    return out


# -- Entries `schur` and `schur-np2` ------------------------------------------------

def schur_solver(meshes, parameters):
    """The annulus of `test_dtn_multiplier_dense_schur._build_solver` on the Quadrature family.

    The multiplier DtN at truncation 3, the `cos 2 theta` load as a
    `SurfaceLoad`, `g = 1`, a bulk-to-shear ratio of 100, and a fluid core
    (`fluid_core=True` of the original; the rigid core is refused by the
    Quadrature family). The `Real` block is the 13 DtN multipliers and the
    core pressure.

    Args:
      meshes: the parent annulus and its mantle submesh.
      parameters: the solver-parameter dictionary.
    """
    parent, sub = meshes
    X = fd.SpatialCoordinate(sub)
    gravity_bcs = {TD.CURVE_OUTER: {"dtn": CylindricalDtN(TD.TRUNCATION)},
                   TD.CURVE_INNER: {"dtn": CylindricalDtN(TD.TRUNCATION)}}
    approximation = gadopt.CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        bulk_shear_ratio=100.0, g=TD.G0, B_mu=TD.B_MU, self_gravity_number=TD.LAMBDA)
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    load = SurfaceLoad(boundary=TD.CURVE_RE,
                       sigma=TD.SIGMA_HAT * fd.cos(2 * fd.atan2(X[1], X[0])),
                       g_surface=TD.G0)
    with stop_annotating():
        return SelfGravitatingGIASolver(
            u, psi, approximation, dt=1.0, gravity_bcs=gravity_bcs, bcs={},
            couplings=[FluidCore(boundary=TD.CURVE_RC, rho_core=2.0), load],
            dtn_representation="multiplier", solver_parameters=parameters, **QUADRATURE)


@contextmanager
def preconditioner_set_up(solver):
    """Set up the solver's outer preconditioner at its present state, without a solve.

    What `NonlinearVariationalSolver.solve` does before the linear solve,
    and no more: the step backend copies the live inputs and the state, the
    solve index and the operator version are published, the options of the
    solver are inserted and its application context is pushed onto the DM,
    the matrix-free Jacobian is formed at the state (its pre-Jacobian
    callback refreshes the local constitutive root), and the KSP is set up
    on it, which sets up the preconditioner tree. The options and the
    context stay in place while the block runs, so the preconditioners can
    be applied inside it.

    Args:
      solver: a `SelfGravitatingGIASolver` on the Quadrature family with its
        state set.

    Yields:
      The outer PETSc `PC`.
    """
    backend = solver._step.backend
    # The mixed solution is one of the inputs, so this also copies the state
    # into `backend.work`.
    for target, value in zip(backend.private_inputs, backend.inputs):
        target.assign(value)
    solver.appctx["gia_solve_index"] = solver.appctx.get("gia_solve_index", 0) + 1
    solver._refresh_operator_version()
    solver.appctx["operator_version"] = None
    nlvs = backend.solver
    ctx = nlvs._ctx
    snes = nlvs.snes
    ctx.set_function(snes)
    ctx.set_jacobian(snes)
    with ExitStack() as stack:
        stack.enter_context(nlvs.inserted_options())
        stack.enter_context(dmhooks.add_hooks(snes.getDM(), nlvs, appctx=ctx))
        with backend.work.dat.vec_ro as w:
            X = w.copy()
        J = ctx._jac.petscmat
        snes.computeJacobian(X, J, J)
        ksp = snes.getKSP()
        ksp.setOperators(J, J)
        ksp.setUp()
        yield ksp.getPC()


def monolithic(function):
    """A copy of the coefficient vector of a mixed Function, as a PETSc Vec."""
    with function.dat.vec_ro as v:
        return v.copy()


class RigidRotation:
    """The rigid rotation of the mantle as a monolithic vector, and its removal.

    The fluid-core annulus has no condition that fixes the rotation of the
    mantle about the axis normal to the plane: the curved facets annihilate
    it only to round-off of the geometry (`|A r| / |r|` is about 5e-9 of
    `|A x| / |x|` for a generic `x`). Block 0 is therefore near-singular, and
    the LU of block 0 resolves the component of a vector along `r` only to
    about `eps cond(A00)`: the identity errors carry 8e-6 along `r` and 5e-13
    in every other direction. That component is nearly the same in the
    forward and the transpose apply, so it would also hide the difference of
    the two. The scenarios of the cases declare this mode as the nullspace
    (`gia_march.NULLSPACE`) and the solver projects it out after every step,
    so the identities are measured on vectors with `r` removed and on errors
    with `r` removed. The removal is the Euclidean projection of the
    monolithic vector, which is a reduction over all ranks.
    """

    def __init__(self, solver):
        X = fd.SpatialCoordinate(solver.mesh)
        r = fd.Function(solver.solution_space)
        r.subfunctions[solver.field_index["displacement"]].interpolate(
            fd.as_vector([-X[1], X[0]]))
        self.r = monolithic(r)
        self.rr = self.r.dot(self.r)

    def remove(self, v):
        """A copy of the Vec `v` with its component along `r` removed."""
        w = v.copy()
        w.axpy(-w.dot(self.r) / self.rr, self.r)
        return w


def reference_complement(solver):
    """`S_ref = A11 - A10 A00^{-1} A01`, from actions and an assembled block 0.

    Independent of the preconditioners under test, and with no matrix block
    that has a `Real` row or column (firedrakeproject/firedrake#5665):

    - `A01 e_k` and `A11 e_k` are the block-0 and the `Real` parts of the
      action of the backend Jacobian on the unit `Real` function `e_k`;
    - row `j` of `A10` is the block-0 part of the action of its adjoint on
      `e_j`;
    - `A00` is the `(u, psi)` block of the backend Jacobian, extracted with
      `ExtractSubBlock` and assembled as one `aij` matrix, and solved by its
      own MUMPS LU.

    The order of the `Real` rows is `solver.real_indices`, the order of the
    preconditioners' index set.

    Args:
      solver: a solver on the Quadrature family whose backend state is set
        (inside `preconditioner_set_up`); it must have no strong boundary
        condition, since none is applied here.

    Returns:
      The dense `n x n` complement.
    """
    if solver.strong_bcs:
        raise ValueError("reference_complement applies no strong boundary condition")
    J = solver._step.backend.J
    Z = solver.solution_space
    real = list(solver.real_indices)
    block0 = [i for i in range(len(Z)) if i not in real]
    n = len(real)
    units = []
    for k in real:
        e = fd.Function(Z)
        e.subfunctions[k].assign(1.0)
        units.append(e)
    forward = [fd.assemble(fd.action(J, e)) for e in units]
    adjoint = [fd.assemble(fd.action(fd.adjoint(J), e)) for e in units]
    A11 = np.array([[block_dot(forward[k], units[j], real[j]) for k in range(n)]
                    for j in range(n)])
    a00 = ExtractSubBlock().split(J, argument_indices=(tuple(block0), tuple(block0)))
    W0 = a00.arguments()[0].function_space()
    solve = fd.LinearSolver(
        fd.assemble(a00, mat_type="aij"),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"},
        options_prefix="operators_reference_block0_")
    A10_Z = np.zeros((n, n))
    for k in range(n):
        rhs = fd.Cofunction(W0.dual())
        for position, i in enumerate(block0):
            C.fill(rhs.subfunctions[position], forward[k].subfunctions[i])
        z = fd.Function(W0)
        solve.solve(z, rhs)
        for j in range(n):
            A10_Z[j, k] = sum(block_dot_pair(adjoint[j].subfunctions[i], z.subfunctions[p])
                              for p, i in enumerate(block0))
    return A11 - A10_Z


def block_dot_pair(a, b):
    """The Euclidean inner product of two vectors of one non-mixed space, over all ranks."""
    with a.dat.vec_ro as av, b.dat.vec_ro as bv:
        return float(av.dot(bv))


def index_vector(seed, n):
    """A length-`n` vector of numbers in [-0.5, 0.5) from integers, the same everywhere."""
    return np.array([C._unit(seed, k) - .5 for k in range(n)])


def complement_rows(prefix, S, S_ref, comm):
    """The rows that compare a complement with the reference and across ranks.

    Returns:
      `<prefix>_n`, `_rel` (Frobenius distance to `S_ref` over its norm),
      `_norm`, `_probe` (`p^T S q` for two fixed index vectors), and
      `_rank_difference` (the Frobenius distance to the complement of the
      last rank over its norm; zero on one rank) and `_rank_sizes_agree`.
    """
    n = S.shape[0]
    p, q = index_vector(1, n), index_vector(2, n)
    other = comm.bcast(S.copy(), root=comm.size - 1)
    sizes = comm.allgather(n)
    return {f"{prefix}_n": n,
            f"{prefix}_rel": float(np.linalg.norm(S - S_ref) / np.linalg.norm(S_ref)),
            f"{prefix}_norm": float(np.linalg.norm(S)),
            f"{prefix}_probe": float(p @ S @ q),
            f"{prefix}_rank_difference": float(np.linalg.norm(S - other) / np.linalg.norm(S)),
            f"{prefix}_rank_sizes_agree": 1.0 if len(set(sizes)) == 1 else 0.0}


def ainvb_rows(meshes):
    """The cached two-block apply of `DtNTwoBlockSchurPC` (`dtn_schur_ainvb`).

    The dictionary of `test_dtn_schur_ainvb.direct_params`: block 0 is
    `preonly` with an assembled MUMPS LU, so `S` is exact and the full
    factorisation is the inverse of the operator. Rows
    (`test_the_cached_complement_matches_an_independent_elimination`,
    `test_the_forward_apply_inverts_the_operator`,
    `test_the_transpose_apply_inverts_the_transposed_operator`,
    `test_the_transpose_apply_is_not_the_forward_apply`,
    `test_the_forward_and_transpose_applies_are_adjoint`,
    `test_two_ranks_build_the_same_complement`): the complement against
    `reference_complement`, `|P^{-1} A x - x| / |x|`,
    `|P^{-T} A^T x - x| / |x|`, the distance of the forward and the
    transpose apply of one vector, and `<P^{-1} a, b>` against
    `<a, P^{-T} b>`. The vectors are coordinate-defined, with the rigid
    rotation removed (`RigidRotation`); the identity errors are written
    with and without their component along it.
    """
    solver = schur_solver(meshes, TA.direct_params())
    set_state(solver)
    comm = solver.solution_space.mesh().comm
    rotation = RigidRotation(solver)
    x = rotation.remove(monolithic(C.coordinate_mixed_function(solver, "schur:x", 1.0)))
    b = rotation.remove(monolithic(C.coordinate_mixed_function(solver, "schur:b", 1.0)))
    out = {}
    with preconditioner_set_up(solver) as pc:
        cache = TA.ainvb_cache(pc)
        A, _ = pc.getOperators()
        # The build is collective, so every rank built the cache exactly once.
        builds = comm.allgather(int(cache.build_count))
        out["ainvb_build_count_min"] = min(builds)
        out["ainvb_build_count_max"] = max(builds)
        S_ref = reference_complement(solver)
        out.update(complement_rows("ainvb", np.asarray(cache.S), S_ref, comm))
        for label, mult, apply in (("forward", A.mult, pc.apply),
                                   ("transpose", A.multTranspose, pc.applyTranspose)):
            image = A.createVecLeft()
            mult(x, image)
            y = A.createVecRight()
            apply(image, y)
            y.axpy(-1.0, x)
            out[f"ainvb_{label}_identity_raw"] = y.norm() / x.norm()
            out[f"ainvb_{label}_identity"] = rotation.remove(y).norm() / x.norm()
        forward = A.createVecRight()
        pc.apply(x, forward)
        transpose = A.createVecRight()
        pc.applyTranspose(x, transpose)
        difference = rotation.remove(forward)
        difference.axpy(-1.0, rotation.remove(transpose))
        out["ainvb_transpose_margin"] = difference.norm() / rotation.remove(forward).norm()
        transpose_b = A.createVecRight()
        pc.applyTranspose(b, transpose_b)
        left, right = forward.dot(b), x.dot(transpose_b)
        out["ainvb_adjoint_left"] = left
        out["ainvb_adjoint_right"] = right
    return out


def dense_rows(meshes):
    """The block-1 preconditioner `DtNMultiplierDenseSchurPC` of the direct preset.

    The dictionary of `test_dtn_multiplier_dense_schur._dense_schur_params`:
    block 0 an assembled MUMPS LU, block 1 GMRES on the Schur complement
    with the dense complement as its preconditioner. Rows (`T1`, `T2`, `T3`,
    `T7`): the complement against `reference_complement`, the forward apply
    solves `S x = rhs` (residual against `S_ref` and solution against
    `S_ref^{-1} rhs`), the transpose apply solves `S^T x = rhs`, and the
    forward and the transpose answers differ.
    """
    solver = schur_solver(meshes, TD._dense_schur_params())
    set_state(solver)
    comm = solver.solution_space.mesh().comm
    out = {}
    with preconditioner_set_up(solver):
        ksp_real, ctx = TD._dense_ctx(solver)
        if getattr(ctx, "_S", None) is None:
            ksp_real.setUp()
        S_ref = reference_complement(solver)
        out.update(complement_rows("dense", np.asarray(ctx._S), S_ref, comm))
        rhs = index_vector(3, ctx._n)
        forward = TD._apply_pc(ksp_real.getPC(), ctx, rhs)
        transpose = TD._apply_pc(ksp_real.getPC(), ctx, rhs, transpose=True)
        out["dense_forward_residual"] = float(
            np.linalg.norm(S_ref @ forward - rhs) / np.linalg.norm(rhs))
        solution = np.linalg.solve(S_ref, rhs)
        out["dense_forward_solution"] = float(
            np.linalg.norm(forward - solution) / np.linalg.norm(solution))
        solution_t = np.linalg.solve(S_ref.T, rhs)
        out["dense_transpose_solution"] = float(
            np.linalg.norm(transpose - solution_t) / np.linalg.norm(solution_t))
        out["dense_transpose_margin"] = float(
            np.linalg.norm(transpose - forward) / np.linalg.norm(forward))
    return out


def run_schur(meshes):
    """The rows of the entries `schur` and `schur-np2` (the same rows on 1 and 2 ranks)."""
    out = {"ranks": fd.COMM_WORLD.size}
    out.update(ainvb_rows(meshes))
    out.update(dense_rows(meshes))
    return out


RUNS = {"sea-transposes": run_sea_transposes, "frame": run_frame, "block0": run_block0,
        "block1": run_block1, "schur": run_schur, "schur-np2": run_schur}
assert set(RUNS) == set(OPERATORS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=sorted(OPERATORS))
    name = parser.parse_args().entry
    print(f"gadopt from {gadopt.__file__}", flush=True)
    values = RUNS[name](unit_meshes())
    M.write_values(CASE_DIR / operators_values(name), values)
