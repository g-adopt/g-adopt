"""Tolerances for the weak boundary condition diagnostics of the viscoelastic solvers.

The entrypoints in this directory do the assembly and the solves under
`doit run_case` and write their numbers to `.dat` files. This file reads those
files and applies the tolerances, so a failure names the case and prints the
number that missed.

Running these tests without running `doit run_case` first fails on the missing
file, which is how every case directory in `tests/` behaves.
"""

from math import log2
from pathlib import Path

import numpy as np
import pytest

from .gia_helpers import (
    BLOCK_SYMMETRY_CASES,
    REFINEMENT_RESOLUTIONS,
    WEAK_U_HISTORIES,
)
from .meta import (
    CONDENSATION_MG_COARSE,
    CONDENSATION_PRESET_CASES,
    CONDENSATION_SOLENOIDAL_DIMS,
    MESHES,
    REFINEMENT_DT_OVER_TAU,
)

CASE_DIR = Path(__file__).parent

# Symmetry is measured relative to the norm of the matrix itself, so this is a
# dimensionless distance from exact symmetry.
SYMMETRY_RTOL = 1e-13
# The variational structure identity is exact in exact arithmetic.
STRUCTURE_RTOL = 1e-12
# The displacement space is P2, so both the gap between the two formulations and
# the normal jump on the weak boundary must fall at least this fast.
REFINEMENT_ORDER = 2.0
# Below this fraction of the displacement norm the coarse gap is round-off, and
# a rate computed from it would be measuring noise instead of a difference
# between the two formulations.
GAP_FLOOR = 1e-8


def load(name):
    """Read one diagnostics file as rows and columns.

    `loadtxt` collapses a file holding a single row, or a single column, to one
    dimension, and the two cases are then indistinguishable. `ndmin=2` keeps the
    shape the file was written with, so every caller can index it as
    `[row][column]`.
    """
    return np.loadtxt(CASE_DIR / name, ndmin=2)


@pytest.mark.parametrize("mesh_key", MESHES)
def test_coupled_displacement_block_symmetry(mesh_key):
    """The displacement block of the coupled Jacobian is symmetric.

    At fixed history the coupled momentum stress is elastic, so a symmetrising
    term built with the effective viscosity instead of the elastic modulus
    leaves this block asymmetric by an amount that grows with dt/tau. The
    default coupled preset preconditions the block with CG, which assumes a
    symmetric operator, so this is a solver-level requirement.
    """
    rows = load(f"block_symmetry-{mesh_key}.dat")
    for (dt_over_tau, exponent), (ratio,) in zip(BLOCK_SYMMETRY_CASES, rows):
        assert ratio <= SYMMETRY_RTOL, (
            f"dt/tau={dt_over_tau}, exponent={exponent}: "
            f"asymmetry {ratio:.3e}")


@pytest.mark.parametrize("mesh_key", MESHES)
def test_pointwise_residual_is_first_variation(mesh_key):
    """The pointwise weak "un" residual is the first variation of its functional.

    `InternalVariableSolver` substitutes the backward-Euler update into the
    stress, so the variation carries the effective viscosity, which is what
    makes its symmetrising term and its penalty share a coefficient.
    """
    (ratio,), = load(f"pointwise_structure-{mesh_key}.dat")
    assert ratio <= STRUCTURE_RTOL, f"structure error {ratio:.3e}"


@pytest.mark.parametrize("mesh_key", MESHES)
def test_coupled_residual_is_first_variation(mesh_key):
    """The coupled weak "un" residual is the first variation at fixed history.

    The coupled stress is elastic in the displacement, so both its variation and
    its penalty carry the elastic modulus. That is what distinguishes this
    reference from the pointwise one.
    """
    (ratio,), = load(f"coupled_structure-{mesh_key}.dat")
    assert ratio <= STRUCTURE_RTOL, f"structure error {ratio:.3e}"


@pytest.mark.parametrize("mesh_key", MESHES)
def test_weak_u_jacobian_symmetric(mesh_key):
    """The weak "u" branch has a symmetric Jacobian for the viscoelastic stress.

    The branch has to work for a stress with a bulk part: the tangent and the
    penalty both pick that part up, and the residual is the first variation of a
    boundary functional, so its Jacobian is a Hessian.
    """
    rows = load(f"weak_u_symmetry-{mesh_key}.dat")
    for history, (ratio,) in zip(WEAK_U_HISTORIES, rows):
        assert ratio <= SYMMETRY_RTOL, f"{history}: asymmetry {ratio:.3e}"


@pytest.mark.parametrize("dt_over_tau", REFINEMENT_DT_OVER_TAU)
def test_coupled_matches_pointwise_under_refinement(dt_over_tau):
    """The coupled and pointwise solutions converge together under refinement.

    The coupled displacement rows differ from the pointwise ones by terms
    proportional to the normal jump on the weak boundary, and those terms vanish
    when the exact solution satisfies the boundary condition. Both the gap and
    the jump must therefore fall at the P2 order.

    This bounds the size of the difference between the two formulations. It does
    not detect an asymmetric displacement block;
    `test_coupled_displacement_block_symmetry` does that.
    """
    rows = load(f"refinement-{dt_over_tau}.dat")
    assert len(rows) == len(REFINEMENT_RESOLUTIONS)
    gaps, normal_jumps, displacement_norms = rows[:, 0], rows[:, 1], rows[:, 2]

    # The coarse gap must be a real difference between the two formulations and
    # not round-off, or the rates below are computed from noise.
    assert gaps[0] >= GAP_FLOOR * displacement_norms[0], (
        f"gap {gaps[0]:.3e} is at round-off relative to the displacement norm "
        f"{displacement_norms[0]:.3e}")

    gap_rate = log2(gaps[0] / gaps[1])
    jump_rate = log2(normal_jumps[0] / normal_jumps[1])
    assert gap_rate >= REFINEMENT_ORDER, f"gaps {gaps}, rate {gap_rate:.2f}"
    assert jump_rate >= REFINEMENT_ORDER, (
        f"normal jumps {normal_jumps}, rate {jump_rate:.2f}")


def test_coupled_iterative_preset_converges():
    """The coupled iterative preset converges with weak "un" boundaries.

    The preset eliminates the internal variables by static condensation and
    solves the condensed displacement operator with CG, which is defined only
    for a symmetric operator. A PETSc converged reason is positive when the
    solve converged.
    """
    (snes_reason, ksp_reason), = load("iterative_preset.dat")
    assert snes_reason > 0, f"SNES diverged, reason {int(snes_reason)}"
    assert ksp_reason > 0, (
        f"condensed displacement CG diverged, reason {int(ksp_reason)}")


# --------------------------------------------------------------------------
# The coupled solver under the static-condensation preset (gia_condensation.py)
#
# Each tolerance below is the one of the unit test that the row comes from, in
# tests/unit/test_internal_variable_history.py,
# tests/unit/test_near_incompressible_pc.py or
# tests/unit/test_coupled_solver_options.py. The docstring of each test names
# that unit test.

# The preset and a direct solve of the same system, relative L2 error.
PRESET_RTOL = 1e-8
# The condensed operator with the boundary term, ||S - S^T|| / ||S||.
CONDENSED_SYMMETRY_RTOL = 1e-12
# The lower bound on the asymmetry where the operator is not symmetric.
MISSING_TERM_ASYMMETRY = 1e-4
TWO_ELEMENT_POWER_LAW_ASYMMETRY = 1e-8
HEXAHEDRAL_POWER_LAW_ASYMMETRY = 1e-4
# Two layouts of the same history, relative L2 error of the displacement and
# absolute integral of the squared history difference.
LAYOUT_RTOL = 1e-12
LAYOUT_HISTORY_ATOL = 1e-24
# Full storage with a residual scaling factor against the symmetric field,
# integral of the squared history difference relative to that of the history.
FULL_STORAGE_HISTORY_RTOL = 1e-16
# A singular condensed operator: the image of the kernel mode relative to
# ||S|| ||mode||, and the centred displacement against a direct solve.
MODE_IMAGE_RTOL = 1e-12
SINGULAR_RTOL = 1e-7
# The gap between the substituted and the coupled solver: an upper bound on
# the coarse mesh and the factor by which one refinement must reduce it.
GAP_COARSE_MAX = 1e-3
GAP_REDUCTION = 4
# The divergence of the solenoidal modes relative to their norm: on affine
# cells, after the filter on a curved mesh, and the lower bound on the
# divergence of the unfiltered degree-two modes on a curved mesh.
AFFINE_DIVERGENCE_RTOL = 1e-11
FILTERED_DIVERGENCE_RTOL = 1e-8
CURVED_LEAK_MIN = 1e-2
# The divergence-free modes must save at least this fraction of the CG
# iterations of the rigid-body modes.
INCOMPRESSIBLE_ITERATION_FRACTION = 0.8
# The iteration caps of the elasticity solves of the unit tests.
GAMG_MAX_ITERATIONS = 5000
MG_MAX_ITERATIONS = 2000


def load_values(name):
    """Read a `name value` file into a dictionary.

    A value that parses as a number is returned as a float, any other value
    (a PETSc type name, `None`) as a string.
    """
    values = {}
    for line in (CASE_DIR / name).read_text().splitlines():
        key, value = line.split()
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    return values


def assert_matches_reference(values, prefix, rtol):
    """Assert `<prefix>_error < rtol` with a reference norm above zero.

    A zero reference norm would make the relative error meaningless, so the
    reference must be nonzero before the error is read.
    """
    reference = values[f"{prefix}_reference_norm"]
    error = values[f"{prefix}_error"]
    assert reference > 0, f"{prefix}: reference norm {reference!r} is zero"
    assert error < rtol, (
        f"{prefix}: relative error {error!r} against the reference "
        f"(norm {reference!r}), tolerance {rtol!r}")


@pytest.mark.parametrize(
    "n_elements, bc_kind", CONDENSATION_PRESET_CASES,
    ids=[f"n{n}-{kind}" for n, kind in CONDENSATION_PRESET_CASES])
def test_static_condensation_preset_matches_direct_solve(n_elements, bc_kind):
    """The preset converges and agrees with a direct solve for 1 to 3 elements.

    From `test_internal_variable_history.py::
    test_static_condensation_preset_matches_direct_solve`: the preset is
    `gadopt.InternalVariableSCPC` with CG on the condensed field, both solves
    converge, and each field agrees with the direct solve to 1e-8 after two
    steps. Three elements is the case Firedrake's `SCPC` refuses in a layout
    with one field per element.
    """
    values = load_values("condensation-preset.dat")
    key = f"preset-n{n_elements}-{bc_kind}"
    assert values[f"{key}-pc_python_type"] == "gadopt.InternalVariableSCPC", (
        f"preset {values[f'{key}-pc_python_type']!r}")
    assert values[f"{key}-condensed_ksp_type"] == "cg", (
        f"condensed Krylov method {values[f'{key}-condensed_ksp_type']!r}, "
        "expected 'cg' for a Newtonian rheology")
    assert values[f"{key}-snes_reason"] > 0, (
        f"SNES reason {values[f'{key}-snes_reason']!r}")
    assert values[f"{key}-condensed_ksp_reason"] > 0, (
        f"condensed KSP reason {values[f'{key}-condensed_ksp_reason']!r}")
    for field in ("u", "m"):
        assert_matches_reference(values, f"{key}-{field}", PRESET_RTOL)


def test_power_law_newton_with_static_condensation():
    """A power law runs Newton on GMRES under the preset and matches a direct solve.

    From `test_internal_variable_history.py::
    test_power_law_newton_with_static_condensation`: `newtonls` with GMRES on
    the condensed field, the rule reports the operator as not symmetric, Newton
    converges in at least one iteration and in as many as the direct solve,
    the condensed operator is assembled once per Newton iteration with
    `operator_version` None, and each field agrees with the direct solve to
    1e-8.
    """
    values = load_values("condensation-preset.dat")
    assert values["power-snes_type"] == "newtonls", repr(values["power-snes_type"])
    assert values["power-condensed_ksp_type"] == "gmres", (
        f"condensed Krylov method {values['power-condensed_ksp_type']!r}, "
        "expected 'gmres' for a power law")
    assert values["power-rule_says_symmetric"] == 0
    assert values["power-snes_reason"] > 0, repr(values["power-snes_reason"])
    newton_its = values["power-newton_its"]
    assert newton_its >= 1, f"Newton iterations {newton_its!r}"
    assert newton_its == values["power-direct_newton_its"], (
        f"Newton iterations {newton_its!r} under the preset, "
        f"{values['power-direct_newton_its']!r} under the direct solve")
    assert values["power-assembly_count"] == newton_its, (
        f"condensed operator assembled {values['power-assembly_count']!r} times "
        f"in {newton_its!r} Newton iterations")
    assert values["power-operator_version"] == "None", (
        repr(values["power-operator_version"]))
    for field in ("u", "m"):
        assert_matches_reference(values, f"power-{field}", PRESET_RTOL)


def test_extruded_hex_with_weak_boundary_matches_direct():
    """Static condensation on an extruded hexahedral mesh with the boundary term.

    From `test_internal_variable_history.py::
    test_extruded_hex_with_weak_boundary_matches_direct`: the condensed
    operator that the preconditioner assembled is symmetric to 1e-12, and each
    field agrees with the direct solve to 1e-8.
    """
    values = load_values("condensation-preset.dat")
    ratio = values["extruded-condensed_asymmetry"]
    assert ratio < CONDENSED_SYMMETRY_RTOL, (
        f"asymmetry {ratio!r}, tolerance {CONDENSED_SYMMETRY_RTOL!r}")
    assert values["extruded-condensed_ksp_reason"] > 0
    for field in ("u", "m"):
        assert_matches_reference(values, f"extruded-{field}", PRESET_RTOL)


def test_history_layouts_agree():
    """The `(d, d)` and the `(n, d, d)` layouts of the history give one answer.

    From four unit tests in `test_internal_variable_history.py`:
    `test_single_element_legacy_layout_matches_combined` (coupled solver,
    displacement to 1e-12, squared history difference below 1e-24),
    `test_full_storage_and_scaling_factor_under_the_preset` (condensed
    operator symmetric to 1e-12, displacement to 1e-8, squared history
    difference below 1e-16 of the squared history),
    `test_substituted_solver_layouts_agree` (displacement to 1e-12, squared
    history difference below 1e-24, a nonzero history, two updates) and
    `test_substituted_solver_single_element_layouts_agree` (displacement to
    1e-12).
    """
    values = load_values("condensation-preset.dat")

    assert_matches_reference(values, "layout-coupled-u", LAYOUT_RTOL)
    assert values["layout-coupled-m_reference_squared"] > 0
    difference = values["layout-coupled-m_squared_difference"]
    assert difference < LAYOUT_HISTORY_ATOL, (
        f"coupled history difference {difference!r}, "
        f"tolerance {LAYOUT_HISTORY_ATOL!r}")

    ratio = values["layout-full-condensed_asymmetry"]
    assert ratio < CONDENSED_SYMMETRY_RTOL, (
        f"full storage asymmetry {ratio!r}, tolerance {CONDENSED_SYMMETRY_RTOL!r}")
    assert_matches_reference(values, "layout-full-u", PRESET_RTOL)
    reference = values["layout-full-m_reference_squared"]
    difference = values["layout-full-m_squared_difference"]
    assert reference > 0
    assert difference < FULL_STORAGE_HISTORY_RTOL * reference, (
        f"full storage history difference {difference!r}, reference "
        f"{reference!r}, relative tolerance {FULL_STORAGE_HISTORY_RTOL!r}")

    assert_matches_reference(values, "layout-substituted-u", LAYOUT_RTOL)
    difference = values["layout-substituted-m_squared_difference"]
    assert difference < LAYOUT_HISTORY_ATOL, (
        f"substituted history difference {difference!r}, "
        f"tolerance {LAYOUT_HISTORY_ATOL!r}")
    assert values["layout-substituted-m0_norm"] > 0, "the history did not advance"
    assert values["layout-substituted-n_updates"] == 2

    assert_matches_reference(values, "layout-substituted_single-u", LAYOUT_RTOL)


def test_condensed_operator_symmetric_because_of_boundary_term():
    """With the boundary term the condensed operator is symmetric; without it, not.

    From `test_internal_variable_history.py::
    test_condensed_operator_symmetric_because_of_boundary_term`: asymmetry
    below 1e-12 with the term, a nonzero boundary term, and asymmetry above
    1e-4 without it at `dt = 25`.
    """
    values = load_values("condensation-operator.dat")
    with_term = values["symmetry-with_term_asymmetry"]
    without_term = values["symmetry-without_term_asymmetry"]
    assert with_term < CONDENSED_SYMMETRY_RTOL, (
        f"asymmetry with the term {with_term!r}, "
        f"tolerance {CONDENSED_SYMMETRY_RTOL!r}")
    assert values["symmetry-boundary_term_norm"] > 0
    assert without_term > MISSING_TERM_ASYMMETRY, (
        f"asymmetry without the term {without_term!r}, expected above "
        f"{MISSING_TERM_ASYMMETRY!r}")


def test_power_law_condensed_operator_symmetry():
    """The power-law condensed operator: where it is symmetric and where it is not.

    From `test_internal_variable_history.py::
    test_power_law_condensed_operator_is_not_symmetric` (two elements in 2-D
    above 1e-8; one element on tetrahedra below 1e-12, which the rule still
    reports as not symmetric) and
    `test_one_element_power_law_is_asymmetric_on_hexahedra` (the same state
    on hexahedra above 1e-4).
    """
    values = load_values("condensation-operator.dat")
    two_elements = values["power_symmetry-two_elements_asymmetry"]
    tetrahedra = values["power_symmetry-tetrahedra_asymmetry"]
    hexahedra = values["power_symmetry-hexahedra_asymmetry"]
    assert two_elements > TWO_ELEMENT_POWER_LAW_ASYMMETRY, (
        f"two-element asymmetry {two_elements!r}, expected above "
        f"{TWO_ELEMENT_POWER_LAW_ASYMMETRY!r}")
    assert tetrahedra < CONDENSED_SYMMETRY_RTOL, (
        f"tetrahedral asymmetry {tetrahedra!r}, "
        f"tolerance {CONDENSED_SYMMETRY_RTOL!r}")
    assert values["power_symmetry-tetrahedra_rule_says_symmetric"] == 0
    assert hexahedra > HEXAHEDRAL_POWER_LAW_ASYMMETRY, (
        f"hexahedral asymmetry {hexahedra!r}, expected above "
        f"{HEXAHEDRAL_POWER_LAW_ASYMMETRY!r}")


def test_condensed_operator_reuse_and_rebuild():
    """The condensed operator is built once per change of the Jacobian.

    From `test_internal_variable_history.py::
    test_condensed_operator_reuse_and_rebuild` (assembly counts 1, 2, 2, 3, 4
    after three solves, a change of `dt`, a solve at the same `dt`, a change
    of the viscosity and `invalidate_jacobian`; a fresh solver agrees to
    1e-8) and `test_constant_exponent_one_keeps_operator_reuse` (`ksponly`,
    one assembly in three solves).
    """
    values = load_values("condensation-operator.dat")
    expected = {
        "reuse-count_after_three_solves": 1,
        "reuse-count_after_dt_change": 2,
        "reuse-count_after_same_dt": 2,
        "reuse-count_after_viscosity_change": 3,
        "reuse-count_after_invalidate": 4,
    }
    measured = {key: values[key] for key in expected}
    assert measured == expected, f"assembly counts {measured!r}, expected {expected!r}"
    assert values["reuse-u_norm"] > 0
    error = values["reuse-fresh_u_error"]
    assert error < PRESET_RTOL, (
        f"fresh solver error {error!r}, tolerance {PRESET_RTOL!r}")
    assert values["reuse_constant_exponent-snes_type"] == "ksponly", (
        repr(values["reuse_constant_exponent-snes_type"]))
    count = values["reuse_constant_exponent-assembly_count"]
    assert count == 1, f"assembly count {count!r} in three solves, expected 1"


def test_singular_condensed_operator_solved_with_cg_and_nullspace():
    """A translation reaches the condensed operator, and CG solves the singular system.

    From `test_internal_variable_history.py::
    test_singular_condensed_operator_solved_with_cg_and_nullspace`: all three
    nullspaces are set on the condensed operator, the translation is in its
    kernel to 1e-12, CG converges, and the centred displacement agrees with a
    direct solve to 1e-7.
    """
    values = load_values("condensation-operator.dat")
    for kind in ("nullspace", "transpose_nullspace", "near_nullspace"):
        assert values[f"singular-{kind}_set"] == 1, f"{kind} not set"
    image = values["singular-mode_image"]
    assert image < MODE_IMAGE_RTOL, (
        f"image of the mode {image!r}, tolerance {MODE_IMAGE_RTOL!r}")
    assert values["singular-condensed_ksp_reason"] > 0
    assert values["singular-condensed_ksp_type"] == "cg"
    reference = values["singular-u_reference_norm"]
    difference = values["singular-u_difference_norm"]
    assert reference > 0
    assert difference < SINGULAR_RTOL * reference, (
        f"centred difference {difference!r}, reference {reference!r}, "
        f"relative tolerance {SINGULAR_RTOL!r}")


def test_substituted_and_coupled_solvers_converge_to_each_other():
    """The gap between the substituted and the coupled solver shrinks under refinement.

    From `test_internal_variable_history.py::
    test_substituted_and_coupled_solvers_converge_to_each_other`: the gap is
    below 1e-3 on 4 by 4 cells and falls by a factor of four on 8 by 8 cells.
    """
    values = load_values("condensation-operator.dat")
    coarse, fine = values["gap-coarse"], values["gap-fine"]
    assert coarse < GAP_COARSE_MAX, f"coarse gap {coarse!r}"
    assert fine < coarse / GAP_REDUCTION, (
        f"gaps {coarse!r} and {fine!r}, reduction below {GAP_REDUCTION!r}")


@pytest.mark.parametrize("dim", CONDENSATION_SOLENOIDAL_DIMS)
def test_solenoidal_modes_are_divergence_free_on_affine_mesh(dim):
    """Every generated mode is divergence-free on an affine simplex mesh.

    From `test_near_incompressible_pc.py::
    test_solenoidal_modes_are_divergence_free_on_affine_mesh`: at least one
    mode, and each below 1e-11 relative divergence.
    """
    values = load_values("condensation-pc.dat")
    assert values[f"modes-{dim}d-count"] > 0
    divergence = values[f"modes-{dim}d-max_divergence"]
    assert divergence < AFFINE_DIVERGENCE_RTOL, (
        f"largest relative divergence {divergence!r}, "
        f"tolerance {AFFINE_DIVERGENCE_RTOL!r}")


def test_divfree_filter_drops_nonsolenoidal_modes_on_curved_mesh():
    """On a curved mesh the filter drops the modes that leak divergence.

    From `test_near_incompressible_pc.py::
    test_divfree_filter_drops_nonsolenoidal_modes_on_curved_mesh`: the
    unfiltered degree-two modes leak above 1e-2, the filter drops some, and
    the survivors are below 1e-8.
    """
    values = load_values("condensation-pc.dat")
    leak = values["modes-curved-unfiltered_max_divergence"]
    assert leak > CURVED_LEAK_MIN, (
        f"largest unfiltered divergence {leak!r}, expected above {CURVED_LEAK_MIN!r}")
    assert (values["modes-curved-filtered_count"]
            < values["modes-curved-unfiltered_count"])
    divergence = values["modes-curved-filtered_max_divergence"]
    assert divergence < FILTERED_DIVERGENCE_RTOL, (
        f"largest filtered divergence {divergence!r}, "
        f"tolerance {FILTERED_DIVERGENCE_RTOL!r}")


def test_near_incompressible_modes_lower_the_iteration_count():
    """The divergence-free modes lower the CG iterations of GAMG.

    From `test_near_incompressible_pc.py::
    test_pc_reduces_iterations_versus_rigid_modes` (fewer iterations with the
    divergence-free modes than with the rigid modes, fewer with those than
    with none, and at most 0.8 of the rigid count) and
    `test_pc_default_does_not_saturate_gamg` (between 0 and 5000 iterations
    with the default mode count).
    """
    values = load_values("condensation-pc.dat")
    none = values["iterations-none"]
    rigid = values["iterations-rigid"]
    incompressible = values["iterations-incompressible"]
    assert incompressible < rigid < none, (
        f"iterations: incompressible {incompressible!r}, rigid {rigid!r}, "
        f"none {none!r}")
    assert incompressible <= INCOMPRESSIBLE_ITERATION_FRACTION * rigid, (
        f"incompressible {incompressible!r} against rigid {rigid!r}, "
        f"fraction {INCOMPRESSIBLE_ITERATION_FRACTION!r}")
    its = values["iterations-default_modes"]
    assert 0 < its < GAMG_MAX_ITERATIONS, f"default modes: {its!r} iterations"


@pytest.mark.parametrize("coarse", CONDENSATION_MG_COARSE)
def test_mg_patch_parameters_converge(coarse):
    """The patch multigrid parameters converge on a nearly incompressible operator.

    From `test_near_incompressible_pc.py::test_mg_patch_parameters_converge`:
    between 0 and 2000 iterations.
    """
    values = load_values("condensation-pc.dat")
    its = values[f"mg-{coarse}-iterations"]
    assert 0 < its < MG_MAX_ITERATIONS, f"{coarse} coarse solve: {its!r} iterations"


def test_the_taped_solve_builds_the_derivative_solvers_with_these_options():
    """The adjoint options reach the derivative solvers that Firedrake builds.

    From `test_coupled_solver_options.py::TestAdjointSolverOptions::
    test_the_taped_solve_builds_the_derivative_solvers_with_these_options`:
    the forward runs `newtonls`; every derivative solver runs `ksponly`, has
    none of the forward-only SNES options, and has every flat Krylov option
    of `adjoint_solver_parameters`.
    """
    values = load_values("condensation-pc.dat")
    assert values["taped-forward_snes_type"] == "newtonls"
    n_solvers = int(values["taped-n_derivative_solvers"])
    assert n_solvers >= 1
    assert values["taped-krylov_options_compared"] > 0
    for i in range(n_solvers):
        assert values[f"taped-{i}-snes_type"] == "ksponly", (
            f"derivative solver {i}: {values[f'taped-{i}-snes_type']!r}")
        assert values[f"taped-{i}-forward_only_options"] == 0, (
            f"derivative solver {i}: "
            f"{values[f'taped-{i}-forward_only_options']!r} forward-only options")
        assert values[f"taped-{i}-krylov_mismatches"] == 0, (
            f"derivative solver {i}: "
            f"{values[f'taped-{i}-krylov_mismatches']!r} Krylov options differ")


def test_spd_and_modes_are_independently_reachable():
    """The SPD flag and the divergence-free near-nullspace reach one block together.

    From `test_coupled_solver_options.py::TestAssembledPCFactoring::
    test_spd_and_modes_are_now_independently_reachable`: 11 near-nullspace
    vectors in 3-D, the SPD flag set, and a nonzero solution.
    """
    values = load_values("condensation-pc.dat")
    assert values["spd-near_nullspace_vectors"] == 11, (
        repr(values["spd-near_nullspace_vectors"]))
    assert values["spd-flag"] == 1
    assert values["spd-solution_norm"] > 0
