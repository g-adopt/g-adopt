"""Tests of the `operators` case: identities of the coupled operator, without a solve.

`gia_operators.py` writes, for each row, the measured value and the value it
is checked against to `out/operators-<entry>.dat`. Each test here holds the
tolerance of the unit test that the row comes from, named in its docstring.
No value is recorded from an earlier run.

A transpose row writes `x_a . J_ab x_b` (`_ab`), `x_b . J_ba x_a` (`_ba`)
and their Cauchy-Schwarz scale (`_scale`); the unit tests compared the dense
blocks entry by entry relative to the larger block, and the pairing is the
same statement for coordinate-defined directions.

This file does not import Firedrake; it runs where the outputs are.
"""

import pytest

import gia_entries as E


def values(entry):
    """The `name value` lines of `out/operators-<entry>.dat` as a dictionary."""
    path = E.CASE_DIR / E.operators_values(entry)
    out = {}
    with open(path) as f:
        for line in f:
            name, value = line.split()
            out[name] = float(value)
    return out


def assert_transposes(v, prefix):
    """The pair `prefix` transposes to `TRANSPOSE_RTOL` and its pairing is not round-off."""
    ab, ba, scale = v[f"{prefix}_ab"], v[f"{prefix}_ba"], v[f"{prefix}_scale"]
    assert scale > 0.0, f"{prefix}: both blocks are zero (scale {scale!r})"
    assert abs(ab) >= E.PAIRING_FLOOR * scale, (
        f"{prefix}: pairing {ab!r} below {E.PAIRING_FLOOR!r} of its scale {scale!r}")
    bound = E.TRANSPOSE_RTOL * scale
    assert abs(ab - ba) <= bound, (
        f"{prefix}: x_a.J_ab x_b = {ab!r}, x_b.J_ba x_a = {ba!r}, difference "
        f"{abs(ab - ba)!r} above {E.TRANSPOSE_RTOL!r} x scale {scale!r} = {bound!r}")


# -- sea-transposes -----------------------------------------------------------------

@pytest.mark.parametrize("pair", E.SEA_PAIRS, ids=lambda p: "-".join(p))
@pytest.mark.parametrize("state", E.SEA_STATES)
def test_the_sheet_blocks_transpose(state, pair):
    """Every coupling block the sea-level sheet creates is the transpose of its partner.

    `test_sea_level.py::TestTransposes::test_the_pair_transposes` (1e-14 of
    the larger block), on the saturated `cap` state and the live-coastline
    `shoreline` state. A sign of the sheet in one row breaks the pairs of
    that row (M1: the displacement row, pairs `u-psi` and `u-Shift`).
    """
    assert_transposes(values("sea-transposes"), f"{state}_{pair[0]}_{pair[1]}")


@pytest.mark.parametrize("pair", [
    pytest.param(("u", "rot"), marks=pytest.mark.xfail(
        strict=True,
        reason="On the Quadrature family the pair (u, m3) transposes only to "
               "6.4e-10 to 6.9e-10 of its scale (measured on the cap, shoreline "
               "and ocean states and without sea level; absolute difference "
               "1.62e-10 in every case): the rotational body force "
               "(gadopt.momentum_equation.rotational_potential_term) is "
               "integrated with the mechanics measure at quad_degree 4, the "
               "mantle term of Rotation.inertia_form with dx_m at the estimated "
               "degree; the two integrals of rho grad(p_3) . u differ by 2.5e-9 "
               "relative.")),
    ("psi", "rot"), ("Shift", "rot")], ids=lambda p: "-".join(p))
def test_the_rotation_pair_transposes(pair):
    """The rotation row against the sheet, on the live-coastline state.

    `test_sea_level.py::TestRotationTransposes::
    test_the_rotation_pair_transposes` (1e-14 of the larger block). Without
    the sheet in `Rotation.inertia_form` the closure row loses its sea-level
    part and the `Shift` pair fails (M8).
    """
    assert_transposes(values("sea-transposes"), f"rotation_{pair[0]}_{pair[1]}")


def test_the_shift_rotation_block_is_nonzero():
    """`(Shift, m3)` and `(m3, Shift)` both carry the sheet.

    Both entries are above `SHIFT_ROTATION_FLOOR` (1e-4) of the
    `(Shift, Shift)` diagonal; the measured ratio is 4.8e-3. Without the
    sheet in `inertia_form` (M8) the entry `(m3, Shift)` falls to 7e-8 of
    the diagonal; without `psi_rot` in the sea level the entry
    `(Shift, m3)` is zero.
    """
    v = values("sea-transposes")
    diagonal = abs(v["shift_shift_entry"])
    assert diagonal > 0.0
    floor = E.SHIFT_ROTATION_FLOOR * diagonal
    for name in ("shift_rot_entry", "rot_shift_entry"):
        assert abs(v[name]) > floor, f"{name} = {v[name]!r}, must exceed {floor!r}"


def test_the_mechanics_row_matches_a_normal_stress():
    """The displacement row of the ocean sheet equals a normal stress of the same weight.

    `test_sea_level.py::TestSheetSigns::
    test_the_mechanics_row_matches_a_normal_stress`: max norm of the
    difference at most 1e-6 of the row of a plain `normal_stress`
    `B_mu g(Re) rho_w S`. A flipped sheet sign (M1) is a relative
    difference of 2.
    """
    v = values("sea-transposes")
    scale = v["normal_stress_scale"]
    assert scale > 0.0
    bound = E.NORMAL_STRESS_RTOL * scale
    assert v["normal_stress_difference"] <= bound, (
        f"mechanics row: difference {v['normal_stress_difference']!r} above {bound!r}")


# -- frame --------------------------------------------------------------------------

FRAME_2D = [f"frame_{rep}_{'rot' if rot else 'norot'}" for rep, rot in E.FRAME_CONFIGURATIONS]
SPHERE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="On the 24-column sphere with the Quadrature family the pair (u, "
           "lambda_i) transposes only to 5.3e-9 to 5.4e-9 of its scale (pairings "
           "19.116666312 against 19.116666858 for lambda_x). The CentreOfMass "
           "residual alone is symmetric to 2e-16; in the whole residual the "
           "column term beta lambda rho w_i dx_m of the displacement row is "
           "integrated at the mechanics quad_degree 4, the constraint row at "
           "the estimated degree (the difference is beta times the degree-4 "
           "error of int rho u_i dx, 4.3e-7). On the 2-D annulus, where dx_m is "
           "an intersected measure, the pair agrees to 1e-16.")


@pytest.mark.parametrize("multiplier", ["lambda_x", "lambda_y"])
@pytest.mark.parametrize("prefix", FRAME_2D)
def test_the_frame_blocks_form_a_saddle_pair(prefix, multiplier):
    """`(u, lambda_i)` is the transpose of `(lambda_i, u)` on the annulus.

    `test_gia_gravity.py::TestCentreOfMassFrame::
    test_blocks_form_an_exact_saddle_pair` (1e-14 of the larger block), on
    both DtN representations, with and without rotation.
    """
    v = values("frame")
    assert_transposes({k.replace(f"{prefix}_u_{multiplier}", "pair"): x
                       for k, x in v.items()}, "pair")


@pytest.mark.parametrize("multiplier", [
    pytest.param(m, marks=SPHERE_XFAIL) for m in ("lambda_x", "lambda_y", "lambda_z")])
def test_the_sphere_frame_blocks_form_a_saddle_pair(multiplier):
    """The three multipliers of the 24-column sphere, assembly only.

    `test_gia_gravity.py::TestCentreOfMassFrame::
    test_extruded_sphere_assembles_the_saddle_pair` (1e-14 of the larger
    block).
    """
    v = values("frame")
    assert_transposes({k.replace(f"frame_sphere_u_{multiplier}", "pair"): x
                       for k, x in v.items()}, "pair")


@pytest.mark.parametrize("prefix", FRAME_2D + ["frame_sphere"])
def test_the_frame_multipliers_do_not_couple(prefix):
    """The multiplier-multiplier block is exactly zero: the form has no such term.

    The second assertion of `test_blocks_form_an_exact_saddle_pair`
    (`== 0.0`).
    """
    value = values("frame")[f"{prefix}_multiplier_block_max"]
    assert value == 0.0, f"{prefix}: largest multiplier-multiplier entry {value!r}"


def test_the_sphere_translation_moment_is_the_total_mass():
    """`D[u = e_i] = M e_i` on the sphere, per direction.

    `test_extruded_sphere_assembles_the_saddle_pair`: `D_i` within 1e-2 of
    the whole mass, the other two moments below 1e-10 of it.
    """
    v = values("frame")
    mass = v["sphere_mass"]
    assert mass > 0.0
    for i in range(3):
        moment, cross = v[f"sphere_moment_{i}"], v[f"sphere_moment_cross_{i}"]
        assert abs(moment - mass) < E.SPHERE_MOMENT_RTOL * mass, (
            f"D_{i}[e_{i}] = {moment!r} against the mass {mass!r}")
        assert cross < E.SPHERE_MOMENT_CROSS_RTOL * mass, (
            f"largest other moment of e_{i}: {cross!r}, mass {mass!r}")


def test_the_inertia_of_a_cos2_sheet_vanishes():
    """`int sigma (x^2 + y^2) dS` of the `cos 2 phi` load is zero.

    `test_gia_gravity.py::TestInertiaTranspose::
    test_inertia_of_a_cos2_sheet_vanishes` (absolute 1e-14), on the load's
    interior facet measure of the parent, read at `u = 0`.
    """
    v = values("frame")
    assert v["inertia_cos2_scale"] > 0.0
    assert abs(v["inertia_cos2"]) < E.INERTIA_COS2_ATOL, (
        f"dI_33 of the cos 2 phi sheet: {v['inertia_cos2']!r} "
        f"(int |sigma| r^2 = {v['inertia_cos2_scale']!r})")


# -- block0 -------------------------------------------------------------------------

def assert_symmetric(v, prefix):
    """The pairing matrix `prefix` is symmetric to `BLOCK0_SYMMETRY_RTOL` of its largest entry."""
    largest = v[f"{prefix}_pairing_max"]
    assert largest > 0.0
    bound = E.BLOCK0_SYMMETRY_RTOL * largest
    assert v[f"{prefix}_asymmetry"] <= bound, (
        f"{prefix}: asymmetry {v[f'{prefix}_asymmetry']!r} above {bound!r}")


def test_the_eliminated_block_is_symmetric():
    """Block 0 of the Quadrature family, with the history eliminated, is symmetric.

    `test_gia_gravity.py::TestEliminatedBlockSymmetry::
    test_the_eliminated_block_is_symmetric` (Quadrature half, 1e-12).
    """
    assert_symmetric(values("block0"), "eliminated")


def test_a_live_coastline_leaves_block_zero_symmetric():
    """With a live coastline the condensed block 0 is still symmetric.

    `test_sea_level.py::TestLiveCoastlineRebuildRules::
    test_block_zero_symmetry_is_unchanged`, measured on the operator
    (1e-12). The sea-level part of the block is above `SEA_BLOCK0_FLOOR` of
    the whole, so the measurement includes the live sheet.
    """
    v = values("block0")
    assert_symmetric(v, "coastline")
    floor = E.SEA_BLOCK0_FLOOR * v["coastline_pairing_max"]
    assert v["coastline_sea_pairing_max"] > floor, (
        f"sea-level part {v['coastline_sea_pairing_max']!r}, must exceed {floor!r}")


# -- block1 -------------------------------------------------------------------------

BLOCK1 = [f"block1_t{t}_{r}" for t in E.BLOCK1_TRUNCATIONS for r in ("norot", "rot")]


@pytest.mark.parametrize("prefix", BLOCK1 + ["block1_sphere"])
def test_the_real_block_is_the_claimed_diagonal(prefix):
    """The `Real` block equals its derived diagonal, entry by entry.

    `test_gia_gravity.py::TestBlockOneIsTheDiagonalItClaims::
    test_the_block_is_the_claimed_diagonal` (truncations 3 and 5, rotation
    on and off) and `TestBlockOneInThreeDimensions::
    test_the_block_is_the_claimed_diagonal`: `DIAG_TOL` and `OFFDIAG_TOL`
    1e-12. The fluid core adds the core pressure, a Lagrange multiplier
    whose derived entry is zero.
    """
    v = values("block1")
    assert v[f"{prefix}_count"] == v[f"{prefix}_expected_count"]
    assert v[f"{prefix}_diag_min"] > 0.0, "a zero derived-nonzero entry makes the check vacuous"
    assert v[f"{prefix}_zero_rows"] == 1.0, "only the core pressure has a zero diagonal"
    for name, tol in (("diag_rel", E.DIAG_TOL), ("zero_diag", E.DIAG_TOL),
                      ("offdiag", E.OFFDIAG_TOL), ("zero_offdiag", E.OFFDIAG_TOL)):
        assert v[f"{prefix}_{name}"] <= tol, (
            f"{prefix} {name}: {v[f'{prefix}_{name}']!r} above {tol!r}")


@pytest.mark.parametrize("prefix, silent", [
    ("alignment_t3", E.ALIGNMENT_SILENT_2D), ("alignment_t5", E.ALIGNMENT_SILENT_2D),
    ("alignment_sphere", E.ALIGNMENT_SILENT_SPHERE)])
def test_each_multiplier_row_is_the_mode_its_key_names(prefix, silent):
    """Driving `psi` with a mode moves exactly the rows of its key, by the discrete integral.

    `TestBlockOneIsTheDiagonalItClaims::
    test_each_multiplier_row_is_the_mode_its_key_names` (truncations 3 and
    5) and `TestBlockOneInThreeDimensions::
    test_each_multiplier_row_is_the_mode_its_key_names` (`L = 1`): fired
    rows to 1e-12 of `theta_psi int psi_h e_k ds`, silent rows below 1e-6 of
    the weakest fired row, the analytic deviation below 1e-2.
    """
    v = values("block1")
    assert v[f"{prefix}_order"] == 1.0, "the multipliers are not the first Real fields"
    assert v[f"{prefix}_fired_min"] > 0.0
    assert v[f"{prefix}_discrete"] <= E.ALIGNMENT_PREDICTED, v[f"{prefix}_discrete"]
    assert v[f"{prefix}_silent"] < silent, v[f"{prefix}_silent"]
    assert v[f"{prefix}_other_rows"] < silent, v[f"{prefix}_other_rows"]
    assert v[f"{prefix}_analytic"] < E.ALIGNMENT_ANALYTIC, v[f"{prefix}_analytic"]


def test_the_polar_pair_is_uncoupled_and_carries_C_minus_A():
    """`m1`, `m2` carry `C - A`, `m3` carries `C` with the opposite sign, nothing couples.

    `TestBlockOneInThreeDimensions::
    test_the_polar_pair_is_uncoupled_and_carries_C_minus_A`: the pair equal
    to 1e-14, the ratio `m3 / m1 = -10` to 1e-12 (`C - A = 0.1 C`), and the
    corners below `OFFDIAG_TOL` of the smallest rotation entry. The values
    of the entries themselves are checked by the derived diagonal.
    """
    v = values("block1")
    m1, m2, m3 = v["sphere_m1"], v["sphere_m2"], v["sphere_m3"]
    assert m1 > 0.0 and m3 < 0.0
    assert abs(m1 - m2) <= E.POLAR_PAIR_RTOL * abs(m1), (m1, m2)
    ratio = m3 / m1
    assert abs(ratio - E.POLAR_RATIO) <= E.POLAR_RATIO_RTOL * abs(E.POLAR_RATIO), ratio
    floor = E.OFFDIAG_TOL * v["sphere_rotation_min"]
    for name in ("sphere_rotation_multiplier_max", "sphere_multiplier_rotation_max",
                 "sphere_rotation_offdiag_max"):
        assert v[name] <= floor, f"{name} = {v[name]!r} above {floor!r}"


def test_the_potential_row_is_not_annihilated_at_zero_B_mu():
    """At `B_mu = 0` the potential rows survive and only the body force is gone.

    `test_gia_gravity.py::TestNullCoupling::
    test_the_potential_row_is_not_annihilated`, with action norms in place
    of the dense block maxima: `|J_psi,psi x|`, `|J_mult,psi x|` and
    `|J_psi,u x|` are positive, `|J_u,psi x|` is exactly zero.
    """
    v = values("block1")
    assert v["null_psi_psi"] > 0.0
    assert v["null_multiplier_psi"] > 0.0
    assert v["null_u_psi"] == 0.0, f"the body force survives B_mu = 0: {v['null_u_psi']!r}"
    assert v["null_psi_u"] > 0.0


# -- schur --------------------------------------------------------------------------

SCHUR_ENTRIES = ("schur", "schur-np2")


@pytest.mark.parametrize("entry", SCHUR_ENTRIES)
@pytest.mark.parametrize("label", ["ainvb", "dense"])
def test_the_complement_matches_an_independent_elimination(entry, label):
    """The complement of each preconditioner is `A11 - A10 A00^{-1} A01`.

    `test_dtn_schur_ainvb.py::
    test_the_cached_complement_matches_an_independent_elimination` and
    `test_dtn_multiplier_dense_schur.py::
    test_T1_parity_against_independent_reference` (`PARITY_FLOOR` 1e-10);
    the reference is built from Jacobian actions and an assembled block 0.
    """
    v = values(entry)
    assert v[f"{label}_n"] == 14.0, "13 DtN multipliers and the core pressure"
    assert v[f"{label}_rel"] < E.SCHUR_PARITY_FLOOR, v[f"{label}_rel"]


@pytest.mark.parametrize("entry", SCHUR_ENTRIES)
@pytest.mark.parametrize("direction", ["forward", "transpose"])
def test_the_cached_apply_inverts_the_operator(entry, direction):
    """`P^{-1} A x = x` and `P^{-T} A^T x = x` with an exact block 0.

    `test_the_forward_apply_inverts_the_operator` and
    `test_the_transpose_apply_inverts_the_transposed_operator`
    (`IDENTITY_FLOOR` 1e-7), with the rigid rotation of the mantle removed
    from the vector and the error (`gia_operators.RigidRotation`).
    """
    v = values(entry)
    assert v["ainvb_build_count_min"] == v["ainvb_build_count_max"] == 1.0
    value = v[f"ainvb_{direction}_identity"]
    assert value < E.SCHUR_IDENTITY_FLOOR, (
        f"{direction}: |P^-1 A x - x| / |x| = {value!r} (with the rigid rotation: "
        f"{v[f'ainvb_{direction}_identity_raw']!r})")


@pytest.mark.parametrize("entry", SCHUR_ENTRIES)
def test_the_transpose_apply_is_not_the_forward_apply(entry):
    """The forward and the transpose apply of one vector differ.

    `test_the_transpose_apply_is_not_the_forward_apply`
    (`TRANSPOSE_MARGIN` 1e-5). M11 makes them equal.
    """
    margin = values(entry)["ainvb_transpose_margin"]
    assert margin > E.SCHUR_TRANSPOSE_MARGIN, margin


@pytest.mark.parametrize("entry", SCHUR_ENTRIES)
def test_the_forward_and_transpose_applies_are_adjoint(entry):
    """`<P^{-1} a, b> = <a, P^{-T} b>`.

    `test_the_forward_and_transpose_applies_are_adjoint` (`PARITY_FLOOR`
    1e-10).
    """
    v = values(entry)
    left, right = v["ainvb_adjoint_left"], v["ainvb_adjoint_right"]
    assert abs(left - right) <= E.SCHUR_PARITY_FLOOR * abs(left), (
        f"<P^-1 a, b> = {left!r}, <a, P^-T b> = {right!r}")


@pytest.mark.parametrize("entry", SCHUR_ENTRIES)
def test_the_dense_applies_solve_the_complement(entry):
    """The dense block-1 preconditioner solves `S x = rhs` and `S^T x = rhs`.

    `test_dtn_multiplier_dense_schur.py::test_T2_forward_solve` and
    `test_T3_transpose_solves_S_transpose` (1e-10 against the independent
    complement; the forward and transpose answers differ by more than
    1e-3).
    """
    v = values(entry)
    for name in ("dense_forward_residual", "dense_forward_solution",
                 "dense_transpose_solution"):
        assert v[name] < E.SCHUR_PARITY_FLOOR, f"{name} = {v[name]!r}"
    assert v["dense_transpose_margin"] > E.DENSE_TRANSPOSE_MARGIN, v["dense_transpose_margin"]


@pytest.mark.parametrize("label", ["ainvb", "dense"])
def test_two_ranks_build_the_same_complement(label):
    """Two ranks build the same complement, and the same as one rank.

    `test_dtn_schur_ainvb.py::test_two_ranks_build_the_same_complement` and
    `test_dtn_multiplier_dense_schur.py::test_T7_parallel_build_keys_on_global_size`
    (`PARITY_FLOOR` 1e-10 across ranks; the same global size on each). The
    2-rank complement also agrees with the serial one in norm and in one
    fixed bilinear probe.
    """
    parallel, serial = values("schur-np2"), values("schur")
    assert parallel["ranks"] == 2.0
    assert parallel[f"{label}_rank_sizes_agree"] == 1.0
    assert parallel[f"{label}_rank_difference"] < E.SCHUR_PARITY_FLOOR
    for name in ("norm", "probe"):
        a, b = parallel[f"{label}_{name}"], serial[f"{label}_{name}"]
        assert abs(a - b) <= E.SCHUR_PARITY_FLOOR * serial[f"{label}_norm"], (
            f"{label} {name}: 2 ranks {a!r}, 1 rank {b!r}")
