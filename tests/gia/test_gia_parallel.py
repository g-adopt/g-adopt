"""Tests of the `parallel` case: the checks that need more than one rank.

`gia_parallel.py` writes the measured values of each entry of
`gia_entries.PARALLEL` to `out/parallel-<entry>.dat`. Each test here holds
the tolerance of the unit test that the entry comes from, named in its
docstring. No value is recorded from an earlier run. A rank that raised or
waited in a collective leaves no file, so a missing file is a failure.

This file does not import Firedrake; it runs where the outputs are.
"""

import pytest

import gia_entries as E


def values(entry):
    """The `name value` lines of `out/parallel-<entry>.dat` as a dictionary."""
    path = E.CASE_DIR / E.parallel_values(entry)
    out = {}
    with open(path) as f:
        for line in f:
            name, value = line.split()
            out[name] = float(value)
    return out


@pytest.mark.parametrize("entry", list(E.PARALLEL))
def test_the_entry_ran_on_its_ranks(entry):
    """Each entry ran on the number of ranks that `PARALLEL` gives it.

    An entry for two ranks that ran on one would pass most checks below
    without testing what it claims.
    """
    ranks = values(entry)["ranks"]
    assert ranks == E.PARALLEL[entry]["cores"], (
        f"{entry}: ran on {ranks!r} rank(s), expected {E.PARALLEL[entry]['cores']!r}")


# -- dtn-guard-np2 ------------------------------------------------------------------

def test_clean_rows_pass_on_every_rank():
    """The positive control: rows equal to the reference raise on no rank.

    `test_dtn_lowrank_parallel.py::test_clean_rows_pass_on_every_rank`.
    Without it, the poisoned calls below could be detecting the setup.
    """
    v = values("dtn-guard-np2")
    assert v["clean_raised_ranks"] == 0.0, (
        f"{v['clean_raised_ranks']!r} rank(s) raised on clean rows, expected 0")


def test_a_rank_with_empty_rows_does_not_strand_the_others():
    """Zeroed rows on one rank: every rank raises the same error and goes on.

    `test_dtn_lowrank_parallel.py::
    test_a_rank_with_empty_rows_does_not_strand_the_others`. A rank-local
    scale makes the rank with zero rows return early while the other rank
    waits in the collective reference assemblies; the entry then leaves no
    file.
    """
    v = values("dtn-guard-np2")
    for name in ("empty_raised_ranks", "empty_reached_ranks"):
        assert v[name] == v["ranks"], f"{name} = {v[name]!r}, expected {v['ranks']!r}"


def test_every_rank_names_the_same_mode():
    """A different mode changed on each rank: every rank gives one message.

    `test_dtn_lowrank_parallel.py::test_every_rank_names_the_same_mode`. A
    rank-local worst mode names a different mode on each rank.
    """
    v = values("dtn-guard-np2")
    assert v["mode_raised_ranks"] == v["ranks"], (
        f"{v['mode_raised_ranks']!r} rank(s) raised, expected {v['ranks']!r}")
    assert v["mode_messages_agree"] == 1.0, "the ranks name different modes"


# -- checkpoint entries -------------------------------------------------------------

@pytest.mark.parametrize("field", E.CHECKPOINT_FIELDS)
@pytest.mark.parametrize("mesh", E.CHECKPOINT_MESHES)
def test_round_trip_two_ranks(mesh, field):
    """Encode, zero and restore give the values back exactly on two ranks.

    `test_viscoelastic_checkpointing.py::test_round_trip_two_ranks`. The
    submesh is the cells of one rank, so the other rank owns none of it and
    reshapes an empty array.
    """
    v = values("checkpoint-round-trip-np2")
    assert v["empty_submesh_rank"] == 1.0, "every rank owns a cell of the submesh"
    error = v[f"{mesh}_{field}_error"]
    assert error == 0.0, f"{mesh} {field}: max error {error!r}, expected 0"


def test_quadrature_fields_read_on_another_number_of_ranks():
    """Written on one rank and read on two, written on two and read on one.

    `test_viscoelastic_checkpointing.py::
    test_quadrature_fields_read_on_another_number_of_ranks`
    (`CROSS_PARTITION_RTOL` 1e-13). The control reads the one-rank file on
    two ranks without the pairing by position and must see an error above
    `CROSS_PARTITION_CONTROL`, which shows that the partitions differ. On
    the writer's number of ranks the values come back bit for bit.
    """
    two_ranks, one_rank = values("checkpoint-read-np2"), values("checkpoint-read")
    for label, error in (("one rank -> two", two_ranks["one_error"]),
                         ("two ranks -> one", one_rank["two_error"])):
        assert error <= E.CROSS_PARTITION_RTOL, (
            f"{label}: relative error {error!r} above {E.CROSS_PARTITION_RTOL!r}")
    control = two_ranks["one_control"]
    assert control > E.CROSS_PARTITION_CONTROL, (
        f"control without pairing: error {control!r}, expected above "
        f"{E.CROSS_PARTITION_CONTROL!r}")
    assert two_ranks["two_exact"] == 1.0, "the two-rank file is not bit for bit on two ranks"


def test_a_file_without_point_coordinates_is_refused_on_another_number_of_ranks():
    """A file without coordinates: refused on one rank, bit for bit on two.

    `test_viscoelastic_checkpointing.py::
    test_a_file_without_point_coordinates_is_refused_on_another_number_of_ranks`.
    The number of ranks of the writer is the one Firedrake stores with the
    distribution of the mesh.
    """
    one_rank, two_ranks = values("checkpoint-read"), values("checkpoint-read-np2")
    assert one_rank["without_points_writer_ranks"] == 2.0, (
        f"writer ranks {one_rank['without_points_writer_ranks']!r}, expected 2")
    assert one_rank["without_points_refused"] == 1.0, (
        f"the read on one rank was not refused with {E.WRITTEN_RANKS_MESSAGE!r}")
    assert two_ranks["two_without_points_exact"] == 1.0, (
        "the file without coordinates is not bit for bit on two ranks")


# -- projection-np2 -----------------------------------------------------------------

def test_small_parallel_replay_and_gradient():
    """The recorded kernel projection on two ranks: replay and guarded Taylor test.

    `test_gia_nullspace_projection_adjoint.py::
    test_small_parallel_replay_and_gradient`: the replay at the recorded
    control equals the recording exactly; the replay at a shifted control
    equals a fresh run to `PROJECTION_SHIFTED_RTOL`; the guards of
    `test_gravity_adjoint.assert_taylor_with_guards` (derivative and
    response floors, rate at least 1.98 with the gradient, rate in
    [0.85, 1.15] without it). The shifted functional depends on the
    projection by more than `PROJECTION_EFFECT_FLOOR`, so the replay checks
    see the block.
    """
    v = values("projection-np2")
    J = v["J"]
    assert v["blocks"] == 1.0, f"{v['blocks']!r} KernelProjectionBlock(s) on the tape"
    assert v["J_replay"] == J, f"replay {v['J_replay']!r} against recording {J!r}"
    fresh = v["J_fresh"]
    effect = abs(v["J_unprojected"] - fresh)
    assert effect > E.PROJECTION_EFFECT_FLOOR * abs(fresh), (
        f"the projection changes J by {effect!r}, at most "
        f"{E.PROJECTION_EFFECT_FLOOR!r} of {fresh!r}")
    gap = abs(v["J_replay_shifted"] - fresh)
    assert gap <= E.PROJECTION_SHIFTED_RTOL * abs(fresh), (
        f"shifted replay {v['J_replay_shifted']!r} against fresh {fresh!r}: "
        f"difference {gap!r} above {E.PROJECTION_SHIFTED_RTOL!r} relative")
    assert abs(v["dJ"]) >= E.PROJECTION_DERIVATIVE_FLOOR * abs(J), (
        f"dJ {v['dJ']!r} below {E.PROJECTION_DERIVATIVE_FLOOR!r} |J| = {abs(J)!r}")
    assert v["response"] >= E.PROJECTION_RESPONSE_FLOOR * abs(J), (
        f"response {v['response']!r} below {E.PROJECTION_RESPONSE_FLOOR!r} |J|")
    assert v["taylor_rate"] >= E.PROJECTION_RATE_MIN, (
        f"Taylor rate {v['taylor_rate']!r} below {E.PROJECTION_RATE_MIN!r}")
    low, high = E.PROJECTION_NO_GRADIENT_BAND
    assert low <= v["taylor_rate_no_gradient"] <= high, (
        f"rate without gradient {v['taylor_rate_no_gradient']!r} not in [{low!r}, {high!r}]")


# -- rotation-radius-np2 ------------------------------------------------------------

def test_the_radius_check_agrees_across_ranks():
    """The 2-rank build with the `k_s` spelling and the radius `RE` succeeds.

    `test_gia_rotation_moments.py::test_the_radius_check_agrees_across_ranks`:
    the coupling keeps `surface_radius = RE` and closes the `m_3` row with
    the polar moment of the mantle (`ROTATION_FORMULA_RTOL`). A refusal on
    one rank leaves no file. The largest local radius of the rank that does
    not reach the outer boundary is written too; when it is further from
    `RE` than the radius tolerance of the check, a check on the local radius
    would refuse there (measured 2.0037 against 2.2037 on the annulus of the
    unit fixture).
    """
    v = values("rotation-radius-np2")
    assert v["surface_radius"] == v["RE"], (
        f"surface_radius {v['surface_radius']!r}, expected {v['RE']!r}")
    closure, moment = v["closure_m3"], v["polar_moment"]
    assert abs(closure - moment) <= E.ROTATION_FORMULA_RTOL * abs(moment), (
        f"closure of m3 {closure!r} against polar moment {moment!r}, "
        f"rtol {E.ROTATION_FORMULA_RTOL!r}")


# -- gravity-np2 --------------------------------------------------------------------

def test_parallel_parity():
    """Low-rank against multiplier DtN on two ranks.

    `test_dtn_lowrank_parity.py::test_parallel_parity` (potential 1e-14
    relative in L2, trace coefficients 1e-12 absolute).
    """
    v = values("gravity-np2")
    assert v["parity_psi_error"] <= E.GRAVITY_PARITY_PSI_RTOL, (
        f"potential {v['parity_psi_error']!r} above {E.GRAVITY_PARITY_PSI_RTOL!r}")
    assert v["parity_coefficient_error"] <= E.GRAVITY_PARITY_COEFFICIENT_ATOL, (
        f"coefficients {v['parity_coefficient_error']!r} above "
        f"{E.GRAVITY_PARITY_COEFFICIENT_ATOL!r}")


def test_parallel_coefficients_are_rank_consistent():
    """Every rank reports the same trace coefficients, exactly.

    `test_dtn_lowrank_parity.py::test_parallel_coefficients_are_rank_consistent`.
    """
    v = values("gravity-np2")
    assert v["coefficients_count"] > 0, "no trace coefficient"
    assert v["coefficients_rank_difference"] == 0.0, (
        f"the ranks differ by {v['coefficients_rank_difference']!r}, expected 0")


@pytest.mark.parametrize("representation", E.GRAVITY_REPRESENTATIONS)
def test_parallel_repeated_monopole_solves_are_stable(representation):
    """The enclosed mass does not drift when a solver is reused on two ranks.

    `test_gravity_solver.py::TestCylindrical::
    test_parallel_repeated_monopole_solves_are_stable`: the first mass
    against the closed form to `GRAVITY_MONOPOLE_MASS_RTOL`, three more
    solves against the first to `GRAVITY_MONOPOLE_DRIFT_RTOL` in the mass
    and in the norm of the potential.
    """
    v = values("gravity-np2")
    prefix = f"monopole_{representation}"
    mass, expected = v[f"{prefix}_mass"], v[f"{prefix}_mass_expected"]
    assert abs(mass - expected) <= E.GRAVITY_MONOPOLE_MASS_RTOL * abs(expected), (
        f"mass {mass!r} against {expected!r}, rtol {E.GRAVITY_MONOPOLE_MASS_RTOL!r}")
    for name in ("mass_drift", "norm_drift"):
        drift = v[f"{prefix}_{name}"]
        assert drift <= E.GRAVITY_MONOPOLE_DRIFT_RTOL, (
            f"{name} {drift!r} above {E.GRAVITY_MONOPOLE_DRIFT_RTOL!r}")


def test_parallel_matches_the_field_number_split():
    """The split by index sets against the split by field numbers on two ranks.

    `test_gravity_solver.py::TestIndexSetSplit::
    test_parallel_matches_the_field_number_split` (1e-12 against the field
    numbers, 1e-9 against the direct preset).
    """
    v = values("gravity-np2")
    for name, bound in (("split_psi_error", E.GRAVITY_SPLIT_RTOL),
                        ("split_coefficient_error", E.GRAVITY_SPLIT_RTOL),
                        ("direct_psi_error", E.GRAVITY_DIRECT_RTOL),
                        ("direct_coefficient_error", E.GRAVITY_DIRECT_RTOL)):
        assert v[name] < bound, f"{name} {v[name]!r}, bound {bound!r}"


@pytest.mark.parametrize("application", range(E.GRAVITY_MATFREE_APPLICATIONS))
def test_parallel_repeated_matfree_application_is_stable(application):
    """Each application of the matrix-free Jacobian equals the assembled one.

    `test_gravity_solver.py::TestIndexSetSplit::
    test_parallel_repeated_matfree_application_is_stable` (1e-12 relative).
    """
    error = values("gravity-np2")[f"matfree_error_{application}"]
    assert error <= E.GRAVITY_MATFREE_RTOL, (
        f"application {application}: {error!r} above {E.GRAVITY_MATFREE_RTOL!r}")
