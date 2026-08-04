"""One definition of the GAMG settings, and a test that keeps it that way.

The six algebraic-multigrid options existed as **eight** literal copies across
three library modules and two drivers. Measured, all eight agreed - so this was
duplication without drift, and the risk it carries is entirely in the future:
the next person to tune `pc_gamg_threshold` changes one copy, every shipped
preset silently disagrees with every other, and the symptom is a solve that got
slower on one path with nothing in any diff to say why.

`gadopt.solver_options_manager.GAMG_PARAMETERS` is now the single definition
and `gamg_parameters(prefix)` applies it. These tests assert that every shipped
preset carries exactly those settings at its own prefix, so re-introducing a
literal that disagrees fails here rather than in a benchmark six weeks later.
"""
import pytest

from gadopt.gia_gravity import selfgrav_dtn_iterative_solver_parameters
from gadopt.gravity_solver import (iterative_gravity_solver_parameters,
                                   lowrank_gravity_solver_parameters)
from gadopt.solver_options_manager import GAMG_PARAMETERS, gamg_parameters
from gadopt.stokes_integrators import (coupled_gia_solver_parameters,
                                       iterative_stokes_solver_parameters)


def gamg_keys_of(parameters, prefix):
    """The GAMG entries of `parameters` at `prefix`, as a bare dictionary.

    Note that `pc_type` is a *generic* option name that happens to be one of
    the six, so a block set to `bjacobi` also has a `pc_type` at its prefix and
    a naive "does any key overlap" test calls it a GAMG block. `uses_gamg`
    below is the discriminator; this helper only extracts.
    """
    return {key[len(prefix):]: value for key, value in parameters.items()
            if key.startswith(prefix)
            and key[len(prefix):] in GAMG_PARAMETERS}


def uses_gamg(parameters, prefix):
    """Whether the block at `prefix` is actually preconditioned by GAMG."""
    return parameters.get(prefix + "pc_type") == "gamg"


#: (name, dictionary, prefix) for every shipped preset that hands a block to
#: GAMG. The prefix is part of the assertion: the same six settings are needed
#: bare, behind `AssembledPC` (`assembled_`) and behind a fieldsplit as well
#: (`fieldsplit_0_assembled_`), and attaching them at the wrong depth is a
#: silent no-op rather than an error.
SHIPPED = [
    ("iterative_gravity", iterative_gravity_solver_parameters, "assembled_"),
    ("lowrank_gravity", lowrank_gravity_solver_parameters, ""),
    ("iterative_stokes", iterative_stokes_solver_parameters["fieldsplit_0"],
     "assembled_"),
    ("coupled_gia", coupled_gia_solver_parameters,
     "fieldsplit_0_assembled_"),
]


class TestOneDefinition:
    @pytest.mark.parametrize("name,parameters,prefix", SHIPPED,
                             ids=[row[0] for row in SHIPPED])
    def test_shipped_preset_uses_the_shared_settings(
            self, name, parameters, prefix):
        assert gamg_keys_of(parameters, prefix) == dict(GAMG_PARAMETERS)

    @pytest.mark.parametrize("condensed", [True, False])
    def test_the_coupled_sweep_uses_them_on_every_split(self, condensed):
        """Each split of the coupled block-0 sweep, not just the first.

        The sweep is two splits condensed and three uncondensed, and each gets
        its own `assembled_` block. A shared constant applied to one and a
        literal left on another is exactly the drift this guards.
        """
        parameters = selfgrav_dtn_iterative_solver_parameters(
            condensed=condensed)
        n_splits = sum(
            1 for key in parameters
            if key.startswith("dtn_fieldsplit_0_pc_fieldsplit_")
            and key.endswith("_fields"))
        assert n_splits == (2 if condensed else 3)
        for split in range(n_splits):
            prefix = f"dtn_fieldsplit_0_fieldsplit_{split}_assembled_"
            if uses_gamg(parameters, prefix):
                assert gamg_keys_of(parameters, prefix) == dict(
                    GAMG_PARAMETERS)
            else:
                # The `m` split is bjacobi/ilu and not GAMG: the (m, m) block
                # is block-diagonal per cell, so ILU(0) on the cell blocks is
                # exact. Asserted rather than skipped, because putting GAMG on
                # the DG1 tensor block instead of the displacement is a
                # mis-split that does not raise and has already cost a run.
                assert parameters[prefix + "pc_type"] == "bjacobi"

    def test_exactly_the_displacement_and_potential_splits_use_gamg(self):
        """Two GAMG blocks either way: `u` and `psi`. Never the `m` block."""
        for condensed in (True, False):
            parameters = selfgrav_dtn_iterative_solver_parameters(
                condensed=condensed)
            using = [
                split for split in range(3)
                if uses_gamg(
                    parameters,
                    f"dtn_fieldsplit_0_fieldsplit_{split}_assembled_")]
            assert len(using) == 2, (condensed, using)

    def test_the_check_rejects_a_drifted_copy(self):
        """The rejecting partner: an accepting assertion alone proves nothing.

        A `gamg_keys_of` that silently returned `{}` would make every test
        above pass against any dictionary at all.
        """
        drifted = dict(gamg_parameters("assembled_"))
        drifted["assembled_pc_gamg_threshold"] = 0.02
        assert gamg_keys_of(drifted, "assembled_") != dict(GAMG_PARAMETERS)
        assert gamg_keys_of({"unrelated_option": 1}, "assembled_") == {}


class TestGamgParameters:
    def test_prefix_is_applied_to_every_key(self):
        assert gamg_parameters("assembled_") == {
            "assembled_" + k: v for k, v in GAMG_PARAMETERS.items()}

    def test_the_default_prefix_is_empty(self):
        assert gamg_parameters() == dict(GAMG_PARAMETERS)

    def test_each_call_returns_a_fresh_dictionary(self):
        """Otherwise one caller's tweak reaches every other preset in-process.

        Returning the module constant itself would make
        `params.update(gamg_parameters())`-then-edit a global change with no
        indication at the edit site.
        """
        first = gamg_parameters()
        first["pc_gamg_threshold"] = 0.99
        assert gamg_parameters()["pc_gamg_threshold"] == 0.01
        assert GAMG_PARAMETERS["pc_gamg_threshold"] == 0.01

    def test_square_graph_is_a_level_count_and_not_a_boolean(self):
        """Pinned because it reads like a boolean and is not.

        `pc_gamg_square_graph` is a deprecated alias for
        `pc_gamg_aggressive_coarsening` and counts LEVELS (PETSc
        `src/ksp/pc/impls/gamg/agg.c:379-386`), so the shipped 100 means "on
        every level there will ever be". Anyone "tidying" it to `True` or `1`
        is changing the coarsening schedule, and this test says so.
        """
        assert GAMG_PARAMETERS["pc_gamg_square_graph"] == 100
        assert not isinstance(GAMG_PARAMETERS["pc_gamg_square_graph"], bool)
