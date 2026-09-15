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
from gadopt.stokes_integrators import (cpu_gamg_parameters,
                                       gamg_common_parameters)


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
]

#: route name -> (preset keyword arguments, the prefix of each GAMG block).
#:
#: Every route sweeps a displacement block and a potential block, and each
#: hands its operator to GAMG at its own depth:
#:
#: - condensed layout: block 0 is a two-way sweep over `u` and `psi`, each
#:   under a `gadopt` `AssembledPC` subclass, which nests the operator's
#:   options under `assembled_`.
#: - uncondensed layout, default block 0 (`gadopt.CondensedBlockPC`): the class
#:   eliminates `M` itself and runs its own `(u, psi)` fieldsplit on assembled
#:   matrices, so GAMG sits directly under `condensed_fieldsplit_N_` with no
#:   `AssembledPC` in between.
#: - uncondensed layout, `block0="pair"`: split 0 is the pair `(u, M)` under
#:   `gadopt.InternalVariableSCPC`, whose condensed displacement matrix goes to
#:   GAMG under `condensed_field_`; split 1 is `psi` under `assembled_`.
#:
#: The prefix is part of the assertion because attaching the settings at the
#: wrong depth is a silent no-op rather than an error.
SWEEP_ROUTES = {
    "condensed": (
        dict(condensed=True),
        ("dtn_fieldsplit_0_fieldsplit_0_assembled_",
         "dtn_fieldsplit_0_fieldsplit_1_assembled_")),
    "uncondensed-condensed-block0": (
        dict(condensed=False),
        ("dtn_fieldsplit_0_condensed_fieldsplit_0_",
         "dtn_fieldsplit_0_condensed_fieldsplit_1_")),
    "uncondensed-pair-block0": (
        dict(condensed=False, block0="pair"),
        ("dtn_fieldsplit_0_fieldsplit_0_condensed_field_",
         "dtn_fieldsplit_0_fieldsplit_1_assembled_")),
}


def gamg_prefixes(parameters):
    """Every prefix in `parameters` whose block is preconditioned by GAMG.

    Read off the dictionary itself rather than tested against a list of
    candidate prefixes, so that a GAMG block at a depth nobody expected is
    found instead of missed. A third GAMG block is what a mis-split looks
    like: the internal variable is block-diagonal per cell and belongs to an
    exact cell-local inverse, and handing that block to smoothed aggregation
    does not raise, does not warn, and shows up only as block 0 running to its
    iteration cap.
    """
    suffix = "pc_type"
    return sorted(key[:-len(suffix)] for key, value in parameters.items()
                  if key.endswith(suffix) and value == "gamg")


class TestOneDefinition:
    def test_stokes_integrators_carries_the_same_six_settings(self):
        """The Stokes and GIA presets take their GAMG block from two module
        constants in `stokes_integrators` that `_configure_iterative_solver`
        nests at run time. They must not drift from `GAMG_PARAMETERS`."""
        assert dict(gamg_common_parameters | cpu_gamg_parameters) == GAMG_PARAMETERS

    @pytest.mark.parametrize("name,parameters,prefix", SHIPPED,
                             ids=[row[0] for row in SHIPPED])
    def test_shipped_preset_uses_the_shared_settings(
            self, name, parameters, prefix):
        assert gamg_keys_of(parameters, prefix) == dict(GAMG_PARAMETERS)

    @pytest.mark.parametrize("route", list(SWEEP_ROUTES),
                             ids=list(SWEEP_ROUTES))
    def test_the_coupled_sweep_uses_them_on_every_split(self, route):
        """Each block of the coupled block-0 sweep, not just the first.

        Two blocks on every route, each handing its operator to GAMG at its
        own prefix (`SWEEP_ROUTES`). A shared constant applied to one and a
        literal left on another is exactly the drift this guards.
        """
        keywords, prefixes = SWEEP_ROUTES[route]
        parameters = selfgrav_dtn_iterative_solver_parameters(**keywords)
        for prefix in prefixes:
            assert uses_gamg(parameters, prefix), (route, prefix)
            assert gamg_keys_of(parameters, prefix) == dict(GAMG_PARAMETERS)

    @pytest.mark.parametrize("route", list(SWEEP_ROUTES),
                             ids=list(SWEEP_ROUTES))
    def test_exactly_the_displacement_and_potential_blocks_use_gamg(
            self, route):
        """Two GAMG blocks on every route: the displacement and `psi`.

        The internal variable reaches GAMG on no route: the condensed space
        does not hold it, and both uncondensed routes eliminate it cell by
        cell with Slate. So a third GAMG block anywhere in the dictionary is a
        mis-split, and a missing one is a block left on PETSc's default
        preconditioner.
        """
        keywords, prefixes = SWEEP_ROUTES[route]
        parameters = selfgrav_dtn_iterative_solver_parameters(**keywords)
        assert gamg_prefixes(parameters) == sorted(prefixes)

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
