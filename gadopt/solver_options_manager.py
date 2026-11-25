from collections.abc import Mapping, Callable
import pprint
import textwrap
import weakref

from typing import Union
from .utility import DEBUG, WARNING, log_level, log


def debug_print(class_name: str, string: str):
    """Print a debugging message.

    When `log_level` is set to DEBUG or higher, print a formatted message to stderr.
    The `class_name` variable is used to prefix each line, where `class_name` is
    generally used to identify the lowest level class in the method resolution order
    (e.g. StokesSolver, EnergySolver, etc)
    """
    if DEBUG >= log_level:
        log(textwrap.indent(string, f"{class_name}: "))


def warning_print(class_name: str, string: str):
    """Print a warning message.

    When `log_level` is set to WARNING or higher, print a formatted message to stderr.
    The `class_name` variable is used to prefix each line, where `class_name` is
    generally used to identify the lowest level class in the method resolution order
    (e.g. StokesSolver, EnergySolver, etc)
    """
    if WARNING >= log_level:
        log(textwrap.indent(string, f"{class_name}: "))


class DeleteParam:
    """An empty class to indicate a solver option will be deleted.

    Since `None` is a valid value for PETSc solver options, a separate object must be
    used to denote that a parameter is to be deleted from the solver configuration at
    __init__ time. Use as follows:
    ```
    stokes_solver = StokesSolver( ...
        solver_parameters_extra = { 'ksp_monitor' : DeleteParam }
    )
    ```
    The resulting `stokes_solver.solver_parameters` will not contain the
    `ksp_monitor` key.
    """

    pass


GAMG_PARAMETERS = {
    "pc_type": "gamg",
    "mg_levels_pc_type": "sor",
    "pc_gamg_threshold": 0.01,
    "pc_gamg_square_graph": 100,
    "pc_gamg_coarse_eq_limit": 1000,
    "pc_gamg_mis_k_minimum_degree_ordering": True,
}
"""G-ADOPT's algebraic multigrid settings, in one place.

Every SPD block G-ADOPT hands to GAMG uses these: the Stokes velocity block,
the gravitational potential block, the coupled self-gravitating solver's
displacement and potential splits, and the low-rank DtN operator's
Robin-shifted stiffness.

They were **six literals in three library modules**, and measured identical in
all six -- duplication without drift. Two driver copies made eight. The one
that had *actually* drifted was in a driver and had gone unnoticed:
`b1_elastic`'s mechanics-only arm wrote five of the six out by hand and omitted
`pc_gamg_mis_k_minimum_degree_ordering`, so it aggregated in a different order
from the coupled solve it exists to be compared against. A missing GAMG option
is a different preconditioner, not an error, so nothing said so.

Two of the settings are also not guessable from their keys:

**`pc_gamg_square_graph` is a deprecated alias for
`pc_gamg_aggressive_coarsening`, and it counts LEVELS rather than being a
boolean** (PETSc `src/ksp/pc/impls/gamg/agg.c:379-386`). So `100` does not mean
"on", it means "use aggressive coarsening on up to 100 levels", i.e. on every
level there will ever be. Anyone reading it as a boolean and setting it to `1`
is changing the coarsening schedule, not tidying a value.

**`mg_levels_pc_type: sor` is measured rather than conventional.** On the
low-rank gravity path, `chebyshev/sor` as shipped takes 11 CG iterations
against 15 for `chebyshev/jacobi` and 15 for `richardson/sor`.

Apply with `gamg_parameters(prefix)` rather than by hand: the same six settings
are needed bare, under `assembled_` (behind `AssembledPC`/`SPDAssembledPC`) and
under `fieldsplit_0_assembled_`, and writing the prefix out is how a set of
options comes to be attached to the wrong block.
"""


def gamg_parameters(prefix: str = "") -> dict:
    """`GAMG_PARAMETERS` with every key prefixed.

    Arguments:
      prefix: string prepended to each key, e.g. `"assembled_"` when the block
        is inverted behind `AssembledPC` or `SPDAssembledPC`, or
        `"fieldsplit_0_assembled_"` when it also sits inside a fieldsplit.
        The default returns the settings unprefixed, for a preconditioner
        applied directly to an assembled operator.

    Returns:
      A fresh dictionary, so a caller may mutate it without reaching every
      other caller -- which a shared module-level constant would otherwise
      invite.
    """
    return {prefix + key: value for key, value in GAMG_PARAMETERS.items()}


def nearly_incompressible_mg_parameters(
    prefix: str = "",
    *,
    coarse: str = "gamg",
    coarse_rtol: float = 1e-2,
    patch_construct_dim: int = 0,
    levels_ksp_max_it: int = 2,
) -> dict:
    r"""Geometric-multigrid options robust to a large bulk/shear ratio.

    The companion to `NearlyIncompressibleAssembledPC` for the part of the
    near-incompressibility problem a near-nullspace cannot reach: the *slope*.
    Enriching GAMG's near-nullspace with the divergence-free modes
    (`gadopt.NearlyIncompressibleAssembledPC`) lowers the iteration count by a
    constant factor but leaves the $\sqrt{\lambda/\mu}$ growth in place, because
    smoothed-aggregation transfers do not preserve the divergence-free space.
    Bending that curve needs a multigrid whose *smoother* captures the
    divergence-free near-kernel on every level, and the standard choice is a
    vertex-star patch relaxation (Arnold-Falk-Winther / Schoberl). Patches are
    built from the mesh topology, so this is a **geometric** multigrid: the
    solver's mesh must be the finest of a `MeshHierarchy` (an
    `ExtrudedMeshHierarchy` / semi-coarsened hierarchy on the production mesh).

    The coarse level is the one part that need not be geometric. With
    ``coarse="gamg"`` it is solved by `NearlyIncompressibleAssembledPC` itself --
    GAMG with the divergence-free near-nullspace -- iterated to ``coarse_rtol``.
    Both the enrichment *and* the iteration are needed: the coarse operator
    inherits the same near-incompressible ill-conditioning, so a single plain
    GAMG V-cycle on it stalls and the whole solve diverges at high ratio
    (measured). ``coarse="lu"`` uses a direct coarse solve instead, a little
    stronger but not flexible about coarse size. This composition -- patch
    smoothers over a GAMG-preconditioned coarse solve -- is what keeps the
    hierarchy shallow, which is why it also serves as the radial semi-coarsening
    multigrid the anisotropic lithosphere wants (IDEA.md B2).

    Measured on a 2-D box in the physically relevant band (K/mu ~ 1e2-1e4, where
    the coupled solver diverges today): 2-3x fewer iterations than the enriched
    GAMG alone, and no divergence through 1e4. Full ratio-independence needs a
    deeper hierarchy than a toy carries; the production extruded hierarchy is
    where that is established, on Gadi.

    The patch relaxation is multiplicative with a symmetrised sweep and dense LU
    sub-solves, wrapped in a Chebyshev Krylov smoother (``levels_ksp_max_it``
    sweeps). The outer Krylov must be flexible -- FGMRES -- because the coarse
    Krylov makes the preconditioner nonlinear; set that on the solver, not here.

    Arguments:
      prefix: string prepended to every key, e.g. `"fieldsplit_0_assembled_"`
        when the block sits inside a fieldsplit behind `AssembledPC`.
      coarse: `"gamg"` (flexible, GAMG + divergence-free near-nullspace) or
        `"lu"` (direct).
      coarse_rtol: relative tolerance of the coarse Krylov when `coarse="gamg"`.
      patch_construct_dim: topological dimension the star patches are built
        around; 0 is vertex stars.
      levels_ksp_max_it: Chebyshev smoothing sweeps per level.

    Returns:
      A fresh options dict, prefixed.
    """
    patch = {
        "pc_patch_construct_type": "star",
        "pc_patch_construct_dim": patch_construct_dim,
        "pc_patch_local_type": "multiplicative",
        "pc_patch_symmetrise_sweep": True,
        "pc_patch_sub_mat_type": "seqdense",
        "pc_patch_save_operators": True,
        "sub_ksp_type": "preonly",
        "sub_pc_type": "lu",
    }

    params = {
        "pc_type": "mg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": levels_ksp_max_it,
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.PatchPC",
        **{"mg_levels_patch_" + k: v for k, v in patch.items()},
    }

    if coarse == "lu":
        params["mg_coarse_pc_type"] = "lu"
    elif coarse == "gamg":
        params.update({
            "mg_coarse_ksp_type": "gmres",
            "mg_coarse_ksp_rtol": coarse_rtol,
            "mg_coarse_ksp_max_it": 200,
            "mg_coarse_pc_type": "python",
            "mg_coarse_pc_python_type": "gadopt.NearlyIncompressibleAssembledPC",
            **gamg_parameters("mg_coarse_assembled_"),
        })
    else:
        raise ValueError('coarse must be "gamg" or "lu"')

    return {prefix + key: value for key, value in params.items()}


# Type alias
ConfigType = Mapping[str, Union[str | float | None | type[DeleteParam], "ConfigType"]]
_InternalConfigType = Mapping[str, Union[str | float | None, "_InternalConfigType"]]
_InternalMutableConfigType = dict[str, Union[str | float | None, "_InternalMutableConfigType"]]


class SolverConfigurationMixin:
    """Manage PETSc solver options in G-ADOPT.

    This class is designed to be subclassed by the base class for any solvers included
    in G-ADOPT. It provides methods for handling and modifying solver parameters passed
    to Firedrake's `[Non]LinearVariationalSolver` solver object during initialisation
    of a G-ADOPT solver object.
    """

    _solver_options_initialised = False

    def reset_solver_config(
        self,
        default_config: ConfigType,
        extra_config: ConfigType | None = None,
    ) -> None:
        """Resets the `solver_parameters` dict.

        Empties the existing `solver_parameters` dict and creates a new one by first
        running `add_to_solver_config` on the empty dict with `default_config`, and
        then again on the resulting dict with `extra_config`. `default_config` is
        mandatory, `extra_config` is optional. Will invoke `callback_ref` if it is set.
        """
        debug_print(self._class_name, "Input default solver configuration:")
        debug_print(self._class_name, pprint.pformat(default_config, indent=2))
        self.solver_parameters = {}
        debug_print(self._class_name, "Processing default config")
        self.add_to_solver_config(default_config, extra_config is None)
        if extra_config is not None:
            debug_print(self._class_name, "Processing extra config")
            self.add_to_solver_config(extra_config, True)

    def print_solver_config(self) -> None:
        """Prints the solver_parameters dict.

        Uses pprint to write the final `solver_paramseters` dict in a human-readable
        way. Useful for debugging purposes when a user wishes to verify PETSc solver
        settings.
        """
        pprint.pprint(self.solver_parameters, indent=2)

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register a function to call whenever `solver_parameters` is updated

        The function provided to `register_update_callback` must take no arguments and
        return nothing. When a subclass provides this function, a user does not need to
        be aware of the underlying Problem/Solver objects in order to ensure that a
        configuration update during an in-progress simulation takes effect properly.

        A weakref is used here in order to prevent circular references, which would
        prevent Python's automatic garbage collection from cleaning up G-ADOPT solver
        objects.
        """
        self.callback_ref = weakref.WeakMethod(callback)

    def process_mapping(
        self,
        key_prefix: str,
        inmap: _InternalConfigType,
        delta_map: ConfigType,
    ) -> _InternalMutableConfigType:
        """Copy `inmap` into a dictionary and apply the changes in `delta_map`

        If any element of `delta_map` is another Mapping, recursively calls itself to
        process the changes in that mapping to the corresponding element in inmap. If
        an element `delta_map` is `DeleteParam`, remove it from the dict if it exists
        in `inmap`.

        The `key_prefix` argument is provided for logging purposes to indicate the
        current mapping being processed
        """
        outmap = dict(inmap)
        for k, v in delta_map.items():
            if v is DeleteParam:
                if k in outmap:
                    debug_print(self._class_name, f"Deleting {key_prefix}[{k}]")
                    del outmap[k]
                else:
                    debug_print(
                        self._class_name,
                        f"Requested deletion of {key_prefix}[{k}] but this key was not found in original mapping",
                    )
            elif isinstance(v, Mapping):
                kp = f"{key_prefix}[{k}]"
                if k in inmap:
                    outmap[k] = self.process_mapping(kp, inmap[k], v)
                else:
                    outmap[k] = self.process_mapping(kp, {}, v)
            else:
                debug_print(self._class_name, f"Adding {key_prefix}[{k}] = {v}")
                if k in outmap and isinstance(outmap[k], Mapping):
                    warning_print(
                        self._class_name,
                        (
                            f"WARNING: key '{k}' holds a parameter dict in the original mapping"
                            ", but is being overwritten with a scalar parameter. This may have "
                            "unintended consequences for this solver's parameters"
                        ),
                    )
                outmap[k] = v

        return outmap

    def add_to_solver_config(self, in_config: ConfigType | None, reinit=False) -> None:
        """Updates the `solver_parameters` dict

        Takes a single Mapping argument that added to the existing `solver_parameters`
        dict, overwriting any existing parameters of the same name. On the first call
        to this method the `SolverConfigurationMixin` objects are created; a logging
        prefix based on the class hierarchy, a reference to a callback initialised to
        `None` and an empty solver configuration. This structure allows a subclass to
        build its solver parameters step-by-step.

        Any Mapping type can be passed into this function, and the function will take
        care of copying the mapping to a mutable dictionary.

        If the `callback_ref` attribute is set, it will be called on completion of the
        update.
        """
        # "Initialise" solver config attrs on first call
        if not self._solver_options_initialised:
            self._class_name = self.__class__.__name__
            self.solver_parameters = {}
            self.callback_ref = None
            self._solver_options_initialised = True
            debug_print(self._class_name, "Initialised SolverConfigurationMixin")

        if in_config is not None:
            self.solver_parameters = self.process_mapping("solver_parameters", self.solver_parameters, in_config)
            debug_print(self._class_name, "Solver configuration after additions:")
            debug_print(self._class_name, pprint.pformat(self.solver_parameters, indent=2))
            if reinit and self.callback_ref is not None:
                debug_print(self._class_name, "Running callback")
                self.callback_ref()()
        else:
            debug_print(self._class_name, "in_config is empty, doing nothing")

    def is_iterative_solver(self) -> bool:
        """
        Decide if a solver is iterative or not

        Return a boolean that indicates whether this solver is an iterative solver or a
        direct solver.
        """
        return self.solver_parameters.get("pc_type") not in ["lu", "cholesky"]
