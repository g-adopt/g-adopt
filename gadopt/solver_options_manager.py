from collections.abc import Mapping, Callable
import pprint
import textwrap
import weakref

from .utility import DEBUG, WARNING, log_level, log


def debug_print(class_name: str, string: str):
    """Print a debugging message.

    When `log_level` is set to DEBUG, print a formatted message to stderr.
    The `class_name` variable is used to prefix each line, where `class_name`
    is generally used to identify the lowest level class in the method
    resolution order (e.g. StokesSolver, EnergySolver, etc)
    """
    if DEBUG >= log_level:
        log(textwrap.indent(string, f"{class_name}: "))


def warning_print(class_name: str, string: str):
    """Print a warning message.

    When `log_level` is set to DEBUG, print a formatted message to stderr.
    The `class_name` variable is used to prefix each line, where `class_name`
    is generally used to identify the lowest level class in the method
    resolution order (e.g. StokesSolver, EnergySolver, etc)
    """
    if WARNING >= log_level:
        log(textwrap.indent(string, f"{class_name}: "))


class DeleteParam:
    """An empty class to indicate a solver option will be deleted.

    Since `None` is a valid value for PETSc solver options, a separate object
    must be used to denote that a parameter is to be deleted from the solver
    configuration at __init__ time. Use as follows:
    ```
    stokes_solver = StokesSolver( ...
        solver_parameters_extra = { 'ksp_monitor' : DeleteParam }
    )
    ```
    The resulting `stokes_solver.solver_parameters` will not contain the
    `ksp_monitor` key.
    """

    pass


# Type alias
ConfigType = Mapping[
    str, str | float | int | None | type[DeleteParam] | Mapping[str, str | float | int | None | type[DeleteParam]]
]


class SolverOptions:
    """Manage PETSc solver options in G-ADOPT.

    This class is designed to be subclassed by the base class for any solvers
    included in G-ADOPT. It provides methods for handling and modifying solver
    parameters passed to Firedrake's `[Non]LinearVariationalSolver solver object
    during initialisation of a G-ADOPT solver object.
    """

    _solver_options_initialised = False

    def reset_solver_config(
        self,
        default_config: ConfigType,
        extra_config: ConfigType | None = None,
    ) -> None:
        """Resets the `solver_parameters` dict.

        Empties the existing `solver_parameters` dict and creates a new one by
        first running `add_to_solver_config` on the empty dict with `default_config`,
        and then again on the resulting dict with `extra_config`. `default_config`
        is mandatory, `extra_config` is optional. Will invoke `callback_ref`
        if it is set.
        """
        debug_print(self._top_level_class_name, "Input default solver configuration:")
        debug_print(self._top_level_class_name, pprint.pformat(default_config, indent=2))
        self.solver_parameters = {}
        debug_print(self._top_level_class_name, "Processing default config")
        self.add_to_solver_config(default_config, extra_config is None)
        if extra_config is not None:
            debug_print(self._top_level_class_name, "Processing extra config")
            self.add_to_solver_config(extra_config, True)

    def print_solver_config(self) -> None:
        """Prints the solver_parameters dict.

        Uses pprint to write the final `solver_paramseters` dict in a
        human-readable way. Useful for debugging purposes when a user
        wishes to verify PETSc solver settings.
        """
        pprint.pprint(self.solver_parameters, indent=2)

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register a function to call whenever `solver_parameters` is updated

        The function provided to `register_update_callback` must take no arguments and return
        nothing. When a subclass provides this function, a user does not need to
        be aware of the underlying Problem/Solver objects in order to ensure that
        a configuration update during an in-progress simulation takes effect properly.

        A weakref is used here in order to prevent circular references, which would
        prevent Python's automatic garbage collection from cleaning up G-ADOPT
        solver objects.
        """
        self.callback_ref = weakref.WeakMethod(callback)

    def process_mapping(
        self,
        key_prefix: str,
        inmap: Mapping[str, str | float | int | None | Mapping],
        delta_map: Mapping[str, str | float | int | None | type[DeleteParam] | Mapping],
    ) -> dict[str, str | float | int | None | dict]:
        """Copy a Mapping object into a dictionary

        If an element of a Mapping is another Mapping, recursively calls itself to
        process that Mapping. If an element of the mapping is `DeleteParam`, remove
        it from the dict if it exists in the Mapping
        """
        outmap = dict(inmap)
        for k, v in delta_map.items():
            if v is DeleteParam:
                if k in outmap:
                    debug_print(self._top_level_class_name, f"Deleting {key_prefix}[{k}]")
                    del outmap[k]
                else:
                    debug_print(
                        self._top_level_class_name,
                        f"Requested deletion of {key_prefix}[{k}] but this key was not found in original mapping",
                    )
            elif isinstance(v, Mapping):
                kp = f"{key_prefix}[{k}]"
                if k in inmap:
                    outmap[k] = self.process_mapping(kp, inmap[k], v)
                else:
                    outmap[k] = self.process_mapping(kp, {}, v)
            else:
                debug_print(self._top_level_class_name, f"Adding {key_prefix}[{k}] = {v}")
                if k in outmap and isinstance(outmap[k], Mapping):
                    warning_print(
                        self._top_level_class_name,
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

        Takes a single Mapping argument that added to the existing `solver_parameters` dict,
        overwriting any existing parameters of the same name. On the first call to this method
        the `SolverOptions` objects are created; a logging prefix based on the class hierarchy,
        a reference to a callback initialised to `None` and an empty solver configuration. This
        structure allows a subclass to build its solver parameters step-by-step.

        Any Mapping type can be passed into this function, and the function will take care of copying
        the mapping to a mutable dictionary.

        If the `callback_ref` attribute is set, it will be called on completion of the update.
        """
        # "Initialise" solver config attrs on first call
        if not self._solver_options_initialised:
            self._top_level_class_name = self.__class__.__mro__[0].__name__
            self.solver_parameters = {}
            self.callback_ref = None
            self._solver_options_initialised = True
            debug_print(self._top_level_class_name, "Initialised SolverOptions")

        if in_config is not None:
            self.solver_parameters = self.process_mapping("solver_parameters", self.solver_parameters, in_config)
            debug_print(self._top_level_class_name, "Solver configuration after additions:")
            debug_print(self._top_level_class_name, pprint.pformat(self.solver_parameters, indent=2))
            if reinit and self.callback_ref is not None:
                debug_print(self._top_level_class_name, "Running callback")
                self.callback_ref()()
        else:
            debug_print(self._top_level_class_name, "in_config is empty, doing nothing")

    def is_iterative_solver(self) -> bool:
        """
        Decide if a solver is iterative or not

        Return a boolean that indicates whether this solver is an iterative
        solver or a direct solver.
        """
        return self.solver_parameters.get("pc_type") not in ["lu", "cholesky"]
