from collections.abc import Mapping, Callable
import pprint
import sys
import textwrap

from .utility import (
    DEBUG,
    log_level,
)


def debug_print(class_name: str, string: str):
    if DEBUG >= log_level:
        print(textwrap.indent(string, f"{class_name}: "), file=sys.stderr)


class DeleteParam:
    pass


# Type alias
DefaultConfigType = Mapping[str, str | float | int | None | Mapping[str, str | float | int | None]]
ExtraConfigType = Mapping[str, str | float | int | None | type[DeleteParam] | Mapping[str, str | float | int | None]]


class SolverOptions:
    def init_solver_config(
        self,
        default_config: DefaultConfigType,
        extra_config: ExtraConfigType | None = None,
        callback: Callable[[], None] | None = None,
    ):
        self._top_level_class_name = self.__class__.__mro__[0].__name__
        if callback is not None:
            debug_print(self._top_level_class_name, f"Registering callback: {callback.__name__}()")
            self.register_update_callback(callback)
        self.reset_solver_config(default_config, extra_config)

    def reset_solver_config(
        self,
        default_config: DefaultConfigType,
        extra_config: ExtraConfigType | None = None,
    ) -> None:
        self.default_config = default_config
        self.extra_config = extra_config
        debug_print(self._top_level_class_name, "Input default solver configuration:")
        debug_print(self._top_level_class_name, pprint.pformat(self.default_config, indent=2))
        self.solver_parameters = {}
        # Convert default_config from an immutable Mapping to a mutable dict
        debug_print(self._top_level_class_name, "Processing default config")
        self.update_solver_config(self.default_config)
        if self.extra_config:
            debug_print(self._top_level_class_name, "Processing extra config")
            self.update_solver_config(self.extra_config)

        if self.update_callback is not None:
            self.update_callback()

    def print_solver_config(self) -> None:
        pprint.pprint(self.solver_parameters, indent=2)

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        self.update_callback = callback

    def process_mapping(
        self,
        key_prefix: str,
        inmap: Mapping[str, str | float | int | Mapping],
        delta_map: Mapping[str, str | float | int | type[DeleteParam] | Mapping],
    ) -> dict[str, str | float | int | dict]:
        outmap = dict(inmap)
        for k, v in delta_map.items():
            if v is DeleteParam:
                if k in outmap:
                    debug_print(self._top_level_class_name, f"Deleting {key_prefix}[{k}]")
                    del outmap[k]
            elif isinstance(v, Mapping):
                kp = f"{key_prefix}[{k}]"
                if k in inmap:
                    outmap[k] = self.process_mapping(kp, inmap[k], v)
                else:
                    outmap[k] = self.process_mapping(kp, {}, v)
            else:
                debug_print(self._top_level_class_name, f"Adding {key_prefix}[{k}] = {v}")
                outmap[k] = v

        return outmap

    def update_solver_config(self, extra_config: ExtraConfigType) -> None:
        self.solver_parameters = self.process_mapping("solver_parameters", self.solver_parameters, extra_config)
