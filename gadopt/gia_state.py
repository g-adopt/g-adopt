r"""Checkpoints of the self-gravitating GIA solver: write, read, convert.

`SelfGravitatingGIASolver.state()` names every Function a restart needs.
This module stores them in a Firedrake `CheckpointFile` and reads them back,
and converts the files that the earlier solver and its drivers wrote.

`CheckpointFile` stores one mesh per Function and knows nothing of
`Submesh`, so every field is stored on the parent mesh in the encoding of
`gadopt.viscoelastic_checkpointing.CheckpointField`: a field of the
mechanics mesh in its own element on the parent, copied degree of freedom by
degree of freedom (`broken=False`), a Quadrature history as one DG0 vector of
exact point values per cell, a `Real` field as itself. The copy, and not the
interpolation into the broken element that `selfgrav_common` uses, is what
makes the round trip bit for bit: the interpolation evaluates the basis at
the nodes and changes the last bit of some values.
A restart loads the parent mesh from the file, cuts the mechanics mesh from
it, builds the solver on those meshes and calls

    solver.restore(read_state(checkpoint, solver))

The round trip copies nodal values and is exact.

## Files of the earlier solver

`state_from_checkpoint` reads the layouts of `selfgrav_common` (gia-mip):
`solution_{i}` for every sub-function of the earlier mixed solution,
`internal_variables_{j}` for a history outside it, `time_current_{i}` and
`time_previous_{i}` for the `BDF2History` levels, and the metadata of
`gadopt.viscoelastic_timestepping.time_history_metadata`. The field order of
the earlier mixed space is the order of this solver's mixed space, so an
index names the same field in both. The Exp01 driver stores the `Real` rows
as `real_{i}` in its metadata and the fields under its own names, which the
`aliases` argument maps.
"""

from collections.abc import Mapping
from numbers import Number

from firedrake import Function
from pyadjoint.tape import stop_annotating

from .gia_couplings import scalar_value
from .viscoelastic_checkpointing import CheckpointField

__all__ = ["decode_into", "read_state", "state_from_checkpoint", "write_state"]


def _same_space(a, b) -> bool:
    """Whether two Functions live on one mesh with one element."""
    return (a.function_space().mesh() is b.function_space().mesh()
            and a.ufl_element() == b.ufl_element())


def decode_into(target: Function, value, parent, name: str) -> None:
    """Write `value` into `target` with a recorded assignment.

    Args:
      target: a Function of the solver's state.
      value: a number (for a `Real` field), a Function on the target's own
        mesh and element, or the parent-mesh encoding of `CheckpointField`.
      parent: the parent mesh of the encoding.
      name: the state name, for messages.
    """
    if isinstance(value, Number):
        target.assign(value)
        return
    if _same_space(target, value):
        target.assign(value)
        return
    if target.ufl_element().family() == "Real":
        target.assign(scalar_value(value))
        return
    if value.function_space().mesh() is not parent:
        raise ValueError(
            f"State entry {name!r} is on neither the target's space nor the "
            "parent mesh of the solver; load it on the mesh the solver uses.")
    # Decoded into a fresh Function with the unrecorded writes of the
    # encoding, then assigned with a recorded assignment. A value in the
    # target's own element is the copy that `write_state` makes; any other
    # element is the broken encoding of the earlier drivers.
    # On the collapsed space: a component space of the mixed space spans
    # both meshes, and an assignment across meshes needs a plain space.
    scratch = Function(target.function_space().collapse())
    broken = value.ufl_element() != target.ufl_element()
    with stop_annotating():
        CheckpointField(name, scratch, parent, broken=broken).restore(value)
    target.assign(scratch)


def write_state(checkpoint, solver, *, idx=None, timestepping_info=None,
                prefix: str = "") -> None:
    """Store `solver.state()` in an open `CheckpointFile`.

    Args:
      checkpoint: a `CheckpointFile` open for writing or appending.
      solver: the `SelfGravitatingGIASolver`.
      idx: the timestepping index of `save_function`, or `None`.
      timestepping_info: the timestepping values of `save_function`.
      prefix: prepended to every stored name, to keep two states apart.
    """
    parent = solver.potential_mesh
    with stop_annotating():
        for name, field in solver.state().items():
            stored = CheckpointField(prefix + name, field, parent,
                                     broken=False).encode()
            kwargs = {} if idx is None else {"idx": idx}
            if timestepping_info is not None:
                kwargs["timestepping_info"] = timestepping_info
            checkpoint.save_function(stored, **kwargs)


def read_state(checkpoint, solver, *, idx=None, prefix: str = "") -> dict:
    """Load the state of `solver`'s layout from an open `CheckpointFile`.

    The solver must be built on the parent mesh that `checkpoint.load_mesh`
    returned, because `CheckpointFile` loads a Function only onto that mesh.

    Returns:
      `{name: Function on the parent}`, for `solver.restore`.
    """
    parent = solver.potential_mesh
    kwargs = {} if idx is None else {"idx": idx}
    return {name: checkpoint.load_function(parent, prefix + name, **kwargs)
            for name in solver.state()}


def state_from_checkpoint(solver, fields: Mapping, metadata: Mapping | None = None,
                          *, aliases: Mapping | None = None) -> dict:
    """Convert a checkpoint of the earlier solver into a state of `solver`.

    Args:
      solver: the new solver, built on the parent mesh the fields were
        loaded on, with the configuration of the run that wrote the file.
      fields: `{stored name: Function on the parent}`, for example the
        `fields` of `selfgrav_common.load_checkpoint`.
      metadata: the time-history metadata of the file
        (`time_history_metadata`), and the `real_{i}` values of the Exp01
        layout. `None` for a backward-Euler file without metadata.
      aliases: `{state name: stored name}` for fields stored under other
        names, for example `{"displacement": "u", "potential": "psi",
        "history": "m"}`.

    Returns:
      A state for `solver.restore`.

    Raises:
      ValueError: when the file's scheme or history family disagrees with
        the solver, or when a state entry has no source in the file.
    """
    metadata = dict(metadata or {})
    aliases = dict(aliases or {})
    order = 2 if solver.time_scheme == "bdf2" else 1
    written = metadata.get("time_scheme_order", 1.0)
    if float(written) != order:
        raise ValueError(f"The checkpoint has time order {written}; the solver "
                         f"runs order {order}.")
    quadrature = float(metadata.get("material_solver_kind", 0.0)) == 1.0
    if quadrature != (solver.family == "Quadrature"):
        raise ValueError("The checkpoint's history family does not match the "
                         f"solver's {solver.family!r} family.")
    if quadrature and float(metadata.get("material_quad_degree", -1)) != solver.quad_degree:
        raise ValueError("The checkpoint's quadrature degree does not match "
                         "the solver's quad_degree.")
    state = {}
    # The earlier mixed solution, by index; its field order is this solver's.
    for i, name in enumerate(solver.field_names):
        key = f"solution_{i}"
        if key in fields:
            state[name] = fields[key]
        elif f"real_{i}" in metadata:
            state[name] = float(metadata[f"real_{i}"])
    # The earlier DG layout held the history as field 1 of the solution; the
    # stored histories outside the solution and the accepted BDF2 level come
    # after it and win, because the accepted level is what the next step
    # reads.
    if "internal_variables_0" in fields:
        state["history"] = fields["internal_variables_0"]
    if "time_current_0" in fields:
        state["history"] = fields["time_current_0"]
    if "time_previous_0" in fields:
        state["history_previous"] = fields["time_previous_0"]
    for name, stored in aliases.items():
        state[name] = fields[stored]
    previous = metadata.get("time_previous_dt", -1.0)
    state["previous_dt"] = float(previous)
    expected = set(solver.state())
    if order == 2 and "history_previous" not in state:
        raise ValueError("A BDF2 restart needs the previous history level "
                         "(time_previous_0).")
    missing = expected - set(state)
    if missing:
        raise ValueError(f"The checkpoint has no source for {sorted(missing)}.")
    return {name: state[name] for name in expected}
