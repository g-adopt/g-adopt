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

A Quadrature history is stored with the physical coordinates of its
quadrature points (`CheckpointField.save`), and `read_state` pairs the
values by position. The restart can therefore run on another number of
ranks, or on a new partition, than the run that wrote the file. The other
fields are stored in polynomial elements, which `CheckpointFile` maps to
any partition itself.

## Files of the earlier solver

`state_from_checkpoint` reads the layouts of `selfgrav_common` (gia-mip):
`solution_{i}` for every sub-function of the earlier mixed solution,
`internal_variables_{j}` for a history outside it, `time_current_{i}` and
`time_previous_{i}` for the `BDF2History` levels (`HISTORY_LEVEL_FIELD`),
and the time-history metadata whose keys this module defines
(`TIME_SCHEME_ORDER_KEY` and the other `*_KEY` names). The field order of
the earlier mixed space is the order of this solver's mixed space, so an
index names the same field in both. The Exp01 driver stores the `Real` rows
as `real_{i}` in its metadata and the fields under its own names, which the
`aliases` argument maps.

Those files hold no point coordinates, so their Quadrature history can be
read only on the number of ranks that wrote them. `state_from_checkpoint`
takes that number (`writer_ranks`, from
`gadopt.viscoelastic_checkpointing.written_rank_count`) and refuses
another one.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Number
from pathlib import Path

import h5py
from firedrake import Function
from pyadjoint.tape import stop_annotating

from .gia_couplings import scalar_value
from .viscoelastic_checkpointing import CheckpointField

__all__ = ["decode_into", "has_state", "read_state", "state_from_checkpoint",
           "stored_configuration", "write_state"]

#: The keys of the time-history metadata in a file of the earlier solver,
#: stored as the timestepping information of `CheckpointFile.save_function`:
#: the time order (1.0 or 2.0), the version of the record (1.0), the
#: accepted previous time step (-1.0 before the first BDF2 step), the history
#: family (1.0 for a Quadrature history, absent or 0.0 for DG) and the
#: quadrature degree of a Quadrature history. `state_from_checkpoint` reads
#: them; this module is their one definition.
TIME_SCHEME_ORDER_KEY = "time_scheme_order"
TIME_HISTORY_VERSION_KEY = "time_history_version"
TIME_PREVIOUS_DT_KEY = "time_previous_dt"
MATERIAL_SOLVER_KIND_KEY = "material_solver_kind"
MATERIAL_QUAD_DEGREE_KEY = "material_quad_degree"

#: The stored name of one accepted history level in a file of the earlier
#: solver: `level` is `"current"` or `"previous"`, `index` the number of the
#: history Function (0 for the one history of the GIA solver).
HISTORY_LEVEL_FIELD = "time_{level}_{index}"


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
        A Quadrature encoding is copied in its stored node order, which is
        right on the partition that encoded it only; `read_state` and
        `state_from_checkpoint` decode the Quadrature entries of a file
        themselves, with the checks of `CheckpointField.load`.
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


#: The HDF5 group of the configuration record of a state.
_GROUP = "/gadopt_gia_state"

#: The keys of the configuration record, in the order of `configuration`.
#: `write_state` stores each as an attribute of `_GROUP`, and
#: `stored_configuration` reads back these keys and no other attribute of
#: that group, so a caller can compare its result with `configuration`.
CONFIGURATION_KEYS = ("time_scheme", "internal_variable_family", "history_degree",
                      "dtn_representation", "fields")


@contextmanager
def _h5(checkpoint):
    """Give the `h5py.File` of an open `CheckpointFile`, or open a path read-only.

    A path is opened with plain `h5py` on each rank that calls, which is a
    read with no collective step, so a caller can test a file before it
    builds any mesh or solver.
    """
    if isinstance(checkpoint, (str, Path)):
        with h5py.File(checkpoint, "r") as f:
            yield f
    else:
        yield checkpoint.h5pyfile


def _decoded(value):
    """An HDF5 attribute as a Python value: `str` for text, a plain number."""
    # h5py returns a string attribute as bytes or str and a number as a
    # NumPy scalar, depending on how it was written.
    if isinstance(value, bytes):
        value = value.decode()
    if hasattr(value, "item"):
        value = value.item()
    return value


def has_state(checkpoint, prefix: str = "") -> bool:
    """Whether a checkpoint holds a state written by `write_state`.

    A driver uses it to choose between `read_state` and the conversion of a
    file of the earlier solver (`state_from_checkpoint`).

    Args:
      checkpoint: a path to the file, or a `CheckpointFile` open for reading.
      prefix: the prefix given to `write_state`.

    Returns:
      `True` when the file has the configuration record of `write_state`
      under `prefix`, `False` otherwise.
    """
    with _h5(checkpoint) as f:
        return (_GROUP + prefix) in f


def stored_configuration(checkpoint, prefix: str = "") -> dict | None:
    """The configuration that `write_state` recorded, read before a solver exists.

    The keys are the ones of the configuration of a solver
    (`CONFIGURATION_KEYS`): `time_scheme`, `internal_variable_family`,
    `history_degree`, `dtn_representation` and `fields`. Any other
    attribute of the record group is left out, so the result compares equal
    to `configuration(solver)` for a matching solver. A driver compares them with its own options before it builds
    the solver, so that it can stop with a message that names its option;
    `read_state` checks the same keys against the solver.

    Args:
      checkpoint: a path to the file, or a `CheckpointFile` open for reading.
      prefix: the prefix given to `write_state`.

    Returns:
      `{key: value}` for the keys of `CONFIGURATION_KEYS` that the record
      holds (strings, and an integer degree), or `None` when the file holds
      no state of `write_state` under `prefix`. A key that the record lacks
      is absent from the result; `read_state` refuses such a file.
    """
    with _h5(checkpoint) as f:
        group = _GROUP + prefix
        if group not in f:
            return None
        attrs = f[group].attrs
        return {key: _decoded(attrs[key]) for key in CONFIGURATION_KEYS if key in attrs}


def configuration(solver) -> dict:
    """What a state is only valid for: scheme, history family and degree, DtN.

    The history degree is the DG degree or the quadrature degree; a state of
    another degree has other degrees of freedom. The DtN representation
    decides whether the state holds multiplier fields.
    """
    return {"time_scheme": solver.time_scheme,
            "internal_variable_family": solver.family,
            "history_degree": int(solver.history.degree),
            "dtn_representation": solver.dtn_representation,
            "fields": ",".join(sorted(solver.state()))}


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
    checkpoint.require_group(_GROUP + prefix)
    for key, value in configuration(solver).items():
        checkpoint.set_attr(_GROUP + prefix, key, value)
    kwargs = {} if idx is None else {"idx": idx}
    if timestepping_info is not None:
        kwargs["timestepping_info"] = timestepping_info
    with stop_annotating():
        for name, field in solver.state().items():
            # `save` also stores the quadrature point coordinates of a
            # Quadrature history, once per file, for a read on any partition.
            CheckpointField(prefix + name, field, parent,
                            broken=False).save(checkpoint, **kwargs)


def read_state(checkpoint, solver, *, idx=None, prefix: str = "") -> dict:
    """Load the state of `solver`'s layout from an open `CheckpointFile`.

    The solver must be built on the parent mesh that `checkpoint.load_mesh`
    returned, because `CheckpointFile` loads a Function only onto that mesh.
    The number of ranks and the partition may differ from the writer's.

    Returns:
      `{name: Function}`, for `solver.restore`: a Quadrature history decoded
      into its own space (paired by the stored point positions), every other
      entry as its encoding on the parent.

    Raises:
      ValueError: when the file was written by a solver of another time
        scheme, history family, history degree or DtN representation, or
        when a Quadrature history cannot be read on this partition
        (`CheckpointField.load`).
    """
    record = stored_configuration(checkpoint, prefix) or {}
    for key, value in configuration(solver).items():
        if key not in record:
            raise ValueError(f"The checkpoint has no {key!r}; it was not "
                             "written by gadopt.gia_state.write_state.")
        stored = record[key]
        if stored != value:
            raise ValueError(f"The checkpoint was written with {key}={stored!r}; "
                             f"this solver has {value!r}.")
    parent = solver.potential_mesh
    kwargs = {} if idx is None else {"idx": idx}
    state = {}
    with stop_annotating():
        for name, target in solver.state().items():
            if _is_quadrature(target):
                # Decoded here, where the file and its point coordinates are
                # at hand; `restore` then assigns a Function of the target's
                # own space.
                scratch = Function(target.function_space())
                state[name] = CheckpointField(prefix + name, scratch, parent).load(
                    checkpoint, **kwargs)
            else:
                state[name] = checkpoint.load_function(parent, prefix + name, **kwargs)
    return state


def _is_quadrature(target) -> bool:
    """Whether a state entry is a Function of a Quadrature element."""
    return (isinstance(target, Function)
            and target.ufl_element().family() == "Quadrature")


def state_from_checkpoint(solver, fields: Mapping, metadata: Mapping | None = None,
                          *, aliases: Mapping | None = None,
                          writer_ranks: int | None = None) -> dict:
    """Convert a checkpoint of the earlier solver into a state of `solver`.

    Args:
      solver: the new solver, built on the parent mesh the fields were
        loaded on, with the configuration of the run that wrote the file.
      fields: `{stored name: Function on the parent}`, for example the
        `fields` of `selfgrav_common.load_checkpoint`.
      metadata: the time-history metadata of the file (the `*_KEY` names
        of this module), and the `real_{i}` values of the Exp01 layout.
        `None` for a backward-Euler file without metadata.
      aliases: `{state name: stored name}` for fields stored under other
        names, for example `{"displacement": "u", "potential": "psi",
        "history": "m"}`.
      writer_ranks: the number of ranks that wrote the file
        (`gadopt.viscoelastic_checkpointing.written_rank_count`). Required
        for a Quadrature history, which these files store in the node order
        of the writer's partition, with no point coordinates.

    Returns:
      A state for `solver.restore`; a Quadrature history decoded into its
      own space.

    Raises:
      ValueError: when the file's scheme or history family disagrees with
        the solver, when a state entry has no source in the file, or when a
        Quadrature history is read without `writer_ranks` or on another
        number of ranks than `writer_ranks`.
    """
    metadata = dict(metadata or {})
    aliases = dict(aliases or {})
    order = 2 if solver.time_scheme == "bdf2" else 1
    written = metadata.get(TIME_SCHEME_ORDER_KEY, 1.0)
    if float(written) != order:
        raise ValueError(f"The checkpoint has time order {written}; the solver "
                         f"runs order {order}.")
    quadrature = float(metadata.get(MATERIAL_SOLVER_KIND_KEY, 0.0)) == 1.0
    if quadrature != (solver.family == "Quadrature"):
        raise ValueError("The checkpoint's history family does not match the "
                         f"solver's {solver.family!r} family.")
    if quadrature and float(metadata.get(MATERIAL_QUAD_DEGREE_KEY, -1)) != solver.quad_degree:
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
    current = HISTORY_LEVEL_FIELD.format(level="current", index=0)
    previous_level = HISTORY_LEVEL_FIELD.format(level="previous", index=0)
    if current in fields:
        state["history"] = fields[current]
    if previous_level in fields:
        state["history_previous"] = fields[previous_level]
    for name, stored in aliases.items():
        state[name] = fields[stored]
    previous = metadata.get(TIME_PREVIOUS_DT_KEY, -1.0)
    state["previous_dt"] = float(previous)
    expected = set(solver.state())
    if order == 2 and "history_previous" not in state:
        raise ValueError("A BDF2 restart needs the previous history level "
                         f"({previous_level}).")
    missing = expected - set(state)
    if missing:
        raise ValueError(f"The checkpoint has no source for {sorted(missing)}.")
    targets = solver.state()
    parent = solver.potential_mesh
    for name in expected:
        target, value = targets[name], state[name]
        if not (_is_quadrature(target) and isinstance(value, Function)
                and not _same_space(target, value)):
            continue
        if writer_ranks is None:
            raise ValueError(
                f"The Quadrature history {name!r} of a file of the earlier "
                "solver is stored in the node order of the writer's partition. "
                "Pass writer_ranks=written_rank_count(checkpoint, mesh_name) "
                "(gadopt.viscoelastic_checkpointing), so that a read on another "
                "number of ranks is refused.")
        scratch = Function(target.function_space())
        with stop_annotating():
            CheckpointField(name, scratch, parent).restore(
                value, writer_ranks=writer_ranks)
        state[name] = scratch
    return {name: state[name] for name in expected}
