"""This module provides the base class for simulations with adaptive meshes.

The class drives the mesh-adaptivity cycle of Animate: fields flagged for adaptation are
turned into a metric, the mesh is regenerated from the metric, and the simulation
objects are rebuilt on the new mesh by the subclass, which provides the physics.
"""

import abc
from collections import deque
from collections.abc import Callable
from pathlib import Path
from shutil import rmtree
from sys import float_info

import firedrake as fd
from animate.adapt import adapt
from animate.metric import RiemannianMetric
from mpi4py import MPI
from ufl.indexed import Indexed

from .utility import function_name, log


class AdaptiveSimulation(abc.ABC):
    """Base class for a time-stepped simulation on an adaptively refined mesh.

    Subclasses provide the physics through the abstract methods `initialise` and
    `set_up`, optionally overriding `time_loop`, `diagnostics`, and `plot_diagnostics`,
    and define the instance attributes documented under `set_up` and `time_loop`. This
    class provides everything else: the mesh-adaptivity cycle, the interpolation of
    fields onto adapted meshes, and the checkpointing and output of the simulation
    state. Near a steady state, mesh adaptation is disabled once remeshes perturb the
    solution more than it changes per time step (`remesh_jump_limit`).

    The lifecycle of a simulation is:

        __init__: `initialise` on the initial mesh, followed by the initial
            mesh-adaptivity loops with `initialise` re-run between loops, and the
            initial checkpoint and output
        run: [`time_loop`, `adapt_mesh`, `set_up`] until the end condition is met

    A typical subclass defines no `__init__`, passing the mesh and the simulation
    parameters to this constructor. A subclass that does define its own `__init__` must
    set everything `initialise` requires before delegating to this constructor, since
    `initialise` runs within it.
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        metric_fields: dict[str, dict[str, bool | float | list[bool | float]]],
        metric_parameters: dict[str, int | float],
        time_step: float,
        adapt_calls: int = 3,
        initial_adapt_loops: int = 3,
        iterations: int = 16,
        remesh_jump_limit: float | None = 2.0,
        checkpoint_frequency: int | float = 1,  # noqa: PYI041
        output_frequency: int | float = 1,  # noqa: PYI041
        time_normalisation: float = 1.0,
    ) -> None:
        """Stores the simulation setup and performs the initial solve.

        Args:
          mesh:
            Firedrake mesh on which the simulation starts.
          metric_fields:
            Fields driving the mesh adaptation, keyed by field name, with the per-field
            metric modifiers `scale` and `log` documented under `set_up` as values;
            components of vector fields provide lists thereof.
          metric_parameters:
            Parameters passed to Animate's RiemannianMetric, as documented under
            `set_parameters` in `animate/metric.py`. The target complexity and the
            minimum and maximum metric magnitudes must be specified.
          time_step:
            Initial time step, also the time-step `Constant` handed to the subclass's
            solvers through the `time_step` attribute.
          adapt_calls:
            Number of times the mesh is adapted, with fields interpolated between calls,
            whenever `adapt_mesh` is invoked.
          initial_adapt_loops:
            Number of mesh-adaptivity loops performed on the initial fields, with
            `initialise` re-run between loops.
          iterations:
            Number of time steps between mesh adaptations, i.e. the iteration budget of
            `time_loop`.
          remesh_jump_limit:
            Jump in the measure returned by `diagnostics` across a remesh beyond which
            mesh adaptation is disabled, once exceeded by two consecutive remeshes; None
            disables the freezing. The default of 2.0 freezes adaptation once a remesh
            perturbs the solution by more than it changes per time step.
          checkpoint_frequency:
            Frequency of checkpointing: an int counts time steps and a float spans
            simulation time, the exact runtime type picking the semantics.
          output_frequency:
            Frequency of output: an int counts time steps and a float spans simulation
            time, the exact runtime type picking the semantics.
          time_normalisation:
            Divisor applied to the simulation time when writing output, e.g. the number
            of time units per million years.

        Notes:
          `initialise` is invoked within this constructor; a subclass defining its own
          `__init__` must set everything `initialise` requires before delegating here.

        """
        self.mesh = mesh
        self.metric_fields = metric_fields
        self.metric_parameters = metric_parameters
        self.adapt_calls = adapt_calls
        self.initial_adapt_loops = initial_adapt_loops
        self.iterations = iterations
        self.remesh_jump_limit = remesh_jump_limit
        self.checkpoint_frequency = checkpoint_frequency
        self.output_frequency = output_frequency
        self.time_normalisation = time_normalisation

        # Time-stepping objects
        self.time = fd.Constant(0.0)  # Initial time
        self.time_step = fd.Constant(time_step)  # Initial time step
        self.step = 0  # A counter to keep track of the simulation time-loop iterations
        self.checkpoint_counter = 0  # A counter to keep track of checkpoint files
        self.output_counter = 0  # A counter to keep track of output files

        # Steady-state tracking: the measure of the solution's change returned by
        # `diagnostics`, the measure just before the most recent remesh, the jump
        # factors of the two most recent remeshes, and whether a remesh just occurred
        self.measure = None
        self.measure_pre_remesh = None
        self.remesh_jumps = deque(maxlen=2)
        self.just_remeshed = False
        # Whether mesh adaptation has been disabled near a steady state
        self.remesh_frozen = False

        self.initialise()  # Initialise solutions on the initial mesh

        for iteration in range(self.initial_adapt_loops):
            # Adapt original mesh based on initial fields
            log(f"Initial adapt iteration #{iteration}")
            self.adapt_mesh(initial=True)

            self.initialise()  # Initialise solutions on the adapted mesh

        # Write initial checkpoint and output
        self.write_checkpoint(initial=True)
        self.write_output(initial=True)

        # Calculate initial simulation diagnostics
        self.measure = self.diagnostics(initial=True)

    def adapt_mesh(self, initial: bool = False) -> None:
        """Adapts the mesh to the fields flagged for adaptation.

        For each adapt call, a Hessian-based metric is computed per flagged field,
        component-wise for vector fields, and the metrics are combined into an overall
        metric from which the mesh is regenerated. Flagged fields are interpolated onto
        each new mesh before the next call; fields excluded from the metric are left to
        `set_up`, which recomputes them.

        Args:
          initial:
            Whether the adaptation is part of the initial loops, in which case only the
            first adapt call is performed and no fields are interpolated, since
            `initialise` re-runs afterwards.

        """

        def add_metric(
            field: fd.Function | Indexed, scale: float = 1.0, use_log: bool = False
        ) -> None:
            """Computes the Hessian-based metric of a scalar field and appends it to
            `metrics`.

            The Hessian is that of the base-10 logarithm of the field if `use_log` is
            set, and the metric is scaled by `scale` and by the maximum value of its
            Frobenius norm across the mesh.
            """
            # Firedrake function for a metric over the mesh where the field lives
            metric = RiemannianMetric(M, name=f"Metric ({function_name(field)})")
            metric.set_parameters(self.metric_parameters)  # Set metric parameters
            # Field Hessian
            metric.compute_hessian(fd.ln(field) / fd.ln(10.0) if use_log else field)
            metric.enforce_spd()  # Ensure boundedness (symmetric positive-definite)

            # Frobenius norm
            metric_magnitude.interpolate(fd.sqrt(fd.inner(metric, metric)))
            metric_magnitude_max = MPI.COMM_WORLD.allreduce(
                metric_magnitude.dat.data_ro.max(), MPI.MAX
            )
            # Scale metric by the maximum value of its Frobenius norm across the mesh,
            # floored at the machine epsilon to guard against a vanishing Hessian, e.g.
            # for a constant field, which would otherwise divide by zero
            metric.assign(
                scale * metric / max(metric_magnitude_max, float_info.epsilon)
            )

            metrics.append(metric)

        for field_specs in self.fields.values():
            if field_specs["include_in_metric"] and "expr" in field_specs:
                field_specs["field"].interpolate(field_specs["expr"])

        for call_count in range(self.adapt_calls):
            log(f"Adapt call #{call_count}")

            M = fd.TensorFunctionSpace(self.mesh, "CG", 1)
            N = fd.FunctionSpace(self.mesh, "CG", 1)
            metric_magnitude = fd.Function(N)

            metrics = []
            for field_specs in self.fields.values():
                if not field_specs["include_in_metric"]:
                    continue

                if isinstance(field_specs["field"].ufl_element(), fd.VectorElement):
                    # Components of vector fields are passed separately, with the
                    # modifiers provided per component
                    for dim in range(field_specs["field"].ufl_shape[0]):
                        add_metric(
                            field_specs["field"][dim],
                            field_specs["metric_mods"]["scale"][dim],
                            field_specs["metric_mods"]["log"][dim],
                        )
                else:
                    add_metric(
                        field_specs["field"],
                        field_specs["metric_mods"]["scale"],
                        field_specs["metric_mods"]["log"],
                    )

            if not metrics:
                raise ValueError("No fields are flagged for mesh adaptation.")

            overall_metric = metrics[0].copy(deepcopy=True)  # Overall metric
            overall_metric.rename("Metric (overall)")
            # Minimum in all directions across all metrics
            overall_metric.intersect(*metrics[1:])
            # Ensure boundedness (symmetric positive-definite)
            overall_metric.enforce_spd()
            # Rescale metric to achieve the desired target complexity
            overall_metric.normalise()

            # Adapting the mesh can generate new nodes outside of the original mesh
            # Increase tolerance to ensure point location does not fail
            self.mesh.tolerance = 0.1
            # Generate new mesh based on overall metric
            self.mesh = adapt(self.mesh, overall_metric, serialise=True)

            if initial:
                break

            self.interpolate_fields()

    def diagnostics(self, initial: bool = False) -> float | None:
        """Computes simulation diagnostics.

        Optional method doing nothing by default; subclasses are free to override it,
        e.g. to compute and log integral diagnostics, such as G-ADOPT's geodynamical
        diagnostics, or to track a convergence measure for `steady_state`.

        The method may return a scalar measure of the solution's change over the last
        time step, e.g. the change of a solution field, which the base class tracks
        across remeshes: once the measure jumps across two consecutive remeshes by more
        than `remesh_jump_limit`, mesh adaptation is disabled, since remeshing then
        perturbs the solution by more than it changes per time step.

        The method is invoked with `initial` set, once the simulation is initialised,
        i.e. after the initial mesh-adaptivity loops and the initial checkpoint and
        output, and after every time step, immediately before `plot_diagnostics`.

        Args:
          initial:
            Whether the diagnostics are the initial ones, in which case the simulation
            is at its initial state, e.g. for creating a log file.

        Returns:
          A measure of the solution's change over the last time step, or None.

        """

    @abc.abstractmethod
    def initialise(self) -> None:
        """Initialises the simulation on the current mesh.

        This method must call `set_up` with `initial=True`, initialise the solution
        fields, and, for a transient simulation, drive the solvers to compute a
        consistent initial state. It is invoked on the initial mesh and again after
        every initial mesh-adaptivity loop.
        """

    def interpolate_fields(self) -> None:
        """Interpolates the simulation fields onto the current mesh.

        All fields flagged for adaptation are interpolated, as are fields without an
        expression; the remaining fields, e.g. derived diagnostics, are left to
        `set_up`, which recomputes them from their expressions.
        """
        for field_specs in self.fields.values():
            if not field_specs["include_in_metric"] and "expr" in field_specs:
                continue

            field = field_specs["field"]
            space = fd.FunctionSpace(self.mesh, field.ufl_element())
            new_field = fd.Function(space, name=field.name())
            new_field.interpolate(field)
            field_specs["field"] = new_field

    def plot_diagnostics(self) -> None:
        """Produces graphs of simulation diagnostics.

        Optional method doing nothing by default; subclasses are free to override it,
        e.g. to plot time series of the values computed by `diagnostics`.

        The method is invoked by `time_loop` after every time step, following
        `diagnostics`. Unlike the latter, it is not invoked upon initialisation, since
        time series are empty at that point.

        """

    def run(
        self,
        steady_state: Callable[[], bool] | None = None,
        time_end: float | None = None,
    ) -> None:
        """Runs the simulation to completion, adapting the mesh en route.

        The simulation proceeds in blocks of `iterations` time steps, with the mesh
        adapted to the current fields and the simulation objects rebuilt on the new mesh
        by `set_up` between blocks. It completes either once the simulation time reaches
        `time_end` or once the `steady_state` callable returns True, with both
        conditions checked after each time step.

        Near a steady state, mesh adaptation is disabled: once the measure returned by
        `diagnostics` jumps across two consecutive remeshes by more than
        `remesh_jump_limit`, the mesh is frozen and only `time_loop` is repeated, since
        remeshing then perturbs the solution by more than it changes per time step.

        Note that the simulation terminates on the `exit_status` attribute alone, which
        the base `time_loop` sets once the end condition is met: a subclass overriding
        `time_loop` must likewise set `exit_status` to True once the end condition,
        `exit_loop`, is met, or the simulation will not terminate.

        Args:
          steady_state:
            Callable invoked after each time step; the simulation completes once it
            returns True, i.e. once a steady state is reached.
          time_end:
            Simulation end time; takes precedence over `steady_state`.

        Raises:
          ValueError: neither `steady_state` nor `time_end` is provided.

        """
        if time_end is not None:  # The simulation stops after reaching a certain time

            def exit_loop() -> bool:
                return float(self.time) >= time_end

            self.exit_loop = exit_loop
        elif steady_state is not None:  # The simulation stops if a condition is met
            self.exit_loop = steady_state
        else:
            raise ValueError("One of `steady_state` or `time_end` must be provided.")

        self.exit_status = False
        while True:  # Mesh adaptivity loop
            self.time_loop()  # Run the time loop

            if self.exit_status:  # Exit loop once simulation completes
                break

            # Disable mesh adaptation once the measure jumps across two consecutive
            # remeshes by more than the limit, since remeshing then undoes more
            # convergence than it buys
            if (
                not self.remesh_frozen
                and self.remesh_jump_limit is not None
                and len(self.remesh_jumps) == 2
                and all(jump >= self.remesh_jump_limit for jump in self.remesh_jumps)
            ):
                self.remesh_frozen = True
                log("Near steady state: mesh adaptation disabled")

            if self.remesh_frozen:  # Continue on the current mesh, without adapting it
                continue

            self.measure_pre_remesh = self.measure  # Measure at the block's end
            self.adapt_mesh()  # Adapt mesh based on current fields
            self.just_remeshed = True  # The next measure follows a remesh
            self.set_up()  # Set simulation objects on the new mesh

    @abc.abstractmethod
    def set_up(self, initial: bool = False) -> None:
        """Sets up the simulation objects on the current mesh.

        This method must define the following instance attributes:

        - `solutions`: a list of Functions holding the solution fields, which are
          checkpointed under their names; subclasses choose the granularity, e.g. a
          mixed function is checkpointed as a single entry
        - `fields`: a dictionary keyed by field name, with a dictionary per field,
          containing the following keys:
            - `field`: Function holding the field on the current mesh
            - `output`: Function into which the scaled field is interpolated for writing
            - `expr` (optional): UFL expression from which the field is interpolated,
              e.g. a derived diagnostic; fields with an entry are interpolated from it
              for output, and before mesh adaptation
            - `include_in_metric`: whether the field drives mesh adaptation
            - `metric_mods` (fields in the metric only): the per-field metric modifiers
              `scale` and `log`, scaling the Hessian and switching to its base-10
              logarithm; components of vector fields provide lists thereof
            - `scale` (optional): scalar or callable applied to the field for output;
              defaults to 1.0, i.e. no scaling

        When called with `initial=False`, i.e. after mesh adaptation, this method must
        assign the current state from `fields` before discarding and regenerating them.
        """

    def time_loop(self) -> None:
        r"""Advances the simulation in time over a number of time steps.

        The method is invoked repeatedly by `run` and must return once the end condition
        set by the latter, `exit_loop`, is met or the iteration budget (`iterations`) is
        exhausted; between invocations, the mesh is adapted and `set_up` re-run, unless
        mesh adaptation has been disabled near a steady state.

        Subclasses relying on this implementation must define the `solvers` instance
        attribute, holding the solvers invoked per time step, in order. They may also
        define `tstep_adapt`, an object updating the time step, e.g. G-ADOPT's
        TimestepAdaptor, which is invoked when present.

        Checkpoints and output are written at the frequencies given to the constructor.

        """

        def due(event: str) -> bool:
            # Whether a checkpoint or output is due, based on the step count or the
            # simulation time, for int and float frequencies, respectively
            frequency = getattr(self, f"{event}_frequency")
            if isinstance(frequency, int):
                return self.step % frequency == 0
            elif isinstance(frequency, float):
                return float(self.time) >= getattr(self, f"{event}_counter") * frequency
            else:
                raise TypeError(f"'{event}_frequency' must be an int or a float")

        for _ in range(self.iterations):  # Time loop
            log(f"Time loop iteration #{self.step}")

            if hasattr(self, "tstep_adapt"):
                self.tstep_adapt.update_timestep()  # Update time step

            for solver in self.solvers:  # Perform solves
                solver.solve()

            # Increment iteration count and time
            self.step += 1
            self.time.assign(self.time + self.time_step)

            # Write checkpoint and output
            due_events = [event for event in ("checkpoint", "output") if due(event)]
            for event in due_events:
                getattr(self, f"write_{event}")()

            # Compute and log diagnostics
            self.measure = self.diagnostics()
            if self.just_remeshed:  # The measure follows a remesh
                self.just_remeshed = False
                # Record the jump the measure suffers across the remesh
                if self.measure is not None and self.measure_pre_remesh:
                    self.remesh_jumps.append(self.measure / self.measure_pre_remesh)
            self.plot_diagnostics()

            if self.exit_loop():  # Exit loop once simulation completes
                self.exit_status = True
                log("Simulation complete")

                # Write checkpoint and output, unless already written this time step
                for event in ("checkpoint", "output"):
                    if event not in due_events:
                        getattr(self, f"write_{event}")()

                break

    def write_checkpoint(self, initial: bool = False) -> None:
        """Checkpoints the mesh and all solution fields to an HDF5 file.

        Each entry of `solutions` is stored under its name. Subclasses choose the
        granularity: a mixed function is checkpointed as a single entry, and scalar
        fields as individual ones.

        Args:
          initial:
            Whether the checkpoint is the initial one, in which case the checkpoint
            directory is recreated.

        """
        if initial and MPI.COMM_WORLD.rank == 0:
            check_dir = Path("checkpoints").resolve()
            if check_dir.is_dir():
                rmtree(check_dir)
            check_dir.mkdir()

        if initial:
            MPI.COMM_WORLD.barrier()

        with fd.CheckpointFile(
            f"checkpoints/checkpoint_{self.checkpoint_counter}.h5", "w"
        ) as checkpoint_file:
            checkpoint_file.save_mesh(self.mesh)
            for solution in self.solutions:
                checkpoint_file.save_function(solution, name=solution.name())

        self.checkpoint_counter += 1

    def write_output(self, initial: bool = False) -> None:
        """Writes all fields to a time series of VTK files.

        Each field is scaled, with scalars multiplying the field and callables applied
        to it, and interpolated into its output Function. The simulation time is divided
        by the time normalisation.

        Args:
          initial:
            Whether the output is the initial one, in which case the VTK file is
            created.

        """
        if initial:
            self.output_file = fd.VTKFile("output.pvd", adaptive=True)

        output_fields = []
        for field_specs in self.fields.values():
            scale = field_specs.get("scale", 1.0)
            field_data = field_specs.get("expr", field_specs["field"])

            if isinstance(scale, Callable):
                field_specs["output"].interpolate(scale(field_data))
            else:
                field_specs["output"].interpolate(scale * field_data)

            output_fields.append(field_specs["output"])

        self.output_file.write(
            *output_fields, time=float(self.time) / self.time_normalisation
        )

        self.output_counter += 1
