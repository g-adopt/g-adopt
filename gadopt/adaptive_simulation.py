r"""This module provides the base class for simulations with adaptive meshes.

The class drives the mesh-adaptivity cycle of Animate: fields flagged for adaptation are
turned into a metric, the mesh is regenerated from the metric, and the simulation
objects are rebuilt on the new mesh by the subclass, which provides the physics.
"""

import abc
from collections.abc import Callable
from pathlib import Path
from shutil import rmtree

import firedrake as fd
from animate.adapt import adapt
from animate.metric import RiemannianMetric
from mpi4py import MPI

from .utility import function_name, log


class AdaptiveSimulation(abc.ABC):
    r"""Base class for a time-stepped simulation on an adaptively refined mesh.

    Subclasses provide the physics through three methods, `initialise`, `set_up`, and
    `time_loop`, and define the `solutions` and `fields` instance attributes documented
    under `set_up`. This class provides everything else: the mesh-adaptivity cycle, the
    interpolation of fields onto adapted meshes, and the checkpointing and output of the
    simulation state.

    The lifecycle of a simulation is:

        __init__: `initialise` on the initial mesh, followed by the initial
            mesh-adaptivity loops with `initialise` re-run between loops, and the
            initial checkpoint and output
        run: [`time_loop`, `adapt_mesh`, `set_up`] until the end time

    A typical subclass defines no `__init__`, passing the mesh and the simulation
    parameters to this constructor. A subclass that does define its own `__init__` must
    set everything `initialise` requires before delegating to this constructor, since
    `initialise` runs within it.
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        metric_parameters: dict[str, int | float],
        time_end: float,
        time_normalisation: float,
        time_step: float,
        adapt_calls: int = 3,
        initial_adapt_loops: int = 3,
    ) -> None:
        """Stores the simulation setup and performs the initial solve.

        Args:
          mesh:
            Firedrake mesh on which the simulation starts.
          metric_parameters:
            Parameters passed to Animate's RiemannianMetric, as documented under
            `set_parameters` in `animate/metric.py`. The target complexity and the
            minimum and maximum metric magnitudes must be specified.
          time_end:
            Simulation end time; `run` exits once it is reached.
          time_normalisation:
            Divisor applied to the simulation time when writing output, e.g. the number
            of time units per million years.
          time_step:
            Initial time step, also the time-step `Constant` handed to the subclass's
            solvers through the `time_step` attribute.
          adapt_calls:
            Number of times the mesh is adapted, with fields interpolated between calls,
            whenever `adapt_mesh` is invoked.
          initial_adapt_loops:
            Number of mesh-adaptivity loops performed on the initial fields, with
            `initialise` re-run between loops.

        Notes:
          `initialise` is invoked within this constructor; a subclass defining its own
          `__init__` must set everything `initialise` requires before delegating here.

        """
        self.mesh = mesh
        self.metric_parameters = metric_parameters
        self.time_end = time_end
        self.time_normalisation = time_normalisation
        self.adapt_calls = adapt_calls
        self.initial_adapt_loops = initial_adapt_loops

        # Time-stepping objects
        self.time = fd.Constant(0.0)  # Initial time
        self.time_step = fd.Constant(time_step)  # Initial time step
        self.step = 0  # A counter to keep track of the simulation time-loop iterations
        self.checkpoint_counter = 0  # A counter to keep track of checkpoint files
        self.output_counter = 0  # A counter to keep track of output files

        self.initialise()  # Initialise solutions on the initial mesh

        for iteration in range(self.initial_adapt_loops):
            # Adapt original mesh based on initial fields
            log(f"Initial adapt iteration #{iteration}")
            self.adapt_mesh(initial=True)

            self.initialise()  # Initialise solutions on the adapted mesh

        # Write initial checkpoint and output
        self.write_checkpoint(initial=True)
        self.write_output(initial=True)

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

        def add_metric(field: fd.Function, scale: float = 1.0, log: bool = False):
            # Firedrake function for a metric over the mesh where the field lives
            metric = RiemannianMetric(M, name=f"Metric ({function_name(field)})")
            metric.set_parameters(self.metric_parameters)  # Set metric parameters
            # Field Hessian
            metric.compute_hessian(fd.ln(field) / fd.ln(10.0) if log else field)
            metric.enforce_spd()  # Ensure boundedness (symmetric positive-definite)

            # Frobenius norm
            metric_magnitude.interpolate(fd.sqrt(fd.inner(metric, metric)))
            metric_magnitude_max = MPI.COMM_WORLD.allreduce(
                metric_magnitude.dat.data_ro.max(), MPI.MAX
            )
            # Scale metric by the maximum value of its Frobenius norm across the mesh
            metric.assign(scale * metric / metric_magnitude_max)

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
                    for dim in range(field_specs["field"].ufl_shape[0]):
                        add_metric(
                            field_specs["field"][dim],
                            field_specs["metric_mods"]["scale"][dim],
                            field_specs["metric_mods"]["log"][dim],
                        )
                else:
                    add_metric(field_specs["field"], **field_specs["metric_mods"])

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

            field = field_specs.pop("field")
            space = fd.FunctionSpace(self.mesh, field.ufl_element())
            new_field = fd.Function(space, name=field.name())
            new_field.interpolate(field)
            field_specs["field"] = new_field

    def run(self) -> None:
        """Runs the simulation to the end time, adapting the mesh en route.

        The time loop is repeated until the end time is reached, with the mesh adapted
        to the current fields and the simulation objects rebuilt on the new mesh between
        loops.
        """
        while True:  # Mesh adaptivity loop
            self.time_loop()  # Run the time loop

            # Exit loop after reaching target time
            if float(self.time) >= self.time_end:
                break

            self.adapt_mesh()  # Adapt mesh based on current fields
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
            - `metric_mods` (fields in the metric only): keyword arguments of the
              per-field metric, `scale` and `log`, scaling the Hessian and switching to
              its base-10 logarithm; components of vector fields provide lists thereof
            - `scale` (optional): scalar or callable applied to the field for output;
              defaults to 1.0, i.e. no scaling

        When called with `initial=False`, i.e. after mesh adaptation, this method must
        assign the current state from `fields` before discarding and regenerating them.
        """

    @abc.abstractmethod
    def time_loop(self) -> None:
        """Advances the simulation in time over a number of time steps.

        The method is invoked repeatedly by `run` and must return once the end time is
        reached or the subclass's iteration budget is exhausted; between invocations,
        the mesh is adapted and `set_up` re-run.
        """

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
