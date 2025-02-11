r"""This module provides classes to perform an adjoint inverse optimisation and checkpoint
intermediate results. Users instantiate the `LinMoreOptimiser` class by providing
relevant parameters and call the `run` method to perform the optimisation.

"""

from pathlib import Path

import firedrake.utils
import pyadjoint.optimization.rol_solver as pyadjoint_rol
import ROL
from firedrake import CheckpointFile, Function, TestFunction, TrialFunction, assemble, dot, solve
from firedrake.adjoint import *  # noqa: F401
from mpi4py import MPI
from pyadjoint import Block, MinimizationProblem
from gadopt.utility import InteriorBC, CombinedSurfaceMeasure

# emulate the previous behaviour of firedrake_adjoint by automatically
# starting the tape
continue_annotation()

# As a part of ROL checkpointing, when a CheckpointedROLVector is encountered,
# we leverage Firedrake's checkpoint functionality to write the underlying vector
# to disk. Then we only need to serialise the filename for this checkpoint, and the
# inner product type. The loading process is a bit more complicated, because
# the vectors need to be loaded on a consistent mesh. Because of this, all
# the vectors that are loaded from a checkpoint register themselves in this list,
# where they can be processed to load the underlying data on the correct mesh.
_vector_registry = []


class ROLSolver(pyadjoint_rol.ROLSolver):
    """A ROLSolver that may use a class other than ROLVector to hold its vectors.

    In the non-checkpointing case, this reduces down to the original ROLSolver class.
    However, if ROL checkpointing is enabled, every vector within the solver needs to be
    a CheckpointedROLVector. We can achieve this by overwriting the base
    `self.rolvector` member.
    """

    def __init__(
        self,
        problem: MinimizationProblem,
        parameters: dict,
        inner_product: str = "L2",
        vector_class=pyadjoint_rol.ROLVector,
        vector_args: list = [],
    ):
        """Initialises the solver instance.

        Args:
          problem:
            PyAdjoint minimisation problem to solve.
          parameters:
            Dictionary containing the parameters governing ROL.
          inner_product:
            String representing the inner product to use for vector operations.
          vector_class:
            PyAdjoint ROL vector representing the underlying vector class.
          vector_args:
            List of arguments for initialisation of the vector class.
        """
        super().__init__(problem, parameters)

        x = [p.tape_value() for p in self.problem.reduced_functional.controls]
        self.rolvector = vector_class(x, *vector_args, inner_product=inner_product)

        # we need to recompute these with the new rolvector instance
        self.bounds = self.__get_bounds()
        self.constraints = self.__get_constraints()


class LinMoreOptimiser:
    def __init__(self, problem, parameters, checkpoint_dir=None, auto_checkpoint=True):
        """The management class for Lin-More trust region optimisation using ROL.

        This class sets up ROL to use the Lin-More trust region method, with a limited-memory
        BFGS secant for determining the steps. A pyadjoint problem has to be set up first,
        containing the optimisation functional and other constraints (like bounds).

        This optimiser also supports checkpointing ROL's state, to allow resumption of
        a previous optimisation without having to refill the L-BFGS memory. The underlying
        objects will be configured for checkpointing if `checkpoint_dir` is specified,
        and optionally the automatic checkpointing each iteration can be disabled by the
        `auto_checkpoint` parameter.

        Params:
            problem (pyadjoint.MinimizationProblem): The actual problem to solve.
            parameters (dict): A dictionary containing the parameters governing ROL.
            checkpoint_dir (Optional[str]): A path to hold any checkpoints that are saved.
            auto_checkpoint (Optional[bool]): Whether to automatically checkpoint each iteration.
        """

        self.iteration = -1

        solver_kwargs = {}
        if checkpoint_dir is not None:
            self._base_checkpoint_dir = Path(checkpoint_dir)
            self._ensure_checkpoint_dir()

            self._mesh = problem.reduced_functional.controls[0].control.function_space().mesh()
            solver_kwargs["vector_class"] = CheckpointedROLVector
            solver_kwargs["vector_args"] = [self]

            self.auto_checkpoint = auto_checkpoint
        else:
            self._base_checkpoint_dir = None
            self.auto_checkpoint = False

        self.rol_solver = ROLSolver(problem, parameters, inner_product="L2", **solver_kwargs)
        self.rol_parameters = ROL.ParameterList(parameters, "Parameters")

        try:
            self.rol_secant = ROL.lBFGS(parameters["General"]["Secant"]["Maximum Storage"])
        except KeyError:
            # Use the default storage value
            self.rol_secant = ROL.lBFGS()

        self.rol_algorithm = ROL.LinMoreAlgorithm(self.rol_parameters, self.rol_secant)
        self.callbacks = []

        self._add_statustest()

    def _ensure_checkpoint_dir(self):
        if MPI.COMM_WORLD.rank == 0:
            self.checkpoint_dir.mkdir(exist_ok=True)

        MPI.COMM_WORLD.Barrier()

    @property
    def checkpoint_dir(self):
        if self.iteration == -1:
            return self._base_checkpoint_dir

        return self._base_checkpoint_dir / str(self.iteration)

    def checkpoint(self):
        """Checkpoints the current ROL state to disk."""

        ROL.serialise_algorithm(self.rol_algorithm, MPI.COMM_WORLD.rank, str(self.checkpoint_dir))

        checkpoint_path = self.checkpoint_dir / "solution_checkpoint.h5"
        with CheckpointFile(str(checkpoint_path), "w") as f:
            for i, func in enumerate(self.rol_solver.rolvector.dat):
                f.save_function(func, name=f"dat_{i}")

    def restore(self, iteration=None):
        """Restores the ROL state from disk.

        The last stored iteration in `checkpoint_dir` is used unless a given iteration
        is specifed.
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            stored_iterations = [int(path.parts[-1]) for path in self._base_checkpoint_dir.glob('*[0-9]/')]
            self.iteration = sorted(stored_iterations)[-1]

        self.rol_algorithm = ROL.load_algorithm(MPI.COMM_WORLD.rank, str(self.checkpoint_dir))
        self._add_statustest()

        self.rol_solver.rolvector.checkpoint_path = self.checkpoint_dir / "solution_checkpoint.h5"
        self.rol_solver.rolvector.load(self._mesh)

        # ROL algorithms run in a loop like `while (statusTest()) { ... }`
        # so we will double up on saving the restored iteration
        # by rolling back the iteration counter, we make sure we overwrite the checkpoint
        # we just restored, to keep the ROL iteration count, and our checkpoint iteration
        # count in sync
        self.iteration -= 1

        # The various ROLVector objects can load all their metadata, but can't actually
        # restore from the Firedrake checkpoint. They register themselves, so we can access
        # them through a flat list.
        vec = self.rol_solver.rolvector.dat
        for v in _vector_registry:
            x = [p.copy(deepcopy=True) for p in vec]
            v.dat = x
            v.load(self._mesh)
            v._optimiser = self

        _vector_registry.clear()

    def run(self):
        """Runs the actual ROL optimisation.

        This will continue until the status test flags the optimisation to complete.
        """

        with stop_annotating():
            self.rol_algorithm.run(
                self.rol_solver.rolvector,
                self.rol_solver.rolobjective,
                self.rol_solver.bounds,
            )

    def _add_statustest(self):
        class StatusTest(ROL.StatusTest):
            def check(inner_self, status):
                self.iteration += 1

                if self.auto_checkpoint:
                    self._ensure_checkpoint_dir()
                    self.checkpoint()

                for callback in self.callbacks:
                    callback()

                return super().check(status)

        # Don't chain with the default status test
        self.rol_algorithm.setStatusTest(StatusTest(self.rol_parameters), False)

    def add_callback(self, callback):
        """Adds a callback to run after every optimisation iteration."""

        self.callbacks.append(callback)


class CheckpointedROLVector(pyadjoint_rol.ROLVector):
    """An extension of ROLVector that supports checkpointing to disk.

    The ROLVector itself is the Python-side implementation of the ROL.Vector interface;
    it defines all the operations ROL may perform on vectors (e.g., scaling, addition)
    and links ROL to the underlying Firedrake vectors.

    When the serialisation library hits a ROL.Vector on the C++ side, it will pickle
    this object, so we provide implementations of `__getstate__` and `__setstate__`
    that will correctly participate in the serialisation pipeline.
    """

    def __init__(
        self,
        dat: Function,
        optimiser: LinMoreOptimiser,
        inner_product: str = "L2",
    ):
        """Initialises the checkpointed ROL vector instance.

        Args:
          dat:
            Firedrake function.
          optimiser (LinMoreOptimiser):
            The managing optimiser for controlling checkpointing paths.
          inner_product:
            String representing the inner product to use for vector operations.
        """
        super().__init__(dat, inner_product)

        self.comm = dat[0].comm
        self._optimiser = optimiser

    def clone(self):
        dat = []
        for x in self.dat:
            dat.append(x._ad_copy())
        res = CheckpointedROLVector(
            dat, self._optimiser, inner_product=self.inner_product
        )
        res.scale(0.0)
        return res

    def save(self, checkpoint_path):
        """Checkpoints the data within this vector to disk.

        Called when this object is pickled as part of the ROL
        state serialisation.
        """

        with CheckpointFile(str(checkpoint_path), "w") as f:
            for i, func in enumerate(self.dat):
                f.save_function(func, name=f"dat_{i}")

    def load(self, mesh):
        """Loads the checkpointed data for this vector from disk.

        Called by the parent Optimiser after the ROL state has
        been deserialised. The pickling routine will register
        this vector within the registry.
        """

        with CheckpointFile(str(self.checkpoint_path), "r") as f:
            for i in range(len(self.dat)):
                self.dat[i] = f.load_function(mesh, name=f"dat_{i}")

    def __setstate__(self, state):
        """Sets the state from the result of unpickling.

        This happens during the restoration of a checkpoint. self.dat needs to be
        separately set, then self.load() can be called.
        """

        # initialise C++ state
        super().__init__(state)
        self.checkpoint_path, self.inner_product = state

        _vector_registry.append(self)

    def __getstate__(self):
        """Returns a state tuple suitable for pickling"""

        checkpoint_id = firedrake.utils._new_uid(self.comm)
        checkpoint_filename = f"vector_checkpoint_{checkpoint_id}.h5"
        checkpoint_path = self._optimiser.checkpoint_dir / checkpoint_filename
        self.save(checkpoint_path)

        return (checkpoint_path, self.inner_product)


minimisation_parameters = {
    "General": {
        "Print Verbosity": 1 if MPI.COMM_WORLD.rank == 0 else 0,
        "Output Level": 1 if MPI.COMM_WORLD.rank == 0 else 0,
        "Krylov": {
            "Iteration Limit": 10,
            "Absolute Tolerance": 1e-4,
            "Relative Tolerance": 1e-2,
        },
        "Secant": {
            "Type": "Limited-Memory BFGS",
            "Maximum Storage": 10,
            "Use as Hessian": True,
            "Barzilai-Borwein": 1,
        },
    },
    "Step": {
        "Type": "Trust Region",
        "Trust Region": {
            "Lin-More": {
                "Maximum Number of Minor Iterations": 10,
                "Sufficient Decrease Parameter": 1e-2,
                "Relative Tolerance Exponent": 1.0,
                "Cauchy Point": {
                    "Maximum Number of Reduction Steps": 10,
                    "Maximum Number of Expansion Steps": 10,
                    "Initial Step Size": 1.0,
                    "Normalize Initial Step Size": True,
                    "Reduction Rate": 0.1,
                    "Expansion Rate": 10.0,
                    "Decrease Tolerance": 1e-8,
                },
                "Projected Search": {
                    "Backtracking Rate": 0.5,
                    "Maximum Number of Steps": 20,
                },
            },
            "Subproblem Model": "Lin-More",
            "Initial Radius": 1.0,
            "Maximum Radius": 1e20,
            "Step Acceptance Threshold": 0.05,
            "Radius Shrinking Threshold": 0.05,
            "Radius Growing Threshold": 0.9,
            "Radius Shrinking Rate (Negative rho)": 0.0625,
            "Radius Shrinking Rate (Positive rho)": 0.25,
            "Radius Growing Rate": 10.0,
            "Sufficient Decrease Parameter": 1e-2,
            "Safeguard Size": 100,
        },
    },
    "Status Test": {
        "Gradient Tolerance": 0,
        "Iteration Limit": 100,
    },
}


class DiagnosticBlock(Block):
    """
    Writes sensitivity with respect to a function for diagnostic purposes.

    Useful for outputting gradients in time dependent simulations
    or inversions
    """

    def __init__(self, f, function, riesz_options={'riesz_representation': 'L2'}):
        """Initialises the Diagnostic block.

        Args:
          f:
            Filename of VTK pvd file to write to.
          function:
            Calculate gradient of reduced functional wrt to this function.
          riesz_options:
            Dictionary specifying riesz represenation (defaults to L2).
        """
        super().__init__()
        self.add_dependency(function)
        self.add_output(function.block_variable)
        self.f = f
        self.f_name = function.name()
        self.riesz_options = riesz_options

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return block_variable.checkpoint

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        out = inputs[0]._ad_convert_type(adj_inputs[0], options=self.riesz_options)
        out.rename('adjoint_'+self.f_name)
        self.f.write(out)
        return 0


class RieszL2BoundaryRepresentation:
    """Callable that Converts l2-representatives to L2-boundary representatives

    Necessary when visualing sensitivity wrt a function that is only
    defined on a surface. Using the usual L2 gradient (with a continuous
    function space) projects spurious oscillating noise into cells that
    are not connected to the boundary, which should have zero
    gradient. This changes the volume integral in the mass term to a
    surface integral so that the L2 gradient correction only accounts
    for the surface area of the element not the volume.
    """
    def __init__(self, Q, bids):
        self.Q = Q
        v = TestFunction(Q)
        g = TrialFunction(Q)
        bc = InteriorBC(Q, 0, bids)
        ds = CombinedSurfaceMeasure(v.function_space().mesh(), degree=6)
        self.M = assemble(dot(v, g)*ds(bids), bcs=bc)

    def __call__(self, value):
        ret = Function(self.Q)
        solve(self.M, ret, value)
        return ret
