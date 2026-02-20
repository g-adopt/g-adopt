r"""This module provides a class to simplify computing of diagnostics typically
encountered in geodynamical simulations. Users instantiate the class by providing
relevant parameters and call individual class methods to compute associated diagnostics.

"""

import firedrake as fd
import numpy as np
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.core.terminal import Terminal
from mpi4py import MPI
from functools import cache, cached_property, partial, _make_key

from firedrake.ufl_expr import extract_unique_domain
from .utility import CombinedSurfaceMeasure, vertical_component
from collections.abc import Sequence
from typing import Literal
from collections import defaultdict

__all__ = ["BaseDiagnostics", "GeodynamicalDiagnostics", "GIADiagnostics"]


# Free functions to allow caching of attributes that may be common among
# multiple functions
@cache
def get_dx(mesh, quad_degree: int) -> fd.Measure:
    return fd.dx(domain=mesh, degree=quad_degree)


@cache
def get_ds(mesh, quad_degree: int) -> fd.Measure:
    if mesh.extruded:
        return CombinedSurfaceMeasure(mesh, quad_degree)
    else:
        return fd.ds(domain=mesh, degree=quad_degree)


@cache
def get_normal(mesh):
    return fd.FacetNormal(mesh)


@cache
def get_volume(dx):
    return fd.assemble(fd.Constant(1) * dx)


@cache
def extract_functions(func_or_op: Expr) -> set[fd.Function]:
    """Extract all Firedrake functions associated with a UFL expression.

    This function recursively searches through any UFL expression for Firedrake
    `Function` objects.

    Args:
        func_or_op: The UFL expression to search through

    Raises:
        TypeError: An object that was neither a UFL Operator or UFL Terminal
        was encountered

    Returns:
        The set of found Firedrake Functions
    """
    if isinstance(func_or_op, fd.Function):
        return {func_or_op}
    elif isinstance(func_or_op, Operator):
        funcs = set()
        for f in func_or_op.ufl_operands:
            funcs |= extract_functions(f)
        return funcs
    elif isinstance(func_or_op, Terminal):
        # Some other UFL object
        return set()
    else:
        raise TypeError("Invalid type")


def ts_cache(
    _func=None,
    *,
    input_funcs: str | Sequence[str] | None = None,
    make_key=partial(_make_key, typed=False),
):
    """Cache the results of a diagnostic function on a per-timestep basis

    This function creates a decorator that caches the results of any diagnostic
    function found in a BaseDiagnostic object (or any subclass thereof) for as long
    as the underlying Firedrake functions remain unmodified. The modification of
    Firedrake functions is tracked by the `dat_version` attribute of the `dat` object
    which is based on the 'state' of the underlying PETSc object (see e.g.
    https://petsc.org/release/manualpages/Sys/PetscObjectStateGet/). Pyop2 also
    maintains a similar counter for non-PETSc objects.

    The purpose of this decorator is to allow multiple calls to the same diagnostic
    function within a timestep to reuse already computed quantities (e.g. Nusselt
    numbers on top/bottom boundaries in energy conservation calculations) or for
    underlying diagnostic algorithms to calculate multiple diagnostics at once when
    it is efficient to do so (e.g. min/max field values - in large parallel applications
    small reductions are dominated by network communication time, so it costs almost no
    extra calculate both the minimum and maximum value of the same field
    simultaneously).

    Args:
        _func (optional):
            Used to determine if the decorator is being called with or without
            parentheses.
        input_funcs (optional):
            A string or Sequence of strings of Firedrake functions that the cached
            results depend on. The decorator will automatically detect Firedrake
            functions in its arguments, this allows custom diagnostics that do not
            take functions as arguments to correctly track dependent functions.
            Default behaviour is to track automatically detected functions only.
        make_key (optional):
            A function to turn *args and **kwargs into a valid  dictionary key.
            Defaults to the same method used by functools.cache with typed=False.

    Raises:
        TypeError:
            The decorator has been used on an object that is not a G-ADOPT
            BaseDiagnostic object
            An attribute specified in `input_funcs` is not a Firedrake function

        AttributeError:
            The BaseDiagnostic object does not have an attribute named in the
            `input_funcs` argument.
    """

    def ts_cache_decorator(diag_func):
        cache = {}
        funcs = defaultdict(set)
        object_state = defaultdict(lambda: defaultdict(lambda: int(-1)))
        check_funcs = set()
        if input_funcs is not None:
            check_funcs |= set(
                (input_funcs,) if isinstance(input_funcs, str) else input_funcs
            )

        def wrapper(*args, **kwargs):
            key = make_key(args, kwargs)
            if key not in cache:
                # Do all sanity checking on the first call to the decorator
                if len(args) == 0 or not isinstance(args[0], BaseDiagnostics):
                    raise TypeError(
                        "This decorator can only be used on G-ADOPT Diagnostics functions"
                    )
                # Find all Firedrake functions in the arguments to this decorator
                for arg in args[1:]:
                    if isinstance(arg, Expr):
                        funcs[key] |= extract_functions(arg)
                for arg in kwargs.values():
                    if isinstance(arg, Expr):
                        funcs[key] |= extract_functions(arg)
                # Add any functions that were specified in the arguments to the
                # decorator factory
                for f in check_funcs:
                    if hasattr(args[0], f):
                        func = getattr(args[0], f)
                    else:
                        raise AttributeError(
                            f"No function named {f} found registered to this diagnostic object"
                        )
                    if not isinstance(func, fd.Function):
                        raise TypeError(
                            f"This diagnostic object has an attribute named {f} but it is not a Firedrake function"
                        )
                    funcs[key].add(func)
            if (
                any(object_state[key][f] != f.dat.dat_version for f in funcs[key])
                or not funcs[key]
            ):
                cache[key] = diag_func(*args, **kwargs)
                for f in funcs[key]:
                    object_state[key][f] = f.dat.dat_version
            return cache[key]

        return wrapper

    # See if we're being called as @ts_cache or @ts_cache().
    if _func is None:
        # We're called with parens.
        return ts_cache_decorator
    # We're called as @ts_cache without parens.
    return ts_cache_decorator(_func)


class FunctionContext:
    """Hold objects that can be derived from a Firedrake Function.

    This class gathers references to objects that can be pulled from a Firedrake
    Function object and calculates quantities based on those objects that will remain
    constant for the duration of a simulation. The set of objects/quantities stored
    are: mesh, function_space, dx and ds measures, the FacetNormal of the mesh
    (as the `.normal` attribute) and the volume of the domain.

    Typical usage example:

      function_contexts[F] = FunctionContext(quad_degree, F)

    Args:
        quad_degree: Quadrature degree to use when approximating integrands involving
        `func`
        func: Function

    """

    def __init__(self, quad_degree: int, func: fd.Function):
        self._function = func
        self._quad_degree = quad_degree

    @cached_property
    def function(self):
        """The function associated with the instance"""
        return self._function

    @cached_property
    def mesh(self):
        """The mesh on which the function has been defined"""
        return extract_unique_domain(self.function)

    @cached_property
    def function_space(self):
        """The function space on which the function has been defined"""
        return self.function.function_space()

    @cached_property
    def dx(self):
        """
        The volume integration measure defined by the mesh and
        `quad_degree` passed when creating this instance
        """
        return get_dx(self.mesh, self._quad_degree)

    @cached_property
    def ds(self):
        """
        The surface integration measure defined by the mesh and
        `quad_degree` passed when creating this instance
        """
        return get_ds(self.mesh, self._quad_degree)

    @cached_property
    def normal(self):
        """The facet normal of the mesh belonging to this instance"""
        return get_normal(self.mesh)

    @cached_property
    def volume(self):
        """The volume of the mesh belonging to this instance"""
        return get_volume(self.dx)

    @cached_property
    def boundary_ids(self):
        """The boundary IDs of the mesh associated with this instance"""
        return tuple(self.mesh.topology.exterior_facets.unique_markers) + (
            ("top", "bottom") if self.mesh.extruded else ()
        )

    @cache
    def check_boundary_id(self, boundary_id: Sequence[int | str] | int | str) -> None:
        """Check if a boundary id or tuple of boundary ids is valid"""
        # strings are Sequences, so have to handle this first otherwise this function
        # searches for 't' 'o' 'p' in ( 'top', 'bottom' )
        if isinstance(boundary_id, str):
            if boundary_id not in self.boundary_ids:
                raise KeyError("Invalid boundary ID for function")
        elif isinstance(boundary_id, Sequence):
            if not all(id in self.boundary_ids for id in boundary_id):
                raise KeyError("Invalid boundary ID for function")
        else:
            if boundary_id not in self.boundary_ids:
                raise KeyError("Invalid boundary ID for function")

    @cache
    def surface_area(self, boundary_id: Sequence[int | str] | int | str):
        """The surface area of the mesh on the boundary belonging to boundary_id"""
        self.check_boundary_id(boundary_id)
        return get_volume(self.ds(boundary_id))

    @cache
    def get_boundary_nodes(
        self, boundary_id: Sequence[int | str] | int | str
    ) -> list[int]:
        """Return the list of nodes on the boundary owned by this process

        Creates a `DirichletBC` object, then uses the `.nodes` attribute for that
        object to provide a list of indices that reside on the boundary of the domain
        of the function associated with this `FunctionContext` instance. The
        `dof_dset.size` parameter of the `FunctionSpace` is used to exclude nodes in
        the halo region of the domain.

        Args:
            boundary_id: Integer ID of the domain boundary

        Returns:
            List of integers corresponding to nodes on the boundary identified by
            `boundary_id`
        """
        self.check_boundary_id(boundary_id)
        bc = fd.DirichletBC(self.function_space, 0, boundary_id)
        return [n for n in bc.nodes if n < self.function_space.dof_dset.size]


class BaseDiagnostics:
    """A base class containing useful operations for diagnostics

    For each Firedrake function passed as a keyword argument in the `funcs` parameter,
    store that function as an attribute of the class accessible by its keyword, e.g.:

      `diag = BaseDiagnostics(quad_degree, z=z)`

    sets the Firedrake function `z` to the `diag.z` parameter.
    If the function is a `MixedFunction`, the subfunctions will be accessible by an
    index, e.g.:

      `diag = BaseDiagnostics(quad_degree, z=z)`

    sets the subfunctions of `z` to `diag.z_0`, `diag.z_1`, etc. A `FunctionContext`
    is created for each function. These attributes are accessed by the
    `diag._function_contexts` dict.

    This class is intended to be subclassed by domain-specific diagnostic classes

    Args:
        quad_degree: Quadrature degree to use when approximating integrands managed by
        this object
        **funcs: Firedrake functions to associate with this instance
    """

    def __init__(self, quad_degree: int, **funcs: fd.Function | None):
        """Initialise a BaseDiagnostics object.

        Sets the `quad_degree` for measures used by this object and passes the
        remaining keyword arguments through to `register_functions`.
        """
        self._function_contexts: dict[fd.Function | Operator, FunctionContext] = {}
        self._quad_degree = quad_degree
        self._mixed_functions: list[str] = []
        self.register_functions(**funcs)

    def register_functions(
        self, *, quad_degree: int | None = None, **funcs: fd.Function | None
    ):
        """Register a function with this BaseDiagnostics object.

        Creates a `FunctionContext` object for each function passed in as a keyword
        argument. Also creates an attribute on the instance to access the input function
        named for the key of the keyword argument. i.e:

        ```
        > diag.register_functions(self, F=F)
        > type(diag.F)
        <class 'firedrake.function.Function'>
        ```
        If an input function is set to `None`, the attribute will still be created
        but set to 0.0. If a mixed function is entered, each subfunction will have
        a `FunctionContext` object associated with it, and the attribute will be named
        with an additional number to denote the index of the subfunction i.e.:

        ```
        > diag.register_functions(self, F)
        > type(diag.F)
        AttributeError: 'Requested 'F', which lives on a mixed space. Instead, access subfunctions via F_0, F_1, ..."
        > type(diag.F_0)
        <class 'firedrake.function.Function'>
        > type(diag.F_1)
        <class 'firedrake.function.Function'>

        Args:
            quad_degree (optional): The quadrature degree for the measures to be used
            by this function. If `None`, the `quad_degree` passed at object
            instantiation time is used. Defaults to None.
            **funcs: key-value pairs of Firedrake functions to associate with this
            instance
        """
        if quad_degree is None:
            quad_degree = self._quad_degree
        for name, func in funcs.items():
            # Handle optional functions in diagnostics
            if func is None:
                setattr(self, name, 0.0)
                continue
            if len(func.subfunctions) == 1:
                if not hasattr(self, name):
                    setattr(self, name, func)
                    self._init_single_func(quad_degree, func)
            else:
                self._mixed_functions.append(name)
                for i, subfunc in enumerate(func.subfunctions):
                    if not hasattr(self, f"{name}_{i}"):
                        setattr(self, f"{name}_{i}", subfunc)
                        self._init_single_func(quad_degree, subfunc)

    def _init_single_func(self, quad_degree: int, func: fd.Function):
        """Create a FunctionContext for a single function"""
        self._function_contexts[func] = FunctionContext(quad_degree, func)

    def __getattr__(self, name: str):
        if name in self._mixed_functions:
            subfn_string = ", ".join([i for i in dir(self) if i.startswith(name + "_")])
            raise AttributeError(
                f"Requested '{name}', which lives in a mixed space. Instead, access subfunctions via {subfn_string}"
            )
        return super().__getattribute__(name)

    #
    # Section 1. Core functions whose output remains constant throughout a model run
    #            Generally intended for internal use only, though can be used to
    #            cache common UFL expressions (e.g. get_upward_component)
    #

    @cache
    def _check_present(self, func_or_op: Expr) -> None:
        """Check if a Firedrake function is known to this instance.

        Determine if a function is present in this BaseDiagnostics instance. If a UFL
        operator is passed in, check that all operands that are functions are present
        in this BaseDiagnostics instance.

        Args:
            func_or_op: The UFL expression to check

        Raises:
            KeyError: The functions associated with the expression were not found in
            this instance.
        """
        for func in extract_functions(func_or_op):
            if func not in self._function_contexts:
                raise KeyError(
                    f"Function {func} is not present in this diagnostic object"
                )

    @cache
    def _check_dim_valid(self, f: Expr) -> None:
        """
        Determine if the 'dim' argument can be used when searching for an expression
        min/max (i.e. if f is a vector/tensor function).
        """
        # The official UFL-sanctioned method of determining if an expression is
        # non-scalar
        # https://github.com/firedrakeproject/ufl/blob/master/ufl/core/expr.py#L290
        if not (f.ufl_shape or f.ufl_free_indices):
            raise TypeError(
                "Requested a min/max over function component but the function is scalar"
            )

    @cache
    def _check_boundary_id(
        self, f: fd.Function, boundary_id: Sequence[int | str] | int | str | None
    ) -> None:
        """Check if a provided boundary ID is valid.

        Args:
            f: Function to check
            boundary_id (optional): The boundary ID to check. If set to `None`,
            assume we're performing a volume integral, so no boundary ID is necessary.
            Otherwise check the boundary ID against those derived from the mesh
            belonging to this instance.

        Raises:
            KeyError: Mesh does not have a boundary corresponding to `boundary_id`
        """
        if boundary_id is not None:
            self._function_contexts[f].check_boundary_id(boundary_id)

    @cache
    def _get_measure(
        self,
        func_or_op: fd.Function | Operator,
        boundary_id: Sequence[int | str] | int | str | None = None,
    ) -> fd.Measure:
        """Get the integration measure associated with this UFL expression.

        If a boundary ID is provided, return the surface measure corresponding to that
        boundary. If not, return the volume measure.

        Args:
            func_or_op: UFL Expression to check
            boundary_id (optional): Boundary ID. If not provided or set to None,
            returns the volume measure. Defaults to None.

        Returns:
            The surface measure corresponding to the provided boundary ID or the
            volume measure.
        """
        self._check_boundary_id(func_or_op, boundary_id)
        for func in extract_functions(func_or_op):
            self._check_present(func)
            if boundary_id is None:
                return self._function_contexts[func].dx
            else:
                return self._function_contexts[func].ds(boundary_id)

    @cache
    def _get_func_for_op(self, op: Operator) -> fd.Function:
        """Return a function on which to interpolate a UFL operator.

        In some cases, UFL operators need to be interpolated onto a function in order
        to calculate diagnostic quantities. This function creates a new function on
        either an existing, compatible function space for the operator, or, if no
        suitable function space exists within the `FunctionContext` objects known to
        this instance, creates a new function space with the same base element as the
        function space used to construct the UFL expression.

        Args:
            op: A UFL expression

        Raises:
            TypeError: The function space detected for the UFL operator is not one of a
            Scalar, Vector or Tensor function space.

        Returns:
            A function on which to interpolate the diagnostic quantity described by
            `op`.
        """

        if op not in self._function_contexts:
            target_shape = op.ufl_shape
            for func in extract_functions(op):
                if func.ufl_shape == target_shape:
                    fs = self._function_contexts[func].function_space
                    break
            else:
                # The choice of function space isn't critically important, however we
                # want the basic element to match whatever the input space was. If
                # we've fallen through to here, use the last func to come out of
                # `extract_functions` as our starting point. If the FunctionSpace
                # element is not scalar, reduce it to a scalar element. When a given
                # scalar element is passed into VectorFunctionSpace or TensorFunction
                # space, Firedrake will automatically construct the necessary
                # vector/tensor element from it.
                if func.ufl_shape:
                    # Assume all sub-elements are the same for vector/tensor spaces
                    sub_elem = func.ufl_element().sub_elements[0]
                else:
                    sub_elem = func.ufl_element()
                match len(target_shape):
                    case 0:
                        # Scalar target function
                        fs = fd.FunctionSpace(func.ufl_domain(), sub_elem)
                    case 1:
                        # Vector target function
                        fs = fd.VectorFunctionSpace(func.ufl_domain(), sub_elem)
                    case 2:
                        # Tensor target function
                        fs = fd.TensorFunctionSpace(func.ufl_domain(), sub_elem)
                    case _:
                        # Don't know
                        raise TypeError("Unknown function space type")
            self._function_contexts[op] = FunctionContext(
                self._function_contexts[func]._quad_degree, fd.Function(fs)
            )
        return self._function_contexts[op].function

    @cache
    def get_upward_component(self, f: fd.Function) -> Operator:
        """Get the upward (against gravity) component of a function.

        Returns a UFL expression for the upward component of a function. Uses the
        G-ADOPT `vertical_component` function and caches the result such that the
        UFL expression only needs to be constructed once per run.

        Args:
            f: Function

        Returns:
            UFL expression for the vertical component of `f`
        """
        self._check_present(f)
        self._check_dim_valid(f)  # Can't take upward component of a scalar function
        return vertical_component(f)

    #
    # Section 2. Implementations
    #            Shared implementations for user-facing functions go here
    #

    @ts_cache
    def _minmax(
        self,
        func_or_op: fd.Function | Operator,
        boundary_id: Sequence[int | str] | int | str | None = None,
        dim: int | None = None,
    ) -> np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]:
        """Calculate the minimum and maximum value of a Firedrake function

        Use the `dat.dat_ro` object of a Firedrake function to extract the values
        of that function, then use an MPI_Allreduce to simultaneously calculate the
        minimum and maximum of that function. If a UFL operator is passed in, first
        interpolate that onto a Firedrake function before calculating. If a boundary
        ID is provided, calculate the min/max along that boundary only.

        Args:
            func_or_op: UFL Expression on on which to find min/max
            boundary_id (optional): Boundary ID. If not provided or set to `None`,
            will find min/max across entire domain. Defaults to None.
            dim (optional): For vector functions, the dimension to over which to
            calculate min/max. If not provided or set to `None`, calculate min/max
            across all components. Defaults to None.

        Returns:
            The minimum and negative maximum value of the function over the
            specified domain
        """
        self._check_present(func_or_op)
        if isinstance(func_or_op, Operator):
            f = self._get_func_for_op(func_or_op)
            f.interpolate(func_or_op)
        elif isinstance(func_or_op, fd.Function):
            f = func_or_op
        func_ctx = self._function_contexts[func_or_op]
        self._check_boundary_id(func_or_op, boundary_id)
        if boundary_id:
            f_data = f.dat.data_ro[func_ctx.get_boundary_nodes(boundary_id)]
        else:
            f_data = f.dat.data_ro
        if dim is not None:
            self._check_dim_valid(f)
            f_data = f_data[:, dim]
        buf = np.array((f_data.min(initial=np.inf), -f_data.max(initial=-np.inf)))
        f.comm.Allreduce(MPI.IN_PLACE, buf.data, MPI.MIN)
        return buf

    #
    # Section 3. User-facing functionality
    #            Common quantities to be calculated across functions
    #            Every function in this section should take a boundary ID
    #            if appropriate
    #

    def min(
        self,
        func_or_op: fd.Function | Operator,
        boundary_id: Sequence[int | str] | int | str | None = None,
        dim: int | None = None,
    ) -> float:
        """
        Calculate the minimum value of a function. See `_minmax`
        docstring for more information.
        """
        return self._minmax(func_or_op, boundary_id, dim)[0]

    def max(
        self,
        func_or_op: fd.Function | Operator,
        boundary_id: Sequence[int | str] | int | str | None = None,
        dim: int | None = None,
    ) -> float:
        """
        Calculate the maximum value of a function See `_minmax`
        docstring for more information.
        """
        return -self._minmax(func_or_op, boundary_id, dim)[1]

    @ts_cache
    def integral(
        self,
        f: fd.Function,
        boundary_id: Sequence[int | str] | int | str | None = None,
    ) -> float:
        """Calculate the integral of a function over the domain associated with it

        Args:
            f: Function.
            boundary_id (optional): Boundary ID. If not provided or set to `None`
            will integrate across entire domain. If provided, will integrate along
            the specified boundary only. Defaults to None.

        Returns:
            Result of integration
        """
        self._check_present(f)
        measure = self._get_measure(f, boundary_id)
        return fd.assemble(f * measure)

    @ts_cache
    def l1norm(
        self, f: fd.Function, boundary_id: Sequence[int | str] | int | str | None = None
    ) -> float:
        """Calculate the L1norm of a function over the domain associated with it

        Args:
            f: Function.
            boundary_id (optional): Boundary ID .If not provided or set to `None`,
            will integrate across entire domain. If provided, will integrate along
            the specified boundary only. Defaults to None.

        Returns:
            float: L1 norm
        """
        self._check_present(f)
        measure = self._get_measure(f, boundary_id)
        return fd.assemble(abs(f) * measure)

    @ts_cache
    def l2norm(
        self, f: fd.Function, boundary_id: Sequence[int | str] | int | str | None = None
    ) -> float:
        """Calculate the L2norm of a function over the domain associated with it

        Args:
            f: Function.
            boundary_id (optional): Boundary ID. If not provided or set to `None`,
            will integrate across entire domain. If provided, will integrate along
            the specified boundary only. Defaults to None.

        Returns:
            float: L2 norm
        """
        self._check_present(f)
        measure = self._get_measure(f, boundary_id)
        return fd.sqrt(fd.assemble(fd.dot(f, f) * measure))

    @ts_cache
    def rms(self, f: fd.Function) -> float:
        """Calculate the RMS of a function over the domain associated with it

        For the purposes of this function, RMS is defined as L2norm/volume

        Args:
            f: Function.
            boundary_id (optional): Boundary ID. If not provided or set to `None`,
            will integrate across entire domain. If provided, will integrate along
            the specified boundary only. Defaults to None.

        Returns:
            float: RMS
        """
        return self.l2norm(f) / fd.sqrt(self._function_contexts[f].volume)


class GeodynamicalDiagnostics(BaseDiagnostics):
    """Typical simulation diagnostics used in geodynamical simulations.

    Arguments:
      z:            Firedrake function for mixed Stokes function space (velocity, pressure)
      T:            Firedrake function for temperature
      bottom_id:    Bottom boundary identifier
      top_id:       Top boundary identifier
      quad_degree:  Degree of polynomial quadrature approximation

    Note:
      All diagnostics are returned as floats.

    Functions:
      u_rms: Root-mean-square velocity
      u_rms_top: Root-mean-square velocity along the top boundary
      Nu_top: Nusselt number at the top boundary
      Nu_bottom: Nusselt number at the bottom boundary
      T_avg: Average temperature in the domain
      T_min: Minimum temperature in domain
      T_max: Maximum temperature in domain
      ux_max: Maximum velocity (first component, optionally over a given boundary)

    """

    def __init__(
        self,
        z: fd.Function,
        T: fd.Function | None = None,
        /,
        bottom_id: Sequence[int | str] | int | str | None = None,
        top_id: Sequence[int | str] | int | str | None = None,
        *,
        quad_degree: int = 4,
    ):
        u, p = z.subfunctions[:2]
        super().__init__(quad_degree, u=u, p=p, T=T)

        if bottom_id:
            self.bottom_id = bottom_id
            self.ds_b = self._function_contexts[self.u].ds(bottom_id)
        if top_id:
            self.top_id = top_id
            self.ds_t = self._function_contexts[self.u].ds(top_id)

    def u_rms(self):
        return self.rms(self.u)

    def u_rms_top(self) -> float:
        return self.l2norm(self.u, self.top_id)

    @ts_cache(input_funcs=["T"])
    def Nu_top(self, scale: float = 1.0):
        return (
            -scale
            * fd.assemble(
                fd.dot(fd.grad(self.T), self._function_contexts[self.T].normal)
                * self.ds_t
            )
            / self._function_contexts[self.T].surface_area(self.top_id)
        )

    @ts_cache(input_funcs="T")
    def Nu_bottom(self, scale: float = 1.0):
        return (
            scale
            * fd.assemble(
                fd.dot(fd.grad(self.T), self._function_contexts[self.T].normal)
                * self.ds_b
            )
            / self._function_contexts[self.T].surface_area(self.bottom_id)
        )

    def T_avg(self):
        return self.integral(self.T) / self._function_contexts[self.T].volume

    def T_min(self):
        return self.min(self.T)

    def T_max(self):
        return self.max(self.T)

    def ux_max(
        self, boundary_id: Sequence[int | str] | int | str | None = None
    ) -> float:
        return self.max(self.u, boundary_id, 0)


class GIADiagnostics(BaseDiagnostics):
    """Typical simulation diagnostics used in glacial isostatic adjustment simulations.

    Arguments:
      d:            Firedrake function for displacement
      bottom_id:    Bottom boundary identifier
      top_id:       Top boundary identifier
      quad_degree:  Degree of polynomial quadrature approximation

    Note:
      All diagnostics are returned as floats.

    Functions:
      u_rms: Root-mean-square displacement
      u_rms_top: Root-mean-square displacement along the top boundary
      ux_max: Maximum displacement (first component, optionally over a given boundary)
      uv_min: Minimum vertical displacement, optionally over a given boundary
      uv_max: Maximum vertical displacement, optionally over a given boundary
      l2_norm_top: L2 norm of displacement on top surface
      l1_norm_top: L1 norm of displacement on top surface
      integrated_displacement: integral of displacement on top surface
    """

    def __init__(
        self,
        u: fd.Function,
        /,
        bottom_id: Sequence[int | str] | int | str | None = None,
        top_id: Sequence[int | str] | int | str | None = None,
        *,
        quad_degree: int = 4,
    ):
        super().__init__(quad_degree, u=u)

        if bottom_id:
            self.ds_b = self._function_contexts[self.u].ds(bottom_id)
        if top_id:
            self.ds_t = self._function_contexts[self.u].ds(top_id)
            self.top_id = top_id

    def u_rms(self):
        return self.rms(self.u)

    def u_rms_top(self) -> float:
        return self.l2norm(self.u, self.top_id)

    def ux_max(self, boundary_id: int | None = None) -> float:
        return self.max(self.u, boundary_id, 0)

    def uv_min(self, boundary_id: int | None = None) -> float:
        "Minimum value of vertical component of velocity/displacement"
        return self.min(self.get_upward_component(self.u), boundary_id)

    def uv_max(self, boundary_id: int | None = None) -> float:
        "Maximum value of vertical component of velocity/displacement"
        return self.max(self.get_upward_component(self.u), boundary_id)

    def l2_norm_top(self) -> float:
        return self.l2norm(self.get_upward_component(self.u), self.top_id)

    def l1_norm_top(self) -> float:
        return self.l1norm(self.get_upward_component(self.u), self.top_id)

    def integrated_displacement(self) -> float:
        return self.integral(self.get_upward_component(self.u), self.top_id)
