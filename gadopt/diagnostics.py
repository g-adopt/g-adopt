r"""This module provides a class to simplify computing of diagnostics typically
encountered in geodynamical simulations. Users instantiate the class by providing
relevant parameters and call individual class methods to compute associated diagnostics.

"""

import firedrake as fd
import numpy as np
import ufl.core.expr
import ufl.core.operator
import ufl.core.terminal
from mpi4py import MPI
from functools import cache, cached_property

from firedrake.ufl_expr import extract_unique_domain
from .utility import CombinedSurfaceMeasure, get_boundary_ids, vertical_component

__all__ = ["BaseDiagnostics", "GeodynamicalDiagnostics", "GIADiagnostics"]


# Free functions to allow caching of attributes that may be common among
# multiple functions
@cache
def get_dx(mesh, quad_degree: int) -> fd.Measure:
    return fd.dx(domain=mesh, degree=quad_degree)


@cache
def get_ds(mesh, function_space, quad_degree: int) -> fd.Measure:
    if function_space.extruded:
        return CombinedSurfaceMeasure(mesh, quad_degree)
    else:
        return fd.ds(domain=mesh, degree=quad_degree)


@cache
def get_normal(mesh):
    return fd.FacetNormal(mesh)


@cache
def get_volume(dx):
    return fd.assemble(fd.Constant(1) * dx)


class FunctionContext:
    """Hold objects that can be derived from a Firedrake Function.

    This class gathers references to objects that can be pulled from a Firedrake
    Function object and calculates quantities based on those objects that will remain
    constant for the duration of a simulation. The set of objects/quantities stored
    are: mesh, function_space, dx and ds measures, the FacetNormal of the mesh
    (as the `.normal` attribute) and the volume of the domain.

    Typical usage example:

      function_contexts[F] = FunctionContext(quad_degree,F)
    """

    def __init__(self, quad_degree: int, func: fd.Function):
        """Initialises the FunctionContext object for `func`

        Args:
            quad_degree: quad degree used to construct measures on
                         domains associated with `func`
            func: Function
        """
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
        return get_ds(self.mesh, self.function_space, self._quad_degree)

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
        return get_boundary_ids(self.mesh)

    @cache
    def get_boundary_nodes(self, boundary_id: int) -> list[int]:
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
    """

    def __init__(self, quad_degree: int, **funcs: fd.Function | None):
        """Initialise a BaseDiagnostics object.

        Sets the `quad_degree` for measures used by this object and passes the
        remaining keyword arguments through to `register_functions`.

        Args:
            quad_degree: The quadrature degree for the measures held by this object
            **funcs: key-value pairs of Firedrake functions to associate with this
            instance
        """
        self._function_contexts: dict[
            fd.Function | ufl.core.operator.Operator, FunctionContext
        ] = {}
        self._quad_degree = quad_degree
        self.register_functions(**funcs)

    def register_functions(
        self, *, quad_degree: int | None = None, **funcs: fd.Function | None
    ):
        """Register a function with this BaseDiagnostics object.

        Creates a `FunctionContext` object for each function passed in as a keyword
        argument. Also creates an attribute on the instance to access the input function
        named for the key of the keyword argument. i.e:

        ```
        > diag.register_functions(self,F=F)
        > type(diag.F)
        <class 'firedrake.function.Function'>
        ```
        If an input function is set to `None`, the attribute will still be created
        but set to 0.0. If a mixed function is entered, each subfunction will have
        a `FunctionContext` object associated with it, and the attribute will be named
        with an additional number to denote the index of the subfunction i.e.:

        ```
        > diag.register_functions(self,F)
        > type(diag.F)
        AttributeError: 'BaseDiagnostics' object has no attribute 'F'
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
                for i, subfunc in enumerate(func.subfunctions):
                    if not hasattr(self, f"{name}_{i}"):
                        setattr(self, f"{name}_{i}", subfunc)
                        self._init_single_func(quad_degree, subfunc)

    def _init_single_func(self, quad_degree: int, func: fd.Function):
        """Create a FunctionContext for a single function"""
        self._function_contexts[func] = FunctionContext(quad_degree, func)

    #
    # Section 1. Core functions whose output remains constant throughout a model run
    #            Generally intended for internal use only, though can be used to
    #            cache common UFL expressions (e.g. get_radial_component)
    #
    @cache
    def _extract_functions(self, func_or_op: ufl.core.expr.Expr) -> set[fd.Function]:
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
        elif isinstance(func_or_op, ufl.core.operator.Operator):
            funcs = set()
            for f in func_or_op.ufl_operands:
                funcs |= self._extract_functions(f)
            return funcs
        elif isinstance(func_or_op, ufl.core.terminal.Terminal):
            # Some other UFL object
            return set()
        else:
            raise TypeError("Invalid type")

    @cache
    def _check_present(self, func_or_op: ufl.core.expr.Expr) -> None:
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
        for func in self._extract_functions(func_or_op):
            if func not in self._function_contexts:
                raise KeyError(
                    f"Function {func} is not present in this diagnostic object"
                )

    @cache
    def _check_dim_valid(self, f: ufl.core.expr.Expr) -> None:
        """
        Determine if the 'dim' argument can be used when searching for an expression
        min/max (i.e. if f is a vector/tensor function).
        """
        # The official UFL-sanctioned method of determining if an expression is
        # non-scalar
        # https://github.com/firedrakeproject/ufl/blob/master/ufl/core/expr.py#L290
        if not (f.ufl_shape or f.ufl_free_indices):
            raise TypeError(
                "Requested a min/max over function dimension for a scalar function"
            )

    @cache
    def _check_boundary_id(
        self, f: fd.Function, boundary_id: int | str | None = None
    ) -> None:
        """Check if a provided boundary ID is valid.

        Args:
            f: Function to check
            boundary_id (optional): The boundary ID to check. If set to `None`,
            assume we're performing a volume integral, so no boundary ID is necessary.
            Otherwise check the boundary ID against those derived from the mesh
            belonging to this instance. Defaults to None.

        Raises:
            KeyError: Mesh does not have a boundary corresponding to `boundary_id`
        """
        if boundary_id is None:
            return
        if boundary_id not in self._function_contexts[f].boundary_ids:
            raise KeyError("Invalid boundary ID for function")

    @cache
    def _get_measure(
        self,
        func_or_op: fd.Function | ufl.core.operator.Operator,
        boundary_id: int | str | None = None,
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
        for func in self._extract_functions(func_or_op):
            self._check_present(func)
            if boundary_id is None:
                return self._function_contexts[func].dx
            else:
                return self._function_contexts[func].ds(boundary_id)

    @cache
    def _get_func_for_op(self, op: ufl.core.operator.Operator) -> fd.Function:
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
            for func in self._extract_functions(op):
                if func.ufl_shape == target_shape:
                    fs = self._function_contexts[func].function_space
                    break
            else:
                # The choice of function space isn't critically important, however we
                # want the basic element to match whatever the input space was. If
                # we've fallen through to here, use the last func to come out of
                # `_extract_functions` as our starting point. If the FunctionSpace
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
    def get_radial_component(self, f: fd.Function) -> ufl.core.operator.Operator:
        """Get the radial component of a function.

        Returns a UFL expression for the radial component of a function. Uses the
        G-ADOPT `vertical_component` function and caches the result such that the
        UFL expression only needs to be constructed once per run.

        Args:
            f: Function

        Returns:
            UFL expression for the vertical component of `f`
        """
        self._check_present(f)
        self._check_dim_valid(f)  # Can't take radial component of a scalar function
        return vertical_component(f)

    #
    # Section 2. Implementations
    #            Shared implementations for user-facing functions go here
    #

    def _minmax(
        self,
        func_or_op: fd.Function | ufl.core.operator.Operator,
        boundary_id: int | None = None,
        dim: int | None = None,
    ) -> np.ndarray[float, float]:
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
        if isinstance(func_or_op, ufl.core.operator.Operator):
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
        func_or_op: fd.Function | ufl.core.operator.Operator,
        boundary_id: int | None = None,
        dim: int | None = None,
    ) -> float:
        """
        Calculate the minimum value of a function. See `_minmax`
        docstring for more information.
        """
        return self._minmax(func_or_op, boundary_id, dim)[0]

    def max(
        self,
        func_or_op: fd.Function | ufl.core.operator.Operator,
        boundary_id: int | None = None,
        dim: int | None = None,
    ) -> float:
        """
        Calculate the maximum value of a function See `_minmax`
        docstring for more information.
        """
        return -self._minmax(func_or_op, boundary_id, dim)[1]

    def integral(self, f: fd.Function, boundary_id: int | str | None = None) -> float:
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

    def l1norm(self, f: fd.Function, boundary_id: int | str | None = None) -> float:
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

    def l2norm(self, f: fd.Function, boundary_id: int | str | None = None) -> float:
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
      u_rms: Root-mean squared velocity
      u_rms_top: Root-mean squared velocity along the top boundary
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
        bottom_id: int | str | None = None,
        top_id: int | str | None = None,
        *,
        quad_degree: int = 4,
    ):
        u, p = z.subfunctions[:2]
        super().__init__(quad_degree, u=u, p=p, T=T)

        if bottom_id:
            self.ds_b = self._function_contexts[self.u].ds(bottom_id)
            self.bottom_surface = fd.assemble(fd.Constant(1) * self.ds_b)
        if top_id:
            self.top_id = top_id
            self.ds_t = self._function_contexts[self.u].ds(top_id)
            self.top_surface = fd.assemble(fd.Constant(1) * self.ds_t)

    def u_rms(self):
        return self.rms(self.u)

    def u_rms_top(self) -> float:
        return self.l2norm(self.u, self.top_id)

    def Nu_top(self, scale: float = 1.0):
        return (
            -scale
            * fd.assemble(
                fd.dot(fd.grad(self.T), self._function_contexts[self.T].normal)
                * self.ds_t
            )
            / self.top_surface
        )

    def Nu_bottom(self, scale: float = 1.0):
        return (
            scale
            * fd.assemble(
                fd.dot(fd.grad(self.T), self._function_contexts[self.T].normal)
                * self.ds_b
            )
            / self.bottom_surface
        )

    def T_avg(self):
        return self.integral(self.T) / self._function_contexts[self.T].volume

    def T_min(self):
        return self.min(self.T)

    def T_max(self):
        return self.max(self.T)

    def ux_max(self, boundary_id: int | None = None) -> float:
        return self.max(self.u, boundary_id, 0)


class GIADiagnostics(BaseDiagnostics):
    def __init__(
        self,
        u: fd.Function,
        /,
        bottom_id: int | str | None = None,
        top_id: int | str | None = None,
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
        return self.min(self.get_radial_component(self.u), boundary_id)

    def uv_max(self, boundary_id: int | None = None) -> float:
        "Maximum value of vertical component of velocity/displacement"
        return self.max(self.get_radial_component(self.u), boundary_id)

    def l2_norm_top(self) -> float:
        return self.l2norm(self.get_radial_component(self.u), self.top_id)

    def l1_norm_top(self) -> float:
        return self.l1norm(self.get_radial_component(self.u), self.top_id)

    def integrated_displacement(self) -> float:
        return self.integral(self.get_radial_component(self.u), self.top_id)
