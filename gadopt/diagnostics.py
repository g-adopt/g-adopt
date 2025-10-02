r"""This module provides a class to simplify computing of diagnostics typically
encountered in geodynamical simulations. Users instantiate the class by providing
relevant parameters and call individual class methods to compute associated diagnostics.

"""

from firedrake import (
    Constant,
    DirichletBC,
    FacetNormal,
    Function,
    assemble,
    dot,
    ds,
    dx,
    grad,
    norm,
    sqrt,
)
from firedrake.ufl_expr import extract_unique_domain
from mpi4py import MPI

from .utility import CombinedSurfaceMeasure


class FunctionAttributeHolder:
    """Hold Firedrake Function attributes

    This class gathers function attributes and calculates quantities based on those
    functions that will remain constant for the duration of a simulation. The set of
    attributes stored is the mesh, function_space, dx and ds measures, the FacetNormal
    of the mesh (as the `.normal` attribute) and the volume of the domain. An empty
    dict is also created to hold boundary nodes, however, this is only constructed
    on as needed by the simulation. Note that the function itself is not stored in by
    this class. Functions are hashable and can therefore be used as dict keys in to
    reference a FunctionAttributeHolder object.

    Args:
      quad_degree:
        The quadrature degree for the measures held by this object
      func:
        The Firedrake function these attributes belong to
    """

    def __init__(self, quad_degree: int, func: Function):
        self.mesh = extract_unique_domain(func)
        self.function_space = func.function_space()
        self.dx = dx(domain=self.mesh, degree=quad_degree)
        self.ds = (
            CombinedSurfaceMeasure(self.mesh, quad_degree)
            if self.function_space.extruded
            else ds(self.mesh)
        )
        self.normal = FacetNormal(self.mesh)
        self.volume = assemble(Constant(1) * self.dx)
        # Fill in as necessary
        self.boundary_nodes: dict[int, list[int]] = {}

    def get_boundary_nodes(self, boundary_id: int) -> list[int]:
        """Return the list of nodes on the boundary owned by this process

        Creates a `DirichletBC` object, then uses the `.nodes` attribute for that
        object to provide a list of indices that reside on the boundary of the domain
        of the function associated with this `FunctionAttributeHolder` object. The
        `dof_dset.size` parameter of the `FunctionSpace` is used to exclude nodes in
        the halo region of the domain. The result is cached for reuse.

        Args:
          boundary_id:
            Integer ID of the domain boundary

        Returns:
          List of integers corresponding to nodes on the boundary identified by
          `boundary_id`
        """
        if boundary_id not in self.boundary_nodes:
            bc = DirichletBC(self.function_space, 0, boundary_id)
            self.boundary_nodes[boundary_id] = [
                n for n in bc.nodes if n < self.function_space.dof_dset.size
            ]
        return self.boundary_nodes[boundary_id]


class BaseDiagnostics:
    """A base class containing useful operations for diagnostics

    For each Firedrake function passed as a keyword argument in the `funcs` parameter,
    store that function as an attibute of the class accessible by its keyword, e.g.:

      `diag = BaseDiagnostics(quad_degree, z=z)`

    sets the Firedrake function `z` to the `diag.z` parameter.
    If the function is a `MixedFunction`, the subfunctions will be accessible by an
    index, e.g.:

      `diag = BaseDiagnostics(quad_degree, z=z)`

    sets the subfunctions of `z` to `diag.z_0`, `diag.z_1`, etc. A
    `FunctionAttributeHolder` is created for each function. These attributes are
    accessed by the `diag._attrs` dict.

    Args:
      quad_degree:
        The quadrature degree for the measures held by this object

      **funcs:
        key-value pairs of Firedrake functions and the class member that will be used
        to reference that function.
    """

    def __init__(self, quad_degree: int, **funcs: Function):
        self._attrs: dict[Function, FunctionAttributeHolder] = {}

        for name, func in funcs.items():
            if len(func.subfunctions) == 1:
                setattr(self, name, func)
                self._init_single_func(quad_degree, func)
            else:
                for i, subfunc in enumerate(func.subfunctions):
                    setattr(self, f"{name}_{i}", func)
                    self._init_single_func(quad_degree, subfunc)

    def _init_single_func(self, quad_degree: int, func: Function):
        self._attrs[func] = FunctionAttributeHolder(quad_degree, func)

    def _function_min(
        self,
        f: Function,
        boundary_id: int | None = None,
        dim: int | None = None,
    ):
        """Calculate the minimum value of a function

        Args:
          f:
            Firedrake function
          boundary_id:
            Optional, if passed the minimum will be calculated on the specified
            boundary, otherwise it will be the minimum over the whole domain
          dim:
            Optional, if passed, will calculate the minimum only for the `dim`
            component of a vector function.

        Returns:
          Minimum value of f across the specified domain/component
        """
        if boundary_id:
            f_data = f.dat.data_ro[self._attrs[f].get_boundary_nodes(boundary_id)]
        else:
            f_data = f.dat.data_ro
        if dim:
            f_data = f_data[:, dim]
        return f.comm.allreduce(f_data.min(), MPI.MIN)

    def _function_max(
        self,
        f: Function,
        boundary_id: int | None = None,
        dim: int | None = None,
    ):
        """Calculate the maximum value of a function

        Args:
          f:
            Firedrake function
          boundary_id:
            Optional, if passed the maximum will be calculated on the specified
            boundary, otherwise it will be the maximum over the whole domain
          dim:
            Optional, if passed, will calculate the maximum only for the `dim`
            component of a vector function.

        Returns:
          Maximum value of f across the specified domain/component
        """
        if boundary_id:
            f_data = f.dat.data_ro[self._attrs[f].get_boundary_nodes(boundary_id)]
        else:
            f_data = f.dat.data_ro
        if dim:
            f_data = f_data[:, dim]
        return f.comm.allreduce(f_data.max(), MPI.MAX)

    def _function_avg(self, f: Function):
        """Calculate the average value of a function

        Args:
          f:
            Firedrake function

        Returns:
          Average value of f across the entire domain associated with it
        """
        return assemble(f * self._attrs[f].dx) / self._attrs[f].volume


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
        z: Function,
        T: Function = 0.0,
        /,
        bottom_id: int | str | None = None,
        top_id: int | str | None = None,
        *,
        quad_degree: int = 4,
    ):
        u, p = z.subfunctions[:2]
        T = T

        if isinstance(T, Function):
            super().__init__(quad_degree, u=u, p=p, T=T)
        else:
            super().__init__(quad_degree, u=u, p=p)

        if bottom_id:
            self.ds_b = self._attrs[self.u].ds(bottom_id)
            self.bottom_surface = assemble(Constant(1) * self.ds_b)
        if top_id:
            self.ds_t = self._attrs[self.u].ds(top_id)
            self.top_surface = assemble(Constant(1) * self.ds_t)

    def u_rms(self):
        return norm(self.u) / sqrt(self._attrs[self.u].volume)

    def u_rms_top(self) -> float:
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self, scale: float = 1.0):
        return (
            -scale
            * assemble(dot(grad(self.T), self._attrs[self.T].normal) * self.ds_t)
            / self.top_surface
        )

    def Nu_bottom(self, scale: float = 1.0):
        return (
            scale
            * assemble(dot(grad(self.T), self._attrs[self.T].normal) * self.ds_b)
            / self.bottom_surface
        )

    def T_avg(self):
        return self._function_avg(self.T)

    def T_min(self):
        return self._function_min(self.T)

    def T_max(self):
        return self._function_max(self.T)

    def ux_max(self, boundary_id: int | None = None) -> float:
        return self._function_max(self.u, boundary_id, 0)
