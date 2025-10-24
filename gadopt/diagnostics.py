r"""This module provides a class to simplify computing of diagnostics typically
encountered in geodynamical simulations. Users instantiate the class by providing
relevant parameters and call individual class methods to compute associated diagnostics.

"""

import firedrake as fd
import numpy as np
from mpi4py import MPI
from functools import cache

from firedrake.ufl_expr import extract_unique_domain
from .utility import CombinedSurfaceMeasure

__all__ = ["BaseDiagnostics", "GeodynamicalDiagnostics"]


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
    """Hold objects that can be derived from a Firedrake Function

    This class gathers references to objects that can be pulled from a Firedrake
    Function object and calculates quantities based on those objects that will remain
    constant for the duration of a simulation. The set of objects/quantities stored
    are: mesh, function_space, dx and ds measures, the FacetNormal of the mesh
    (as the `.normal` attribute) and the volume of the domain.

    Args:
      quad_degree:
        The quadrature degree for the measures held by this object
      func:
        The Firedrake function these attributes belong to
    """

    def __init__(self, quad_degree: int, func: fd.Function):
        self.function = func
        self.mesh = extract_unique_domain(func)
        self.function_space = func.function_space()
        self.dx = get_dx(self.mesh, quad_degree)
        self.ds = get_ds(self.mesh, self.function_space, quad_degree)
        self.normal = get_normal(self.mesh)
        self.volume = get_volume(get_dx(self.mesh, quad_degree))

    @cache
    def get_boundary_nodes(self, boundary_id: int) -> list[int]:
        """Return the list of nodes on the boundary owned by this process

        Creates a `DirichletBC` object, then uses the `.nodes` attribute for that
        object to provide a list of indices that reside on the boundary of the domain
        of the function associated with this `FunctionContext` object. The
        `dof_dset.size` parameter of the `FunctionSpace` is used to exclude nodes in
        the halo region of the domain.

        Args:
          boundary_id:
            Integer ID of the domain boundary

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

    Args:
      quad_degree:
        The quadrature degree for the measures held by this object

      **funcs:
        key-value pairs of Firedrake functions and the class member that will be used
        to reference that function.
    """

    def __init__(self, quad_degree: int, **funcs: fd.Function):
        self._function_contexts: dict[fd.Function, FunctionContext] = {}

        for name, func in funcs.items():
            if len(func.subfunctions) == 1:
                setattr(self, name, func)
                self._init_single_func(quad_degree, func)
            else:
                for i, subfunc in enumerate(func.subfunctions):
                    setattr(self, f"{name}_{i}", func)
                    self._init_single_func(quad_degree, subfunc)

    def _init_single_func(self, quad_degree: int, func: fd.Function):
        """
        Create a FunctionContext for a single function
        """
        self._function_contexts[func] = FunctionContext(quad_degree, func)

    @cache
    def _check_present(self, func: fd.Function) -> None:
        """
        Determine if a function is present in this BaseDiagnostics object
        """
        if func not in self._function_contexts:
            raise KeyError(f"Function {func} is not present in this diagnostic object")

    @cache
    def _check_dim_valid(self, f: fd.Function) -> None:
        """
        Determine if the 'dim' argument can be used when searching for a function
        min/max (i.e. if f is a vector/tensor function).
        """
        if len(f.dat.shape) < 2:
            raise KeyError(
                "Requested a min/max over function dimension for a scalar function"
            )

    def _function_min(
        self,
        f: fd.Function,
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
        self._check_present(f)
        if boundary_id:
            f_data = f.dat.data_ro[
                self._function_contexts[f].get_boundary_nodes(boundary_id)
            ]
        else:
            f_data = f.dat.data_ro
        if dim is not None:
            self._check_dim_valid(f)
            f_data = f_data[:, dim]
        return f.comm.allreduce(f_data.min(initial=np.inf), MPI.MIN)

    def _function_max(
        self,
        f: fd.Function,
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
        self._check_present(f)
        if boundary_id:
            f_data = f.dat.data_ro[
                self._function_contexts[f].get_boundary_nodes(boundary_id)
            ]
        else:
            f_data = f.dat.data_ro
        if dim is not None:
            self._check_dim_valid(f)
            f_data = f_data[:, dim]
        return f.comm.allreduce(f_data.max(initial=-np.inf), MPI.MAX)

    def _function_avg(self, f: fd.Function):
        """Calculate the average value of a function

        Args:
          f:
            Firedrake function

        Returns:
          Average value of f across the entire domain associated with it
        """
        self._check_present(f)
        return (
            fd.assemble(f * self._function_contexts[f].dx)
            / self._function_contexts[f].volume
        )


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

        if T is None:
            self.T = 0.0
            super().__init__(quad_degree, u=u, p=p)
        else:
            super().__init__(quad_degree, u=u, p=p, T=T)

        if bottom_id:
            self.ds_b = self._function_contexts[self.u].ds(bottom_id)
            self.bottom_surface = fd.assemble(fd.Constant(1) * self.ds_b)
        if top_id:
            self.ds_t = self._function_contexts[self.u].ds(top_id)
            self.top_surface = fd.assemble(fd.Constant(1) * self.ds_t)

    def u_rms(self):
        return fd.norm(self.u) / fd.sqrt(self._function_contexts[self.u].volume)

    def u_rms_top(self) -> float:
        return fd.sqrt(fd.assemble(fd.dot(self.u, self.u) * self.ds_t))

    def Nu_top(self, scale: float = 1.0):
        return (
            -scale
            * fd.assemble(
                fd.dot(fd.grad(self.T), self._function_contexts[self.T].normal) * self.ds_t
            )
            / self.top_surface
        )

    def Nu_bottom(self, scale: float = 1.0):
        return (
            scale
            * fd.assemble(
                fd.dot(fd.grad(self.T), self._function_contexts[self.T].normal) * self.ds_b
            )
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
