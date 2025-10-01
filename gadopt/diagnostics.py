r"""This module provides a class to simplify computing of diagnostics typically
encountered in geodynamical simulations. Users instantiate the class by providing
relevant parameters and call individual class methods to compute associated diagnostics.

"""

from firedrake import Constant, DirichletBC, FacetNormal, Function, assemble, dot, ds, dx, grad, norm, sqrt
from firedrake.ufl_expr import extract_unique_domain
from mpi4py import MPI
from functools import cache

from firedrake.ufl_expr import extract_unique_domain
from .utility import CombinedSurfaceMeasure

__all__ = ["BaseDiagnostics", "GeodynamicalDiagnostics"]

class FunctionAttributeHolder:
    def __init__(self, quad_degree: int, func: Function):
        self.mesh = extract_unique_domain(func)
        self.function_space = func.function_space()
        self.dx = dx(domain=self.mesh, degree=quad_degree)
        self.ds = CombinedSurfaceMeasure(self.mesh, quad_degree) if self.function_space.extruded else ds(self.mesh)
        self.normal = FacetNormal(self.mesh)
        self.volume = assemble(Constant(1) * self.dx)
        # Fill in as necessary
        self.boundary_nodes: dict[int, list[int]] = {}

    def get_boundary_nodes(self, boundary_id: int) -> list[int]:
        if boundary_id not in self.boundary_nodes:
            bc = DirichletBC(self.function_space, 0, boundary_id)
            tmp = Function(self.function_space)
            self.boundary_nodes[boundary_id] = [n for n in bc.nodes if n < len(tmp.dat.data_ro)]
        return self.boundary_nodes[boundary_id]


class BaseDiagnostics:
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

    def _function_max(self, f: Function, boundary_id: int | None = None):
        if boundary_id:
            f_data = f.dat.data_ro[self._attrs[f].get_boundary_nodes(boundary_id), 0]
        else:
            f_data = f.dat.data_ro

        return f.comm.allreduce(f_data.max(), MPI.MAX)

    def _function_min(self, f: Function, boundary_id: int | None = None):
        if boundary_id:
            f_data = f.dat.data_ro[self._attrs[f].get_boundary_nodes(boundary_id), 0]
        else:
            f_data = f.dat.data_ro

        return f.comm.allreduce(f_data.min(), MPI.MIN)

    def _function_avg(self, f: Function):
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
      uv_min: Minimum velocity (vertical component, optionally over a given boundary)

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
        return fd.sqrt(fd.assemble(fd.dot(self.u, self.u) * self.ds_t))

    def Nu_top(self):
        return -assemble(dot(grad(self.T), self._attrs[self.T].normal) * self.ds_t) / self.top_surface

    def Nu_bottom(self):
        return assemble(dot(grad(self.T), self._attrs[self.T].normal) * self.ds_b) / self.bottom_surface

    def T_avg(self):
        return assemble(self.T * self._attrs[self.T].dx) / self._attrs[self.T].volume

    def T_min(self):
        return self._function_min(self.T)

    def T_max(self):
        return self._function_max(self.T)

    def ux_max(self, boundary_id: int | None = None) -> float:
        return self._function_max(self.u, boundary_id)
