r"""This module provides a class to simplify computing of diagnostics typically encountered
in geodynamical simulations. Users instantiate the class by providing relevant
parameters and call individual class methods to compute associated diagnostics.

"""

from firedrake import (
    Constant, DirichletBC, FacetNormal, Function,
    assemble, dot, ds, dx, grad, norm, sqrt,
)
from firedrake.functionspaceimpl import MixedFunctionSpace
from firedrake.ufl_expr import extract_unique_domain
from mpi4py import MPI
import numpy as np

from .utility import CombinedSurfaceMeasure


class GeodynamicalDiagnostics:
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
      uz_min: Minimum velocity (final component, optionally over a given boundary)

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
        mesh = extract_unique_domain(z)

        # Allows sub-iv solver (just disp no mixed space)
        # to use same diagnostics. Would be better to separate
        # out into Base classes
        is_mixed_space = isinstance(
            z.function_space().topological, MixedFunctionSpace
        )
        if is_mixed_space:
            self.u, self.p = z.subfunctions[:2]
        else:
            self.u = z

        self.T = T

        self.dx = dx(domain=mesh, degree=quad_degree)
        self.domain_volume = assemble(Constant(1) * self.dx)

        if self.u.function_space().extruded:
            self.ds = CombinedSurfaceMeasure(mesh, quad_degree)
        else:
            self.ds = ds(mesh)

        if bottom_id:
            self.ds_b = self.ds(bottom_id)
            self.bottom_surface = assemble(Constant(1) * self.ds_b)
        if top_id:
            self.ds_t = self.ds(top_id)
            self.top_surface = assemble(Constant(1) * self.ds_t)

        self.n = FacetNormal(mesh)

    def u_rms(self):
        return norm(self.u) / sqrt(self.domain_volume)

    def u_rms_top(self) -> float:
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self):
        return -assemble(dot(grad(self.T), self.n) * self.ds_t) / self.top_surface

    def Nu_bottom(self):
        return assemble(dot(grad(self.T), self.n) * self.ds_b) / self.bottom_surface

    def T_avg(self):
        return assemble(self.T * self.dx) / self.domain_volume

    def T_min(self):
        T_data = self.T.dat.data_ro
        return self.T.comm.allreduce(T_data.min(), MPI.MIN)

    def T_max(self):
        T_data = self.T.dat.data_ro
        return self.T.comm.allreduce(T_data.max(), MPI.MAX)

    def ux_max(self, boundary_id=None) -> float:
        if boundary_id:
            bcu = DirichletBC(self.u.function_space(), 0, boundary_id)
            ux_data = self.u.dat.data_ro_with_halos[bcu.nodes, 0]
        else:
            ux_data = self.u.dat.data_ro[:, 0]

        return self.u.comm.allreduce(ux_data.max(), MPI.MAX)

    def uz_min(self, boundary_id=None) -> float:
        if boundary_id:
            bcu = DirichletBC(self.u.function_space(), 0, boundary_id)
            uz_data = self.u.dat.data_ro_with_halos[bcu.nodes, -1]
        else:
            uz_data = self.u.dat.data_ro[:, -1]

        return self.u.comm.allreduce(uz_data.min(initial=np.inf), MPI.MIN)
