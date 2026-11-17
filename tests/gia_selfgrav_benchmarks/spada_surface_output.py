"""Surface-only Spada output on the simulation's curved boundary facets."""

from pathlib import Path

import firedrake as fd
import numpy as np
import selfgrav_common as common
from mpi4py import MPI
from spada_surface_io import SurfaceHDF5Series

from gadopt.momentum_equation import rotational_potential


class NodalRestriction:
    """Copy CG1/2/3 nodal values to a direct simplex submesh, without a search.

    PETSc's subpoint map identifies parent entities; scalar CG sections
    identify their nodes. For CG3 the two edge nodes follow the edge cone's
    orientation. Use scalar sections even for vector fields so indices count
    nodes independently of Firedrake's component/block storage conventions.
    Halo reads supply parent values when a submesh node has changed owner.
    """

    def __init__(self, source_mesh, target_mesh, degree):
        if degree not in (1, 2, 3):
            raise ValueError("Surface restriction supports nodal CG degrees 1, 2 and 3")
        self.source_mesh, self.target_mesh = source_mesh, target_mesh
        self.degree = degree
        source = fd.FunctionSpace(source_mesh, "CG", degree)
        target = fd.FunctionSpace(target_mesh, "CG", degree)
        source_section = source.dm.getLocalSection()
        target_section = target.dm.getLocalSection()
        parent_points = target_mesh.topology_dm.getSubpointIS().indices
        self.indices = np.full(target_section.getStorageSize(), -1, dtype=np.int64)
        for point in range(*target_section.getChart()):
            count = target_section.getDof(point)
            if not count:
                continue
            parent = int(parent_points[point])
            if source_section.getDof(parent) != count:
                raise ValueError("Source and submesh CG entity nodes do not match")
            first = source_section.getOffset(parent)
            nodes = np.arange(first, first + count, dtype=np.int64)
            if count == 2:
                source_cone = source_mesh.topology_dm.getCone(parent)
                target_cone = parent_points[target_mesh.topology_dm.getCone(point)]
                if np.array_equal(source_cone, target_cone[::-1]):
                    nodes = nodes[::-1]
                elif not np.array_equal(source_cone, target_cone):
                    raise ValueError("Cannot match submesh edge orientation")
            first_target = target_section.getOffset(point)
            self.indices[first_target:first_target + count] = nodes
        if np.any(self.indices < 0):
            raise ValueError("Some submesh nodes have no parent node")

    def __call__(self, source, target):
        """Collectively copy owned target nodes, fetching source halos first."""
        source_values = source.dat.data_ro_with_halos
        values = target.dat.data_wo
        values[:] = source_values[self.indices[:len(values)]]


class SurfaceOutput:
    """One HDF5 (default) or VTK series, in metres and m/yr, with SI geometry.

    The mesh has exactly the mantle's tagged boundary triangles. Copy P2
    coordinates through the topology map before creating its geometry:
    Submesh alone can inherit only linear coordinates. The dimensional
    output mesh shares topology with this nondimensional trace mesh, and
    its output Functions share Dats with the trace Functions. No simulation
    coordinates are changed, and no volume-to-surface point search occurs.
    """

    def __init__(self, directory, case, parent, mantle, *, boundary_id=None,
                 output_format="hdf5", restart_time=None):
        if output_format not in ("hdf5", "vtk"):
            raise ValueError("Surface output format must be hdf5 or vtk")
        self.output_format = output_format
        boundary_id = common.SURF_RE if boundary_id is None else boundary_id
        self.boundary_id = boundary_id
        boundary = fd.Submesh(mantle, 2, boundary_id, name=f"spada_{case}_boundary")
        geometry_degree = mantle.coordinates.function_space().ufl_element().degree()
        coordinates = fd.Function(fd.VectorFunctionSpace(boundary, "CG", geometry_degree))
        NodalRestriction(mantle, boundary, geometry_degree)(mantle.coordinates, coordinates)
        self.surface = fd.Mesh(coordinates, name=f"spada_{case}_surface_nd")
        coordinates_m = fd.Function(coordinates.function_space()).assign(coordinates * common.D_SCALE)
        self.mesh = fd.Mesh(coordinates_m, name=f"spada_{case}_surface_m")
        self.parent, self.mantle = parent, mantle
        self.displacement_degree = common.DISPLACEMENT_DEGREE
        self.potential_degree = common.POTENTIAL_DEGREE
        vector = fd.VectorFunctionSpace(self.surface, "CG", self.displacement_degree)
        scalar = fd.FunctionSpace(self.surface, "CG", self.potential_degree)
        self._u, self._previous_u = fd.Function(vector), fd.Function(vector)
        self._psi, self._previous_psi = fd.Function(scalar), fd.Function(scalar)
        self._mantle_psi = fd.Function(fd.FunctionSpace(mantle, "CG", self.potential_degree))
        self._u_trace = NodalRestriction(mantle, self.surface, self.displacement_degree)
        self._psi_trace = NodalRestriction(mantle, self.surface, self.potential_degree)
        self._psi_to_mantle = (None if parent.topology is mantle.topology else
                               NodalRestriction(parent, mantle, self.potential_degree))

        x = fd.SpatialCoordinate(self.surface)
        self.radial = x / fd.sqrt(fd.dot(x, x))
        rho = fd.sqrt(x[0]**2 + x[1]**2)
        safe_rho = fd.conditional(rho > 1e-12, rho, 1.)
        self.east = fd.conditional(rho > 1e-12,
                                  fd.as_vector((-x[1]/safe_rho, x[0]/safe_rho, 0.)),
                                  fd.as_vector((0., 1., 0.)))
        self.north = fd.cross(self.radial, self.east)
        self.gravity = common.gravity_exact_ufl(fd.Constant(common.RE))
        self._data, self.fields = {}, {}
        for suffix in ("m", "rate_m_per_yr"):
            self._field(f"displacement_{suffix}", self.displacement_degree, vector=True)
            for component in ("radial", "northward", "eastward"):
                self._field(f"{component}_displacement_{suffix}", self.displacement_degree)
            for kind in ("gravitational", "rotational", "total"):
                degree = self.potential_degree if kind == "gravitational" else self.displacement_degree
                self._field(f"geoid_{kind}_{suffix}", degree)
        for name in ("rate_valid", "rate_dt_yr", "rate_interval_start_kyr", "rate_interval_end_kyr"):
            self._field(name, 1)
        extension = "h5" if output_format == "hdf5" else "pvd"
        self.path = str(Path(directory) / f"spada_{case}_surface.{extension}")
        self.cell_count = self.surface.comm.allreduce(self.surface.cell_set.size, op=MPI.SUM)
        if output_format == "hdf5":
            self.series = SurfaceHDF5Series(self.path, self.mesh, self.metadata(),
                                            restart_time=restart_time)
        else:
            self.vtk = fd.VTKFile(self.path, comm=mantle.comm,
                                  mode="a" if restart_time is not None else "w",
                                  target_degree=self.displacement_degree)

    def _field(self, name, degree, *, vector=False):
        space = fd.VectorFunctionSpace if vector else fd.FunctionSpace
        values = fd.Function(space(self.surface, "CG", degree), name=name)
        self._data[name] = values
        self.fields[name] = fd.Function(space(self.mesh, "CG", degree), val=values.dat, name=name)

    def metadata(self):
        return {"file": Path(self.path).name, "format": self.output_format,
                "mesh_name": self.mesh.name,
                "surface_cells": int(self.cell_count),
                "boundary_id": int(self.boundary_id),
                "coordinates_unit": "m", "time_unit": "kyr", "vtk_degree": self.displacement_degree,
                "geometry": "exact nodal restriction of simulation boundary geometry",
                "rates": "adjacent-step backward difference; NaN at t=0",
                "geoid_gravitational": "psi/g0", "geoid_total": "(psi + psi_rot)/g0",
                "fields": list(self.fields)}

    def _potential_trace(self, source, target):
        if self._psi_to_mantle is None:
            self._psi_trace(source, target)
        else:
            self._psi_to_mantle(source, self._mantle_psi)
            self._psi_trace(self._mantle_psi, target)

    def _rotation_potential(self, values):
        return rotational_potential([fd.Constant(values.get(name, 0.))
                                     for name in ("m1", "m2", "m3")],
                                    self.surface, Omega_sq=common.OMEGA_SQ)

    def _displacement_fields(self, value, suffix):
        self._data[f"displacement_{suffix}"].interpolate(value)
        for component, basis in (("radial", self.radial), ("northward", self.north),
                                 ("eastward", self.east)):
            self._data[f"{component}_displacement_{suffix}"].interpolate(fd.dot(value, basis))

    def _geoid_fields(self, gravitational, rotational, suffix):
        self._data[f"geoid_gravitational_{suffix}"].interpolate(gravitational)
        self._data[f"geoid_rotational_{suffix}"].interpolate(rotational)
        self._data[f"geoid_total_{suffix}"].interpolate(gravitational + rotational)

    def write(self, solver, rates, t_kyr):
        """Write current fields and rates on the same undeformed surface."""
        self._u_trace(solver.displacement, self._u)
        self._potential_trace(solver.potential, self._psi)
        rotation = solver.rotation_values()
        rotational = self._rotation_potential(rotation)
        scale = common.D_SCALE / self.gravity
        self._displacement_fields(self._u * common.D_SCALE, "m")
        self._geoid_fields(self._psi * scale, rotational * scale, "m")
        if rates.dt_yr > 0:
            self._u_trace(rates.displacement, self._previous_u)
            self._potential_trace(rates.potential, self._previous_psi)
            self._displacement_fields((self._u - self._previous_u) * common.D_SCALE / rates.dt_yr,
                                      "rate_m_per_yr")
            previous_rotation = self._rotation_potential(rates.rotation)
            self._geoid_fields((self._psi - self._previous_psi) * scale / rates.dt_yr,
                               (rotational - previous_rotation) * scale / rates.dt_yr,
                               "rate_m_per_yr")
        else:
            for name, field in self._data.items():
                if name.endswith("rate_m_per_yr"):
                    field.assign(float("nan"))
        self._data["rate_valid"].assign(float(rates.dt_yr > 0))
        for name in ("rate_dt_yr", "rate_interval_start_kyr", "rate_interval_end_kyr"):
            self._data[name].assign(rates.metadata()[name])
        if self.output_format == "hdf5":
            self.series.write(self.fields.values(), t_kyr, rates.metadata())
        else:
            self.vtk.write(*self.fields.values(), time=t_kyr)
