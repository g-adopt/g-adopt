import numpy as np
from mpi4py import MPI
import firedrake as fd

from gadopt import CubedSphereMesh, ExtrudedMesh, layerwise_nodal_control_volumes


base_radius = 2.2
mesh = ExtrudedMesh(
    CubedSphereMesh(base_radius, refinement_level=1, degree=2),
    layers=2,
    layer_height=-0.5,
    extrusion_type="radial",
)
areas = layerwise_nodal_control_volumes(mesh)

level_count = 2 * (mesh.layers - 1) + 1
area_data = areas.dat.data_ro.reshape((-1, level_count))
local_sums = area_data.sum(axis=0)
layer_sums = mesh.comm.allreduce(local_sums)
radii = np.linspace(base_radius, 1.2, level_count)
expected_factors = (radii / base_radius) ** 2
local_pointwise_error = np.max(
    np.abs(area_data - area_data[:, :1] * expected_factors[None, :]),
    initial=0.0,
)
pointwise_error = mesh.comm.allreduce(local_pointwise_error, op=MPI.MAX)

# Exercise the separate degree-one facet implementation on geometry above the
# degree supported by the reconstructed degree-two path.
linear_mesh = ExtrudedMesh(
    CubedSphereMesh(base_radius, refinement_level=1, degree=3),
    layers=2,
    layer_height=-0.5,
    extrusion_type="radial",
)
linear_areas = layerwise_nodal_control_volumes(linear_mesh, degree=1)
linear_level_count = linear_mesh.layers
linear_area_data = linear_areas.dat.data_ro.reshape((-1, linear_level_count))
linear_radii = np.linspace(base_radius, 1.2, linear_level_count)
linear_expected_factors = (linear_radii / base_radius) ** 2
local_linear_pointwise_error = np.max(
    np.abs(
        linear_area_data
        - linear_area_data[:, :1] * linear_expected_factors[None, :]
    ),
    initial=0.0,
)
linear_pointwise_error = linear_mesh.comm.allreduce(
    local_linear_pointwise_error, op=MPI.MAX
)
linear_layer_sums = linear_mesh.comm.allreduce(linear_area_data.sum(axis=0))
linear_facet_sums = np.array(
    [
        fd.assemble(fd.Constant(1) * fd.ds_b(domain=linear_mesh)),
        fd.assemble(fd.Constant(1) * fd.dS_h(domain=linear_mesh)),
        fd.assemble(fd.Constant(1) * fd.ds_t(domain=linear_mesh)),
    ]
)
linear_sum_error = np.max(
    np.abs(linear_layer_sums - linear_facet_sums) / linear_facet_sums
)

if mesh.comm.rank == 0:
    np.savetxt(
        "layer_sums.dat",
        np.column_stack(
            (
                radii,
                layer_sums,
                np.full(level_count, pointwise_error),
                np.full(level_count, linear_pointwise_error),
                np.full(level_count, linear_sum_error),
                np.full(level_count, mesh.comm.size),
            )
        ),
    )
