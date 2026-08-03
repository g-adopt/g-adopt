import numpy as np
from mpi4py import MPI

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

if mesh.comm.rank == 0:
    np.savetxt(
        "layer_sums.dat",
        np.column_stack(
            (
                radii,
                layer_sums,
                np.full(level_count, pointwise_error),
                np.full(level_count, mesh.comm.size),
            )
        ),
    )
