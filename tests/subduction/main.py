import firedrake as fd
from mpi4py import MPI

import parameters as prms
from simulation import initialise, time_loop
from utility import adapt_mesh, generate_mesh, write_output


match prms.mesh_generation:  # Mesh generation
    case "firedrake":
        mesh = fd.RectangleMesh(
            *prms.mesh_elements, *prms.domain_dims, quadrilateral=False
        )
    case "gmsh":
        if MPI.COMM_WORLD.rank == 0:
            generate_mesh(prms.domain_dims, prms.mesh_layers)

        mesh = fd.Mesh("mesh.msh")

# Time-stepping objects
time_now = fd.Constant(0.0)  # Initial time
time_step = fd.Constant(prms.time_step)  # Initial time step
step = 0  # A counter to keep track of the simulation time-loop iterations

for _ in range(prms.initial_adapt_calls):
    # Initialise solutions for mesh adaptivity
    stokes, T, psi, _ = initialise(mesh, time_step)

    # Fields involved in initial mesh adaptivity
    mesh_fields = {
        stokes: {"add_to_metric": [True, False], "scaling": [1e-3, None]},
        T: {"add_to_metric": True, "scaling": 1e0},
        psi: {"add_to_metric": True, "scaling": 1e0},
    }
    if prms.free_surface:
        mesh_fields[stokes]["add_to_metric"].append(False)
        mesh_fields[stokes]["scaling"].append(None)

    # Adapt original mesh based on initial fields
    mesh, mesh_fields = adapt_mesh(mesh, mesh_fields, initial=True)

# Initialise solutions on the adapted mesh
stokes, T, psi, field_expr = initialise(mesh, time_step)

# Write initial output
output_file = fd.VTKFile("output_adaptive.pvd", adaptive=True)
write_output(
    output_file,
    float(time_now) * prms.time_scale / prms.myr_to_seconds,
    *stokes.subfunctions,
    T,
    psi,
    field_expressions=field_expr,
)

# Fields involved in mesh adaptivity loop
mesh_fields = {
    stokes: {"add_to_metric": [True, False], "scaling": [1e-1, None]},
    T: {"add_to_metric": True, "scaling": 1e0},
    psi: {"add_to_metric": True, "scaling": 1e0},
}
if prms.free_surface:
    mesh_fields[stokes]["add_to_metric"].append(False)
    mesh_fields[stokes]["scaling"].append(None)

while True:  # Mesh adaptivity loop
    mesh_fields, step = time_loop(
        mesh, mesh_fields, time_now, time_step, step, output_file
    )

    if float(time_now) >= prms.time_end:  # Exit loop after reaching target time
        break

    mesh, mesh_fields = adapt_mesh(mesh, mesh_fields)  # Adapt mesh based on new fields
