import numpy as np
import shapely as sl

from gadopt import *


def cosine_curve(x, amplitude, wavelength, vertical_shift):
    return amplitude * np.cos(2 * np.pi / wavelength * x) + vertical_shift


nx, ny = 80, 80
lx, ly = 0.9142, 1

mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W])
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

z = Function(Z)
u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
psi = Function(K, name="Level set")

interface_deflection = 0.1
interface_wavelength = 2 * lx
material_interface_y = 0.2

isd_params = (interface_deflection, interface_wavelength, material_interface_y)

interface_x = np.linspace(0, lx, 1000)
interface_y = cosine_curve(interface_x, *isd_params)
line_string = sl.LineString([*np.column_stack((interface_x, interface_y))])
sl.prepare(line_string)

node_coords_x, node_coords_y = node_coordinates(psi)
node_relation_to_curve = [
    (
        node_coord_y > cosine_curve(node_coord_x, *isd_params),
        line_string.distance(sl.Point(node_coord_x, node_coord_y)),
    )
    for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y)
]

signed_dist_to_interface = Function(K)
signed_dist_to_interface.dat.data[:] = [
    dist if is_above else -dist for is_above, dist in node_relation_to_curve
]

min_mesh_edge_length = min(lx / nx, ly / ny)
epsilon = Constant(min_mesh_edge_length / 4)

psi.interpolate((1 + tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

Ra_c_buoyant = 0
Ra_c_dense = 1
Ra_c = material_field(psi, [Ra_c_buoyant, Ra_c_dense], interface="arithmetic")

approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

Z_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(
    approximation,
    z,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)
stokes_solver.solve()

delta_t = Function(R).assign(1.0)
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6)

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)

time_now, time_end = 0, 100

output_file = VTKFile("forward_output.pvd")
output_file.write(*z.subfunctions, psi, time=time_now)

step = 0
while True:
    t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve(step, equation="advection")
    stokes_solver.solve()

    time_now += float(delta_t)
    step += 1

    output_file.write(*z.subfunctions, psi, time=time_now)

    if time_now >= time_end:
        break

with CheckpointFile("forward_checkpoint.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)

    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")
