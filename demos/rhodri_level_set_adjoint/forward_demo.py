import numpy as np
import shapely as sl
from gadopt import *

def cosine_curve(x, amplitude, wavelength, vertical_shift):
    return amplitude * np.cos(2 * np.pi / wavelength * x) + vertical_shift

nx, ny = 80, 80
lx, ly = 0.9142, 1

mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W])
Q = FunctionSpace(mesh, "CG", 2)
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

z = Function(Z)
u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
T = Function(Q, name="Temperature")
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

buoyant_material = Material(RaB=-1)
dense_material = Material(RaB=0)
materials = [buoyant_material, dense_material]

Ra = 0
RaB = field_interface(
    [psi], [material.RaB for material in materials], method="arithmetic"
)

approximation = BoussinesqApproximation(Ra, RaB=RaB)

delta_t = Function(R).assign(0.1)

Z_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

output_file = VTKFile("forward_output.pvd")

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)

subcycles = 1
level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, subcycles, epsilon)
level_set_solver.reini_params["iterations"] = 0

timesteps = 200

for timestep in range(0, timesteps):

    output_file.write(*z.subfunctions, T, psi)
    
    stokes_solver.solve()

    level_set_solver.solve(timestep)

    
with CheckpointFile("forward_checkpoint.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")
