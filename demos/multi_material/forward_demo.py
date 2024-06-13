import numpy as np
import shapely as sl

from gadopt import *
from gadopt.inverse import *


def cosine_curve(x, amplitude, wavelength, vertical_shift):
    return amplitude * np.cos(2 * np.pi / wavelength * x) + vertical_shift


nx, ny = 40, 40
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

interface_deflection = 0.02
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

time_now = 0
delta_t = Function(R).assign(1)
output_frequency = 10
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)

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

step = 0
output_counter = 0
time_end = 1000
while True:
    if time_now >= output_counter * output_frequency:
        output_file.write(*z.subfunctions, T, psi)
        output_counter += 1

    if time_end is not None:
        t_adapt.maximum_timestep = min(output_frequency, time_end - time_now)
    t_adapt.update_timestep()
    time_now += float(delta_t)
    step += 1

    stokes_solver.solve()

    level_set_solver.solve(step)

    if time_now >= time_end:
        log("Reached end of simulation -- exiting time-step loop")
        break

with CheckpointFile("forward_checkpoint.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")

objective = assemble(inner(psi, psi) * dx)
log(f"\n\n{objective}\n\n")

reduced_functional = ReducedFunctional(objective, Control(psi))
log(f"\n\n{reduced_functional(psi)}\n\n")

perturbation = [Function(K).interpolate(0.5)]
log(taylor_test(reduced_functional, psi, perturbation))
