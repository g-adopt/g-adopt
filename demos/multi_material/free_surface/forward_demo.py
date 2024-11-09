import numpy as np
import shapely as sl
from mpi4py import MPI

from gadopt import *


def cosine_curve(x, amplitude, wavelength, vertical_shift):
    return amplitude * np.cos(2 * np.pi / wavelength * x) + vertical_shift


nx, ny = 128, 32
lx, ly = 3e6, 7e5

mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W, W])
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

z = Function(Z)
u, p, eta = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
z.subfunctions[2].rename("Free surface")
psi = Function(K, name="Level set")

slab = sl.Polygon(
    [
        (1e6, ly),
        (1e6, 5e5),
        (1.1e6, 5e5),
        (1.1e6, 6e5),
        (lx + 1e5, 6e5),
        (lx + 1e5, ly),
        (1e6, ly),
    ]
)
free_surface = sl.LineString([(0, ly), (lx, ly)])

sl.prepare(slab)
sl.prepare(free_surface)

material_interface = sl.union(free_surface, slab.boundary)

node_relation_to_interface = [
    (slab.contains(sl.Point(x, y)), material_interface.distance(sl.Point(x, y)))
    for x, y in node_coordinates(psi)
]

signed_dist_to_interface = Function(K)
signed_dist_to_interface.dat.data[:] = [
    dist if is_inside else -dist for is_inside, dist in node_relation_to_interface
]

local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

psi.interpolate((1 + tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

mu_slab = 1e23
mu_mantle = 1e21
mu = material_field(psi, [mu_mantle, mu_slab], interface="geometric")

rho_slab = 3300
rho_mantle = 3200
rho_material = material_field(psi, [rho_mantle, rho_slab], interface="arithmetic")

approximation = Approximation(
    "BA",
    dimensional=True,
    parameters={"g": 9.81, "mu": mu, "rho": rho_mantle, "rho_material": rho_material},
)

delta_t = Function(R).assign(1e11)
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6)

stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"free_surface": {"eta_index": 0, "rho_ext": 0}},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(z, approximation, bcs=stokes_bcs, timestep_fs=delta_t)
stokes_solver.solve()

level_set_solver = LevelSetSolver(psi, u, delta_t, eSSPRKs10p3, epsilon)

time_now, time_end = 0, 25e6 * 365.25 * 8.64e4

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
