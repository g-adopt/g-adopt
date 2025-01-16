from numpy import linspace

from gadopt import *

nx, ny = 64, 64
lx, ly = 0.9142, 1

mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W])
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

z = Function(Z)
u, p = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
psi = Function(K, name="Level set")

interface_coords_x = linspace(0, lx, 1000)
callable_args = (
    interface_deflection := 0.1,
    perturbation_wavelength := 2 * lx,
    initial_interface_y := 0.2,
)
signed_distance_array = signed_distance(
    psi,
    interface_geometry="curve",
    interface_callable="cosine",
    interface_args=(interface_coords_x, *callable_args),
)
epsilon = interface_thickness(psi)
psi.dat.data[:] = conservative_level_set(signed_distance_array, epsilon)

Ra_c = material_field(psi, [Ra_c_buoyant := 0, Ra_c_dense := 1], interface="sharp")
approximation = Approximation("BA", dimensional=False, parameters={"Ra_c": Ra_c})

time_now, time_end = 0, 150
delta_t = Function(R).assign(1.0)
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6)

Z_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(
    z,
    approximation,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)
stokes_solver.solve()

adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)

output_file = VTKFile("forward_output.pvd")
output_file.write(*z.subfunctions, psi, time=time_now)

step = 0
while True:
    t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve()
    stokes_solver.solve()

    time_now += float(delta_t)
    step += 1

    output_file.write(*z.subfunctions, psi, time=time_now)

    if time_now >= time_end:
        break

with CheckpointFile("forward_checkpoint.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(psi, name="Level set")
