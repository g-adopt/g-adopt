from numpy import linspace

from gadopt import *

nx, ny = 64, 64
lx, ly = 0.9142, 1

mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W])
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

stokes = Function(Z, name="Stokes")
stokes.subfunctions[0].rename("Velocity")
stokes.subfunctions[1].rename("Pressure")
u, p = split(stokes)
psi = Function(K, name="Level set")

interface_coords_x = linspace(0, lx, 1000)
callable_args = (
    interface_deflection := 0.02,
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

time_now, time_end = 0, 200
delta_t = Function(R).assign(1.0)
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6)

stokes_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    boundary.bottom: {"u": 0},
    boundary.top: {"u": 0},
    boundary.left: {"ux": 0},
    boundary.right: {"ux": 0},
}

stokes_solver = StokesSolver(
    stokes,
    approximation,
    bcs=stokes_bcs,
    nullspace={"nullspace": stokes_nullspace, "transpose_nullspace": stokes_nullspace},
)
stokes_solver.solve()

adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)

output_file = VTKFile("forward_output.pvd")
output_file.write(*stokes.subfunctions, psi, time=time_now)

step = 0
while True:
    t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve()
    stokes_solver.solve()

    time_now += float(delta_t)
    step += 1

    output_file.write(*stokes.subfunctions, psi, time=time_now)

    if time_now >= time_end:
        break

with CheckpointFile("forward_checkpoint.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(stokes, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")
