from gadopt import *

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

interface_coords = [
    (lx, ly),
    (1e6, ly),
    (1e6, 5e5),
    (1.1e6, 5e5),
    (1.1e6, 6e5),
    (lx, 6e5),
]
boundary_coords = [(lx, ly)]

signed_distance_array = signed_distance(
    psi,
    interface_coordinates=interface_coords,
    interface_geometry="Polygon",
    boundary_coordinates=boundary_coords,
)
epsilon = interface_thickness(psi)
psi.dat.data[:] = conservative_level_set(signed_distance_array, epsilon)

mu = material_field(psi, [mu_mantle := 1e21, mu_slab := 1e23], interface="geometric")
rho_material = material_field(
    psi, [rho_mantle := 3200, rho_slab := 3300], interface="sharp"
)

approximation = Approximation(
    "BA",
    dimensional=True,
    parameters={"g": 9.81, "mu": mu, "rho": rho_mantle, "rho_material": rho_material},
)

delta_t = Function(R).assign(1e11)
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6)

stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"free_surface": {"eta_index": 2, "rho_ext": 0}},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}

stokes_solver = StokesSolver(
    z, approximation, coupled_tstep=delta_t, theta=0.5, bcs=stokes_bcs
)
stokes_solver.solve()

adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)

time_now, time_end = 0, 25e6 * 365.25 * 8.64e4

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

    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")
