from numpy import linspace, pi

from gadopt import *

nx, ny = 64, 64
L = 5e5
mesh = SquareMesh(nx, ny, L, quadrilateral=True)
mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W, W])
K = FunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)

z = Function(Z)
u, p, eta = split(z)
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
z.subfunctions[2].rename("Free-surface height")
psi = Function(K, name="Level set")

interface_coords_x = linspace(0, L, nx * 10 + 1)
interface_args = (
    interface_deflection := L / 100,
    perturbation_wavelength := L,
    initial_interface_y := 4e5,
    perturbation_phase := pi,
)
signed_distance_kwargs = curve_interface(
    interface_coords_x, curve="cosine", curve_args=interface_args
)

signed_distance_array = signed_distance(psi, **signed_distance_kwargs)
epsilon = interface_thickness(psi)
psi.dat.data[:] = conservative_level_set(signed_distance_array, epsilon)

rho_material = material_field(
    psi, [rho_buoyant := 3200, rho_dense := 3300], interface="sharp"
)
mu = material_field(psi, [mu_buoyant := 1e20, mu_dense := 1e21], interface="geometric")
approximation = Approximation(
    "BA",
    dimensional=True,
    parameters={"g": 9.81, "mu": mu, "rho": rho_dense, "rho_material": rho_material},
)

myr_to_seconds = 1e6 * 365.25 * 8.64e4
time_now = 0
delta_t = Function(R).assign(0.01 * myr_to_seconds)
output_frequency = 0.1 * myr_to_seconds
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)

Z_nullspace = create_stokes_nullspace(Z)
stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"free_surface": {"eta_index": 2, "rho_ext": 0}},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
stokes_solver = StokesSolver(
    z,
    approximation,
    coupled_tstep=delta_t,
    theta=0.5,
    bcs=stokes_bcs,
    nullspace={"nullspace": Z_nullspace, "transpose_nullspace": Z_nullspace},
)
stokes_solver.solve()

adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)

output_file = VTKFile("output.pvd")
output_file.write(*z.subfunctions, psi, time=time_now / myr_to_seconds)

output_counter = 1
time_end = 6 * myr_to_seconds
while True:
    if time_end - time_now < output_frequency:
        t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve()
    stokes_solver.solve()

    time_now += float(delta_t)

    if time_now >= output_counter * output_frequency:
        output_file.write(*z.subfunctions, psi, time=time_now / myr_to_seconds)
        output_counter += 1

    if time_now >= time_end:
        output_file.write(*z.subfunctions, psi, time=time_now / myr_to_seconds)
        output_counter += 1

        with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
            final_checkpoint.save_mesh(mesh)
            final_checkpoint.save_function(z, name="Stokes")
            final_checkpoint.save_function(psi, name="Level set")

        log("Reached end of simulation -- exiting time-step loop")
        break
