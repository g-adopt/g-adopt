from numpy import linspace

from gadopt import *

mesh_elements = (64, 64)
domain_dims = (0.9142, 1.0)
mesh = RectangleMesh(*mesh_elements, *domain_dims, quadrilateral=True)
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
u = split(stokes)[0]
psi = Function(K, name="Level set")
time_step = Function(R).assign(1.0)

callable_args = (
    curve_parameter := linspace(0.0, domain_dims[0], 1000),
    interface_deflection := 0.02,
    perturbation_wavelength := 2.0 * domain_dims[0],
    interface_coord_y := 0.2,
)
boundary_coordinates = [domain_dims, (0.0, domain_dims[1]), (0.0, interface_coord_y)]

epsilon = interface_thickness(K, min_cell_edge_length=True)
assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="curve",
    interface_callable="cosine",
    interface_args=callable_args,
    boundary_coordinates=boundary_coordinates,
)

RaB = material_field(
    psi, [RaB_buoyant := 0.0, RaB_dense := 1.0], interface="arithmetic"
)
approximation = BoussinesqApproximation(Ra := 0.0, RaB=RaB)


stokes_nullspace = create_stokes_nullspace(Z)

stokes_bcs = {
    boundary.bottom: {"u": 0.0},
    boundary.top: {"u": 0.0},
    boundary.left: {"ux": 0.0},
    boundary.right: {"ux": 0.0},
}

stokes_solver = StokesSolver(
    stokes,
    approximation,
    bcs=stokes_bcs,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)
stokes_solver.solve()

adv_kwargs = {"u": u, "timestep": time_step}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)

t_adapt = TimestepAdaptor(time_step, u, V, target_cfl=0.6)

time_now = 0.0
output_file = VTKFile("output_forward.pvd")
output_file.write(*stokes.subfunctions, psi, time=time_now)

step = 0
time_end = 200.0
while True:
    t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    level_set_solver.solve()
    stokes_solver.solve()

    step += 1
    time_now += float(time_step)

    output_file.write(*stokes.subfunctions, psi, time=time_now)

    if time_now >= time_end:
        break

with CheckpointFile("forward_checkpoint.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(stokes, name="Stokes")
    final_checkpoint.save_function(psi, name="Level set")
