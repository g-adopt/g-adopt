import gc

from animate import RiemannianMetric, adapt

from gadopt import *


def run_interval(mesh, old_mesh_fields, time_now, step, output_counter):
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure space (scalar)
    Z = MixedFunctionSpace([V, W])  # Stokes space (mixed)
    Q = FunctionSpace(mesh, "DG", 2)  # Temperature space (scalar, discontinuous)
    K = FunctionSpace(mesh, "DG", 2)  # Level-set space (scalar, discontinuous)

    stokes = Function(Z)  # A field over the mixed function space Z
    u, p = stokes.subfunctions
    u.rename("Velocity")  # Firedrake function for velocity
    p.rename("Pressure")  # Firedrake function for pressure
    T = Function(Q, name="Temperature")  # Firedrake function for temperature
    psi = Function(K, name="Level set")  # Firedrake function for level set

    mesh_fields = {"u": u, "p": p, "T": T, "psi": psi}
    for field_name, field in mesh_fields.items():
        field.interpolate(old_mesh_fields[field_name])

    # Compositional Rayleigh number, defined based on each material value and location
    RaB = material_field(psi, [RaB_reference, RaB_dense], interface="arithmetic")
    approximation = BoussinesqApproximation(Ra, RaB=RaB)

    boundary = get_boundary_ids(mesh)  # Object holding references to mesh boundary IDs
    temp_bcs = {boundary.bottom: {"T": 1.0}, boundary.top: {"T": 0.0}}
    energy_solver = EnergySolver(
        T, u, approximation, time_step, ImplicitMidpoint, bcs=temp_bcs
    )

    stokes_bcs = {
        boundary.left: {"ux": 0.0},
        boundary.right: {"ux": 0.0},
        boundary.bottom: {"uy": 0.0},
        boundary.top: {"uy": 0.0},
    }
    stokes_nullspace = create_stokes_nullspace(Z)
    stokes_solver = StokesSolver(
        stokes,
        approximation,
        T,
        bcs=stokes_bcs,
        nullspace=stokes_nullspace,
        transpose_nullspace=stokes_nullspace,
    )
    if step == 0:
        stokes_solver.solve()
        output_file.write(*stokes.subfunctions, T, psi, time=time_now)
        output_counter += 1

    epsilon = interface_thickness(K, min_cell_edge_length=True)
    adv_kwargs = {"u": u, "timestep": time_step}
    reini_kwargs = {"epsilon": epsilon}
    level_set_solver = LevelSetSolver(
        psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs
    )

    gd = GeodynamicalDiagnostics(stokes, T, boundary.bottom, boundary.top)
    material_area = interface_coord_y * domain_dims[0]
    entrainment_height = 0.2  # Height above which entrainment diagnostic is calculated

    # Current level-set advection requires a CFL condition that should not exceed 0.6.
    t_adapt = TimestepAdaptor(
        time_step, u, V, target_cfl=0.6, maximum_timestep=output_frequency
    )

    for _ in range(interval_steps):
        # Update timestep
        if time_end - time_now < output_frequency:
            t_adapt.maximum_timestep = time_end - time_now
        t_adapt.update_timestep()

        level_set_solver.solve()  # Advect and reinitialise level set
        energy_solver.solve()  # Solve energy system
        stokes_solver.solve()  # Solve Stokes sytem

        # Increment iteration count and time
        step += 1
        time_now += float(time_step)

        # Calculate proportion of material entrained above a given height
        entrainment = material_entrainment(
            psi,
            material_size=material_area,
            entrainment_height=entrainment_height,
            side=1,
            direction="above",
        )
        # Log diagnostics
        log_file.log_str(
            f"{step} {time_now} {float(time_step)} {gd.u_rms()} {entrainment}"
        )

        # Write output
        if time_now >= output_counter * output_frequency - 1e-16:
            output_file.write(*stokes.subfunctions, T, psi, time=time_now)
            output_counter += 1

        # Check if simulation has completed
        if time_now >= time_end:
            log_file.close()  # Close logging file

            # Checkpoint solution fields to disk
            with CheckpointFile("final_state.h5", "w") as final_checkpoint:
                final_checkpoint.save_mesh(mesh)
                final_checkpoint.save_function(T, name="Temperature")
                final_checkpoint.save_function(stokes, name="Stokes")
                final_checkpoint.save_function(psi, name="Level set")

            log("Reached end of simulation -- exiting time-step loop")
            break

    return mesh_fields, time_now, step, output_counter


mesh_elements = (200, 100)  # Number of cells in x and y directions
domain_dims = (2.0, 1.0)  # Domain dimensions in x and y directions
mesh = RectangleMesh(*mesh_elements, *domain_dims, quadrilateral=False)
mesh.cartesian = True

K = FunctionSpace(mesh, "DG", 2)  # Level-set function space (scalar, discontinuous)
psi = Function(K, name="Level set")  # Firedrake function for level set

time_now = 0.0  # Initial time
time_end = 0.03  # Will be changed to 0.05 once mesh adaptivity is available
time_step = Constant(1e-6)  # Initial time step
step = 0  # A counter to keep track of the simulation time loop iterations
interval_steps = 10  # Simulation time loop iterations to run before adapting the mesh
output_frequency = 1e-4  # Frequency (based on simulation time) at which to output
output_counter = 0  # A counter to keep track of outputting

Ra = 3e5  # Thermal Rayleigh number
RaB_dense = 4.5e5  # Compositional Rayleigh number in the dense material
RaB_reference = 0.0  # Compositional Rayleigh number in the reference material

u = as_vector((0.0, 0.0))  # Initial velocity state to be interpolated
p = 0.0  # Initial pressure state to be interpolated
# Initial temperature state to be interpolated
x, y = SpatialCoordinate(mesh)  # Extract UFL representation of spatial coordinates
u0 = (
    domain_dims[0] ** (7.0 / 3.0)
    / (1.0 + domain_dims[0] ** 4.0) ** (2.0 / 3.0)
    * (Ra / 2.0 / sqrt(pi)) ** (2.0 / 3.0)
)
v0 = u0
Q_ic = 2.0 * sqrt(domain_dims[0] / pi / u0)
Tu = erf((1.0 - y) / 2.0 * sqrt(u0 / x)) / 2.0
Tl = 1.0 - 1.0 / 2.0 * erf(y / 2.0 * sqrt(u0 / (domain_dims[0] - x)))
Tr = 1.0 / 2.0 + Q_ic / 2.0 / sqrt(pi) * sqrt(v0 / (y + 1.0)) * exp(
    -(x**2) * v0 / (4.0 * y + 4.0)
)
Ts = 1.0 / 2.0 - Q_ic / 2.0 / sqrt(pi) * sqrt(v0 / (2.0 - y)) * exp(
    -((domain_dims[0] - x) ** 2.0) * v0 / (8.0 - 4.0 * y)
)
T = max_value(min_value(Tu + Tl + Tr + Ts - 3.0 / 2.0, 1.0), 0.0)
# Initial level-set state to be interpolated
interface_coord_y = 0.025
signed_distance = interface_coord_y - y
epsilon = interface_thickness(K, min_cell_edge_length=True)
assign_level_set_values(psi, epsilon, signed_distance)

# Fields requiring interpolation after adapting the mesh
mesh_fields = {"u": u, "p": p, "T": T, "psi": psi}
# Fields contributing to metric calculation
metric_fields = {"T": "Temperature", "psi": "Level set"}

metric_parameters = {
    "dm_plex_metric_target_complexity": 30_000,  # Target number of cells
    "dm_plex_metric_p": np.inf,  # Estimated interpolation error from infinity norm
    "dm_plex_metric_gradation_factor": 1.5,  # Variation in edge length across cells
    "dm_plex_metric_a_max": 10.0,  # Maximum cell aspect ratio
    "dm_plex_metric_h_min": 1e-3,  # Minimum edge length
    "dm_plex_metric_h_max": 1.0,  # Maximum edge length
}

output_file = VTKFile("output.pvd", adaptive=True)
log_file = ParameterLog("params.log", mesh)
log_file.log_str("step time dt u_rms entrainment")

while True:
    mesh_fields, time_now, step, output_counter = run_interval(
        mesh, mesh_fields, time_now, step, output_counter
    )

    # Collect and clean objects associated with the old mesh
    PETSc.garbage_cleanup(mesh.comm)
    gc.collect()

    if time_now >= time_end:  # Exit loop after reaching target time
        break

    M = TensorFunctionSpace(mesh, "CG", 1)
    metrics = []
    for field_key, field_name in metric_fields.items():
        metric = RiemannianMetric(M)  # Firedrake function for a metric on a field
        metric.rename(f"Metric ({field_name})")  # Specific name from current field
        metric.set_parameters(metric_parameters)  # Set metric parameters
        metric.compute_hessian(mesh_fields[field_key], method="L2")  # Hessian of field
        metric.enforce_spd()  # Ensure metric boundedness (symmetric positive-definite)
        metrics.append(metric)

    metric = metrics[0].copy(deepcopy=True)  # Firedrake function for overall metric
    metric.rename("Metric (overall)")  # Specific name
    metric.intersect(*metrics[1:])  # Edge length from minimum across all field metrics
    metric.normalise()  # Rescale metric to achieve the desired target complexity

    mesh = adapt(mesh, metric)  # Generate new mesh based on overall metric
    mesh.cartesian = True  # Tag new mesh as Cartesian to inform other G-ADOPT objects
