from argparse import ArgumentParser
from functools import partial
from importlib import import_module
from pathlib import Path
from subprocess import run

import firedrake as fd
from mpi4py import MPI

import gadopt as ga


def write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter):
    """Write checkpointed fields to the checkpoint file."""
    time_output.assign(time_now)

    checkpoint_file.save_mesh(mesh)
    for field_name, field in checkpoint_fields.items():
        if isinstance(field, list):
            for i, field_element in enumerate(field):
                checkpoint_file.save_function(
                    field_element, name=f"{field_name} #{i}", idx=dump_counter
                )
        else:
            checkpoint_file.save_function(field, name=field_name, idx=dump_counter)


def write_output(output_file):
    """Write output fields to the output file."""
    time_output.assign(time_now)

    if simulation.dimensional:
        density.interpolate(rho_material)
    else:
        compo_rayleigh.interpolate(RaB)
    viscosity.interpolate(mu)
    if benchmark == "trim_2023":
        simulation.internal_heating_rate(H, time_now)

    output_file.write(
        time_output,
        *stokes_function.subfunctions,
        temperature,
        *level_set,
        *level_set_grad,
        *output_fields,
    )


# Import simulation module
parser = ArgumentParser()
parser.add_argument("benchmark", help="Path to the benchmark directory")
args = parser.parse_args()

benchmark = args.benchmark.split("/")[1]
simulation = import_module(f"{args.benchmark.rstrip('/').replace('/', '.')}.simulation")

# Set benchmark paths
benchmark_path = Path(args.benchmark)
mesh_path = benchmark_path / "mesh"
output_path = benchmark_path / "outputs"

if simulation.checkpoint_restart:  # Restore mesh and key functions
    old_check_file = f"checkpoint_{simulation.checkpoint_restart}_{simulation.tag}.h5"
    with fd.CheckpointFile(str(output_path / old_check_file), "r") as h5_check:
        mesh = h5_check.load_mesh("firedrake_default")

        dump_counter = h5_check.get_timestepping_history(mesh, "Time")["index"][-1]
        time_output = h5_check.load_function(mesh, "Time", idx=dump_counter)
        stokes_function = h5_check.load_function(mesh, "Stokes", idx=dump_counter)
        temperature = h5_check.load_function(mesh, "Temperature", idx=dump_counter)

        i = 0
        level_set = []
        while True:
            try:
                level_set.append(
                    h5_check.load_function(mesh, f"Level set #{i}", idx=dump_counter)
                )
                i += 1
            except RuntimeError:
                break

    # Thickness of the hyperbolic tangent profile in the conservative level-set approach
    if benchmark == "trim_2023":
        epsilon = 1 / 2 / simulation.k
    else:
        epsilon = ga.interface_thickness(
            level_set[0].function_space(), min_cell_edge_length=True
        )
    if benchmark == "schmalholz_2011":
        epsilon.interpolate(
            mesh.comm.allreduce(mesh.cell_sizes.dat.data.min(), MPI.MIN) / 4
        )

    time_now = time_output.dat.data[0]
else:  # Initialise mesh and key functions
    match simulation.mesh_gen:  # Generate mesh
        case "gmsh":
            mesh_file = mesh_path / "mesh.msh"
            if not mesh_file.exists():
                if MPI.COMM_WORLD.rank == 0:
                    if mesh_file.with_suffix(".geo").exists():
                        run(["gmsh", "-2", str(mesh_file.with_suffix(".geo"))])
                    else:
                        mesh_path.mkdir(exist_ok=True)
                        simulation.generate_mesh(mesh_path)
            mesh = fd.Mesh(str(mesh_file))
        case "firedrake":
            mesh = fd.RectangleMesh(
                *simulation.mesh_elements, *simulation.domain_dims, quadrilateral=True
            )
        case _:
            raise ValueError("'mesh_gen' must be 'firedrake' or 'gmsh'")

    # Set up Stokes function spaces corresponding to the mixed Q2Q1 Taylor-Hood element
    func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
    func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    # Define temperature function space and initialise temperature
    func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
    temperature = fd.Function(func_space_temp, name="Temperature")
    simulation.initialise_temperature(temperature)

    # Set up function spaces and functions used in the level-set approach
    func_space_ls = fd.FunctionSpace(mesh, "DQ", simulation.level_set_func_space_deg)
    level_set = [
        fd.Function(func_space_ls, name=f"Level set #{i}")
        for i in range(len(simulation.materials) - 1)
    ]

    # Thickness of the hyperbolic tangent profile in the conservative level-set approach
    if benchmark == "trim_2023":
        epsilon = 1 / 2 / simulation.k
    else:
        epsilon = ga.interface_thickness(func_space_ls, min_cell_edge_length=True)
    if benchmark == "schmalholz_2011":
        epsilon.interpolate(
            mesh.comm.allreduce(mesh.cell_sizes.dat.data.min(), MPI.MIN) / 4
        )

    # Initialise level set
    for ls, kwargs in zip(level_set, simulation.signed_distance_kwargs_list):
        ga.assign_level_set_values(ls, epsilon, **kwargs)

    time_output = fd.Function(func_space_pres, name="Time")
    time_now = 0
    dump_counter = 0

# Whether we start from checkpoint or not, we annotate our mesh as cartesian
mesh.cartesian = True

# Extract velocity and pressure from the Stokes function
velocity, pressure = fd.split(stokes_function)  # UFL expressions
# Associated Firedrake functions
stokes_function.subfunctions[0].rename("Velocity")
stokes_function.subfunctions[1].rename("Pressure")
# Copy velocity function for steady-state convergence check
if hasattr(simulation, "steady_state_threshold"):
    velocity_old = stokes_function.subfunctions[0].copy(deepcopy=True)

# Continuous function space for material field output
func_space_output = fd.FunctionSpace(mesh, "CG", simulation.level_set_func_space_deg)
output_fields = []
# Set up material fields and the equation system
approximation_parameters = {}
if simulation.dimensional:
    rho_material = ga.material_field(
        level_set,
        [material.rho for material in simulation.materials],
        interface="sharp",
    )
    density = fd.Function(func_space_output, name="Density")
    output_fields.append(density)
    approximation_parameters["rho"] = simulation.materials[0].rho
    approximation_parameters["delta_rho"] = rho_material - simulation.materials[0].rho
    approximation_parameters["RaB"] = 1
else:
    if simulation.materials[0].RaB is None:
        RaB_material = [simulation.Ra * material.B for material in simulation.materials]
    else:
        RaB_material = [material.RaB for material in simulation.materials]
    RaB = ga.material_field(
        level_set,
        RaB_material,
        interface="arithmetic" if benchmark == "trim_2023" else "sharp",
    )
    compo_rayleigh = fd.Function(func_space_output, name="RaB")
    output_fields.append(compo_rayleigh)
    approximation_parameters["RaB"] = RaB

mu = ga.material_field(
    level_set,
    [material.mu(velocity, temperature) for material in simulation.materials],
    interface="sharp" if benchmark == "schmalholz_2011" else "geometric",
)
viscosity = fd.Function(func_space_output, name="Viscosity")
output_fields.append(viscosity)
approximation_parameters["mu"] = mu

if benchmark == "trim_2023":
    H = fd.Function(temperature.function_space(), name="Internal heating rate")
    output_fields.append(H)
    approximation_parameters["H"] = H

if simulation.dimensional:
    approximation_parameters["g"] = simulation.g
    approximation_parameters["T0"] = 0

Ra = getattr(simulation, "Ra", 0)
approximation = ga.BoussinesqApproximation(Ra, **approximation_parameters)

# Timestep object
real_func_space = fd.FunctionSpace(mesh, "R", 0)
timestep = fd.Function(real_func_space).assign(simulation.initial_timestep)

# Set up energy and Stokes solvers
energy_solver = ga.EnergySolver(
    temperature,
    velocity,
    approximation,
    timestep,
    ga.ImplicitMidpoint,
    bcs=simulation.temp_bcs,
)
stokes_nullspace = ga.create_stokes_nullspace(
    stokes_function.function_space(), **simulation.stokes_nullspace_args
)
stokes_solver = ga.StokesSolver(
    stokes_function,
    temperature,
    approximation,
    bcs=simulation.stokes_bcs,
    quad_degree=None,
    solver_parameters=simulation.stokes_solver_params,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)

# Solve initial Stokes system
stokes_solver.solve()

# Set up level-set solvers
adv_kwargs = {"u": velocity, "timestep": timestep}
reini_kwargs = {"epsilon": epsilon, "timestep": 1e-2, "frequency": 5}
if benchmark == "tosi_2015":
    # Speed up simulation by avoiding frequent reinitialisation
    reini_kwargs["frequency"] = 10
level_set_solver = [
    ga.LevelSetSolver(ls, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
    for ls in level_set
]
level_set_grad = [ls_solv.solution_grad for ls_solv in level_set_solver]

# Time-loop objects
t_adapt = ga.TimestepAdaptor(
    timestep,
    velocity,
    stokes_function.subfunctions[0].function_space(),
    target_cfl=0.6,
    maximum_timestep=simulation.dump_period,
)
output_file = fd.VTKFile(
    output_path / f"output_{simulation.checkpoint_restart}_{simulation.tag}.pvd"
)
checkpoint_file = fd.CheckpointFile(
    f"{output_path}/checkpoint_{simulation.checkpoint_restart + 1}_{simulation.tag}.h5",
    mode="w",
)

# Fields to include in checkpoints
checkpoint_fields = {
    "Time": time_output,
    "Stokes": stokes_function,
    "Temperature": temperature,
    "Level set": level_set,
}

# Objects used to calculate simulation diagnostics
diag_vars = {"epsilon": epsilon, "level_set": level_set, "viscosity": viscosity}
geo_diag = ga.GeodynamicalDiagnostics(
    stokes_function, temperature, bottom_id=3, top_id=4
)

if benchmark == "trim_2023":
    # Omit level-set reinitialisation
    disable_reinitialisation = True
    # Function to be coupled with the energy solver
    update_forcings = partial(simulation.internal_heating_rate, H)
else:
    disable_reinitialisation = False
    update_forcings = None

# Perform the time loop
has_end_time = hasattr(simulation, "time_end")
while True:
    # Calculate simulation diagnostics
    simulation.diagnostics(time_now, geo_diag, diag_vars, benchmark_path)

    # Write to output file and increment dump counter
    if time_now >= dump_counter * simulation.dump_period:
        # Write to checkpoint file
        if dump_counter % simulation.checkpoint_period == 0:
            write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        write_output(output_file)
        dump_counter += 1

    # Update timestep
    if has_end_time and simulation.time_end - time_now < simulation.dump_period:
        t_adapt.maximum_timestep = simulation.time_end - time_now
    t_adapt.update_timestep()

    # Solve energy system
    energy_solver.solve(update_forcings=update_forcings, t=time_now)

    # Advect each level set
    for ls_solv in level_set_solver:
        ls_solv.solve(disable_reinitialisation=disable_reinitialisation)

    # Solve Stokes system
    stokes_solver.solve()

    # Progress simulation time and increment time-loop step counter
    time_now += float(timestep)

    # Check if simulation has completed
    if has_end_time:
        exit_loop = time_now >= simulation.time_end
    else:
        exit_loop = fd.norm(velocity - velocity_old) < simulation.steady_state_threshold
        velocity_old = stokes_function.subfunctions[0].copy(deepcopy=True)

    if exit_loop:
        # Calculate final simulation diagnostics
        simulation.diagnostics(time_now, geo_diag, diag_vars, benchmark_path)
        # Save post-processing fields and produce graphs
        simulation.plot_diagnostics(benchmark_path)

        # Write final simulation state to checkpoint file
        write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        checkpoint_file.close()
        # Write final simulation state to output file
        write_output(output_file)

        break
