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
    checkpoint_file.save_mesh(mesh)
    for field_name, field in checkpoint_fields.items():
        if isinstance(field, list):
            for i, field_element in enumerate(field):
                checkpoint_file.save_function(
                    field_element, name=f"{field_name} #{i}", idx=dump_counter
                )
        else:
            checkpoint_file.save_function(field, name=field_name, idx=dump_counter)
    checkpoint_file.set_attr("/", "time", time_now)


def write_output(output_file):
    """Write output fields to the output file."""
    if simulation.dimensional:
        density.interpolate(rho)
    else:
        compo_rayleigh.interpolate(Ra_c)
    viscosity.interpolate(mu)
    if args.benchmark == "trim_2023":
        simulation.internal_heating_rate(Q, time_now)

    output_file.write(
        *stokes_function.subfunctions,
        temperature,
        *level_set,
        *level_set_grad_proj,
        *output_fields,
        time=time_now,
    )


# Import simulation class
parser = ArgumentParser()
parser.add_argument("benchmark")
args = parser.parse_args()
simulation = import_module(f"benchmarks.{args.benchmark}.simulation")

# Set benchmark paths
benchmark_path = Path(f"benchmarks/{args.benchmark}")
mesh_path = benchmark_path / "mesh"
output_path = benchmark_path / "outputs"

if simulation.checkpoint_restart:  # Restore mesh and key functions
    with fd.CheckpointFile(
        output_path / f"checkpoint_{simulation.restart_from_checkpoint}.h5", "r"
    ) as h5_check:
        mesh = h5_check.load_mesh("firedrake_default")
        dump_counter = h5_check.get_timestepping_history(mesh, "Time")["index"][-1]
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

        time_now = h5_check.get_attr("/", "time")

    # Thickness of the hyperbolic tangent profile in the conservative level-set approach
    if args.benchmark == "trim_2023":
        epsilon = fd.Constant(1 / 2 / simulation.k)
    else:  # Empirical calibration that seems to be robust
        local_min_mesh_size = mesh.cell_sizes.dat.data.min()
        epsilon = fd.Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)
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

    # Set up Stokes function spaces corresponding to the mixed Q2Q1 Taylor-Hood element
    func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
    func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    # Define temperature function space and initialise temperature
    func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
    temperature = fd.Function(func_space_temp, name="Temperature")
    if hasattr(simulation, "initialise_temperature"):
        simulation.initialise_temperature(temperature)

    # Set up function spaces and functions used in the level-set approach
    func_space_ls = fd.FunctionSpace(mesh, "DQ", 2)
    level_set = [
        fd.Function(func_space_ls, name=f"Level set #{i}")
        for i in range(len(simulation.materials) - 1)
    ]

    # Thickness of the hyperbolic tangent profile in the conservative level-set approach
    if args.benchmark == "trim_2023":
        epsilon = fd.Constant(1 / 2 / simulation.k)
    else:  # Empirical calibration; adjust if necessary
        local_min_mesh_size = mesh.cell_sizes.dat.data.min()
        epsilon = fd.Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

    # Initialise level set
    signed_dist_to_interface = fd.Function(level_set[0].function_space())
    for ls, isd, params in zip(
        level_set, simulation.initialise_signed_distance, simulation.isd_params
    ):
        signed_dist_to_interface.dat.data[:] = isd(params, ls)
        ls.interpolate((1 + fd.tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

    time_now = 0
    dump_counter = 0

# Annotate mesh as cartesian
mesh.cartesian = True

# Extract velocity and pressure from the Stokes function
velocity, pressure = fd.split(stokes_function)  # UFL expressions
# Associated Firedrake functions
stokes_function.subfunctions[0].rename("Velocity")
stokes_function.subfunctions[1].rename("Pressure")

# Continuous function space for material field output
func_space_output = fd.FunctionSpace(mesh, "CG", 2)
output_fields = []
# Set up material fields and the equation system
approximation_parameters = {}
if simulation.dimensional:
    rho = ga.material_field(
        level_set,
        [material.rho for material in simulation.materials],
        interface="sharp",
    )
    density = fd.Function(func_space_output, name="Density").interpolate(rho)
    output_fields.append(density)
    approximation_parameters["rho"] = simulation.materials[0].rho
    approximation_parameters["rho_diff"] = rho - simulation.materials[0].rho
else:
    Ra_c = ga.material_field(
        level_set,
        [material.Ra_c for material in simulation.materials],
        interface="arithmetic" if args.benchmark == "trim_2023" else "sharp",
    )
    compo_rayleigh = fd.Function(func_space_output, name="Ra_c").interpolate(Ra_c)
    output_fields.append(compo_rayleigh)
    approximation_parameters["Ra_c"] = Ra_c

mu = ga.material_field(
    level_set,
    [material.mu(velocity, temperature) for material in simulation.materials],
    interface="sharp" if args.benchmark == "schmalholz_2011" else "geometric",
)
viscosity = fd.Function(func_space_output, name="Viscosity").interpolate(mu)
output_fields.append(viscosity)
approximation_parameters["mu"] = mu

if args.benchmark == "trim_2023":
    Q = fd.Function(temperature.function_space(), name="Internal heating rate")
    output_fields.append(Q)
    approximation_parameters["Q"] = Q

if simulation.dimensional:
    approximation_parameters["g"] = simulation.g
elif "thermal" in simulation.buoyancy_terms:
    approximation_parameters["Ra"] = simulation.Ra

approximation = ga.EquationSystem(
    approximation="BA",
    dimensional=simulation.dimensional,
    parameters=approximation_parameters,
    buoyancy_terms=simulation.buoyancy_terms,
)

# Timestep object
real_func_space = fd.FunctionSpace(mesh, "R", 0)
timestep = fd.Function(real_func_space).assign(simulation.initial_timestep)

# Set up energy and Stokes solvers
energy_solver = []
if hasattr(simulation, "initialise_temperature"):
    energy_solver.append(
        ga.EnergySolver(
            approximation,
            temperature,
            velocity,
            timestep,
            ga.ImplicitMidpoint,
            bcs=simulation.temp_bcs if hasattr(simulation, "temp_bcs") else None,
        )
    )
stokes_nullspace = ga.create_stokes_nullspace(stokes_function.function_space())
stokes_solver = ga.StokesSolver(
    approximation,
    stokes_function,
    temperature,
    bcs=simulation.stokes_bcs,
    quad_degree=None,
    nullspace={"nullspace": stokes_nullspace, "transpose_nullspace": stokes_nullspace},
)

# Solve initial Stokes system
stokes_solver.solve()
if hasattr(simulation, "steady_state_threshold"):
    velocity_old = stokes_function.subfunctions[0].copy(deepcopy=True)

# Set up level-set solvers
level_set_solver = [
    ga.LevelSetSolver(ls, velocity, timestep, ga.eSSPRKs10p3, epsilon)
    for ls in level_set
]
if args.benchmark == "trim_2023":
    for ls_solver in level_set_solver:
        ls_solver.reini_params["iterations"] = 0
level_set_grad_proj = [ls_solv.ls_grad_proj for ls_solv in level_set_solver]

# Time-loop objects
t_adapt = ga.TimestepAdaptor(
    timestep,
    velocity,
    stokes_function.subfunctions[0].function_space(),
    target_cfl=0.6,
    maximum_timestep=simulation.dump_period,
)
output_file = fd.VTKFile(
    output_path / f"output_{simulation.checkpoint_restart}_check.pvd",
    target_continuity=fd.ufl.sobolevspace.SobolevSpace("H1"),
)
checkpoint_file = fd.CheckpointFile(
    f"{output_path}/checkpoint_{simulation.checkpoint_restart + 1}.h5", mode="w"
)

# Fields to include in checkpoints
checkpoint_fields = {
    "Stokes": stokes_function,
    "Level set": level_set,
    "Temperature": temperature,
}

# Objects used to calculate simulation diagnostics
diag_vars = {"epsilon": epsilon, "level_set": level_set, "viscosity": viscosity}
geo_diag = ga.GeodynamicalDiagnostics(
    stokes_function, temperature, bottom_id=3, top_id=4
)

# Function to be coupled with the energy solver
if args.benchmark == "trim_2023":
    update_forcings = partial(simulation.internal_heating_rate, Q)
else:
    update_forcings = None

# Perform the time loop
step = 1
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
    if has_end_time:
        t_adapt.maximum_timestep = min(
            simulation.dump_period, simulation.time_end - time_now
        )
    t_adapt.update_timestep()

    # Solve energy system
    if energy_solver:
        energy_solver[0].solve(time_now, update_forcings)

    # Advect each level set
    for ls_solv in level_set_solver:
        ls_solv.solve(step)

    # Solve Stokes system
    stokes_solver.solve()

    # Progress simulation time and increment time-loop step counter
    time_now += float(timestep)
    step += 1

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
