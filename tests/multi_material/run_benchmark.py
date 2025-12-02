from argparse import ArgumentParser
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
    checkpoint_file.set_attr("/", "timestep", float(timestep))


def write_output(output_file):
    """Write output fields to the output file."""
    if simulation.dimensional:
        density.interpolate(rho_material)
    else:
        compo_rayleigh.interpolate(RaB)
    viscosity.interpolate(mu)

    myr_to_seconds = 1e6 * 365.25 * 8.64e4
    output_file.write(
        *stokes_function.subfunctions,
        temperature,
        *level_set,
        *level_set_grad,
        *output_fields,
        time=time_now / (myr_to_seconds if simulation.dimensional else 1.0),
    )


# Import simulation module
parser = ArgumentParser()
parser.add_argument("benchmark", help="Path to the benchmark directory")
parser.add_argument(
    "--without-plot",
    action="store_true",
    help="Speed up simulation by skipping time-loop updates of diagnostic plots",
)
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
        dump_counter = h5_check.get_timestepping_history(mesh, "Stokes")["index"][-1]
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
        func_space_real = fd.FunctionSpace(mesh, "R", 0)
        timestep = fd.Function(func_space_real)
        timestep.assign(h5_check.get_attr("/", "timestep"))

    # Thickness of the hyperbolic tangent profile in the conservative level-set approach
    if benchmark == "trim_2023":
        epsilon = 1.0 / 2.0 / simulation.k
    else:
        epsilon = ga.interface_thickness(
            level_set[0].function_space(), min_cell_edge_length=True
        )
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
    func_space_vel = fd.VectorFunctionSpace(mesh, "Q", 2)
    func_space_pres = fd.FunctionSpace(mesh, "Q", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    # Define temperature function space and initialise temperature
    func_space_temp = fd.FunctionSpace(mesh, "DQ", 2, variant="equispaced")
    temperature = fd.Function(func_space_temp, name="Temperature")
    if hasattr(simulation, "initialise_temperature"):
        simulation.initialise_temperature(temperature)

    # Set up function spaces and functions used in the level-set approach
    func_space_ls = fd.FunctionSpace(
        mesh, "DQ", 1 if benchmark == "schmalholz_2011" else 2
    )  # Stokes solver diverges when using DQ2 for the Schmalholz benchmark
    level_set = [
        fd.Function(func_space_ls, name=f"Level set #{i}")
        for i in range(len(simulation.materials) - 1)
    ]

    # Thickness of the hyperbolic tangent profile in the conservative level-set approach
    if benchmark == "trim_2023":
        epsilon = 1.0 / 2.0 / simulation.k
    else:
        epsilon = ga.interface_thickness(func_space_ls, min_cell_edge_length=True)

    # Initialise level set
    for ls, kwargs in zip(level_set, simulation.signed_distance_kwargs_list):
        ga.assign_level_set_values(ls, epsilon, **kwargs)

    time_now = 0.0
    func_space_real = fd.FunctionSpace(mesh, "R", 0)
    timestep = fd.Function(func_space_real).assign(simulation.initial_timestep)
    dump_counter = 0

# Annotate mesh as Cartesian to inform other G-ADOPT objects
mesh.cartesian = True

# Rename Firedrake functions for velocity and pressure
stokes_function.subfunctions[0].rename("Velocity")
stokes_function.subfunctions[1].rename("Pressure")
# Extract velocity indexed expression from the function defined on the mixed space
velocity = fd.split(stokes_function)[0]
# Copy velocity function for steady-state convergence check
if hasattr(simulation, "steady_state_threshold"):
    velocity_old = stokes_function.subfunctions[0].copy(deepcopy=True)

# Continuous function space for material field output
finite_elem_output = fd.FiniteElement("DQ", fd.quadrilateral, 1, variant="equispaced")
func_space_output = fd.FunctionSpace(mesh, finite_elem_output)
output_fields = []
# Set up material fields and the equation system
approximation_parameters = {}
if simulation.dimensional:
    rho_material = ga.material_field(
        level_set,
        [material.rho for material in simulation.materials],
        interface="arithmetic" if benchmark == "woidt_1978" else "sharp",
    )
    density = fd.Function(func_space_output, name="Density")
    output_fields.append(density)
    approximation_parameters["rho"] = simulation.materials[0].rho
    approximation_parameters["delta_rho"] = rho_material - simulation.materials[0].rho
    approximation_parameters["RaB"] = 1.0
else:
    RaB = ga.material_field(
        level_set,
        [material.RaB for material in simulation.materials],
        interface="arithmetic" if benchmark == "trim_2023" else "sharp",
    )
    compo_rayleigh = fd.Function(func_space_output, name="RaB")
    output_fields.append(compo_rayleigh)
    approximation_parameters["RaB"] = RaB

mu = ga.material_field(
    level_set,
    [material.mu(velocity, temperature) for material in simulation.materials],
    # Sharp viscosity interface required to reproduce Schmalholz benchmark diagnostic
    interface="sharp" if benchmark == "schmalholz_2011" else "geometric",
)
viscosity = fd.Function(func_space_output, name="Viscosity")
output_fields.append(viscosity)
approximation_parameters["mu"] = mu

if benchmark == "trim_2023":
    H = fd.Function(temperature.function_space(), name="Internal heating rate")
    simulation.internal_heating_rate(H, time_now)
    output_fields.append(H)
    approximation_parameters["H"] = H

if simulation.dimensional:
    approximation_parameters["g"] = simulation.g
    approximation_parameters["T0"] = 0.0

Ra = getattr(simulation, "Ra", 0.0)
approximation = ga.BoussinesqApproximation(Ra, **approximation_parameters)

# Set up possible energy solver
energy_solver = None
if hasattr(simulation, "initialise_temperature"):
    energy_solver = ga.EnergySolver(
        temperature,
        velocity,
        approximation,
        timestep,
        ga.ImplicitMidpoint,
        bcs=simulation.temp_bcs,
    )
# Set up Stokes solver
stokes_nullspace = ga.create_stokes_nullspace(stokes_function.function_space())
# Update line search algorithm to ensure convergence
upd_ls_type = {"snes_linesearch_type": "cp"} if benchmark == "schmalholz_2011" else None
stokes_solver = ga.StokesSolver(
    stokes_function,
    approximation,
    temperature,
    bcs=simulation.stokes_bcs,
    solver_parameters_extra=upd_ls_type,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)
stokes_solver.solve()  # Determine initial velocity and pressure fields

# Set up level-set solvers
adv_kwargs = {"u": velocity, "timestep": timestep}
reini_kwargs = {"epsilon": epsilon}
if benchmark == "tosi_2015":  # Avoid expensive frequent reinitialisation
    reini_kwargs["frequency"] = 10
level_set_solver = [
    ga.LevelSetSolver(ls, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
    for ls in level_set
]
level_set_grad = [ls_solver.solution_grad for ls_solver in level_set_solver]

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
    "Stokes": stokes_function,
    "Level set": level_set,
    "Temperature": temperature,
}

# Objects used to calculate simulation diagnostics
diag_vars = {"epsilon": epsilon, "level_set": level_set, "viscosity": viscosity}
geo_diag = ga.GeodynamicalDiagnostics(
    stokes_function, temperature, bottom_id=3, top_id=4
)

# Level-set reinitialisation must be excluded for Trim et al. (2023), as the level-set
# field acts as the composition field, which is purely advected.
disable_reinitialisation = True if benchmark == "trim_2023" else False

# Perform the time loop
has_end_time = hasattr(simulation, "time_end")
while True:
    # Calculate simulation diagnostics
    simulation.diagnostics(time_now, geo_diag, diag_vars, benchmark_path)
    if not args.without_plot:
        simulation.plot_diagnostics(benchmark_path)

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
    if energy_solver is not None:
        energy_solver.solve()

    # Advect each level set
    for ls_solver in level_set_solver:
        ls_solver.solve(disable_reinitialisation=disable_reinitialisation)

    # Solve Stokes system
    stokes_solver.solve()

    # Progress simulation time
    time_now += float(timestep)
    if benchmark == "trim_2023":
        simulation.internal_heating_rate(H, time_now)

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
