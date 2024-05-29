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

    RaB.interpolate(RaB_ufl) if dimensionless else density.interpolate(
        dens_diff + ref_dens
    )
    viscosity.interpolate(viscosity_ufl)
    if "Trim_2023" in Simulation.name:
        Simulation.internal_heating_rate(int_heat_rate_ufl, time_now)
    else:
        int_heat_rate.interpolate(int_heat_rate_ufl)

    output_file.write(
        time_output,
        velocity,
        pressure,
        temperature,
        *level_set,
        *level_set_grad_proj,
        RaB,
        density,
        viscosity,
        int_heat_rate,
    )


# Import Simulation class
parser = ArgumentParser()
parser.add_argument("benchmark")
args = parser.parse_args()
Simulation = import_module(args.benchmark).Simulation

if Simulation.restart_from_checkpoint:  # Restore mesh and key functions
    with fd.CheckpointFile(
        f"{Simulation.name}/checkpoint_{Simulation.restart_from_checkpoint}.h5".lower(),
        "r",
    ) as h5_check:
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
    if "Trim_2023" in Simulation.name:
        epsilon = fd.Constant(1 / 2 / Simulation.k)
    else:  # Empirical calibration that seems to be robust
        local_min_mesh_size = mesh.cell_sizes.dat.data.min()
        epsilon = fd.Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

    time_now = time_output.dat.data[0]
else:  # Initialise mesh and key functions
    # Set up geometry; boundary mapping: {1: left, 2: right, 3: bottom, 4: top}
    try:
        mesh_path = Path(Simulation.mesh_file)
        if not mesh_path.exists():
            if MPI.COMM_WORLD.rank == 0:
                if mesh_path.with_suffix(".geo").exists():
                    run(["gmsh", "-2", str(mesh_path.with_suffix(".geo"))])
                else:
                    Simulation.generate_mesh()
        mesh = fd.Mesh(str(mesh_path))
    except AttributeError:
        mesh = fd.RectangleMesh(
            *Simulation.mesh_elements,
            *[sum(z) for z in zip(Simulation.domain_origin, Simulation.domain_dims)],
            *Simulation.domain_origin,
            quadrilateral=True,
        )

    # Set up Stokes function spaces corresponding to the mixed Q2Q1 Taylor-Hood element
    func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
    func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
    func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
    stokes_function = fd.Function(func_space_stokes)

    # Define temperature function space and initialise temperature
    func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
    temperature = fd.Function(func_space_temp, name="Temperature")
    Simulation.initialise_temperature(temperature)

    # Set up function spaces and functions used in the level-set approach
    func_space_ls = fd.FunctionSpace(mesh, "DQ", Simulation.level_set_func_space_deg)
    level_set = [
        fd.Function(func_space_ls, name=f"Level set #{i}")
        for i in range(len(Simulation.materials) - 1)
    ]

    # Thickness of the hyperbolic tangent profile in the conservative level-set approach
    if "Trim_2023" in Simulation.name:
        epsilon = fd.Constant(1 / 2 / Simulation.k)
    else:  # Empirical calibration that seems to be robust
        local_min_mesh_size = mesh.cell_sizes.dat.data.min()
        epsilon = fd.Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

    # Initialise level set
    signed_dist_to_interface = fd.Function(level_set[0].function_space())
    for ls, isd, params in zip(
        level_set, Simulation.initialise_signed_distance, Simulation.isd_params
    ):
        signed_dist_to_interface.dat.data[:] = isd(params, ls)
        ls.interpolate((1 + fd.tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

    time_output = fd.Function(func_space_pres, name="Time")
    time_now = 0
    dump_counter = 0

# Extract velocity and pressure from the Stokes function
velocity_ufl, pressure_ufl = fd.split(stokes_function)  # UFL expressions
velocity, pressure = stokes_function.subfunctions  # Associated Firedrake functions
velocity.rename("Velocity")
pressure.rename("Pressure")

# Set up fields that depend on the material interface
func_space_interp = fd.FunctionSpace(mesh, "CG", Simulation.level_set_func_space_deg)

if "Trim_2023" in Simulation.name:
    ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless = ga.density_RaB(
        Simulation, level_set, func_space_interp, method="arithmetic"
    )
else:
    ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless = ga.density_RaB(
        Simulation, level_set, func_space_interp
    )

viscosity_ufl = ga.field_interface(
    level_set,
    [
        material.viscosity(velocity_ufl, temperature)
        for material in Simulation.materials
    ],
    method="sharp" if "Schmalholz_2011" in Simulation.name else "geometric",
)
viscosity = fd.Function(func_space_interp, name="Viscosity").interpolate(viscosity_ufl)

if "Trim_2023" in Simulation.name:
    int_heat_rate_ufl = fd.Function(
        temperature.function_space(), name="Internal heating rate"
    )
    int_heat_rate = int_heat_rate_ufl
    Simulation.internal_heating_rate(int_heat_rate_ufl, 0)
else:
    int_heat_rate_ufl = ga.field_interface(
        level_set,
        [material.internal_heating_rate() for material in Simulation.materials],
        method="geometric",
    )
    int_heat_rate = fd.Function(
        func_space_interp, name="Internal heating rate"
    ).interpolate(int_heat_rate_ufl)

# Timestep object
timestep = fd.Constant(Simulation.initial_timestep)

# Set up energy and Stokes solvers
approximation = ga.BoussinesqApproximation(
    Simulation.Ra,
    rho=ref_dens,
    alpha=1,
    g=Simulation.g,
    T0=0,
    RaB=RaB_ufl,
    delta_rho=dens_diff,
    kappa=1,
    H=int_heat_rate_ufl,
)
energy_solver = ga.EnergySolver(
    temperature,
    velocity_ufl,
    approximation,
    timestep,
    ga.ImplicitMidpoint,
    bcs=Simulation.temp_bcs,
)
stokes_nullspace = ga.create_stokes_nullspace(
    stokes_function.function_space(), **Simulation.stokes_nullspace_args
)
stokes_solver = ga.StokesSolver(
    stokes_function,
    temperature,
    approximation,
    bcs=Simulation.stokes_bcs,
    mu=viscosity_ufl,
    quad_degree=None,
    solver_parameters=Simulation.stokes_solver_params,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)

# Solve initial Stokes system
stokes_solver.solve()

# Parameters involved in level-set reinitialisation
reini_params = {
    "epsilon": epsilon,
    "tstep": 1e-2,
    "tstep_alg": ga.eSSPRKs3p3,
    "frequency": 5,
    "iterations": 0 if "Trim_2023" in Simulation.name else 1,
}

# Set up level-set solvers
level_set_solver = [
    ga.LevelSetSolver(
        ls, velocity_ufl, timestep, ga.eSSPRKs10p3, Simulation.subcycles, reini_params
    )
    for ls in level_set
]
level_set_grad_proj = [ls_solv.level_set_grad_proj for ls_solv in level_set_solver]

# Time-loop objects
t_adapt = ga.TimestepAdaptor(
    timestep,
    velocity_ufl,
    velocity.function_space(),
    target_cfl=Simulation.subcycles * 0.6,
    maximum_timestep=Simulation.dump_period,
)
output_file = fd.output.VTKFile(
    f"{Simulation.name}/output_{Simulation.restart_from_checkpoint}_check.pvd".lower(),
    target_degree=Simulation.level_set_func_space_deg,
)
checkpoint_file = fd.CheckpointFile(
    f"{Simulation.name}/checkpoint_{Simulation.restart_from_checkpoint + 1}.h5".lower(),
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
diag_vars = {
    "epsilon": epsilon,
    "level_set": level_set,
    "level_set_grad_proj": level_set_grad_proj,
    "density": density,
    "viscosity": viscosity,
    "int_heat_rate": int_heat_rate,
}
geo_diag = ga.GeodynamicalDiagnostics(
    velocity_ufl, pressure_ufl, temperature, bottom_id=3, top_id=4
)

# Function to be coupled with the energy solver
if "Trim_2023" in Simulation.name:
    update_forcings = partial(Simulation.internal_heating_rate, int_heat_rate_ufl)
else:
    update_forcings = None

# Perform the time loop
step = 0
while True:
    # Calculate simulation diagnostics
    Simulation.diagnostics(time_now, geo_diag, diag_vars)

    # Write to output file and increment dump counter
    if time_now >= dump_counter * Simulation.dump_period:
        # Write to checkpoint file
        if dump_counter % Simulation.checkpoint_period == 0:
            write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        write_output(output_file)
        dump_counter += 1

    # Update timestep
    if Simulation.time_end is not None:
        t_adapt.maximum_timestep = min(
            Simulation.dump_period, Simulation.time_end - time_now
        )
    t_adapt.update_timestep()

    # Solve energy system
    energy_solver.solve(t=time_now, update_forcings=update_forcings)

    # Advect each level set
    for ls_solv in level_set_solver:
        ls_solv.solve(step)

    # Solve Stokes system
    stokes_solver.solve()

    # Progress simulation time and increment time-loop step counter
    time_now += float(timestep)
    step += 1

    # Check if simulation has completed
    end_time = Simulation.time_end is not None and time_now >= Simulation.time_end
    steady = Simulation.steady_state_condition(
        velocity, stokes_solver.solution_old.subfunctions[0]
    )
    if end_time or steady:
        # Calculate final simulation diagnostics
        Simulation.diagnostics(time_now, geo_diag, diag_vars)
        # Save post-processing fields and produce graphs
        Simulation.plot_diagnostics()

        # Write final simulation state to checkpoint file
        write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        checkpoint_file.close()
        # Write final simulation state to output file
        write_output(output_file)

        break
