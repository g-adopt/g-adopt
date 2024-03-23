import argparse
import importlib
from functools import partial

import firedrake as fd
from mpi4py import MPI

import gadopt as ga


def write_output(dump_counter, checkpoint_fields=None):
    """Write diagnostics to the output file and optionally generate a checkpoint."""
    time_output.assign(time_now)
    RaB.interpolate(RaB_ufl) if dimensionless else density.interpolate(
        dens_diff + ref_dens
    )
    viscosity.interpolate(viscosity_ufl)
    if "Trim_2023" in Simulation.name:
        Simulation.internal_heating_rate(int_heat_rate_ufl, time_now)

    output_file.write(
        time_output,
        *level_set,
        *level_set_grad_proj,
        velocity,
        pressure,
        RaB,
        density,
        viscosity,
        temperature,
        int_heat_rate,
    )

    if checkpoint_fields is not None:
        with ga.CheckpointFile(
            f"{Simulation.name.lower()}/checkpoint_{dump_counter}.h5", mode="w"
        ) as checkpoint_file:
            checkpoint_file.save_mesh(mesh)
            for field_name, field in checkpoint_fields.items():
                checkpoint_file.save_function(field, name=field_name, idx=0)

    return dump_counter + 1


parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
args = parser.parse_args()
Simulation = importlib.import_module(args.benchmark).Simulation

# Set up geometry
# Boundary mapping: {1: left, 2: right, 3: bottom, 4: top}
try:
    mesh = fd.Mesh(Simulation.mesh_file)
except AttributeError:
    mesh = fd.RectangleMesh(
        *Simulation.mesh_elements,
        *[sum(z) for z in zip(Simulation.domain_origin, Simulation.domain_dimensions)],
        *Simulation.domain_origin,
        quadrilateral=True,
    )

# Set up Stokes function spaces corresponding to the Q2Q1 Taylor-Hood element
func_space_vel = fd.VectorFunctionSpace(mesh, "CG", 2)
func_space_pres = fd.FunctionSpace(mesh, "CG", 1)
func_space_stokes = fd.MixedFunctionSpace([func_space_vel, func_space_pres])
# Define Stokes functions on the mixed function space
stokes_function = fd.Function(func_space_stokes)
velocity_ufl, pressure_ufl = fd.split(stokes_function)  # UFL expressions
velocity, pressure = stokes_function.subfunctions  # Associated Firedrake functions
velocity.rename("Velocity")
pressure.rename("Pressure")

# Define temperature function space and initialise temperature
func_space_temp = fd.FunctionSpace(mesh, "CG", 2)
temperature = fd.Function(func_space_temp, name="Temperature")
Simulation.initialise_temperature(temperature)

# Set up function spaces and functions used in the level-set approach
level_set_func_space_deg = Simulation.level_set_func_space_deg
func_space_ls = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
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
signed_dist_to_interface = fd.Function(func_space_ls)
for ls, isd, params in zip(
    level_set, Simulation.initialise_signed_distance, Simulation.isd_params
):
    signed_dist_to_interface.dat.data[:] = isd(params, ls)
    ls.interpolate((1 + fd.tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

# Set up fields that depend on the material interface
func_space_interp = fd.FunctionSpace(mesh, "CG", level_set_func_space_deg)

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
    int_heat_rate_ufl = fd.Function(func_space_temp, name="Internal heating rate")
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
dt = fd.Constant(Simulation.dt)

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
    dt,
    ga.ImplicitMidpoint,
    bcs=Simulation.temp_bcs,
)
stokes_nullspace = ga.create_stokes_nullspace(
    func_space_stokes, **Simulation.stokes_nullspace_args
)
stokes_solver = ga.StokesSolver(
    stokes_function,
    temperature,
    approximation,
    bcs=Simulation.stokes_bcs,
    mu=viscosity_ufl,
    quad_degree=None,
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
        ls, velocity_ufl, dt, ga.eSSPRKs10p3, Simulation.subcycles, reini_params
    )
    for ls in level_set
]
level_set_grad_proj = [ls_solv.level_set_grad_proj for ls_solv in level_set_solver]

# Time-loop objects
t_adapt = ga.TimestepAdaptor(
    dt,
    velocity_ufl,
    func_space_vel,
    target_cfl=Simulation.subcycles * 0.6,
    maximum_timestep=Simulation.dump_period,
)
time_output = fd.Function(func_space_pres, name="Time")
time_now, step, dump_counter = 0, 0, 0
output_file = fd.output.VTKFile(
    f"{Simulation.name}/output.pvd".lower(), target_degree=level_set_func_space_deg
)

# Fields to include in checkpoints
checkpoint_fields = {"Level set": level_set[0], "Temperature": temperature}

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
    velocity_ufl, pressure_ufl, temperature, bottom_id=3, top_id=4, diag_vars=diag_vars
)

# Function to be coupled with the energy solver
if "Trim_2023" in Simulation.name:
    update_forcings = partial(Simulation.internal_heating_rate, int_heat_rate_ufl)
else:
    update_forcings = None

# Perform the time loop
while True:
    if time_now >= dump_counter * Simulation.dump_period:  # Write to output file
        dump_counter = write_output(dump_counter)

    Simulation.diagnostics(time_now, geo_diag)

    # Update time and timestep
    if Simulation.time_end is not None:
        t_adapt.maximum_timestep = min(
            Simulation.dump_period, Simulation.time_end - time_now
        )
    dt.assign(t_adapt.update_timestep())

    # Solve energy system
    energy_solver.solve(t=time_now, update_forcings=update_forcings)

    for ls_solv in level_set_solver:
        ls_solv.solve(step)

    # Solve Stokes system
    stokes_solver.solve()

    # Increment simulation time and time-loop step counter
    time_now += float(dt)
    step += 1

    # Check if simulation has completed
    end_time = Simulation.time_end is not None and time_now >= Simulation.time_end
    steady = Simulation.steady_state_condition(
        velocity, stokes_solver.solution_old.subfunctions[0]
    )
    if end_time or steady:
        break
# Write final simulation state to output file and generate checkpoint
dump_counter = write_output(dump_counter, checkpoint_fields)
# Calculate final simulation diagnostics
Simulation.diagnostics(time_now, geo_diag)
# Save post-processing fields and produce graphs
Simulation.save_and_plot()
