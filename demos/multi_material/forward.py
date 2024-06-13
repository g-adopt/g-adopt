from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from subprocess import run

from mpi4py import MPI

from gadopt import *
from gadopt.inverse import *


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


def write_output(output_file, output_fields):
    """Write output fields to the output file."""
    RaB.interpolate(RaB_ufl) if dimensionless else density.interpolate(
        dens_diff + ref_dens
    )
    viscosity.interpolate(viscosity_ufl)
    int_heat_rate.interpolate(int_heat_rate_ufl)

    output_file.write(*output_fields)


# Import Simulation class
parser = ArgumentParser()
parser.add_argument("benchmark")
args = parser.parse_args()
Simulation = import_module(args.benchmark).Simulation

# Set up geometry; boundary mapping: {1: left, 2: right, 3: bottom, 4: top}
try:
    mesh_path = Path(Simulation.mesh_file)
    if not mesh_path.exists():
        if MPI.COMM_WORLD.rank == 0:
            if mesh_path.with_suffix(".geo").exists():
                run(["gmsh", "-2", str(mesh_path.with_suffix(".geo"))])
            else:
                Simulation.generate_mesh()
    mesh = Mesh(str(mesh_path))
except AttributeError:
    mesh = RectangleMesh(
        *Simulation.mesh_elements,
        *[sum(z) for z in zip(Simulation.domain_origin, Simulation.domain_dims)],
        *Simulation.domain_origin,
        quadrilateral=True,
    )

# Set up Stokes function spaces corresponding to the mixed Q2Q1 Taylor-Hood element
func_space_vel = VectorFunctionSpace(mesh, "CG", 2)
func_space_pres = FunctionSpace(mesh, "CG", 1)
func_space_stokes = MixedFunctionSpace([func_space_vel, func_space_pres])
stokes_function = Function(func_space_stokes)

# Extract velocity and pressure from the Stokes function
velocity, pressure = split(stokes_function)  # UFL expressions
# Associated Firedrake functions
stokes_function.subfunctions[0].rename("Velocity")
stokes_function.subfunctions[1].rename("Pressure")

# Define temperature function space and initialise temperature
func_space_temp = FunctionSpace(mesh, "CG", 2)
temperature = Function(func_space_temp, name="Temperature")
Simulation.initialise_temperature(temperature)

# Set up function spaces and functions used in the level-set approach
func_space_ls = FunctionSpace(mesh, "DQ", Simulation.level_set_func_space_deg)
level_set = [
    Function(func_space_ls, name=f"Level set #{i}")
    for i in range(len(Simulation.materials) - 1)
]

# Initialise level set
local_min_mesh_size = mesh.cell_sizes.dat.data.min()
# Empirical calibration that seems to be robust
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)
signed_dist_to_interface = Function(level_set[0].function_space())
for ls, isd, params in zip(
    level_set, Simulation.initialise_signed_distance, Simulation.isd_params
):
    signed_dist_to_interface.dat.data[:] = isd(params, ls)
    ls.interpolate((1 + tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

# Set up fields that depend on the material interface
func_space_interp = FunctionSpace(mesh, "CG", Simulation.level_set_func_space_deg)
ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless = density_RaB(
    Simulation, level_set, func_space_interp
)

viscosity_ufl = field_interface(
    level_set,
    [material.viscosity(velocity, temperature) for material in Simulation.materials],
    method="sharp" if "Schmalholz_2011" in Simulation.name else "geometric",
)
viscosity = Function(func_space_interp, name="Viscosity").interpolate(viscosity_ufl)

int_heat_rate_ufl = field_interface(
    level_set,
    [material.internal_heating_rate() for material in Simulation.materials],
    method="geometric",
)
int_heat_rate = Function(func_space_interp, name="Internal heating rate")
int_heat_rate.interpolate(int_heat_rate_ufl)

# Time objects
real_func_space = FunctionSpace(mesh, "R", 0)
time_now = Function(real_func_space).assign(0)
time_now_dump = Function(func_space_pres, name="Time")
timestep = Function(real_func_space).assign(Simulation.initial_timestep)
timestep_dump = Function(func_space_pres, name="Time step")

# Set up energy and Stokes solvers
approximation = BoussinesqApproximation(
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
energy_solver = EnergySolver(
    temperature,
    velocity,
    approximation,
    timestep,
    ImplicitMidpoint,
    bcs=Simulation.temp_bcs,
)
stokes_nullspace = create_stokes_nullspace(
    stokes_function.function_space(), **Simulation.stokes_nullspace_args
)
stokes_solver = StokesSolver(
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

# Set up level-set solvers
level_set_solver = [
    LevelSetSolver(ls, velocity, timestep, eSSPRKs10p3, Simulation.subcycles, epsilon)
    for ls in level_set
]
level_set_solver[0].reini_params["tstep"] *= 10
level_set_grad_proj = [ls_solv.level_set_grad_proj for ls_solv in level_set_solver]

# Time-loop objects
t_adapt = TimestepAdaptor(
    timestep,
    velocity,
    stokes_function.subfunctions[0].function_space(),
    target_cfl=Simulation.subcycles * 0.6,
    maximum_timestep=Simulation.dump_period,
)
output_file = output.VTKFile(
    f"{Simulation.name}/output_{Simulation.restart_from_checkpoint}_check.pvd".lower(),
    target_degree=Simulation.level_set_func_space_deg,
)
checkpoint_file = CheckpointFile(
    f"{Simulation.name}/checkpoint_{Simulation.restart_from_checkpoint + 1}.h5".lower(),
    mode="w",
)

# Fields to include in outputs and checkpoints
output_fields = (
    time_now_dump,
    timestep_dump,
    *stokes_function.subfunctions,
    temperature,
    *level_set,
    *level_set_grad_proj,
    RaB,
    density,
    viscosity,
    int_heat_rate,
)
checkpoint_fields = {
    "Time": time_now_dump,
    "Time step": timestep_dump,
    "Stokes": stokes_function,
    "Temperature": temperature,
    "Level set": level_set,
}

update_forcings = None

# Perform the time loop
step = 0
dump_counter = 0
while True:
    # Write to output file and increment dump counter
    if float(time_now) >= dump_counter * Simulation.dump_period:
        time_now_dump.assign(time_now)
        timestep_dump.assign(timestep)
        # Write to checkpoint file
        if dump_counter % Simulation.checkpoint_period == 0:
            write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        write_output(output_file, output_fields)
        dump_counter += 1

    # Update timestep
    if Simulation.time_end is not None:
        t_adapt.maximum_timestep = min(
            Simulation.dump_period, Simulation.time_end - float(time_now)
        )
    t_adapt.update_timestep()

    # Solve energy system
    energy_solver.solve(t=float(time_now), update_forcings=update_forcings)

    # Advect each level set
    for ls_solv in level_set_solver:
        ls_solv.solve(step)

    # Solve Stokes system
    stokes_solver.solve()

    # Progress simulation time and increment time-loop step counter
    time_now += timestep
    step += 1

    # Check if simulation has completed
    end_time = (
        Simulation.time_end is not None and float(time_now) >= Simulation.time_end / 2
    )
    steady = Simulation.steady_state_condition(stokes_solver)
    if end_time or steady:
        time_now_dump.assign(time_now)
        timestep_dump.assign(timestep)
        # Write final simulation state to checkpoint file
        write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        checkpoint_file.close()
        # Write final simulation state to output file
        write_output(output_file, output_fields)

        break

objective = assemble(inner(level_set[0], level_set[0]) * dx)
log(f"\n\n{objective}\n\n")

reduced_functional = ReducedFunctional(objective, [*[Control(ls) for ls in level_set]])
log(f"\n\n{reduced_functional([*level_set])}\n\n")

perturbation = [
    Function(func_space_ls).interpolate(0.5),
]
log(taylor_test(reduced_functional, [*level_set], perturbation))
