from argparse import ArgumentParser
from importlib import import_module

from mpi4py import MPI

from gadopt import *
from gadopt.inverse import *


def callback():
    initial_misfit = assemble(
        (ls_control[0].block_variable.checkpoint.restore() - ls_ini_forw[0]) ** 2 * dx
    )
    final_misfit = assemble(
        (level_set[0].block_variable.checkpoint.restore() - ls_final_forw[0]) ** 2 * dx
    )

    log(f"Initial misfit; {initial_misfit}\nFinal misfit: {final_misfit}")


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
    int_heat_rate.interpolate(int_heat_rate_ufl)

    output_file.write(
        time_output,
        *stokes_function.subfunctions,
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

tape = get_working_tape()
tape.clear_tape()

with CheckpointFile(
    f"{Simulation.name}/checkpoint_{Simulation.restart_from_checkpoint}.h5".lower(),
    "r",
) as h5_check:
    mesh = h5_check.load_mesh("firedrake_default")

    dump_counter = h5_check.get_timestepping_history(mesh, "Time")["index"][-1]
    time_output = h5_check.load_function(mesh, "Time", idx=dump_counter)
    stokes_function = h5_check.load_function(mesh, "Stokes", idx=dump_counter)
    temperature = h5_check.load_function(mesh, "Temperature", idx=dump_counter)

    func_space_control = FunctionSpace(mesh, "CG", 1)
    ls_ini_forw = []
    ls_final_forw = []
    ls_control = []
    level_set = []
    i = 0
    while True:
        try:
            ls_ini_forw.append(h5_check.load_function(mesh, f"Level set #{i}", idx=0))
            ls_final_forw.append(
                h5_check.load_function(mesh, f"Level set #{i}", idx=dump_counter)
            )
            ls_control.append(
                Function(func_space_control, name=f"Level set control #{i}")
            )
            ls_control[i].project(
                h5_check.load_function(mesh, f"Level set #{i}", idx=dump_counter)
            )
            level_set.append(
                Function(ls_final_forw[i].function_space(), name=f"Level set #{i}")
            )
            level_set[i].project(ls_control[i], bcs=None)

            i += 1
        except RuntimeError:
            break

# Thickness of the hyperbolic tangent profile in the conservative level-set approach
local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

time_now = time_output.dat.data[0]

# Extract velocity and pressure from the Stokes function
velocity, pressure = split(stokes_function)  # UFL expressions
# Associated Firedrake functions
stokes_function.subfunctions[0].rename("Velocity")
stokes_function.subfunctions[1].rename("Pressure")

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
viscosity = Function(func_space_interp, name="Viscosity")
viscosity.interpolate(viscosity_ufl)

int_heat_rate_ufl = field_interface(
    level_set,
    [material.internal_heating_rate() for material in Simulation.materials],
    method="geometric",
)
int_heat_rate = Function(func_space_interp, name="Internal heating rate")
int_heat_rate.interpolate(int_heat_rate_ufl)

# Timestep object
real_func_space = FunctionSpace(mesh, "R", 0)
timestep = Function(real_func_space)
timestep.assign(Simulation.initial_timestep)

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

# Set up level-set solvers
level_set_solver = [
    LevelSetSolver(ls, velocity, timestep, eSSPRKs10p3, Simulation.subcycles, epsilon)
    for ls in level_set
]
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

# Fields to include in checkpoints
checkpoint_fields = {
    "Time": time_output,
    "Stokes": stokes_function,
    "Temperature": temperature,
    "Level set": level_set,
}

update_forcings = None

# Perform the time loop
step = 0
while True:
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
    steady = Simulation.steady_state_condition(stokes_solver)
    if end_time or steady:
        # Write final simulation state to checkpoint file
        write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        checkpoint_file.close()
        # Write final simulation state to output file
        write_output(output_file)

        break

objective = assemble((level_set[0] - ls_final_forw[0]) ** 2 * dx)

reduced_functional = ReducedFunctional(objective, Control(ls_control[0]))

pause_annotation()

ls_lower_bound = Function(ls_control[0].function_space())
ls_lower_bound.assign(0)
ls_upper_bound = Function(ls_control[0].function_space())
ls_upper_bound.assign(1)

minimisation_problem = MinimizationProblem(
    reduced_functional, bounds=(ls_lower_bound, ls_upper_bound)
)

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir="optimisation_checkpoint",
)
optimiser.add_callback(callback)
optimiser.run()

solution = optimiser.rol_solver.rolvector.dat[0]
solution.rename("solution")
solution_file = output.VTKFile(
    f"{Simulation.name}/solution.pvd".lower(),
    target_degree=Simulation.level_set_func_space_deg,
)
solution_file.write(solution)
