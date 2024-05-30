from argparse import ArgumentParser
from importlib import import_module

import firedrake as fd
from mpi4py import MPI

import gadopt as ga
import gadopt.inverse as gai


def callback():
    initial_misfit = fd.assemble(
        (ls_control[0].block_variable.checkpoint.restore() - ls_ini_forw[0]) ** 2
        * fd.dx
    )
    final_misfit = fd.assemble(
        (level_set[0].block_variable.checkpoint.restore() - ls_final_forw[0]) ** 2
        * fd.dx
    )

    ga.log(f"Initial misfit; {initial_misfit}\nFinal misfit: {final_misfit}")


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

with fd.CheckpointFile(
    f"{Simulation.name}/checkpoint_{Simulation.restart_from_checkpoint}.h5".lower(),
    "r",
) as h5_check:
    mesh = h5_check.load_mesh("firedrake_default")

    dump_counter = h5_check.get_timestepping_history(mesh, "Time")["index"][-1]
    time_output = h5_check.load_function(mesh, "Time", idx=dump_counter)
    stokes_function = h5_check.load_function(mesh, "Stokes", idx=dump_counter)
    temperature = h5_check.load_function(mesh, "Temperature", idx=dump_counter)

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
                h5_check.load_function(mesh, f"Level set #{i}", idx=dump_counter)
            )

            level_set.append(fd.Function(ls_control[i].function_space()))
            level_set[i].rename(f"Level set #{i}")
            level_set[i].project(ls_control[i], bcs=None)

            i += 1
        except RuntimeError:
            break

# Thickness of the hyperbolic tangent profile in the conservative level-set approach
local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = fd.Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

time_now = time_output.dat.data[0]

# Extract velocity and pressure from the Stokes function
velocity_ufl, pressure_ufl = fd.split(stokes_function)  # UFL expressions
velocity, pressure = stokes_function.subfunctions  # Associated Firedrake functions
velocity.rename("Velocity")
pressure.rename("Pressure")

# Set up fields that depend on the material interface
func_space_interp = fd.FunctionSpace(mesh, "CG", Simulation.level_set_func_space_deg)
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

int_heat_rate_ufl = ga.field_interface(
    level_set,
    [material.internal_heating_rate() for material in Simulation.materials],
    method="geometric",
)
int_heat_rate = fd.Function(
    func_space_interp, name="Internal heating rate"
).interpolate(int_heat_rate_ufl)

# Timestep object
real_func_space = fd.FunctionSpace(mesh, "R", 0)
timestep = fd.Function(real_func_space).assign(Simulation.initial_timestep)

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
    "iterations": 1,
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
    steady = Simulation.steady_state_condition(
        velocity, stokes_solver.solution_old.subfunctions[0]
    )
    if end_time or steady:
        # Write final simulation state to checkpoint file
        write_checkpoint(checkpoint_file, checkpoint_fields, dump_counter)
        checkpoint_file.close()
        # Write final simulation state to output file
        write_output(output_file)

        break

objective = fd.assemble((level_set[0] - ls_final_forw[0]) ** 2 * fd.dx)

reduced_functional = gai.ReducedFunctional(objective, gai.Control(ls_control[0]))

ls_lower_bound = fd.Function(ls_control[0].function_space()).assign(0)
ls_upper_bound = fd.Function(ls_control[0].function_space()).assign(1)

minimisation_problem = gai.MinimizationProblem(
    reduced_functional, bounds=(ls_lower_bound, ls_upper_bound)
)

optimiser = gai.LinMoreOptimiser(minimisation_problem, gai.minimisation_parameters)
optimiser.add_callback(callback)
optimiser.run()

solution = optimiser.rol_solver.rolvector.dat[0]
solution.rename("solution")
solution_file = fd.output.VTKFile(
    f"{Simulation.name}/solution.pvd".lower(),
    target_degree=Simulation.level_set_func_space_deg,
)
solution_file.write(solution)
