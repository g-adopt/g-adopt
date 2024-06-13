from argparse import ArgumentParser
from importlib import import_module

from mpi4py import MPI

from gadopt import *
from gadopt.inverse import *


def callback():
    optimised_ls = ls_control[0].block_variable.checkpoint
    optimisation_file.write(optimised_ls)

    initial_misfit = assemble((optimised_ls - ls_ini_forw[0]) ** 2 * dx) / domain_area
    final_misfit = assemble((level_set[0] - ls_final_forw[0]) ** 2 * dx) / domain_area

    log(f"Initial misfit: {initial_misfit}\nFinal misfit: {final_misfit}")


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

with CheckpointFile(
    f"{Simulation.name}/checkpoint_{Simulation.restart_from_checkpoint}.h5".lower(),
    "r",
) as h5_check:
    mesh = h5_check.load_mesh("firedrake_default")

    dump_counter = h5_check.get_timestepping_history(mesh, "Time")["index"][-1]

    time_now_dump = h5_check.load_function(mesh, "Time", idx=dump_counter)
    timestep_dump = h5_check.load_function(mesh, "Time step", idx=dump_counter)
    stokes_function = h5_check.load_function(mesh, "Stokes", idx=dump_counter)
    temperature = h5_check.load_function(mesh, "Temperature", idx=dump_counter)

    ls_ini_forw = []
    ls_final_forw = []
    level_set = []
    ls_control = []
    i = 0
    while True:
        try:
            ls_i_ini_forw = h5_check.load_function(mesh, f"Level set #{i}", idx=0)
            ls_i_final_forw = h5_check.load_function(
                mesh, f"Level set #{i}", idx=dump_counter
            )

            ls_ini_forw.append(ls_i_ini_forw)
            ls_final_forw.append(ls_i_final_forw)
            level_set.append(ls_i_final_forw.copy(deepcopy=True))
            ls_control.append(ls_i_final_forw.copy(deepcopy=True))

            i += 1
        except RuntimeError:
            break

    level_set[0].project(ls_control[0])

# Extract velocity and pressure from the Stokes function
velocity, pressure = split(stokes_function)  # UFL expressions
# Associated Firedrake functions
stokes_function.subfunctions[0].rename("Velocity")
stokes_function.subfunctions[1].rename("Pressure")

# Thickness of the hyperbolic tangent profile in the conservative level-set approach
local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

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
time_now = Function(real_func_space).interpolate(time_now_dump)
timestep = Function(real_func_space).interpolate(timestep_dump)

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
level_set_solver[0].reini_params["tstep"] *= 10
level_set_grad_proj = [ls_solv.level_set_grad_proj for ls_solv in level_set_solver]

# Time-step adaptor
t_adapt = TimestepAdaptor(
    timestep,
    velocity,
    stokes_function.subfunctions[0].function_space(),
    target_cfl=Simulation.subcycles * 0.6,
    maximum_timestep=Simulation.dump_period,
)

# File objects
output_file = output.VTKFile(
    f"{Simulation.name}/output_{Simulation.restart_from_checkpoint}_check.pvd".lower(),
    target_degree=Simulation.level_set_func_space_deg,
)
checkpoint_file = CheckpointFile(
    f"{Simulation.name}/checkpoint_{Simulation.restart_from_checkpoint + 1}.h5".lower(),
    mode="w",
)
optimisation_file = output.VTKFile(
    f"{Simulation.name}/optimised_control.pvd".lower(),
    target_degree=Simulation.level_set_func_space_deg,
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
        Simulation.time_end is not None and float(time_now) >= Simulation.time_end
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

domain_area = assemble(1 * dx(domain=mesh))
objective = assemble((level_set[0] - ls_final_forw[0]) ** 2 * dx) / domain_area

reduced_functional = ReducedFunctional(objective, Control(ls_control[0]))
reduced_functional.derivative()

ls_lower_bound = Function(ls_control[0].function_space()).assign(0)
ls_upper_bound = Function(ls_control[0].function_space()).assign(1)

minimisation_problem = MinimizationProblem(
    reduced_functional, bounds=(ls_lower_bound, ls_upper_bound)
)

optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
optimiser.add_callback(callback)
optimiser.run()

optimisation_file.write(*optimiser.rol_solver.rolvector.dat)

# iter  value          gnorm          snorm          delta          #fval     #grad     #hess     #proj     tr_flag   iterCG    flagCG
# 0     6.723550e-02   1.037193e-06   ---            1.000000e+00   1         1         0         2         ---       ---       ---
# 1     6.723550e-02   1.037193e-06   1.037193e-06   1.000000e+01   2         2         9         11        0         1         0
# 2     6.722513e-02   1.037112e-06   1.000000e+01   1.000000e+02   3         3         37        32        0         10        1
# 3     6.712146e-02   1.036312e-06   1.000000e+02   1.000000e+03   4         4         43        39        0         2         3
# 4     6.608915e-02   1.028313e-06   1.000000e+03   1.000000e+04   5         5         49        46        0         2         3
# 5     5.620603e-02   9.483124e-07   1.000000e+04   1.000000e+05   6         6         55        53        0         2         3
# 6     1.374918e-03   1.483075e-07   1.000000e+05   1.000000e+06   7         7         63        61        0         3         3
# 7     5.148654e-07   1.091189e-22   1.850668e+04   1.000000e+07   8         8         67        66        0         1         0
# 8     5.148654e-07   0.000000e+00   3.227889e-11   1.000000e+08   9         9         98        89        0         10        1
