import firedrake as fd
import numpy as np
from benchmarks.gerya_2003 import Simulation
from mpi4py import MPI

from gadopt.approximations import BoussinesqApproximation
from gadopt.energy_solver import EnergySolver
from gadopt.level_set_tools import (
    LevelSetEquation,
    ProjectionSolver,
    ReinitialisationEquation,
    TimeStepperSolver,
)
from gadopt.stokes_integrators import StokesSolver, create_stokes_nullspace
from gadopt.time_stepper import ImplicitMidpoint, eSSPRKs3p3, eSSPRKs10p3
from gadopt.utility import TimestepAdaptor


def recursive_conditional(level_set, material_value):
    ls = level_set.pop()

    if level_set:  # Directly specify material value on only one side of the interface
        return fd.conditional(
            ls < 0.5,
            material_value.pop(),
            recursive_conditional(level_set, material_value),
        )
    else:  # Final level set; specify values for last two materials
        return fd.conditional(ls < 0.5, *reversed(material_value))


def write_output(dump_counter):
    time_output.assign(time_now)
    if B_is_None and RaB_is_None:
        density.interpolate(dens_diff + ref_dens)
    else:
        RaB.interpolate(RaB_ufl)
    viscosity.interpolate(viscosity_ufl)

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

    return dump_counter + 1


# Set up geometry
mesh = fd.RectangleMesh(
    *Simulation.mesh_elements, *Simulation.domain_dimensions, quadrilateral=True
)  # RectangleMesh boundary mapping: {1: left, 2: right, 3: bottom, 4: top}

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
level_set_func_space_deg = 1
func_space_ls = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
func_space_lsgp = fd.VectorFunctionSpace(mesh, "CG", level_set_func_space_deg)
level_set = [
    fd.Function(func_space_ls, name=f"Level set #{i}")
    for i in range(len(Simulation.materials) - 1)
]
level_set_grad_proj = [
    fd.Function(func_space_lsgp, name=f"Level-set gradient (projection) #{i}")
    for i in range(len(level_set))
]

# Thickness of the hyperbolic tangent profile in the conservative level-set approach
epsilon = fd.Constant(
    min(
        dim / elem
        for dim, elem in zip(Simulation.domain_dimensions, Simulation.mesh_elements)
    )
    / 4
)  # Empirical calibration that seems to be robust

# Initialise level set
for ls, isd, params in zip(
    level_set, Simulation.initialise_signed_distance, Simulation.isd_params
):
    signed_dist_to_interface = isd(params, ls)
    ls.dat.data[:] = (
        np.tanh(np.asarray(signed_dist_to_interface) / 2 / float(epsilon)) + 1
    ) / 2

# Set up fields that depend on the material interface
func_space_interp = fd.FunctionSpace(mesh, "CG", level_set_func_space_deg)
density = fd.Function(func_space_interp, name="Density")
RaB = fd.Function(func_space_interp, name="RaB")

B_is_None = all(material.B() is None for material in Simulation.materials)
RaB_is_None = all(material.RaB() is None for material in Simulation.materials)
if B_is_None and RaB_is_None:
    RaB_ufl = fd.Constant(1)
    ref_dens = fd.Constant(Simulation.reference_material.density())
    dens_diff = recursive_conditional(
        level_set.copy(),
        [material.density() - ref_dens for material in Simulation.materials],
    )
    density.interpolate(dens_diff + ref_dens)
elif RaB_is_None:
    ref_dens = fd.Constant(1)
    dens_diff = fd.Constant(1)
    RaB_ufl = recursive_conditional(
        level_set.copy(),
        [Simulation.Ra * material.B() for material in Simulation.materials],
    )
    RaB.interpolate(RaB_ufl)
elif B_is_None:
    ref_dens = fd.Constant(1)
    dens_diff = fd.Constant(1)
    RaB_ufl = recursive_conditional(
        level_set.copy(),
        [material.RaB() for material in Simulation.materials],
    )
    RaB.interpolate(RaB_ufl)
else:
    raise ValueError("Providing B and RaB is redundant.")

viscosity_ufl = recursive_conditional(
    level_set.copy(),
    [material.viscosity(velocity_ufl) for material in Simulation.materials],
)
viscosity = fd.Function(func_space_interp, name="Viscosity").interpolate(viscosity_ufl)

int_heat_rate_ufl = recursive_conditional(
    level_set.copy(),
    [material.internal_heating_rate() for material in Simulation.materials],
)
int_heat_rate = fd.Function(func_space_temp, name="Internal heating rate").interpolate(
    int_heat_rate_ufl
)

dt = fd.Constant(Simulation.dt)

# Set up energy and Stokes solvers
approximation = BoussinesqApproximation(
    Simulation.Ra,
    rho=ref_dens,
    g=Simulation.g,
    alpha=1,
    kappa=1,
    RaB=RaB_ufl,
    delta_rho=dens_diff,
    H=int_heat_rate_ufl,
)
energy_solver = EnergySolver(
    temperature,
    velocity_ufl,
    approximation,
    dt,
    ImplicitMidpoint,
    bcs=Simulation.temp_bcs,
)
stokes_nullspace = create_stokes_nullspace(func_space_stokes)
stokes_solver = StokesSolver(
    stokes_function,
    temperature,
    approximation,
    bcs=Simulation.stokes_bcs,
    mu=viscosity_ufl,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)

# Additional fields required for each level-set solver
level_set_fields = {"velocity": velocity_ufl}
projection_fields = [{"target": ls} for ls in level_set]
reinitialisation_fields = [
    {"level_set_grad": ls_grad_proj, "epsilon": epsilon}
    for ls_grad_proj in level_set_grad_proj
]

# Force projected gradient norm to satisfy the reinitialisation equation
projection_boundary_conditions = [
    fd.DirichletBC(
        func_space_lsgp,
        ls * (1 - ls) / epsilon * fd.Constant((1, 0)),
        "on_boundary",
    )
    for ls in level_set
]

# Set up level-set solvers
level_set_solver = [
    TimeStepperSolver(
        ls, level_set_fields, dt / Simulation.subcycles, eSSPRKs10p3, LevelSetEquation
    )
    for ls in level_set
]
projection_solver = [
    ProjectionSolver(ls_grad_proj, proj_fields, bcs=proj_bcs)
    for ls_grad_proj, proj_fields, proj_bcs in zip(
        level_set_grad_proj, projection_fields, projection_boundary_conditions
    )
]
reinitialisation_solver = [
    TimeStepperSolver(
        ls,
        reini_fields,
        2e-2,  # Trade-off between actual reinitialisation and unwanted interface motion
        eSSPRKs3p3,
        ReinitialisationEquation,
        coupled_solver=proj_solver,
    )
    for ls, reini_fields, proj_solver in zip(
        level_set, reinitialisation_fields, projection_solver
    )
]

# Time-loop objects
t_adapt = TimestepAdaptor(
    dt, velocity_ufl, func_space_vel, target_cfl=Simulation.subcycles * 0.65
)
time_output = fd.Function(func_space_pres, name="Time")
time_now, step, dump_counter = 0, 0, 0
reini_frequency, reini_iterations = 1, 2
output_file = fd.File(
    f"{Simulation.name}/output.pvd".lower(), target_degree=level_set_func_space_deg
)

diag_vars = {
    "level_set": level_set,
    "level_set_grad_proj": level_set_grad_proj,
    "velocity": velocity_ufl,
    "pressure": pressure_ufl,
    "density": density,
    "viscosity": viscosity,
    "temperature": temperature,
    "int_heat_rate": int_heat_rate,
}

# Perform the time loop
while time_now < Simulation.time_end:
    if time_now >= dump_counter * Simulation.dump_period:  # Write to output file
        dump_counter = write_output(dump_counter)

    # if Simulation.name.startswith("Trim_2023"):
    #     update_internal_heating_rate(int_heat_rate, time_now)

    # Update time and timestep
    t_adapt.maximum_timestep = min(
        Simulation.dump_period, Simulation.time_end - time_now
    )
    dt.assign(t_adapt.update_timestep())
    time_now += float(dt)

    # Solve Stokes system
    stokes_solver.solve()

    # Solve energy system
    energy_solver.solve()

    for subcycle in range(Simulation.subcycles):
        # Solve level-set advection
        for ls, ls_solver, reini_solver in zip(
            level_set, level_set_solver, reinitialisation_solver
        ):
            ls_solver.solve()

            if step > reini_frequency:  # Update stored function given previous solve
                reini_solver.ts.solution_old.assign(ls)

            if step % reini_frequency == 0:  # Solve level-set reinitialisation
                for reini_step in range(reini_iterations):
                    reini_solver.solve()

                # Update stored function given previous solve
                ls_solver.ts.solution_old.assign(ls)

    Simulation.diagnostics(time_now, diag_vars)

    # Increment time-loop step counter
    step += 1
# Write final simulation state to output file
dump_counter = write_output(dump_counter)

if MPI.COMM_WORLD.rank == 0:  # Save post-processing fields and produce graphs
    Simulation.save_and_plot()
