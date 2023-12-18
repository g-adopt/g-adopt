import argparse
import importlib

import firedrake as fd

import gadopt as ga


def write_output(dump_counter, checkpoint_fields=None):
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
level_set_func_space_deg = Simulation.level_set_func_space_deg
func_space_ls = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
level_set = [
    fd.Function(func_space_ls, name=f"Level set #{i}")
    for i in range(len(Simulation.materials) - 1)
]

# Thickness of the hyperbolic tangent profile in the conservative level-set approach
if Simulation.name == "Trim_2023":
    epsilon = fd.Constant(1 / 2 / Simulation.k)
else:
    epsilon = fd.Constant(
        min(
            dim / elem
            for dim, elem in zip(Simulation.domain_dimensions, Simulation.mesh_elements)
        )
        / 4
    )  # Empirical calibration that seems to be robust

# Initialise level set
signed_dist_to_interface = fd.Function(func_space_ls)
for ls, isd, params in zip(
    level_set, Simulation.initialise_signed_distance, Simulation.isd_params
):
    signed_dist_to_interface.dat.data[:] = isd(params, ls)
    ls.interpolate((1 + fd.tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

# Set up fields that depend on the material interface
func_space_interp = fd.FunctionSpace(mesh, "CG", level_set_func_space_deg)
density = fd.Function(func_space_interp, name="Density")
RaB = fd.Function(func_space_interp, name="RaB")
# Identify if the equations are written in dimensional form or not and define relevant
# variables accordingly
B_is_None = all(material.B() is None for material in Simulation.materials)
RaB_is_None = all(material.RaB() is None for material in Simulation.materials)
if B_is_None and RaB_is_None:
    RaB_ufl = fd.Constant(1)
    ref_dens = fd.Constant(Simulation.reference_material.density())
    dens_diff = ga.sharp_interface(
        level_set.copy(),
        [material.density() - ref_dens for material in Simulation.materials],
        method="arithmetic",
    )
    density.interpolate(dens_diff + ref_dens)
elif RaB_is_None:
    ref_dens = fd.Constant(1)
    dens_diff = fd.Constant(1)
    RaB_ufl = ga.diffuse_interface(
        level_set.copy(),
        [Simulation.Ra * material.B() for material in Simulation.materials],
        method="arithmetic",
    )
    RaB.interpolate(RaB_ufl)
elif B_is_None:
    ref_dens = fd.Constant(1)
    dens_diff = fd.Constant(1)
    RaB_ufl = ga.sharp_interface(
        level_set.copy(),
        [material.RaB() for material in Simulation.materials],
        method="arithmetic",
    )
    RaB.interpolate(RaB_ufl)
else:
    raise ValueError("Providing B and RaB is redundant.")

viscosity_ufl = ga.diffuse_interface(
    level_set.copy(),
    [material.viscosity(velocity_ufl) for material in Simulation.materials],
    method="geometric",
)
viscosity = fd.Function(func_space_interp, name="Viscosity").interpolate(viscosity_ufl)

if Simulation.name == "Trim_2023":
    int_heat_rate_ufl = fd.Function(func_space_temp, name="Internal heating rate")
    int_heat_rate = int_heat_rate_ufl
    Simulation.internal_heating_rate(int_heat_rate_ufl, 0)
else:
    int_heat_rate_ufl = ga.diffuse_interface(
        level_set.copy(),
        [material.internal_heating_rate() for material in Simulation.materials],
        method="geometric",
    )
    int_heat_rate = fd.Function(
        func_space_interp, name="Internal heating rate"
    ).interpolate(int_heat_rate_ufl)

dt = fd.Constant(Simulation.dt)

# Set up energy and Stokes solvers
approximation = ga.BoussinesqApproximation(
    Simulation.Ra,
    rho=ref_dens,
    g=Simulation.g,
    alpha=1,
    kappa=1,
    RaB=RaB_ufl,
    delta_rho=dens_diff,
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
stokes_nullspace = ga.create_stokes_nullspace(func_space_stokes)
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

# Parameters involved in level-set reinitialisation
reini_params = {
    "epsilon": epsilon,
    "tstep": 1.5e-2,
    "tstep_alg": ga.eSSPRKs3p3,
    "frequency": 1,
    "iterations": 1,
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
    dt, velocity_ufl, func_space_vel, target_cfl=Simulation.subcycles * 0.65
)
time_output = fd.Function(func_space_pres, name="Time")
time_now, step, dump_counter = 0, 0, 0
output_file = fd.File(
    f"{Simulation.name}/output.pvd".lower(), target_degree=level_set_func_space_deg
)

checkpoint_fields = {"Level set": level_set[0], "Temperature": temperature}

diagnostic_fields = {
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

    for ls_solv in level_set_solver:
        ls_solv.solve(step)

    Simulation.diagnostics(time_now, diagnostic_fields)

    if Simulation.name == "Trim_2023":
        Simulation.internal_heating_rate(int_heat_rate_ufl, time_now)

    # Increment time-loop step counter
    step += 1
# Write final simulation state to output file
dump_counter = write_output(dump_counter, checkpoint_fields)

# Save post-processing fields and produce graphs
Simulation.save_and_plot()
