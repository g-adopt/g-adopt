from gadopt import *
from mpi4py import MPI

# Set up geometry:
nx, ny = 40, 40
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)  # Square mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Function to store the solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T0 = Constant(0.091)  # Non-dimensional surface temperature
Di = Constant(0.5)  # Dissipation number.
T.interpolate((1.0 - (T0*exp(Di) - T0)) * ((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1]))))

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(1e5)  # Rayleigh number

# Compressible reference state:
gruneisen = 1.0
rhobar = Function(Q, name="CompRefDensity").interpolate(exp(((1.0 - X[1]) * Di) / gruneisen))
Tbar = Function(Q, name="CompRefTemperature").interpolate(T0 * exp((1.0 - X[1]) * Di) - T0)
# why do we have these as functions?
alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)

approximation = TruncatedAnelasticLiquidApproximation(Ra, Di, rho=rhobar, Tbar=Tbar, alpha=alphabar, cp=cpbar)

time = 0.0
steady_state_tolerance = 1e-9
max_timesteps = 20000
kappa = Constant(1.0)  # Thermal diffusivity

FullT = Function(Q, name="FullTemperature").assign(T+Tbar)


# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Write output files in VTK format:
u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = VTKFile("output.pvd")
ref_file = VTKFile('reference_state.pvd')
dump_period = 100
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4

delta_t = Constant(1e-6)  # Initial time-step
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)


# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)
plog.log_str(
    "timestep time dt maxchange u_rms u_rms_surf ux_max nu_base "
    "nu_top energy avg_t rate_work_g rate_viscous energy_2"
)

gd = GeodynamicalDiagnostics(u, p, FullT, bottom_id, top_id)


temp_bcs = {
    bottom_id: {'T': 1.0 - (T0*exp(Di) - T0)},
    top_id: {'T': 0.0},
}

stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'uy': 0},
    left_id: {'ux': 0},
    right_id: {'ux': 0},
}

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             cartesian=True, constant_jacobian=True,
                             transpose_nullspace=Z_nullspace)

checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        output_file.write(u, p, T, FullT)
        ref_file.write(rhobar, Tbar, alphabar, cpbar, chibar)

    dt = t_adapt.update_timestep()
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    u_rms = gd.u_rms()
    u_rms_surf = gd.u_rms_top()
    bcu = DirichletBC(u.function_space(), 0, top_id)
    ux_max = u.dat.data_ro_with_halos[bcu.nodes, 0].max(initial=0)
    ux_max = u.comm.allreduce(ux_max, MPI.MAX)  # Maximum Vx at surface
    nusselt_number_top = gd.Nu_top()
    nusselt_number_base = gd.Nu_bottom()
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature = gd.T_avg()
    rate_work_against_gravity = assemble(approximation.work_against_gravity(u, T)*dx)
    rate_viscous_dissipation = assemble(approximation.viscous_dissipation(u)*dx)
    energy_conservation_2 = abs(rate_work_against_gravity - rate_viscous_dissipation)

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {u_rms} {u_rms_surf} {ux_max} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {average_temperature} "
                 f"{rate_work_against_gravity} {rate_viscous_dissipation} "
                 f"{energy_conservation_2}")

    # Calculate Full T:
    FullT.assign(T+Tbar)

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Checkpointing:
    if timestep % checkpoint_period == 0:
        checkpoint_file.save_function(T, name="Temperature", idx=timestep)
        checkpoint_file.save_function(z, name="Stokes", idx=timestep)

plog.close()
checkpoint_file.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
