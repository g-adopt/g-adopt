from gadopt import *
from mpi4py import MPI

# Set up geometry:
nx, ny = 80, 80
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
T.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))

delta_t = Constant(1e-6)  # Initial time-step
t_adapt = TimestepAdaptor(delta_t, V, maximum_timestep=0.1, increase_tolerance=1.5)

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(200)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)
# Rheology:
gamma_T, gamma_Z = Constant(ln(10**6)), Constant(ln(10))
mu_star, sigma_y = Constant(0.001), Constant(1.0)
epsilon = sym(grad(u))  # Strain-rate
epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with a tolerance to ensure stability)
mu_lin = exp(-gamma_T*T + gamma_Z*(1 - X[1]))
mu_plast = mu_star + (sigma_y / epsii)
mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)

time = 0.0
steady_state_tolerance = 1e-9
max_timesteps = 10
kappa = Constant(1.0)  # Thermal diffusivity

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Write output files in VTK format:
u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output.pvd")
dump_period = 1
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4

# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms u_rms_surf ux_max nu_top nu_base energy avg_t")

gd = GeodynamicalDiagnostics(u, p, T, bottom_id, top_id)


temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}

stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'uy': 0},
    left_id: {'ux': 0},
    right_id: {'ux': 0},
}

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                             cartesian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        output_file.write(u, p, T)

    dt = t_adapt.update_timestep(u)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    bcu = DirichletBC(u.function_space(), 0, top_id)
    ux_max = u.dat.data_ro_with_halos[bcu.nodes, 0].max(initial=0)
    ux_max = u.comm.allreduce(ux_max, MPI.MAX)  # Maximum Vx at surface
    nusselt_number_top = gd.Nu_top()
    nusselt_number_base = gd.Nu_bottom()
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} {gd.u_rms_top()} {ux_max} "
                 f"{nusselt_number_top} {nusselt_number_base} "
                 f"{energy_conservation} {gd.T_avg()} ")

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
