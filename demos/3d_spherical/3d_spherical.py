from gadopt import *
import scipy.special
import math

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
rmin, rmax, ref_level, nlayers = 1.22, 2.22, 4, 8

# Construct a CubedSphere mesh and then extrude into a sphere:
mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type='radial')
bottom_id, top_id = "bottom", "top"
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
domain_volume = assemble(1*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)


# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
Qlayer = FunctionSpace(mesh2d, "CG", 2)  # used to compute layer average

# Test functions and functions to hold solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
T = Function(Q, name="Temperature")
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
theta = atan2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
phi = atan2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
k = as_vector((X[0]/r, X[1]/r, X[2]/r))  # Radial unit vector (in direction opposite to gravity)

conductive_term = rmin*(rmax - r) / (r*(rmax - rmin))
# evaluate P_lm node-wise using scipy lpmv
l, m, eps_c, eps_s = 3, 2, 0.01, 0.01
Plm = Function(Q, name="P_lm")
cos_phi = interpolate(cos(phi), Q)
Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)
Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
if m == 0:
    Plm.assign(Plm/math.sqrt(2))
T.interpolate(conductive_term +
              (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))

Ra = Constant(7e3)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

delta_t = Constant(1e-6)  # Initial time-step

# For computing layer averages
T_avg = Function(Q, name='Layer_Averaged_Temp')
T_dev = Function(Q, name='Temperature_Deviation')

# Compute layer average for initial stage:
averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

# Define time stepping parameters:
steady_state_tolerance = 1e-6
max_timesteps = 20
time = 0.0

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

# Write output files in VTK format:
u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = VTKFile("output.pvd")
dump_period = 1
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4
# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms nu_top nu_base energy avg_t t_dev_avg")

gd = GeodynamicalDiagnostics(u, p, T, bottom_id, top_id)
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             cartesian=False, constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)

checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        # compute radially averaged temperature profile
        averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))
        # compute deviation from layer average
        T_dev.assign(T-T_avg)
        output_file.write(u, p, T, T_dev)

    if timestep != 0:
        dt = t_adapt.update_timestep()
    else:
        dt = float(delta_t)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    nusselt_number_top = gd.Nu_top() * (rmax*(rmin-rmax)/rmin)
    nusselt_number_base = gd.Nu_bottom() * (rmin*(rmax-rmin)/rmax)
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    T_dev_avg = assemble(T_dev * dx) / domain_volume

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} "
                 f"{nusselt_number_top} {nusselt_number_base} "
                 f"{energy_conservation} {gd.T_avg()} {T_dev_avg} ")

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
