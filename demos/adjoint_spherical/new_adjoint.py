from gadopt import *
from gadopt.gplates import *
from mpi4py import MPI
import numpy as np
import os.path

# Set up geometry:
rmin, rmax, ref_level, nlayers = 1.208, 2.208, 7, 64

# Construct a CubedSphere mesh and then extrude into a sphere (or load from checkpoint):
with CheckpointFile("initial_condition_mat_prop/Final_State.h5", mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")
    z = f.load_function(mesh, "Stokes")
mesh.cartesian = False
bottom_id, top_id = "bottom", "top"

# Set up function spaces - currently using the bilinear Q2Q1 element pair for Stokes and DQ2 T:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
V_scalar = FunctionSpace(mesh, "CG", 2)  # CG2 Scalar function space
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "DQ", 2)  # Temperature function space (DG scalar)
Q_CG = FunctionSpace(mesh, "CG", 2)  # CG Temperature function space for visualisation.
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Test functions and functions to hold solutions:
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
vr = Function(V_scalar, name="Radial_Velocity")  # For diagnostic output

# We next specify the important constants for this problem, including compressibility parameters,
# and set up the TALA approximation.
Ra = Constant(5e8)  # Rayleigh number
Di = Constant(0.9492824165791792)  # Dissipation number
H_int = Constant(9.93)  # Internal heating

X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
rhat = as_vector([X[0]/r, X[1]/r, X[2]/r])  # Radial unit vector (in direction opposite to gravity)

rhobar = Function(Q, name="CompRefDensity")
interpolate_1d_profile(function=rhobar, one_d_filename="initial_condition_mat_prop/rhobar.txt")
rhobar.assign(rhobar / 3200.)
Tbar = Function(Q, name="CompRefTemperature")
interpolate_1d_profile(function=Tbar, one_d_filename="initial_condition_mat_prop/Tbar.txt")
Tbar.assign((Tbar - 1600.) / 3700.)
alphabar = Function(Q, name="IsobaricThermalExpansivity")
interpolate_1d_profile(function=alphabar, one_d_filename="initial_condition_mat_prop/alphabar.txt")
alphabar.assign(alphabar / 4.1773e-05)
cpbar = Function(Q, name="IsobaricSpecificHeatCapacity")
interpolate_1d_profile(function=cpbar, one_d_filename="initial_condition_mat_prop/CpSIbar.txt")
cpbar.assign(cpbar / 1249.7)
gbar = Function(Q, name="GravitationalAcceleration")
interpolate_1d_profile(function=gbar, one_d_filename="initial_condition_mat_prop/gbar.txt")
gbar.assign(gbar / 9.8267)
kappa = Constant(3.0)  # Thermal conductivity = yields a diffusivity of 7.5e-7 at surface.

# We next prepare our viscosity, starting with a radial profile.
mu_rad = Function(Q, name="Viscosity_Radial")  # Depth dependent component of viscosity
interpolate_1d_profile(function=mu_rad, one_d_filename="initial_condition_mat_prop/mu2_radial.txt")

# We set up a Timestep Adaptor, for controlling the time-step length
# (via a CFL criterion) as the simulation advances in time.
time = 0.006721334013564765  # Initial time
sim_end = 0.  # Final time (Myr) for this segment of run
delta_t = Constant(5e-7)  # Initial time-step
timesteps = 1000  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.6, maximum_timestep=1e-5, increase_tolerance=2.0)

# We next set up our Full Temperature field (noting that T has been initialised above from checkpoint)
# and also specify two fields for computing lateral deviations from a radial layer average,
# for ease of visualisation.
FullT = Function(Q, name="FullTemperature").assign(T+Tbar)
T_avg = Function(Q, name='Layer_Averaged_Temp')
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))
T_dev = Function(Q, name='Temperature_Deviation').assign(FullT-T_avg)

# Now that we have the average T profile, we add lateral viscosity variation due to temperature variations:
mu_field = Function(Q, name="Viscosity")
delta_mu_T = Constant(1000.)  # For T_dev[-0.5,0.5], this leads to a viscosity range of [7.07106781e+01,1.41421356e-02]; 1000[30,0.3]
mu = mu_rad * exp(-ln(delta_mu_T) * T_dev)

# These fields are used to set up our Truncated Anelastic Liquid Approximation.
approximation = TruncatedAnelasticLiquidApproximation(Ra, Di, rho=rhobar, Tbar=Tbar, alpha=alphabar, cp=cpbar, g=gbar, H=H_int, mu=mu, kappa=kappa)

# Nullspaces and near-nullspace objects are next set up,
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

# Next, we define the surface velocities. All GPlates functionalities
# are accessible through the module `gadopt.gplates`. We will use the
# interface provided by G-ADOPT for pyGPlates. Similar to pyGPlates,
# the G-ADOPT interface requires specific files for loading and
# processing surface velocities from a reconstruction model. For this
# case we use the 1.8 Byr reconustruction of Cao et al. 2024.

# To generate a pyGplatesConnector in G-ADOPT, you need to provide the
# necessary rotation and topology files.  These files describe the
# plate polygons and their association with the Euler rotation poles
# at each stage of the reconstruction.  Additionally, you need to
# specify the oldest available age in the model, which for the
# reconstruction considered here is a billion years.  There are
# optional arguments with default values that can affect how the
# surface velocities are reconstructed.  `nseeds` is the number of
# points on a sphere used to initially load the plate reconstruction
# data, and `nneighbours` is the number of nearest points used to
# interpolate from the seed points to our 3-D mesh.  A lower `nseeds *
# 1/nneighbours` ratio results in a smoother velocity representation
# at each age, and vice versa.  This is especially useful for
# simulations on coarser grids. Given that this tutorial considers a
# Rayleigh number several orders of magnitude lower than Earth's
# mantle, we also scale plate velocities using an optional
# scaling_factor.

cao_2024_files = ensure_reconstruction("Cao 2024", "../gplates_files")

plate_reconstruction_model = pyGplatesConnector(
    rotation_filenames=cao_2024_files["rotation_filenames"],
    topology_filenames=cao_2024_files["topology_filenames"],
    oldest_age=1800,
    nseeds=1e4,
    nneighbours=4,
    scaling_factor=1.
)

# With the plate reconstruction model loaded using
# `pyGplatesConnector``, we can now generate the velocity field.  This
# is done using `GplatesVelocityFunction`. For all practical purposes,
# it behaves as a UFL-compatible Function. However, defining it
# requires two additional arguments. One is the `gplates_connector`,
# which we defined above, and the second is the subdomain marker of
# the top boundary in the mesh. Other arguments are identical to a
# Firedrake Function, meaning at minimum a FunctionSpace should be
# provided for the Function, which here is `V`, and optionally a name
# for the function.

# Top velocity boundary condition
gplates_velocities = GplatesVelocityFunction(
    V,
    gplates_connector=plate_reconstruction_model,
    top_boundary_marker=top_id,
    name="GPlates_Velocity"
)


# Followed by boundary conditions for velocity and temperature.
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'u': gplates_velocities},
}

temp_bcs = {
    bottom_id: {'T': 1.0 - 930/3700.},  # Take out adiabat
    top_id: {'T': 0.0},
}

# We next setup our output, in VTK format.
# We also open a file for logging and set up our diagnostic outputs.
output_file = VTKFile("output.pvd")
output_frequency = 200

plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt u_rms nu_base nu_top conservation avg_T min_T max_T min_mu max_mu ")

gd = GeodynamicalDiagnostics(z, FullT, bottom_id, top_id, quad_degree=6)

# We can now setup and solve the variational problem, for both the energy and Stokes equations,
# passing in the approximation, nullspace and near-nullspace information configured above.
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
energy_solver.solver_parameters['ksp_converged_reason'] = None
energy_solver.solver_parameters['ksp_rtol'] = 1e-4

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             constant_jacobian=False,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)
stokes_solver.solver_parameters['snes_rtol'] = 1e-2
stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

# Before we begin with the time-stepping, we need to know when to
# stop, which is when we arrive at the present-day.  To achieve this,
# we define `presentday_ndtime` which tells us when the simulation
# should end.  Note that this tutorial terminates after reaching a
# specified number of timesteps, prior to reaching the present-day.

presentday_ndtime = plate_reconstruction_model.age2ndtime(0.0)
sim_end_ndtime = plate_reconstruction_model.age2ndtime(sim_end)

# Set up fields for visualisation on CG meshes - DG is overkill for output.
FullT_CG = Function(Q_CG, name="FullTemperature_CG").interpolate(FullT)
T_CG = Function(Q_CG, name='Temperature_CG').interpolate(T)
T_dev_CG = Function(Q_CG, name='Temperature_Deviation_CG').interpolate(T_dev)
mu_field_CG = Function(Q_CG, name="Viscosity_CG").interpolate(mu)

# We now initiate the time loop:
for timestep in range(0, timesteps):

    # Update varial velocity:
    vr.interpolate(inner(u, rhat))

    if timestep != 0:
        dt = t_adapt.update_timestep()
    else:
        dt = float(delta_t)
    time += dt

    # Update plate velocities:
    gplates_velocities.update_plate_reconstruction(time)

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Calculate Full T and update gradient fields:
    FullT.assign(T+Tbar)
    # Compute deviation from layer average
    averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))
    T_dev.assign(FullT-T_avg)

    # Compute diagnostics:
    nusselt_number_top = gd.Nu_top() * (rmax*(rmax-rmin)/rmin)
    nusselt_number_base = gd.Nu_bottom() * (rmin*(rmax-rmin)/rmax)
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    max_mu = mu_field.dat.data.max()
    max_mu = mu_field.comm.allreduce(max_mu, MPI.MAX)
    min_mu = mu_field.dat.data.min()
    min_mu = mu_field.comm.allreduce(min_mu, MPI.MIN)

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {gd.u_rms()} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {gd.T_avg()} "
                 f"{gd.T_min()} {gd.T_max()}, {min_mu} {max_mu}  ")

    # Do not go over present-day:
    if time > presentday_ndtime:
        log("time > presentday_ndtime")
        # Write output and interpolate to CG:
        mu_field_CG.interpolate(mu)
        FullT_CG.interpolate(FullT)
        T_CG.interpolate(T)
        T_dev_CG.interpolate(T_dev)
        output_file.write(*z.subfunctions, vr, FullT_CG, T_CG, T_dev_CG, mu_field_CG)
        break

    # Do not go over x Myr or model time:
    if time > sim_end_ndtime:
        log("time > sim_end_ndtime")
        # Write output and interpolate to CG:
        mu_field_CG.interpolate(mu)
        FullT_CG.interpolate(FullT)
        T_CG.interpolate(T)
        T_dev_CG.interpolate(T_dev)
        output_file.write(*z.subfunctions, vr, FullT_CG, T_CG, T_dev_CG, mu_field_CG)
        break

plog.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
