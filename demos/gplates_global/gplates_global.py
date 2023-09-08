from gadopt import *
from mpi4py import MPI
import scipy.special
import math
import numpy as np
#import libgplates

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
rmin, rmax, ref_level, nlayers = 1.22, 2.22, 5, 16

# A gaussian shaped radial resolution function:
resolution_func = np.ones((nlayers))


def gaussian(center, c, a):
    return a*np.exp(-(np.linspace(rmin, rmax, nlayers)-center)**2/(2*c**2))


for idx, r_0 in enumerate([rmin, rmax, rmax - 660/6370]):
    # gaussian radius
    c = 0.15
    # how different is the high res area from low res
    res_amplifier = 5.
    resolution_func *= 1/(1+gaussian(center=r_0, c=c, a=res_amplifier))

# Construct a CubedSphere mesh and then extrude into a sphere - note that unlike cylindrical case, popping is done internally here:
mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(mesh2d, layers=nlayers, layer_height=(rmax-rmin)*resolution_func/np.sum(resolution_func), extrusion_type='radial')
bottom_id, top_id = "bottom", "top"
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
domain_volume = assemble(1.*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Test functions and functions to hold solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
muf = Function(W, name="Viscosity")

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())
log("Number of Temperature DOF:", Q.dim())

# GPlates requirements:
X = SpatialCoordinate(mesh)
X_val = interpolate(X, V)
gplates_velocities = Function(V, name="GPlates_Velocity")

# Set up temperature field and initialise:
T = Function(Q, name="Temperature")
Taverage = Function(Q1, name="Average Temperature")
T_dev = Function(Q1, name="Temperature_Deviation")
r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
theta = atan_2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
phi = atan_2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
k = as_vector((X[0]/r, X[1]/r, X[2]/r))  # Radial unit vector (in direction opposite to gravity)
T0 = Constant(0.091)  # Non-dimensional surface temperature
Di = Constant(0.5)  # Dissipation number.
H_int = Constant(10.0)  # Internal heating

# Initial condition for T:
conductive_term = ((1.0 - (T0*exp(Di) - T0)) * (2.22-r))
# Evaluate P_lm node-wise using scipy lpmv
l, m, eps_c, eps_s = 6, 4, 0.02, 0.02
Plm = Function(Q, name="P_lm")
cos_phi = interpolate(cos(phi), Q)
Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)
Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
if m == 0:
    Plm.assign(Plm/math.sqrt(2))
T.interpolate(conductive_term +
              (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))

# Important constants and physical parameters
Ra = Constant(5.0e7)  # Rayleigh number

# Rheological parameters
mulinf = Function(W, name="Viscosity_Lin")
muplastf = Function(W, name="Viscosity_Plast")
muminf = Function(W, name="Viscosity_Min")
mu_lin = 2.0


def step_func(r, center, mag, increasing=True, sharpness=30):
    """
    A step function designed to control viscosity jumps:
    input:
      r: is the radius array
      center: radius of the jump
      increasing: if True, the jump happens towards lower r, otherwise jump happens at higher r
      sharpness: how sharp should the jump should be (larger numbers = sharper).
    """
    if increasing:
        sign = 1
    else:
        sign = -1
    return mag * (0.5 * (1 + tanh(sign*(r-center)*sharpness)))


# Depth dependence: for the lower mantle increase we multiply the profile with a linear function
for line, step in zip([5.*(rmax-r), 1., 1.],
                      [step_func(r, 1.992, 30, False),
                       step_func(r, 2.078, 10, False),
                       step_func(r, 2.2, 10, True)]):
    mu_lin += line*step

# Adding temperature dependence:
delta_mu_T = Constant(100.)
mu_lin *= exp(-ln(delta_mu_T) * T)
mu_star, sigma_y = Constant(1.0), 5.0e5 + 2.5e6*(rmax-r)
epsilon = sym(grad(u))  # strain-rate
epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with a tolerance to ensure stability)
mu_plast = mu_star + (sigma_y / epsii)
mu_min = conditional(le(mu_plast, mu_lin), 1.0, 0.0)
mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)

# Compressible reference state:
rho_0, alpha, gruneisen = 1.0, 1.0, 1.0
weight = r-rmin
rhobar = Function(Q, name="CompRefDensity").interpolate(rho_0 * exp(((1.0 - weight) * Di) / alpha))
Tbar = Function(Q, name="CompRefTemperature").interpolate(T0 * exp((1.0 - weight) * Di) - T0)
alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)
FullT = Function(Q, name="FullTemperature").assign(T+Tbar)

# approximation = BoussinesqApproximation(Ra)
approximation = AnelasticLiquidApproximation(Ra, Di, rho=rhobar, Tbar=Tbar, alpha=alphabar, chi=chibar, cp=cpbar)

delta_t = Constant(1e-6)  # Initial time-step
t_adapt = TimestepAdaptor(delta_t, V, maximum_timestep=5e-6, increase_tolerance=1.25)
max_timesteps = 1
time = 0.0

# Compute layer average for initial stage:
averager = LayerAveraging(
    mesh, np.linspace(rmin, rmax, nlayers * 2), cartesian=False, quad_degree=6)
averager.extrapolate_layer_average(Taverage, averager.get_layer_average(FullT))

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

# Write output files in VTK format:
u, p = z.split()  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output.pvd")
ref_file = File('reference_state.pvd')
dump_period = 10
# Frequency of checkpoint files:
checkpoint_period = dump_period * 5
# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)

temp_bcs = {
    bottom_id: {'T': 1.0 - (T0*exp(Di) - T0)},
    top_id: {'T': 0.0},
}
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},    
#    top_id: {'u': gplates_velocities},
}

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
energy_solver.fields['source'] = rhobar * H_int
energy_solver.solver_parameters['ksp_converged_reason'] = None
energy_solver.solver_parameters['ksp_rtol'] = 1e-4
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                             cartesian=False,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)
stokes_solver.solver_parameters['snes_type'] = "ksponly"
stokes_solver.solver_parameters['snes_rtol'] = 5e-2
stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 5e-4
stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 5e-3

# No-Slip (prescribed) boundary condition for the top surface
#bc_gplates = DirichletBC(Z.sub(0), 0, (top_id))
#boundary_X = X_val.dat.data_ro_with_halos[bc_gplates.nodes]

# Get initial surface velocities:
#libgplates.rec_model.set_time(model_time=time)
#gplates_velocities.dat.data_with_halos[bc_gplates.nodes] = libgplates.rec_model.get_velocities(boundary_X)


# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        # compute radial temperature
        averager.extrapolate_layer_average(Taverage, averager.get_layer_average(FullT))
        # compute deviation from layer average
        T_dev.assign(FullT-Taverage)
        # interpolate viscosity
        muf.interpolate(mu)
        # write
        output_file.write(u, p, FullT, T_dev, muf)
        ref_file.write(rhobar, Tbar, alphabar, cpbar, chibar)

    if timestep != 0:
        dt = t_adapt.update_timestep(u)
    else:
        dt = float(delta_t)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Update surface velocities:
#    libgplates.rec_model.set_time(model_time=time)
#    gplates_velocities.dat.data_with_halos[bc_gplates.nodes] = libgplates.rec_model.get_velocities(boundary_X)

    # Compute diagnostics:
    u_rms = sqrt(assemble(dot(u, u) * dx)) * sqrt(1./domain_volume)
    nusselt_number_top = (assemble(dot(grad(FullT), n) * ds_t) / assemble(Constant(1.0, domain=mesh)*ds_t)) * (rmax*(rmax-rmin)/rmin)
    nusselt_number_base = (assemble(dot(grad(FullT), n) * ds_b) / assemble(Constant(1.0, domain=mesh)*ds_b)) * (rmin*(rmax-rmin)/rmax)
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature = assemble(T * dx) / domain_volume
    max_viscosity = muf.dat.data.max()
    max_viscosity = muf.comm.allreduce(max_viscosity, MPI.MAX)
    min_viscosity = muf.dat.data.min()
    min_viscosity = muf.comm.allreduce(min_viscosity, MPI.MIN)

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {u_rms} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {average_temperature} "
                 f"{min_viscosity} {max_viscosity} ")

    # Checkpointing:
    if timestep % checkpoint_period == 0:
        # Checkpointing during simulation:
        checkpoint_data = DumbCheckpoint(f"Temperature_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(T, name="Temperature")
        checkpoint_data.close()

        checkpoint_data = DumbCheckpoint(f"Stokes_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(z, name="Stokes")
        checkpoint_data.close()

plog.close()

# Write final state:
final_checkpoint_data = DumbCheckpoint("Final_Temperature_State", mode=FILE_CREATE)
final_checkpoint_data.store(T, name="Temperature")
final_checkpoint_data.close()

final_checkpoint_data = DumbCheckpoint("Final_Stokes_State", mode=FILE_CREATE)
final_checkpoint_data.store(z, name="Stokes")
final_checkpoint_data.close()
