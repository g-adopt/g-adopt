from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI
import libgplates
import numpy as np

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
rmin, rmax, ref_level, nlayers = 1.22, 2.22, 6, 32

# Initiating layer heights with 1.
resolution_func = np.ones((nlayers))


# A gaussian shaped function to enable variable radial resolution
def gaussian(center, c, a):
    """Gaussian shaped function to enable variable radial resolution"""
    return a*np.exp(-(np.linspace(rmin, rmax, nlayers)-center)**2/(2*c**2))


# building the resolution function
for idx, r_0 in enumerate([rmin, rmax, rmax - 660/6370]):
    # Gaussian radius
    c = 0.15
    # how different is the high res area from low res
    res_amplifier = 5.
    resolution_func *= 1 / (1 + gaussian(center=r_0, c=c, a=res_amplifier))

# Construct a CubedSphere mesh and then extrude into a sphere - note that unlike cylindrical case, popping is done internally here:
mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(
    mesh2d,
    layers=nlayers,
    layer_height=(rmax-rmin)*resolution_func/np.sum(resolution_func),
    extrusion_type="radial"
)
bottom, top = "bottom", "top"
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
domain_volume = assemble(1.*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)


# Define logging convenience functions:
def log(*args):
    """Log output to stdout from root processor only"""
    PETSc.Sys.Print(*args)


def log_params(f, str):
    """Log diagnostic paramters on root processor only"""
    if mesh.comm.rank == 0:
        f.write(str + "\n")
        f.flush()


# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
Qlayer = FunctionSpace(mesh2d, "CG", 2)  # used to compute layer average
# Test functions and functions to hold solutions:
v, w = TestFunctions(Z)
q = TestFunction(Q)
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
muf = Function(W, name="Viscosity")

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
theta = atan_2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
phi = atan_2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
k = as_vector((X[0]/r, X[1]/r, X[2]/r))  # Radial unit vector (in direction opposite to gravity)
Told, Tnew, Tdev = Function(Q, name="OldTemp"), Function(Q, name="NewTemp"), Function(Q, name="DeltaT")
T0 = Constant(0.091)  # Non-dimensional surface temperature

# Initialise from checkpoint:
tic_dc = DumbCheckpoint("Temperature_State_100", mode=FILE_READ)
tic_dc.load(Told, name="Temperature")
tic_dc.close()
Tnew.assign(Told)

# Initial condition for Stokes:
uic_dc = DumbCheckpoint("Stokes_State_100", mode=FILE_READ)
uic_dc.load(z, name="Stokes")
uic_dc.close()

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
Ttheta = 0.5 * Tnew + (1-0.5) * Told

# helper function to compute horizontal layer averages
Tlayer = Function(Qlayer, name='LayerTemp')  # stores values of temp in one layer
Tavg = Function(Q, name='LayerAveragedTemp')  # averaged temp function returned by function
Rmin_area = assemble(Constant(1.0, domain=mesh2d)*dx)  # area of CMB


def layer_average(T):
    """ Compute layer average temperature """
    vnodes = nlayers*2 + 1  # n/o Q2 nodes in the vertical
    hnodes = Qlayer.dim()  # n/o Q2 nodes in each horizontal layer
    assert hnodes*vnodes == Q.dim()
    for i in range(vnodes):
        Tlayer.dat.data[:] = T.dat.data_ro[i::vnodes]
        # NOTE: this integral is performed on mesh2d, which always has r=Rmin, but we normalize
        Tavg.dat.data[i::vnodes] = assemble(Tlayer*dx) / Rmin_area
    return Tavg


max_timesteps = 500
target_cfl_no = 2.5
maximum_timestep = 1e-4
increase_tolerance = 1.25
time = 0.0

# Timestepping - CFL related stuff:
delta_x = sqrt(CellVolume(mesh))
ref_vel = Function(V, name="Reference_Velocity")


def compute_timestep(u, current_delta_t):
    """Return the timestep, based upon the CFL criterion"""

    ref_vel.interpolate(dot(JacobianInverse(mesh), u))
    velmax = mesh.comm.allreduce(ref_vel.dat.data.max(), MPI.MAX)
    ts_min = 1./velmax
    # Grab (smallest) maximum permitted on all cores:
    ts_max = min(float(current_delta_t)*increase_tolerance, maximum_timestep)
    # Compute timestep
    tstep = min(ts_min*target_cfl_no, ts_max)
    log("Timestep: ", tstep, ts_min*target_cfl_no)
    return tstep


# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    "mat_type": "matfree",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-2,
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": {
        "ksp_type": "cg",
        "ksp_rtol": 1e-4,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "gamg",
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-3,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
    }
}

# Energy Equation Solver Parameters:
energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-4,
    "ksp_converged_reason": None,
    "pc_type": "sor",
}


X_val = interpolate(X, V)

# set up a Function for gplate velocities
gplates_velocities = Function(V, name='SurfaceVelocity')

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(1.5e7)  # Rayleigh number
Di = Constant(0.5)  # Dissipation number.
mulinf = Function(W, name="Viscosity_Lin")
muplastf = Function(W, name="Viscosity_Plast")
muminf = Function(W, name="Viscosity_Min")
delta_mu_660, delta_mu_r, delta_mu_T = Constant(40.), Constant(1.99), Constant(500.)
mu_lin = (delta_mu_660 - (delta_mu_660-1)/2. - (delta_mu_660-1)*tanh((r-delta_mu_r)*10)/2.)*exp(-ln(delta_mu_T) * Tnew)
mu_star, sigma_y = Constant(0.5), 1e4 + 2.4e5*(rmax-r)
epsilon = sym(grad(u))  # strain-rate
epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with a tolerance to ensure stability)
mu_plast = mu_star + (sigma_y / epsii)
mu_min = conditional(le(mu_plast, mu_lin), 1.0, 0.0)
mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)

C_ip = Constant(100.0)  # Fudge factor for interior penalty term used in weak imposition of BCs
p_ip = 2  # Maximum polynomial degree of the _gradient_ of velocity

# Temperature equation related constants:
delta_t = Constant(1e-6)  # Initial time-step
tcond = Constant(1.0)  # Thermal conductivity
H_int = Constant(10.0)  # Internal heating

# Compressible reference state:
rho_0, alpha, cpr, cvr, gruneisen = 1.0, 1.0, 1.0, 1.0, 1.0
weight = r-rmin
rhobar = Function(Q, name="CompRefDensity").interpolate(rho_0 * exp(((1.0 - weight) * Di) / alpha))
Tbar = Function(Q, name="CompRefTemperature").interpolate(T0 * exp((1.0 - weight) * Di) - T0)
alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)
FullT = Function(Q, name="FullTemperature").assign(Tnew+Tbar)

# Stokes equations in UFL form:
I = Identity(3)


def stress(u):
    """ Return the stress tensor """
    return 2 * mu * sym(grad(u)) - 2./3. * I * mu * div(u)


F_stokes = inner(grad(v), stress(u)) * dx - div(v) * p * dx + dot(n, v) * p * ds_tb - (dot(v, k) * (Ra * Ttheta * rhobar * alphabar - (Di/gruneisen) * (cpr/cvr)*rhobar*chibar*p) * dx)
F_stokes += -w * div(rhobar*u) * dx + w * dot(n, rhobar*u) * ds_tb

# Nitsche free-slip BCs
F_stokes += -dot(v, n) * dot(dot(n, stress(u)), n) * ds_b
F_stokes += -dot(u, n) * dot(dot(n, stress(v)), n) * ds_b
F_stokes += C_ip * mu * (p_ip + 1)**2 * FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds_b

# No-Slip (prescribed) boundary condition for the top surface
bc_gplates = DirichletBC(Z.sub(0), gplates_velocities, (top))
boundary_X = X_val.dat.data_ro_with_halos[bc_gplates.nodes]

# Get initial surface velocities:
libgplates.rec_model.set_time(model_time=time)
gplates_velocities.dat.data_with_halos[bc_gplates.nodes] = libgplates.rec_model.get_velocities(boundary_X)


def absv(u):
    """Component-wise absolute value of vector for SU stabilisation"""
    return as_vector([abs(ui) for ui in u])


def beta(Pe):
    """Component-wise beta formula Donea and Huerta (2.47a) for SU stabilisation"""
    return as_vector([1/tanh(Pei+1e-6) - 1/(Pei+1e-6) for Pei in Pe])


# SU(PG) ala Donea & Huerta:
# Columns of Jacobian J are the vectors that span the quad/hex
# which can be seen as unit-vectors scaled with the dx/dy/dz in that direction (assuming physical coordinates x,y,z aligned with local coordinates)
# thus u^T J is (dx * u , dy * v)
# and following (2.44c) Pe = u^T J / 2 (as nu=diffusivity=1)
# beta(Pe) is the xibar vector in (2.44a)
# then we get artifical viscosity nubar from (2.49)
J = Function(TensorFunctionSpace(mesh, 'DQ', 1), name='Jacobian').interpolate(Jacobian(mesh))
Pe = absv(dot(u, J)) / 2
nubar = dot(Pe, beta(Pe))
q_SU = q + nubar / dot(u, u) * dot(u, grad(q))

# Energy equation in UFL form:
F_energy = q * rhobar * cpbar * ((Tnew - Told) / delta_t) * dx + q_SU * rhobar * cpbar * dot(u, grad(Ttheta)) * dx + dot(grad(q), tcond * grad(Tbar + Ttheta)) * dx + q * (alphabar * rhobar * Di * dot(u, k) * Ttheta) * dx - q * ((Di/Ra) * inner(stress(u), grad(u))) * dx - q * rhobar * H_int * dx

# Temperature boundary conditions
bctb, bctt = DirichletBC(Q, 1.0 - (T0 * exp(Di) - T0), bottom), DirichletBC(Q, 0.0, top)

# Nullspaces and near-nullspaces:
p_nullspace = VectorSpaceBasis(constant=True)  # Constant nullspace for pressure
Z_nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), p_nullspace])  # Setting mixed nullspace

# Generating near_nullspaces for GAMG:
x_rotV = Function(V).interpolate(as_vector((0, X[2], -X[1])))
y_rotV = Function(V).interpolate(as_vector((-X[2], 0, X[0])))
z_rotV = Function(V).interpolate(as_vector((-X[1], X[0], 0)))
nns_x = Function(V).interpolate(Constant([1., 0., 0.]))
nns_y = Function(V).interpolate(Constant([0., 1., 0.]))
nns_z = Function(V).interpolate(Constant([0., 0., 1.]))
V_near_nullspace = VectorSpaceBasis([nns_x, nns_y, nns_z, x_rotV, y_rotV, z_rotV])
V_near_nullspace.orthonormalize()
Z_near_nullspace = MixedVectorSpaceBasis(Z, [V_near_nullspace, Z.sub(1)])

# Write output files in VTK format:
u, p = z.split()  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output.pvd")
dump_period = 20
# Frequency of checkpoint files:
checkpoint_period = dump_period * 1
# Open file for logging diagnostic output:
f = open("params.log", "w")

# Setup problem and solver objects so we can reuse (cache) solver setup
stokes_problem = NonlinearVariationalProblem(F_stokes, z, bcs=[bc_gplates])  # velocity BC for the bottom surface is handled through Nitsche, top surface through gplates_velocities
stokes_solver = NonlinearVariationalSolver(stokes_problem, solver_parameters=stokes_solver_parameters, appctx={"mu": mu}, transpose_nullspace=Z_nullspace, near_nullspace=Z_near_nullspace)
energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
energy_solver = NonlinearVariationalSolver(energy_problem, solver_parameters=energy_solver_parameters)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep == 0 or timestep % dump_period == 0:
        # compute radial temperature
        Tavg = layer_average(FullT)
        # compute deviation from layer average
        Tdev.assign(FullT-Tavg)
        # Write output:
        muf.interpolate(mu)
        mulinf.interpolate(mu_lin)
        muplastf.interpolate(mu_plast)
        muminf.interpolate(mu_min)
        output_file.write(u, p, Tnew, Tdev, FullT, muf, mulinf, muplastf, muminf)

    current_delta_t = delta_t
    if timestep != 0:
        delta_t.assign(compute_timestep(u, current_delta_t))  # Compute adaptive time-step
    time += float(delta_t)

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Update surface velocities
    libgplates.rec_model.set_time(model_time=time)
    gplates_velocities.dat.data_with_halos[bc_gplates.nodes] = libgplates.rec_model.get_velocities(boundary_X)

    # Compute diagnostics:
    u_rms = sqrt(assemble(dot(u, u) * dx)) * sqrt(1./domain_volume)
    nusselt_number_top = (assemble(dot(grad(FullT), n) * ds_t) / assemble(Constant(1.0, domain=mesh)*ds_t)) * (rmax*(rmax-rmin)/rmin)
    nusselt_number_base = (assemble(dot(grad(FullT), n) * ds_b) / assemble(Constant(1.0, domain=mesh)*ds_b)) * (rmin*(rmax-rmin)/rmax)
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature = assemble(FullT * dx) / domain_volume
    rate_work_against_gravity = assemble(rhobar * alphabar * Di.values()[0] * Tnew * u[1] * dx) / domain_volume
    rate_viscous_dissipation = (assemble(inner(stress(u), grad(u)) * dx) / domain_volume) * Di.values()[0]/Ra.values()[0]
    energy_conservation_2 = abs(rate_work_against_gravity - rate_viscous_dissipation)
    max_viscosity = muf.dat.data.max()
    max_viscosity = muf.comm.allreduce(max_viscosity, MPI.MAX)
    min_viscosity = muf.dat.data.min()
    min_viscosity = muf.comm.allreduce(min_viscosity, MPI.MIN)

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((Tnew - Told)**2 * dx))

    # Log diagnostics:
    log_params(f, f"{timestep} {time} {float(delta_t)} {maxchange} {u_rms} "
               f"{nusselt_number_base} {nusselt_number_top} "
               f"{energy_conservation} {average_temperature} "
               f"{rate_work_against_gravity} {rate_viscous_dissipation} "
               f"{energy_conservation_2} {min_viscosity} {max_viscosity} ")

    # Set Told = Tnew - assign the values of Tnew to Told and calculated Full T
    Told.assign(Tnew)
    FullT.assign(Tnew + Tbar)

    # Checkpointing:
    if timestep % checkpoint_period == 0:
        # Checkpointing during simulation:
        checkpoint_data = DumbCheckpoint(f"Temperature_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(Tnew, name="Temperature")
        checkpoint_data.close()

        checkpoint_data = DumbCheckpoint(f"Stokes_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(z, name="Stokes")
        checkpoint_data.close()

f.close()

# compute radial temperature
Tavg = layer_average(Tnew)
# compute deviation from layer average
Tdev.assign(Tnew-Tavg)
# Write output:
output_file.write(u, p, Tnew, Tdev, FullT, muf, mulinf, muplastf, muminf)

# Write final state:
final_checkpoint_data = DumbCheckpoint("Final_Temperature_State", mode=FILE_CREATE)
final_checkpoint_data.store(Tnew, name="Temperature")
final_checkpoint_data.close()

final_checkpoint_data = DumbCheckpoint("Final_Stokes_State", mode=FILE_CREATE)
final_checkpoint_data.store(z, name="Stokes")
final_checkpoint_data.close()
