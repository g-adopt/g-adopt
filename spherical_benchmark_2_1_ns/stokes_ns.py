from firedrake import *
from mpi4py import MPI
import math, numpy, scipy.special

#########################################################################################################
################################## Some important constants etc...: #####################################
#########################################################################################################

# Geometric Constants:
rmax                   = 2.22
rmin                   = 1.22
nlayers                = 24

# Mesh and associated physical boundary IDs:
mesh2d = IcosahedralSphereMesh(rmin, refinement_level=4, degree=2)
mesh = ExtrudedMesh(mesh2d, nlayers, (rmax-rmin)/nlayers, extrusion_type='radial')
bottom_id, top_id = 'bottom', 'top' # When using Gmsh generated mesh


# Global Constants:
steady_state_tolerance = 1e-7
max_timesteps          = 1000
target_cfl_no          = 2.5
max_timestep           = 0.05

# Stokes related constants:
Ra                     = Constant(7e3)   # Rayleigh Number
k                      = Constant((0,1))
X                      = SpatialCoordinate(mesh)
r                      = sqrt(X[0]**2+X[1]**2+X[2]**2)
theta                  = atan_2(X[1], X[0]) # Theta (longitude - different symbol to Zhong)
phi                    = atan_2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
k                      = as_vector((X[0]/r, X[1]/r, X[2]/r)) # Radial unit vector (in direction opposite to gravity)

# Related to temperature initial condition:
l     = 3
m     = 2
eps_c = 0.01
eps_s = 0.01

# Temperature related constants:
delta_t                = Constant(1e-6) # Initial time-step
kappa                  = Constant(1.0)  # Thermal diffusivity

# Temporal discretisation - Using a Crank-Nicholson scheme where theta_ts = 0.5:
theta_ts               = 0.5

#### Print function to ensure log output is only written on processor zero (if running in parallel) ####
def log(*args):
    if mesh.comm.rank == 0:
        print(*args)

        
# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    'mat_type': 'matfree',
    'snes_type': 'ksponly',
    'ksp_monitor_true_residual': None,
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_type': 'full',
    'fieldsplit_0': {
        'ksp_type': 'cg',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_pc_type' : 'hypre',
        'ksp_rtol': '1e-5',
    },
    'fieldsplit_1': {
        'ksp_type': 'fgmres',
        'ksp_converged_reason': True,
        'pc_type' : 'python',
        'pc_python_type' : 'firedrake.MassInvPC',
        'Mp_ksp_type': 'cg',
        'Mp_pc_type': 'sor',
        'ksp_rtol': '1e-3',
    }
}

# Temperature Equation Solver Parameters:
temperature_solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_rtol': 1e-5,
        'ksp_type': 'gmres',
        'pc_type': 'sor',
        'mat_type': 'aij'
}

#########################################################################################################
################################## Geometry and Spatial Discretization: #################################
#########################################################################################################

# Set up function spaces - currently using the P2P1 element pair :
V    = VectorFunctionSpace(mesh, "CG", 2) # Velocity function space (vector)
W    = FunctionSpace(mesh, "CG", 1) # Pressure function space (scalar)
Q    = FunctionSpace(mesh, "CG", 2) # Temperature function space (scalar)

# Set up mixed function space and associated test functions:
Z    = MixedFunctionSpace([V, W])
N, M = TestFunctions(Z)
Y    = TestFunction(Q)

# Set up fields on these function spaces - split into each component so that they are easily accessible:
z    = Function(Z) # a field over the mixed function space Z.
u, p = split(z) # can we nicely name mixed function space fields?

# Timestepping - CFL related stuff:
delta_x = sqrt(CellVolume(mesh))
ts_func = Function(Q) # Note that time stepping should be dictated by Temperature related mesh.

def compute_timestep(u):
    # A function to compute the timestep, based upon the CFL criterion
    ts_func.interpolate(delta_x / sqrt(dot(u,u)))
    ts_min = ts_func.dat.data.min()
    ts_min = mesh.comm.allreduce(ts_min, MPI.MIN)
    return min(ts_min*target_cfl_no,max_timestep)

#########################################################################################################
############################ T advection diffusion equation Prerequisites: ##############################
#########################################################################################################

# Set up temperature field and initialise based upon coordinates:
T_old    = Function(Q, name="OldTemperature")
conductive_term = rmin*(rmax - r) / (r*(rmax - rmin))
Plm = Function(Q, name="P_lm")
cos_phi = interpolate(cos(phi), Q)
# evaluate P_lm node-wise using scipy
Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data[:])
# rescale
Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
if m==0:
    Plm.assign(Plm/math.sqrt(2))
T_old.interpolate(conductive_term +
  (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))

T_new   = Function(Q, name="Temperature")
T_new.assign(T_old)

# Temporal discretisation - Using a theta scheme:
T_theta = theta_ts * T_new + (1-theta_ts) * T_old

#########################################################################################################
############################################ Setup Equations ############################################
#########################################################################################################

### Initially deal with Stokes equations ###

# Equation in weak (ufl) form - note that continuity equation is added here - need to better understand why:
# Set up in residual form to ensure non-linear solvers are used.
mu        = Constant(1.0) # Constant viscosity
tau       = mu * (grad(u)+transpose(grad(u))) # Strain-rate tensor:
F_stokes  = inner(grad(N), tau) * dx + dot(N,grad(p)) * dx - (dot(N,k)*Ra*T_theta) * dx
F_stokes += dot(grad(M),u) * dx # Continuity equation

# Set up boundary conditions for Stokes: We first need to extract the velocity field from the mixed 
# function space Z. This is done using Z.sub(0). We subsequently need to extract the x and y components 
# of velocity. This is done using an additional .sub(0) and .sub(1), respectively. Note that the final arguments
# here are the physical boundary ID's.
bcv_val_top  = Function(V).interpolate(as_vector((0.0, 0.0, 0.0)))
bcv_top      = DirichletBC(Z.sub(0), bcv_val_top, top_id)
bcv_val_base = Function(V).interpolate(as_vector((0.0, 0.0, 0.0)))
bcv_base     = DirichletBC(Z.sub(0), bcv_val_base, bottom_id)

### Next deal with Temperature advection-diffusion equation: ###
F_energy = Y * ((T_new - T_old) / delta_t) * dx + Y*dot(u,grad(T_theta)) * dx + dot(grad(Y),kappa*grad(T_theta)) * dx
bct_base = DirichletBC(Q, 1.0, bottom_id)
bct_top  = DirichletBC(Q, 0.0, top_id)

# Write output files in VTK format:
u, p = z.split() # Do this first to extract individual velocity and pressure fields:
u.rename('Velocity')
p.rename('Pressure')
u_file = File('velocity.pvd')
p_file = File('pressure.pvd')
t_file = File('temperature.pvd')
# write initial conditions:
u_file.write(u)
p_file.write(p)
t_file.write(T_new)

# Now perform the time loop:
for timestep in range(0,max_timesteps):

    delta_t.assign(compute_timestep(u)) # Compute adaptive time-step
    log("Timestep Number: ", timestep, " Timestep: ", float(delta_t))

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
    solve(F_stokes==0, z, bcs=[bcv_base,bcv_top], solver_parameters=stokes_solver_parameters, nullspace=nullspace, appctx={'mu': mu})

    # Temperature system:
    solve(F_energy==0, T_new, bcs=[bct_base,bct_top], solver_parameters=temperature_solver_parameters)

    if timestep % 10 == 1:
        # Write output:
        u_file.write(u)
        p_file.write(p)
        t_file.write(T_new)

    # Compute diagnostics:
    domain_volume = assemble(1.*dx(domain=mesh))
    rms_velocity = numpy.sqrt(assemble(dot(u,u) * dx)) * numpy.sqrt(1./domain_volume)

    mean_T_prime = assemble(T_new * dx) / domain_volume    

    n = FacetNormal(mesh)
    nusselt_number_top  = (assemble(dot(grad(T_new),n) * ds_t) / assemble(Constant(1.0,domain=mesh)*ds_t) ) * rmax/rmin
    nusselt_number_base = (assemble(dot(grad(T_new),n) * ds_b) / assemble(Constant(1.0,domain=mesh)*ds_b) ) * rmin/rmax
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    log("RMS Velocity: ",rms_velocity)
    log("T_prime: ",mean_T_prime)
    log("Nu (base, top, cons): ",nusselt_number_base,nusselt_number_top,energy_conservation)

    # Calculate L2-norm of change in temperature:
    maxchange = math.sqrt(assemble((T_new - T_old)**2 * dx))

    # Leave if steady-state has been achieved:
    log("Maxchange:",maxchange," Targetting:",steady_state_tolerance)
    if(maxchange < steady_state_tolerance):
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Set T_old = T_new - assign the values of T_new to T_old
    T_old.assign(T_new)

# Write final state (for adjoint testing):
checkpoint_data = DumbCheckpoint("Final_State",mode=FILE_CREATE)
checkpoint_data.store(T_new,name="Temperature")
checkpoint_data.close()
