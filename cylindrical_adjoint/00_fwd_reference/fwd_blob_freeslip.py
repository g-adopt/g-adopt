"""
    Initialization for the adjoint test
    We let a plume rise
    to the surface
"""
from firedrake import *
from mpi4py import MPI
import math
from firedrake.petsc import PETSc


#### Print function to ensure log output is only written on processor zero (if running in parallel) ####
def log(*args):
    if mesh.comm.rank == 0:
        PETSc.Sys.Print(*args)

#########################################################################################################
################################## Some important constants etc...: #####################################
#########################################################################################################

# Mesh and associated physical boundary IDs:
mesh    = Mesh('../mesh/shell_v2_cylindrical.msh') # This mesh was generated via gmsh

# Output the number of vertices 
log(" Number of Vertices %i" %mesh.num_vertices())

# Top and bottom curve
bottom_id, top_id       = 312, 311 # When using Gmsh generated mesh
left_id, right_id       = 314, 313 # When using Gmsh generated mesh

# Global Constants:
steady_state_tolerance  = 1e-7
max_timesteps           = 99999
simu_time               = 9e6
# conversion of non-dimensional time to years
time_dim                = (6370e3**2*3.18e-8)/1e-6
target_cfl_no           = 1.0
max_timestep            = 0.1

# Stokes related constants:
mu      = Constant(1.0)   # Viscosity
Ra      = Constant(1e8)   # Rayleigh Number
X       = SpatialCoordinate(mesh)
r       = sqrt(X[0]**2+X[1]**2)
er      = as_vector((X[0]/r, X[1]/r))
h       = sqrt(CellVolume(mesh))
n       = FacetNormal(mesh)

# center of the blob for initialsiation
# currently located at theta = 30
blb_ctr = 0.70*as_vector((0.29552020666133955, 0.955336489125606))
blb_gaus = Constant(0.06)
# Temperature related constants:
delta_t = Constant(1e-10) # Initial time-step
kappa   = Constant(1.0)  # Thermal diffusivity
# Stokes Equation Solver Parameters:
stokes_solver_parameters = { 
    'mat_type': 'matfree',
    'snes_type': 'ksponly',
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
        'ksp_rtol': '1e-4',
    }   
}

# Temperature Equation Solver Parameters:
temperature_solver_parameters = { 
        'snes_type': 'ksponly',
        'ksp_rtol': 1e-5,
        'ksp_type': 'gmres',
        'pc_type': 'sor',
        'mat_type': 'aij' #free
}
##
## Stokes Equation Solver Parameters:
#stokes_solver_parameters = {
#    'snes_type': 'ksponly',
#    'ksp_type': 'preonly',
#    'pc_type': 'fieldsplit',
#    'pc_fieldsplit_type': 'schur',
#    'pc_fieldsplit_schur_type': 'full',
#    'fieldsplit_0': {
#        'ksp_type': 'preonly',
#        'pc_type' : 'lu',
#        'pc_factor_mat_solver_type': 'mumps',
#        'ksp_rtol': 1e-6,
#    },
#    'fieldsplit_1': {
#        'ksp_type': 'fgmres',
#        'ksp_converged_reason': True,
#        'pc_type' : 'none',
#        'ksp_rtol': 1e-4,
#    },
#    'pc_factor_mat_solver_type': 'mumps',
#    'mat_type': 'aij'
#}
#
## Temperature Equation Solver Parameters:
#temperature_solver_parameters = {
#        'snes_type': 'ksponly',
#        'ksp_rtol': 1e-3,
#        'ksp_type': 'gmres',
#        'pc_type': 'sor',
#        'mat_type': 'aij' #free
#}

#########################################################################################################
################################## Geometry and Spatial Discretization: #################################
#########################################################################################################

# Set up function spaces - currently using the P2P1 element pair :
V    = VectorFunctionSpace(mesh, "CG", 2)   # Velocity function space (vector)
W    = FunctionSpace(mesh, "CG", 1)         # Pressure function space (scalar)
Q    = FunctionSpace(mesh, "CG", 2)         # Temperature function space (scalar)

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
#
X      = SpatialCoordinate(mesh)

# Set up temperature field and initialise based upon coordinates:
T_old   = Function(Q, name="OldTemperature")

# Initial condition for tempereature
T_ic   = Function(Q, name="T_IC")

# Interpolating a function
T_ic.interpolate(0.7 + 0.3*exp(-0.5*((X-blb_ctr)/blb_gaus)**2));#

#
### Depending on the input initial condition we choose which tempreaeture to set
### A constant average temperature plus a perturbation
###if rstr_flg:
### Set up T_ic, which is for loading from a checkpoint 
##tic_file = DumbCheckpoint("T_new",mode=FILE_READ)
##log('File opened')
##time_ic, k_ic = tic_file.get_timesteps()
##log('getting_time')
##my_time = time_ic[0]
##k_ind = k_ic[0] 
##tic_file.set_timestep(t=my_time, idx=k_ind)
##log('setting time')
##tic_file.load(T_ic, 'Temperature')
##log('temperature loaded')
##tic_file.close()
##else: 

# Assign T_ic to the old temperature
T_old.assign(T_ic)
log('T_old initiated')
# At the beginning T_new and Old are the same
T_new   = Function(Q, name="Temperature")
T_new.assign(T_old)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
theta   = 0.5
T_theta = theta * T_new + (1-theta) * T_old

#########################################################################################################
############################################ Setup Equations ############################################
#########################################################################################################

### Initially deal with Stokes equations ###
# Equation in weak (ufl) form 
# rewrite the weak form to be consistent with Freund and Sternberg
mu        = Constant(1.0) # Constant viscosity
# coefficient of the stabilizing term in Nitsche method
# for now use the same value as in Freund and Sternberg
beta     = Constant(10)
# Rhodri's stoks: 
# tau = mu * (grad(u)+transpose(grad(u))) 
# F_stokes  = inner(grad(v), tau) * dx + dot(v,grad(p)) * dx - (dot(v,k)*Ra*T_ic) * dx
# F_stokes += dot(grad(q),u) * dx # Continuity equation
# stress integral in weak form (wo pressure)
def a(u,N): return 2*inner(mu*sym(grad(u)), grad(N))*dx
# weak form of pressure and continuity equation
def b(N,M): return - div(N)*M*dx
# traction sigma(u, p).n 
def t(u,p): return dot(2*mu*sym(grad(u)),n) - p*n
# buoyancy 
def f(N): return (dot(N,er)*Ra*T_theta) * dx
# Nitsche method 
# notice that the third term is the stabilizer 
# (beta/h) is model dependent
def nitsche_freeslip(b_id): return - dot(N,n)*dot(n,t(u,p))*ds(b_id)\
                                   - dot(u,n)*dot(n,t(N,M))*ds(b_id)\
                                   + beta/h*dot(u,n)*dot(N,n)*ds(b_id)
F_stokes =  a(u,N) + b(N,p) + b(u,M) - f(N)\
            + nitsche_freeslip(b_id=top_id)\
            + nitsche_freeslip(b_id=bottom_id)\
            + nitsche_freeslip(b_id=right_id)


# Set up boundary conditions for Stokes: We first need to extract the velocity field from the mixed 
# function space Z. This is done using Z.sub(0). We subsequently need to extract the x and y components 
# of velocity. This is done using an additional .sub(0) and .sub(1), respectively. Note that the final arguments
# here are the physical boundary ID's. 
# Left and right side walls 
# Since right_id is not aligned with x axis anymore, we impose free-slib BC using Nitsche 
#bcv_r = DirichletBC(Z.sub(0).sub(1), 0.0, (right_id))
# But left_id is aligned with y axis
bcv_l = DirichletBC(Z.sub(0).sub(0), 0.0, (left_id) )

### Next deal with Temperature advection-diffusion equation: ###
F_energy = Y * ((T_new - T_old) / delta_t) * dx + Y*dot(u,grad(T_theta)) * dx + dot(grad(Y),kappa*grad(T_theta)) * dx

bct_base = DirichletBC(Q, 1.0, bottom_id)
bct_top  = DirichletBC(Q, 0.0, top_id)

# Computing the volume
# 
domain_volume = assemble(1.*dx(domain=mesh))

# Write output files in VTK format:
u, p = z.split() # Do this first to extract individual velocity and pressure fields:
u._name = 'Velocity' # This doesn't seem to work!
p._name = 'Pressure'

# Writing out the vtu file
u_file = File('Visual/velocity.pvd')
p_file = File('Visual/pressure.pvd')
t_file = File('Visual/temperature.pvd')##

# initial time and time-step index
my_time = 0.0
k_ind = 0
# initiate a checkpointing for velocity
checkpoint_u = DumbCheckpoint("velocities",\
    single_file=True, mode=FILE_CREATE, comm=mesh.comm)
#for timestep in range(0, max_timesteps):
while my_time < simu_time:
    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0. 
    solve(F_stokes==0, z, bcs=[bcv_l], solver_parameters=stokes_solver_parameters)
    checkpoint_u.set_timestep(t=my_time, idx=k_ind)
    checkpoint_u.store(u)
    # I have moved computation of time steps here, in case we restart 
    delta_t.assign(compute_timestep(u)) # Compute adaptive time-step

    # Check if total time is exceeding simu_time
    # if so fix the time so we end up with simu_time
    if my_time+(float(delta_t)*time_dim) > simu_time:
        delta_t.assign((simu_time-my_time)/time_dim)

    # We use fixed time steps for tests
    # the number 50e3 is somewhat arbitrary, but small enough
    delta_t.assign(100e3/time_dim)

    # Compute new time
    my_time += float(delta_t)*time_dim

    # Temperature system:
    solve(F_energy==0, T_new, solver_parameters=temperature_solver_parameters)

    # Visualisation Output
    if k_ind % 10 == 0:
        log("Output at time", str("%10.3f" %(my_time/1e6)), " Ma")
        t_file.write(T_new)
        u_file.write(u) 
        p_file.write(p)

    # Set T_old = T_new - assign the values of T_new to T_old
    T_old.assign(T_new)

    # increment index 
    k_ind += 1

    # Output necessary diagnostics
    log("idx = ", str("%i" %k_ind), ", time-step: ", str("%10.3f Ma" %(float(delta_t)*time_dim/1e6)), ", time:", str("%10.3f" %(my_time/1e6)), " Ma")

# we should close the velocity file
checkpoint_u.close() 

t_file.write(T_new)
u_file.write(u) 
p_file.write(p)

# Save the final temperature field to be used as reference 
checkpoint_data = DumbCheckpoint("T_final_state",\
    single_file=True, mode=FILE_CREATE, comm=mesh.comm)
checkpoint_data.store(T_new)
checkpoint_data.close() 
