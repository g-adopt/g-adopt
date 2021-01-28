"""
    Doing a taylor test for the adjoint 
"""

from firedrake import *
from mpi4py import MPI
import math, numpy
from firedrake.petsc import PETSc
from firedrake_adjoint import *
from pyadjoint import MinimizationProblem, ROLSolver

#########################################################################################################
################################## Some important constants etc...: #####################################
#########################################################################################################

#logging.set_log_level(1)
#logging.set_level(1)

# Geometric Constants:
y_max = 1.0
x_max = 1.0

#   how many intervals along x/y directions 
disc_n = 100

# Top and bottom ids, for extruded mesh
top_id, bottom_id = 2, 1 

# The mesh
mesh = utility_meshes.PeriodicRectangleMesh(nx=disc_n, ny=disc_n,\
                                            Lx=x_max, Ly=y_max, direction='x',\
                                            quadrilateral=True, reorder=None,\
                                            distribution_parameters=None, diagonal=None)
# spatial coordinates
X  = x, y = SpatialCoordinate(mesh)
h	  = sqrt(CellVolume(mesh))
y_abs     = sqrt(y**2)
yhat  = as_vector((0,y)) / y_abs

# Global Constants:
steady_state_tolerance = 1e-7
max_num_timesteps      = 30
target_cfl_no          = 2.5
max_timestep           = 1.00

# Stokes related constants:
Ra                     = Constant(1e5)   # Rayleigh Number

# Below are callbacks relating to the adjoint solutions (accessed through solve).
# Not sure what the best place would be to initiate working tape!
tape = get_working_tape()

# Temperature related constants:
delta_t                = Constant(1e-6) # Initial time-step
kappa                  = Constant(1.0)  # Thermal diffusivity

# Temporal discretisation - Using a Crank-Nicholson scheme where theta_ts = 0.5:
theta_ts               = 0.5

#### Print function to ensure log output is only written on processor zero (if running in parallel) ####
def log(*args):
    if mesh.comm.rank == 0:
        PETSc.Sys.Print(*args) 

# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'
}

# Temperature Equation Solver Parameters:
temperature_solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_converged_reason': None,
        'ksp_monitor': None,
        'ksp_rtol': 1e-2,
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
Z       = MixedFunctionSpace([V, W])
N, M    = TestFunctions(Z)
Y       = TestFunction(Q)

# Set up fields on these function spaces - split into each component so that they are easily accessible:
z    = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)     # can we nicely name mixed function space fields?

# Timestepping - CFL related stuff:
ts_func = Function(Q) # Note that time stepping should be dictated by Temperature related mesh.

def compute_timestep(u):
    # A function to compute the timestep, based upon the CFL criterion
    ts_func.interpolate( h / sqrt(dot(u,u)))
    ts_min = ts_func.dat.data.min()
    ts_min = mesh.comm.allreduce(ts_min, MPI.MIN)
    #return min(ts_min*target_cfl_no,max_timestep)
    return 1e-4 
                        
#########################################################################################################
############################ T advection diffusion equation Prerequisites: ##############################
#########################################################################################################

# Final state, which will be used as reference for minimization, loaded from a file 
final_state = Function(Q, name='RefTemperature')
final_state_file = DumbCheckpoint("./final_state", mode=FILE_READ)
final_state_file.load(final_state, 'Temperature')
final_state_file.close()

# Initial condition
T_ic   = Function(Q, name="T_IC")
T_ic.assign(0.5)

# Set up temperature field and initialise based upon coordinates:
T_old    = Function(Q, name="OldTemperature")

# Having a single hot blob on 1.5, 0.0
T_old.assign(T_ic)

T_new   = Function(Q, name="Temperature")
T_new.assign(T_old)

# Temporal discretisation - Using a theta scheme:
T_theta = theta_ts * T_new + (1-theta_ts) * T_old


# ********************** Setup Equations ************************ 
# viscosiy 
mu        = Constant(1.0) # Constant viscosity

# deviatoric stresses
def tau(u): return  mu * (grad(u)+transpose(grad(u)))

# traction field 
def trac(u,p): return dot(tau(u),n) - p*n

# Finalise equations
F_stokes  = inner(grad(N), tau(u)) * dx - div(N)*p * dx 
F_stokes += - (dot(N,yhat)*Ra*T_theta) * dx 
F_stokes += - div(u)* M * dx

# Setting free-slip BC for top and bottom
bcu_topbase= DirichletBC(Z.sub(0).sub(1), 0.0, (top_id, bottom_id) )

# Temperature, advection-diffusion equation
F_energy = Y * ((T_new - T_old) / delta_t) * dx + Y*dot(u,grad(T_theta)) * dx + dot(grad(Y),kappa*grad(T_theta)) * dx

# Prescribed temperature for top and bottom
bct_base = DirichletBC(Q, 1.0, bottom_id)
bct_top  = DirichletBC(Q, 0.0, top_id)

# For some reason this only works here!
u, p    = z.split() # Do this first to extract individual velocity, pressure and lagrange multplier fields:
u.rename('Velocity') 
p.rename('Pressure')

# Printing out the degrees of freedom 
log('global number of nodes P1 coeffs/nodes:', W.dim())

# A simulation time to track how far we are
simu_time = 0.0

# Setting adjoint and forward callbacks, and control parameter
#local_cb_post = OptimisationOutputCallbackPost()
#eval_cb_pre = ForwardCallbackPost()
control = Control(T_ic)

# Now perform the time loop:
for timestep in range(0, max_num_timesteps):
    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0. 
    solve(F_stokes==0, z, bcs=[bcu_topbase], solver_parameters=stokes_solver_parameters)

    # updating time-step based on velocities
    delta_t.assign(compute_timestep(u)) # Compute adaptive time-step

    # Temperature system:
    solve(F_energy==0, T_new, bcs=[bct_base, bct_top], solver_parameters=stokes_solver_parameters)

    # updating time
    simu_time += float(delta_t)

    # Set T_old = T_new - assign the values of T_new to T_old
    T_old.assign(T_new)

    # Updating Temperature
    log("Timestep Number: ", timestep, " Timestep: ", str('%10.4e' %simu_time))


# setting up the functional
functional = assemble(0.5*(T_new - final_state)**2 * dx)

# Defining the object for pyadjoint
reduced_functional = ReducedFunctional(functional, control)

# Taylor test:
Delta_temp   = Function(Q, name="Delta_Temperature")
Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
minconv = taylor_test(reduced_functional, T_ic, Delta_temp)
log(minconv)



