"""
    Generation of forward problem - for an adjoint simulation
"""

from firedrake import *
from mpi4py import MPI
import math, numpy
from firedrake.petsc import PETSc

#########################################################################################################
################################## Some important constants etc...: #####################################
#########################################################################################################

logging.set_log_level(1)
logging.set_level(1)

# Geometric Constants:
R_max = 2.22
R_min = 1.22

#  The level of descretisation
disc_n = 3

# Constructing a Circle first
m = CircleManifoldMesh(disc_n*16*8, radius=R_min)

# The extruded mesh has a default height of 1.0
mesh = ExtrudedMesh(m, layers=disc_n*16, extrusion_type='radial')     

# Top and bottom ids, for extruded mesh
top_id, bottom_id = 'top', 'bottom'

## popping all the coordinates on a sphere 
P1    = FunctionSpace(mesh, "CG", 1)
V     = VectorFunctionSpace(mesh, "CG", 2)
x, y  = SpatialCoordinate(mesh)
r     = sqrt(x**2 + y**2)
r_p1  = interpolate(r, P1)
xy_p2 = interpolate(as_vector((x,y))/r*r_p1, V)
mesh  = Mesh(xy_p2)

# now redefine everything based on the super/iso parametric P2 mesh
X     = SpatialCoordinate(mesh)
x, y  = SpatialCoordinate(mesh)
n     = FacetNormal(mesh)
h	  = sqrt(CellVolume(mesh))
r     = sqrt(x**2 + y**2)
rhat  = as_vector((x, y)) / r

# Global Constants:
steady_state_tolerance = 1e-7
max_timesteps          = 20
target_cfl_no          = 2.5
max_timestep           = 1.00

# Stokes related constants:
Ra                     = Constant(1e4)   # Rayleigh Number

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

# Stokes Equation Solver Parameters:
stokes_solver_parameters_iterative = {
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
        'assembled_pc_type': 'gamg',
        'assembled_pc_gamg_threshold_scale': 1.0,
        'assembled_pc_gamg_threshold': 0.01,
        'assembled_pc_gamg_coarse_eq_limit': 800,
        "assembled_mg_coarse_ksp_type": "preonly",
        "assembled_mg_coarse_pc_type": "sor",
        "assembled_mg_coarse_ksp_max_it": 10,
        "assembled_mg_coarse_ksp_rtol": 0., # we always want 10 iterations
        "assembled_mg_coarse_ksp_atol": 0., # we always want 10 iterations
        'ksp_rtol': '1e-4',
        'ksp_atol': '1e-15',
        #'ksp_test_transpose_null_space': None,
        #'ksp_test_null_space': None,
        #'mat_null_space_test_view': None,
        #'ksp_converged_reason': None,
        #'ksp_view': None, 
        'ksp_monitor': None,
    },
    'fieldsplit_1': {
        'ksp_type': 'fgmres',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.MassInvPC',
        'Mp_ksp_type': 'cg',
        'Mp_pc_type': 'sor',
        'ksp_rtol': '1e-2',
        #'ksp_test_transpose_null_space': None,
        #'mat_null_space_test_view': None,
        #'ksp_test_null_space': None,
        #'ksp_converged_reason': None,
        #'ksp_view': None, 
        'ksp_monitor': None,
    }
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
    return min(ts_min*target_cfl_no,max_timestep)
                        
#########################################################################################################
############################ T advection diffusion equation Prerequisites: ##############################
#########################################################################################################

# Set up temperature field and initialise based upon coordinates:
T_old    = Function(Q, name="OldTemperature")

#T_old.interpolate(1/(rmax-rmin)*(rmax-r))
#T_old.interpolate(rmax-r + 0.02*cos(3.*atan_2(y,x))*sin((r-rmin)*pi))

# Having a single hot blob on 1.5, 0.0
blb_ctr_h = as_vector((1.5, 0.0)) 
blb_gaus = Constant(0.50)
# A linear temperature profile from the surface to the CMB, with a gaussian blob somewhere
T_old.interpolate(R_max - r + 0.3*exp(-0.5*((X-blb_ctr_h)/blb_gaus)**2));

T_new   = Function(Q, name="Temperature")
T_new.assign(T_old)

# Temporal discretisation - Using a theta scheme:
T_theta = theta_ts * T_new + (1-theta_ts) * T_old


# Setting a radial profile to compute deviations only
T_anom  = Function(Q, name="DevTemperature")
T_ave   = Function(Q, name="Geotherm")
T_ave.interpolate(R_max - r)
# Anomalous Temperature
#########################################################################################################
############################################ Setup Equations ############################################
#########################################################################################################

### Initially deal with Stokes equations ###

# Equation in weak (ufl) form 
mu        = Constant(1.0) # Constant viscosity

# coefficient of the stabilizing term in Nitsche method
beta     = Constant(20) 

# deviatoric stresses
def tau(u): return  mu * (grad(u)+transpose(grad(u)))

# traction field 
def trac(u,p): return dot(tau(u),n) - p*n


# Finalise equations
F_stokes  = inner(grad(N), tau(u)) * dx - div(N)*p * dx 
F_stokes += - (dot(N,rhat)*Ra*T_theta) * dx 
F_stokes += - div(u)* M * dx

# nitsche free slip BCs
nitsche_fs  = - dot(N,n)*dot(n,trac(u,p))*ds_tb\
              - dot(u,n)*dot(n,trac(N,M))*ds_tb\
              + beta/h*dot(u,n)*dot(N,n)*ds_tb

F_stokes += nitsche_fs

### Next deal with Temperature advection-diffusion equation: ###
F_energy = Y * ((T_new - T_old) / delta_t) * dx + Y*dot(u,grad(T_theta)) * dx + dot(grad(Y),kappa*grad(T_theta)) * dx
bct_base = DirichletBC(Q, 1.0, bottom_id)
bct_top  = DirichletBC(Q, 0.0, top_id)


# Write output files in VTK format:
u_file = File('Paraview/velocity.pvd')
p_file = File('Paraview/pressure.pvd')
t_file = File('Paraview/temperature.pvd')

# For some reason this only works here!
u, p    = z.split() # Do this first to extract individual velocity, pressure and lagrange multplier fields:
u.rename('Velocity') 
p.rename('Pressure')



# we will use this to visualize rotational effects
rot_nullspace = Function(V, name='rotation')

# constructing mixed_nullspace, that will be passed to solve 
# the rotational nullspace around z (point of view) for velocity
rotZ = Function(Z)
rotV, _  = rotZ.split()
rotV.interpolate(as_vector((-y, x)))
rotVSpace = VectorSpaceBasis([rotV])
rotVSpace.orthonormalize()

# constant nullspace for pressure
p_nullspace = VectorSpaceBasis(constant=True)

# generating the nullspace for mixed
nullspace_Z = MixedVectorSpaceBasis(Z, [rotVSpace, p_nullspace])

# Generating near_nullspaces for GAMG
#   along x axis
c0V = Function(V)
c0V.interpolate(Constant([1., 0.]))

#   along y axis
c1V = Function(V)
c1V.interpolate(Constant([0., 1.]))

# rotation around Z axis (beacuse using rotV from above again here messes with the nullspaces, probably because we orthnormalize rotVSpace)
cr0V = Function(V)
cr0V.interpolate(as_vector((-y, x)))

# Construction the near_null_modes 
near_nullspace_V = VectorSpaceBasis([cr0V, c0V, c1V])
near_nullspace_V.orthonormalize()
near_nullspace_Z = MixedVectorSpaceBasis(Z, [near_nullspace_V, Z.sub(1)])


# Printing out the degrees of freedom 
log('Number of coefficients for the pressure space (which is kind of number of nodes globally.)', W.dim())

# A simulation time to track how far we are
simu_time = 0.0

# Now perform the time loop:
for timestep in range(0,max_timesteps):
    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0. 
    #solve(F_stokes==0, z, solver_parameters=stokes_solver_parameters_iterative, appctx={'mu': mu}, nullspace=nullspace)
    solve(F_stokes==0, z, solver_parameters=stokes_solver_parameters_iterative, appctx={'mu': mu}, nullspace=nullspace_Z,\
           near_nullspace=near_nullspace_Z)

    # updating time-step based on velocities
    delta_t.assign(compute_timestep(u)) # Compute adaptive time-step

    # Temperature system:
    solve(F_energy==0, T_new, bcs=[bct_base, bct_top], solver_parameters=temperature_solver_parameters)

    # updating time
    simu_time += float(delta_t)

    # Write output:
    if timestep % 50 == 0:
        log("Output:", simu_time, timestep)
        checkpoint_data = DumbCheckpoint(str("T_new_%4.4i" %simu_time), single_file=True, mode=FILE_CREATE, comm=mesh.comm)
        checkpoint_data.set_timestep(t=simu_time, idx=timestep)
        checkpoint_data.store(T_new)
        checkpoint_data.close()
        u_file.write(u)
        p_file.write(p)
        T_anom.assign(T_new - T_ave)
        t_file.write(T_new, T_anom)

    # Set T_old = T_new - assign the values of T_new to T_old
    T_old.assign(T_new)

    # Updating Temperature
    log("Timestep Number: ", timestep, " Timestep: ", float(delta_t))

## Write final state (for adjoint testing):
#checkpoint_data = DumbCheckpoint("Final_State",mode=FILE_CREATE)
#checkpoint_data.store(T_new,name="Temperature")
#checkpoint_data.close()    
