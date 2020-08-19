from firedrake import *
from mpi4py import MPI
import math
from firedrake.petsc import PETSc
from firedrake_adjoint import *
#from pyadjoint.optimization.optimization import minimize
import numpy as np
import pyadjoint
import moola

#### Print function to ensure log output is only written on processor zero (if running in parallel) ####
def log(*args):
    if mesh.comm.rank == 0:
        PETSc.Sys.Print(*args)

# Mesh and associated physical boundary IDs:
mesh    = Mesh('../mesh/shell_v2_cylindrical.msh') # This mesh was generated via gmsh

# Output the number of vertices 
log(" Number of Vertices %i" %mesh.num_vertices())

# Top and bottom curve
bottom_id, top_id       = 312, 311 # When using Gmsh generated mesh
left_id, right_id       = 314, 313 # When using Gmsh generated mesh

# Global Constants:
steady_state_tolerance  = 1e-7
simu_time               = 9e6

# mesh related
X                       = SpatialCoordinate(mesh)
r                       = sqrt(X[0]**2+X[1]**2)
er                      = as_vector((X[0]/r, X[1]/r))
domain_volume           = assemble(1.*dx(domain=mesh))
h                       = sqrt(CellVolume(mesh))
n                       = FacetNormal(mesh)
# physics related: 
# conversion of non-dimensional time to years
time_dim                = (6370e3**2*3.18e-8)/1e-6
# Stokes related constants
mu                      = Constant(1.0)   # Viscosity
Ra                      = Constant(1e8)   # Rayleigh Number
# stabilizer term in Nitsche method
beta     = Constant(10)

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
# we are having fixed time step, so this we don't need now
target_cfl_no           = 1.0
max_timestep            = 0.1
ts_func = Function(Q)

def compute_timestep(u):
    # A function to compute the timestep, based upon the CFL criterion
    ts_func.interpolate(h / sqrt(dot(u,u)))
    ts_min = ts_func.dat.data.min()
    ts_min = mesh.comm.allreduce(ts_min, MPI.MIN)
    return min(ts_min*target_cfl_no,max_timestep)

# Final state, which will be used as reference for minimization 
final_state = Function(Q, name="FinalState")
final_state_file = DumbCheckpoint("../00_fwd_reference/T_final_state", mode=FILE_READ)
final_state_file.load(final_state, 'Temperature')
final_state_file.close()

# Initial condition for tempereature
T_ic   = Function(Q, name="T_IC")
# Set up temperature field and initialise based upon coordinates:
T_old  = Function(Q, name="OldTemperature")
# Function for top boundary prescribed boundary
u_top  = Function(V, name='function_3[0]')

## initialize from reference as well
#T_ic_file = DumbCheckpoint("../00_fwd_reference/T_final_state", mode=FILE_READ)
#T_ic_file.load(T_ic, 'Temperature')
#T_ic_file.close()

T_ic.assign(0.7)

# open file to read top velocities from
top_bdr_file = DumbCheckpoint("../00_fwd_reference/velocities", mode=FILE_READ)
# get all the time steps stored
bdr_times, bdr_idx = top_bdr_file.get_timesteps()

# Below are callbacks relating to the adjoint solutions (accessed through solve).
# Not sure what the best place would be to initiate working tape!
tape = get_working_tape()

# Assign T_ic to the old temperature
T_old.assign(T_ic)

# At the beginning T_new and Old are the same
T_new   = Function(Q, name="Temperature")
T_new.assign(T_old)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
theta   = 0.5
T_theta = theta * T_new + (1-theta) * T_old

# Weak form of stokes (consistent with Freund and Sternberg)
# Rhodri's stoks: 
#       tau = mu * (grad(u)+transpose(grad(u))) 
#       F_stokes  = inner(grad(v), tau) * dx + dot(v,grad(p)) * dx - (dot(v,k)*Ra*T_ic) * dx
#       F_stokes += dot(grad(q),u) * dx # Continuity equation
# stress rate 
def a(u,N): return 2*inner(mu*sym(grad(u)), grad(N))*dx
# pressure/continuity 
def b(N,M): return - div(N)*M*dx
# traction (\sigma(u, p).n)
def t(u,p): return dot(2*mu*sym(grad(u)),n) - p*n
# r.h.s 
def f(N): return (dot(N,er)*Ra*T_theta) * dx
# Nitsche  boundary terms (third term is the stabilizer) 
# (beta/h) = model dependent
def nitsche_freeslip(b_id): return - dot(N,n)*dot(n,t(u,p))*ds(b_id)\
                                   - dot(u,n)*dot(n,t(N,M))*ds(b_id)\
                                   + beta/h*dot(u,n)*dot(N,n)*ds(b_id)
# setting up stokes
F_stokes =  a(u,N) + b(N,p) + b(u,M) - f(N)\
               + nitsche_freeslip(b_id=right_id)\
               + nitsche_freeslip(b_id=top_id)\
               + nitsche_freeslip(b_id=bottom_id)

# setting up energy
F_energy = Y * ((T_new - T_old) / delta_t) * dx + Y*dot(u,grad(T_theta)) * dx + dot(grad(Y),kappa*grad(T_theta)) * dx

# Dirichlet boundary conditions for energy equation (comment them out for no-flux)
# at cmb
#bct_base = DirichletBC(Q, 1.0, bottom_id)
# surface
#bct_top  = DirichletBC(Q, 0.0, top_id)

# Dirichlet boundary conditions for Stokes:
# Comment out FS boundary for the right wall. In this domain, we are not algined with x axis.
# Therefore we use Nitsche method to impose boundary
#bcv_r = DirichletBC(Z.sub(0).sub(1), 0.0, (right_id))
# left side wall is FS
bcv_l = DirichletBC(Z.sub(0).sub(0), 0.0, (left_id))

# top boundary (prescribed from file), emulating imposition of surface velocities from plate reconstructions
bcv_t = DirichletBC(Z.sub(0), u_top, (top_id))
# Bottom id is now free slip using Nitsche method
#bcv_b = DirichletBC(Z.sub(0),        0.0, (bottom_id))


# Write output files in VTK format:
u, p = z.split() # Do this first to extract individual velocity and pressure fields:
u._name = 'Velocity' # This doesn't seem to work!
p._name = 'Pressure'

# Initialise functional
functional = 0.0


# used for output of adjoint files
class DoublyIndexedFile:
    def __init__(self, name):
        self.name = name
        self.index0 = 0
        self.f = None
        self._open()

    def _open(self):
        name = self.name + '_{}.pvd'.format(self.index0)
        self.f = File(name)

    def next(self):
        self.index0 += 1
        self._open()

    def write(self, function):
        self.f.write(function)

lambda_t_pvd   = DoublyIndexedFile('Visual/lambda_t')
lambda_u_pvd   = DoublyIndexedFile('Visual/lambda_u')
lambda_p_pvd   = DoublyIndexedFile('Visual/lambda_p')
#lambda_t_copy  = Function(Q, name='Visual/lambda_t_copy')

def energy_adj_cb(adj_sol):
    adj_sol.rename('lambda_T')
    lambda_t_pvd.write(adj_sol)
#    lambda_t_copy.assign(adj_sol) # Need to think about this and how it links to callbacks below...

def stokes_adj_cb(adj_sol):
    lambda_u, lambda_p = adj_sol.split()
    lambda_u.rename('lambda_u')
    lambda_p.rename('lambda_p')
    lambda_u_pvd.write(lambda_u)
    lambda_p_pvd.write(lambda_p)

# Below are callbacks allowing us to access various field information (accessed through reducedfunctional).
class OptimisationOutputCallbackPost:
    def __init__(self):
        self.iter_idx = 0
        self.opt_gradient_file    = File('Visual/opt_gradient.pvd')
        self.opt_t_ic_file        = File('Visual/opt_temperature_ic.pvd')
        self.opt_t_final_file     = File('Visual/opt_temperature_final.pvd')
        self.opt_u_final_file     = File('Visual/opt_velocity_final.pvd')
        self.opt_p_final_file     = File('Visual/opt_pressure_final.pvd')
        self.grad_copy            = Function(Q, name="Gradient")
        self.T_copy               = Function(Q, name="Temperature")
        self.z_copy               = Function(Z, name="Stokes")
        self.u_copy               = Function(V, name="Velocity")
        self.p_copy               = Function(W, name="Pressure")

    def __call__(self, cb_functional, dj, controls):
        # output current gradient:
        self.grad_copy.assign(dj)
        self.opt_gradient_file.write(self.grad_copy)
        # output current control (temperature initial condition)
        self.T_copy.assign(controls)
        self.opt_t_ic_file.write(self.T_copy)
        # output current final state temperature
        self.T_copy.assign(T_new.block_variable.checkpoint)
        self.opt_t_final_file.write(self.T_copy)
        # output current final state velocity and pressure
        self.z_copy.assign(z.block_variable.checkpoint)
        self.u_copy.assign(self.z_copy.split()[0])
        self.p_copy.assign(self.z_copy.split()[1])
        self.opt_u_final_file.write(self.u_copy)
        self.opt_p_final_file.write(self.p_copy)
        func_val = assemble((self.T_copy-final_state)**2 * dx)
        grad_val = sqrt(assemble((self.grad_copy)**2 * dx))
        log(str('Iteration= %i, functional=' %(self.iter_idx)), func_val, ', grad(dj)=', grad_val)
        log('Starting Derivative calculation', self.iter_idx)
        self.iter_idx += 1

        global lambda_t_pvd, lambda_u_pvd, lambda_p_pvd
        lambda_t_pvd.next()
        lambda_u_pvd.next()
        lambda_p_pvd.next()

class ForwardCallbackPost:
   def __init__(self):
      self.fwd_idx = 0
   def __call__(self, controls):
      self.fwd_idx +=1
      log("Starting fwd calculation:", self.fwd_idx)


local_cb_post = OptimisationOutputCallbackPost()
eval_cb_pre = ForwardCallbackPost()
# control for inversion
control = Control(T_ic)
# Writing out the vtu file
u_file = File('Visual/velocity.pvd')
p_file = File('Visual/pressure.pvd')
t_file = File('Visual/temperature.pvd')##

# initial time and time-step index
my_time = 0.0
k_ind = 0

while my_time < simu_time:
    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    # update boundary values for velocity
    idx_time = abs(my_time - bdr_times).argmin()
    top_bdr_file.set_timestep(t=bdr_times[idx_time], idx=bdr_idx[idx_time])
    top_bdr_file.load(u_top, 'function_3[0]')
    solve(F_stokes==0, z, bcs=[bcv_l], solver_parameters=stokes_solver_parameters)
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


top_bdr_file.close()

alpha_s = Constant(1.)
alpha_r = Constant(0.01)

T_ave = Constant(0.5)

#misfit_scale = assemble(alpha_s/2 * (final_state - T_ave)**2*dx)
#misfit = misfit_scale*assemble(0.5*(T_new - final_state)**2 * dx)
misfit = assemble(0.5*(T_new - final_state)**2 * dx)

#reg_scale = assemble(alpha_r/2 * dot(grad(final_state - T_ave), grad(final_state - T_ave))*dx)
#regulariser = reg_scale*assemble(0.5*dot(grad(T_ic - T_ave), grad(T_ic-T_ave))*dx)
#log('Reguliser scale', float(reg_scale))
#log('Main misfit scale', float(misfit_scale))

# assemble(f), evaluates f in the corresponding space.
# If f is of Form class, then it evaluate the integrals, (I believe like below) 
functional += misfit
#functional += regulariser 
log("Overall Functional:", functional)

### Reduced funcational - functional with respect to the control.
reduced_functional = ReducedFunctional(functional, control, eval_cb_pre=eval_cb_pre, derivative_cb_post=local_cb_post)

## Taylor test:
#Delta_temp   = Function(Q, name="Delta_Temperature")
#Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
#minconv = taylor_test(reduced_functional, T_ic, Delta_temp)
#log(minconv)
#import sys; sys.exit(0)

v_ic = moola.FiredrakePrimalVector(T_ic)
prob = pyadjoint.MoolaOptimizationProblem(reduced_functional, memoize=0)
solver = moola.BFGS(prob, v_ic, options={
        "jtol": 0,
        "rjtol": 1e-8,
        "gtol": 1e-12,
        "maxiter": 1000,
        "display": 3,
        "Hinit": "default",
})
optimal_sol = solver.solve()
optimal_ic = optimal_sol["control"].data


### Write optimal initial condition
optimal_ic_file = File('optimal_ic.pvd')
optimal_ic_file.write(optimal_ic)

### Calculate and write difference between synthetic and optimal_ic:
diff = Function(Q, name="Diff")
diff.assign(T_ic - optimal_ic)
diff_optimal_ic_file = File('diff_optimal_ic.pvd')
diff_optimal_ic_file.write(diff)

# Save the optimal temperature_ic field 
ckpt_T_ic = DumbCheckpoint("T_ic_optimal",\
        single_file=True, mode=FILE_CREATE, comm=mesh.comm)
ckpt_T_ic.store(T_ic)
ckpt_T_ic.close() 

### Write normed difference between synthetic and predicted IC:
ic_state_functional = assemble(diff**2 * dx)
log("Optimized IC State Functional:", ic_state_functional)
