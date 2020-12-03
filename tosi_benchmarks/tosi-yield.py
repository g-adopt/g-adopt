from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI
import math, numpy

#########################################################################################################
################################## Some important constants etc...: #####################################
#########################################################################################################

mesh   = UnitSquareMesh(160, 160) # This mesh is generated via a firedrake function
bottom_id, top_id = 3, 4 # When using firedrake generated mesh
left_id, right_id = 1, 2 # When using firedrake generated mesh

# Global Constants:
steady_state_tolerance = 5e-8
max_timesteps          = 10000
target_cfl_no          = 2.5
increase_tolerance     = 1.2
maximum_timestep       = 0.1
time                   = 0.0

# Stokes related constants:
lambda_T     = Constant(math.log(10**5))
lambda_Z     = Constant(math.log(10))
eta_star     = Constant(0.001)
yield_stress = Constant(5.0)
Ra           = Constant(100.0)
k            = Constant((0,1)) # Radial unit vector (in direction opposite to gravity):

# Temperature related constants:
delta_t = Constant(1e-6) # Initial time-step
kappa   = Constant(1.0)  # Thermal diffusivity

#### Print function to ensure log output is only written on processor zero (if running in parallel) ####
def log(*args):
    PETSc.Sys.Print(*args)

def log_params(f, str):
    if mesh.comm.rank == 0:
        f.write(str + "\n")
        f.flush()

# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    'mat_type': 'matfree',
#    'snes_type': 'ksponly',
    'snes_type': 'newtonls',
    'snes_linesearch_type': 'bt',
    'snes_max_it': 100,
    'snes_rtol': 1e-1,
    'snes_atol': 1e-50,
    'snes_stol': 1e-5,
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_type': 'full',
    'fieldsplit_0': {
        'ksp_type': 'cg',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_pc_type' : 'gamg',
#        'ksp_converged_reason': None,
        'ksp_rtol': '1e-5',
    },
    'fieldsplit_1': {
        'ksp_type': 'fgmres',
        'ksp_converged_reason': None,
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
E    = FunctionSpace(mesh, "CG", 1) # Strain_rate second invariant function space

# Set up mixed function space and associated test functions:
Z    = MixedFunctionSpace([V, W])
N, M = TestFunctions(Z)
Y    = TestFunction(Q)
E_DG = TestFunction(E)

# Set up fields on these function spaces - split into each component so that they are easily accessible:
z    = Function(Z) # a field over the mixed function space Z.
u, p = split(z)
mu_field       = Function(W, name="Viscosity")
epsii_field    = Function(E, name="Eps_II")

# Timestepping - CFL related stuff:
delta_x = sqrt(CellVolume(mesh))
ts_func = Function(Q) # Note that time stepping should be dictated by Temperature related mesh.

# Initial condition for velocity:
uic_dc = DumbCheckpoint("../../../blankenbach_isoviscous/Ra_1e4/160_Parallel_8/Final_Stokes_State",mode=FILE_READ)
uic_dc.load(z,name="Stokes")
uic_dc.close()

def compute_timestep(u,current_delta_t):
    # A function to compute the timestep, based upon the CFL criterion
    ts_func.interpolate(delta_x / sqrt(dot(u,u)))
    ts_min = ts_func.dat.data.min()
    ts_min = mesh.comm.allreduce(ts_min, MPI.MIN)
    ts_max = min(current_delta_t.dat.data[0]*increase_tolerance,maximum_timestep)
    return min(ts_min*target_cfl_no,ts_max)

#########################################################################################################
############################ T advection diffusion equation Prerequisites: ##############################
#########################################################################################################

# Set up temperature field and initialise based upon coordinates:
X       = SpatialCoordinate(mesh)
T_old   = Function(Q, name="OldTemperature")
tic_dc  = DumbCheckpoint("../../../blankenbach_isoviscous/Ra_1e4/160_Parallel_8/Final_Temperature_State",mode=FILE_READ)
tic_dc.load(T_old,name="Temperature")
tic_dc.close()

T_new   = Function(Q, name="Temperature")
T_new.assign(T_old)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
theta   = 0.5
T_theta = theta * T_new + (1-theta) * T_old

#########################################################################################################
############################################ Setup Equations ############################################
#########################################################################################################

### Initially deal with Stokes equations ###

# Strain-rate:
epsilon    = 0.5 * (grad(u)+transpose(grad(u))) # Strain-rate
epsii      = sqrt(inner(epsilon,epsilon)) # 2nd invariant
# Viscosity
mu_lin     = exp(-lambda_T*T_new + lambda_Z*(1 - X[1])) # with pressure-dependent part
#mu_lin     = exp(-lambda_T * T_new) # temperature-dependent only
mu_plast   = eta_star + (yield_stress / epsii)
mu         = 2. * ( (1./mu_lin) + (1./mu_plast))**-1.
# Equation in weak (ufl) form - note that continuity equation is added here - need to better understand why:
tau        = 2. * mu * epsilon # Stress
F_stokes   = inner(grad(N), tau) * dx + dot(N, grad(p)) * dx - (dot(N,k)*Ra*T_theta) * dx
F_stokes  += dot(grad(M), u) * dx # Continuity equation

# Set up boundary conditions for Stokes: We first need to extract the velocity field from the mixed
# function space Z. This is done using Z.sub(0). We subsequently need to extract the x and y components
# of velocity. This is done using an additional .sub(0) and .sub(1), respectively. Note that the final arguments
# here are the physical boundary ID's.
bcvx = DirichletBC(Z.sub(0).sub(0), 0, (left_id,right_id))
bcvy = DirichletBC(Z.sub(0).sub(1), 0, (bottom_id,top_id))

### Next deal with Temperature advection-diffusion equation: ###
Y_SUPG   = Y + dot(u,grad(Y)) * (delta_x / (2*sqrt(dot(u,u))))
F_energy = Y_SUPG * ((T_new - T_old) / delta_t) * dx + Y_SUPG*dot(u,grad(T_theta)) * dx + dot(grad(Y),kappa*grad(T_theta)) * dx
bct_base = DirichletBC(Q, 1.0, bottom_id)
bct_top  = DirichletBC(Q, 0.0, top_id)

# Pressure nullspace
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# Set up GAMG near nullspace:
rotZ    = Function(Z)
rotV, _ = rotZ.split()
rotV.interpolate(as_vector((-X[1], X[0])))

c0 = Function(Z)
c0V, _ = c0.split()
c1 = Function(Z)
c1V, _ = c1.split()
c0V.interpolate(Constant([1., 0.]))
c1V.interpolate(Constant([0., 1.]))

near_nullmodes = VectorSpaceBasis([c0V, c1V, rotV])
near_nullmodes.orthonormalize()
near_nullmodes_Q = MixedVectorSpaceBasis(Z, [near_nullmodes, Z.sub(1)])

# Write output files in VTK format:
u, p = z.split() # Do this first to extract individual velocity and pressure fields:
u.rename("Velocity")
p.rename("Pressure")
u_file      = File('velocity.pvd')
p_file      = File('pressure.pvd')
t_file      = File('temperature.pvd')
mu_file     = File('viscosity.pvd')
epsii_file  = File('epsII.pvd')

period = 100
f = open('params.log', 'w')
# Now perform the time loop:
for timestep in range(0,max_timesteps):

    current_delta_t = delta_t
    if(timestep != 0):
        delta_t.assign(compute_timestep(u, current_delta_t)) # Compute adaptive time-step
    time += float(delta_t)
    log("Timestep Number: ", timestep, " Timestep: ", float(delta_t), " Time: ", time)

    # Handle Output:
    mu_field.interpolate(mu)
    epsii_field.interpolate(epsii)
    if(timestep == 0 or timestep % period == 1):
        u_file.write(u)
        p_file.write(p)
        t_file.write(T_new)
        mu_file.write(mu_field)
        epsii_file.write(epsii_field)

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    solve(F_stokes==0, z, bcs=[bcvx,bcvy], solver_parameters=stokes_solver_parameters, nullspace=nullspace, near_nullspace=near_nullmodes_Q, appctx={'mu': mu})

    # Temperature system:
    solve(F_energy==0, T_new, bcs=[bct_base,bct_top], solver_parameters=temperature_solver_parameters)

    # Compute diagnostics:
    domain_volume     = assemble(1.*dx(domain=mesh))
    u_rms             = sqrt(assemble(dot(u,u) * dx)) * sqrt(1./domain_volume)
    u_rms_surf        = numpy.sqrt(assemble(u[0] ** 2 * ds(top_id)))    
    bcu               = DirichletBC(u.function_space(), 0, top_id)
    ux_max            = u.dat.data_ro_with_halos[bcu.nodes,0].max(initial=0)
    ux_max            = u.comm.allreduce(ux_max, MPI.MAX)

    # Nusselt number:
    n = FacetNormal(mesh)
    nusselt_number_base = assemble(dot(grad(T_new),n) * ds(bottom_id))
    nusselt_number_top  = assemble(dot(grad(T_new),n) * ds(top_id))
    energy_conservation_1 = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    average_temperature = assemble(T_new * dx) / domain_volume

    max_viscosity = mu_field.dat.data.max()
    max_viscosity = mu_field.comm.allreduce(max_viscosity, MPI.MAX)
    min_viscosity = mu_field.dat.data.min()
    min_viscosity = mu_field.comm.allreduce(min_viscosity, MPI.MIN)    

    rate_work_against_gravity = assemble(T_new * u[1] * dx) / domain_volume
    rate_viscous_dissipation  = assemble(inner(tau, epsilon) * dx) / domain_volume
    energy_conservation_2 = abs(rate_work_against_gravity - rate_viscous_dissipation / Ra.values()[0])

    log("RMS Velocity (domain, surf, surf_max): ", u_rms, u_rms_surf, ux_max)
    log("Nu (base, top, cons): ",nusselt_number_base,nusselt_number_top,energy_conservation_1)
    log("Average Temperature: ", average_temperature)
    log("mu (max, min)", max_viscosity, min_viscosity)
    log("<W>, <phi>, cons: ", rate_work_against_gravity, rate_viscous_dissipation, energy_conservation_2)

    # Calculate L2-norm of change in temperature:
    maxchange = math.sqrt(assemble((T_new - T_old)**2 * dx))

    # Leave if steady-state has been achieved:
    log("Maxchange:",maxchange," Targetting:",steady_state_tolerance)
    log()

    log_params(f, f"{time} {maxchange} {u_rms} {u_rms_surf} {ux_max} {nusselt_number_base} {nusselt_number_top} {average_temperature} {min_viscosity} {max_viscosity} {rate_work_against_gravity} {rate_viscous_dissipation} {energy_conservation_2}")
    
    if(maxchange < steady_state_tolerance):
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Set T_old = T_new - assign the values of T_new to T_old
    T_old.assign(T_new)

f.close()   

# Write final state (for adjoint testing):
checkpoint_data = DumbCheckpoint("Final_State",mode=FILE_CREATE)
checkpoint_data.store(T_new,name="Temperature")
checkpoint_data.close()    
