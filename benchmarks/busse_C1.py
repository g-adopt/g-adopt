from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI

# Set up geometry
a, b, c         = 1.0079, 0.6283, 1.0
nx, ny, nz      = 20, int(b/c * 20), 20
mesh2d          = RectangleMesh(nx, ny, a, b, quadrilateral=True) # Square mesh generated via firedrake
mesh            = ExtrudedMesh(mesh2d, nz)
bottom_id, top_id, left_id, right_id, front_id, back_id  = "bottom", "top", 1, 2, 3, 4
n      = FacetNormal(mesh) # Normals, required for Nusselt number calculation
X      = SpatialCoordinate(mesh)  # Coordinate field
domain_volume = assemble(1.*dx(domain=mesh)) # Required for diagnostics (e.g. RMS velocity)

# Parameters relating to time-stepping:
steady_state_tolerance = 1e-8
max_timesteps          = 25000
target_cfl_no          = 2.5
maximum_timestep       = 0.1
increase_tolerance     = 1.2
time                   = 0.0
dim                    = 3

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
mu      = Constant(1.0)   # Viscosity - constant for this isoviscous case.
Ra      = Constant(3e4)   # Rayleigh number.
k       = Constant((0, 0, 1)) # Radial unit vector (in direction opposite to gravity).

# Temperature equation related constants:
delta_t = Constant(1e-6) # Initial time-step.
kappa   = Constant(1.0)  # Thermal diffusivity.

#### Print function to ensure log output is only written on processor zero (if running in parallel) ####
def log(*args):
    PETSc.Sys.Print(*args)

#### File for logging output diagnostics through simulation.
def log_params(f, str):
    if mesh.comm.rank == 0:
        f.write(str + "\n")
        f.flush()

#########################################################################################################
######################################### Solver parameters:  ###########################################
#########################################################################################################

# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": {
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_pc_gamg_threshold_scale": 1.0,
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_coarse_eq_limit": 800,
        "ksp_rtol": 1e-5,     
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
        "ksp_rtol": 1e-4,
    }
}

# Temperature Equation Solver Parameters:
temperature_solver_parameters = {
        "snes_type": "ksponly",
        "ksp_rtol": 1e-5,
        "ksp_type": "gmres",
        "pc_type": "sor",
        "mat_type": "aij"
}

#########################################################################################################
######################################## Spatial Discretization: ########################################
#########################################################################################################

# Set up function spaces - currently using the trilinear Q2Q1 element pair:
V    = VectorFunctionSpace(mesh, "CG", 2) # Velocity function space (vector)
W    = FunctionSpace(mesh, "CG", 1) # Pressure function space (scalar)
Q    = FunctionSpace(mesh, "CG", 2) # Temperature function space (scalar)

log("Number of Velocity DOF:", V.dim()*3)
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()*3+W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up mixed function space and associated test functions:
Z    = MixedFunctionSpace([V, W])
N, M = TestFunctions(Z)
Y    = TestFunction(Q)

# Set up fields on these function spaces - split into each component so that they are easily accessible:
z    = Function(Z) # a field over the mixed function space Z.
u, p = split(z) # returns a symbolic UFL expression for velocity (u) and pressure (p)

#########################################################################################################
############################################ Time Stepping: #############################################
#########################################################################################################

# Timestepping - CFL related stuff:
delta_x = sqrt(CellVolume(mesh))
ts_func = Function(Q) # Note that time stepping should be dictated by Temperature related mesh.

def compute_timestep(u, current_delta_t):
    # A function to compute the timestep, based upon the CFL criterion.
    ts_func.interpolate(delta_x / sqrt(dot(u,u)))
    ts_min = ts_func.dat.data.min()
    ts_min = mesh.comm.allreduce(ts_min, MPI.MIN)
    ts_max = min(current_delta_t.dat.data[0]*increase_tolerance, maximum_timestep)
    return min(ts_min*target_cfl_no, ts_max)

#########################################################################################################
############################ T advection diffusion equation Prerequisites: ##############################
#########################################################################################################

# Set up temperature field and initialise:
T_old   = Function(Q, name="OldTemperature")
pi      = 3.141592653589793238
T_old.interpolate((1 - X[2]) + 0.1*(cos(pi*X[0]/a) + cos(pi*X[1]/b))*sin(pi*X[2]))

# To start set T_new to T_old.
T_new   = Function(Q, name="Temperature")
T_new.assign(T_old)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
theta   = 0.5
T_theta = theta * T_new + (1-theta) * T_old

#########################################################################################################
############################################ Setup Equations ############################################
#########################################################################################################

### Initially deal with Stokes equations ###

# Equation in weak (ufl) form - note that continuity equation is added here - need to better understand why:
stress    = 2.*mu*sym(grad(u))
F_stokes  = inner(grad(N), stress) * dx + dot(N,grad(p)) * dx - (dot(N,k)*Ra*T_theta) * dx
F_stokes += dot(grad(M),u) * dx # Continuity equation

# Set up boundary conditions for Stokes: We first extract the velocity
# field from the mixed function space Z. This is done using
# Z.sub(0). We subsequently extract the x and y components of
# velocity. This is done using an additional .sub(0) and .sub(1),
# respectively. Note that the final arguments here are the physical
# boundary ID's.
bcv_fb = DirichletBC(Z.sub(0).sub(1), 0, (front_id, back_id)) # Free slip BC on front and back
bcv_lr = DirichletBC(Z.sub(0).sub(0), 0, (left_id, right_id))  # Free slip BC on left and right
bcv_bt = DirichletBC(Z.sub(0), 0, (bottom_id,top_id)) # Zero slip boundary condition on bottom and top

# Pressure nullspace
p_nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# Next deal with Temperature advection-diffusion equation - in this example, we do not use stabilisation:
F_energy = Y * ((T_new - T_old) / delta_t) * dx + Y * dot(u,grad(T_theta)) * dx + dot(grad(Y),kappa*grad(T_theta)) * dx

# Temperature boundary conditions
bct_base = DirichletBC(Q, 1.0, bottom_id)
bct_top  = DirichletBC(Q, 0.0, top_id)

# Write output files in VTK format:
u, p = z.split() # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output.pvd")
dump_period = 50
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4
# Open file for logging diagnostic output:
f = open("params.log", "w")

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    current_delta_t = delta_t
    if timestep != 0:
        delta_t.assign(compute_timestep(u, current_delta_t)) # Compute adaptive time-step
    time += float(delta_t)

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    solve(F_stokes==0, z, bcs=[bcv_bt, bcv_fb, bcv_lr], solver_parameters=stokes_solver_parameters, appctx={'mu': mu}, nullspace=p_nullspace)

    # Temperature system:
    solve(F_energy==0, T_new, bcs=[bct_base,bct_top], solver_parameters=temperature_solver_parameters)

    # Write output:
    if timestep == 0 or timestep % dump_period == 0:
        output_file.write(u, p, T_new)

    # Compute diagnostics:
    u_rms                 = sqrt(assemble(dot(u,u) * dx)) * sqrt(1./domain_volume) # RMS velocity
    u_rms_surf            = sqrt(assemble(u[0] ** 2 * ds_t)) # RMS velocity at surface
    bcu                   = DirichletBC(u.function_space(), 0, top_id)
    ux_max                = u.dat.data_ro_with_halos[bcu.nodes,0].max(initial=0)
    ux_max                = u.comm.allreduce(ux_max, MPI.MAX) # Maximum Vx at surface
    nusselt_number_top    = -1 * assemble(dot(grad(T_new),n) * ds_t) * (1./assemble(T_new * ds_b))
    nusselt_number_base   = assemble(dot(grad(T_new),n) * ds_b) * (1./assemble(T_new * ds_b))
    energy_conservation   = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature   = assemble(T_new * dx) / domain_volume

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T_new - T_old)**2 * dx))

    # Log diagnostics:
    log_params(f, f"{timestep} {time} {maxchange} {u_rms} {u_rms_surf} {ux_max} "
               f"{nusselt_number_base} {nusselt_number_top} "
               f"{energy_conservation} {average_temperature} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Set T_old = T_new - assign the values of T_new to T_old
    T_old.assign(T_new)

    # Checkpointing:
    if timestep % checkpoint_period == 0:
	# Checkpointing during simulation:
        checkpoint_data = DumbCheckpoint(f"Temperature_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(T_new, name="Temperature")
        checkpoint_data.close()

        checkpoint_data = DumbCheckpoint(f"Stokes_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(z, name="Stokes")
        checkpoint_data.close()

f.close()

# Write final state:
final_checkpoint_data = DumbCheckpoint("Final_Temperature_State", mode=FILE_CREATE)
final_checkpoint_data.store(T_new, name="Temperature")
final_checkpoint_data.close()

final_checkpoint_data = DumbCheckpoint("Final_Stokes_State", mode=FILE_CREATE)
final_checkpoint_data.store(z, name="Stokes")
final_checkpoint_data.close()

