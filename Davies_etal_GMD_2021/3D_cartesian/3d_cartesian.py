from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry
a, b, c = 1.0079, 0.6283, 1.0
nx, ny, nz = 20, int(b/c * 20), 20
mesh2d = RectangleMesh(nx, ny, a, b, quadrilateral=True)  # Rectangular 2D mesh
mesh = ExtrudedMesh(mesh2d, nz)
bottom_id, top_id, left_id, right_id, front_id, back_id = "bottom", "top", 1, 2, 3, 4
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
# Test functions and functions to hold solutions:
v, w = TestFunctions(Z)
q = TestFunction(Q)
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
Told, Tnew = Function(Q, name="OldTemp"), Function(Q, name="NewTemp")
pi = 3.141592653589793238
Told.interpolate(0.5*(erf((1-X[2])*4)+erf(-X[2]*4)+1) + 0.2*(cos(pi*X[0]/a)+cos(pi*X[1]/b))*sin(pi*X[2]))
Tnew.assign(Told)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
Ttheta = 0.5 * Tnew + (1-0.5) * Told

# Define time stepping parameters:
steady_state_tolerance = 1e-7  # Set to 1e-9 for simulations in paper.
max_timesteps = 50000
target_cfl_no = 1.0
maximum_timestep = 0.1
increase_tolerance = 1.5
time = 0.0

# Timestepping - CFL related stuff:
ref_vel = Function(V, name="Reference_Velocity")


def compute_timestep(u, current_delta_t):
    """Return the timestep, based upon the CFL criterion"""

    ref_vel.interpolate(dot(JacobianInverse(mesh), u))
    ts_min = 1. / mesh.comm.allreduce(ref_vel.dat.data.max(), MPI.MAX)
    # Grab (smallest) maximum permitted on all cores:
    ts_max = min(float(current_delta_t)*increase_tolerance, maximum_timestep)
    # Compute timestep:
    tstep = min(ts_min*target_cfl_no, ts_max)
    return tstep


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
        "ksp_rtol": 1e-5,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-4,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_rtol": 1e-5,
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
    }
}

# Energy Equation Solver Parameters:
energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
    "ksp_converged_reason": None,
    "pc_type": "sor",
}


# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(3e4)  # Rayleigh number
mu = Constant(1.0)  # Viscosity
k = Constant((0, 0, 1))  # Unit vector (in direction opposite to gravity).

# Temperature equation related constants:
delta_t = Constant(1e-6)  # Initial time-step
kappa = Constant(1.0)  # Thermal diffusivity

# Stokes equations in UFL form:
stress = 2 * mu * sym(grad(u))
F_stokes = inner(grad(v), stress) * dx - div(v) * p * dx - (dot(v, k) * Ra * Ttheta) * dx
F_stokes += -w * div(u) * dx  # Continuity equation
# Energy equation in UFL form:
F_energy = q * (Tnew - Told) / delta_t * dx + q * dot(u, grad(Ttheta)) * dx + dot(grad(q), kappa * grad(Ttheta)) * dx

# Set up boundary conditions for Stokes: We first extract the velocity
# field from the mixed function space Z. This is done using
# Z.sub(0). We subsequently extract the x and y components of
# velocity. This is done using an additional .sub(0) and .sub(1),
# respectively. Note that the final arguments here are the physical
# boundary ID's.
bcvfb = DirichletBC(Z.sub(0).sub(1), 0, (front_id, back_id))  # Free slip BC on front and back
bcvlr = DirichletBC(Z.sub(0).sub(0), 0, (left_id, right_id))  # Free slip BC on left and right
bcvbt = DirichletBC(Z.sub(0), 0, (bottom_id, top_id))  # Zero slip boundary condition on bottom and top

# Temperature boundary conditions
bctb, bctt = DirichletBC(Q, 1.0, bottom_id), DirichletBC(Q, 0.0, top_id)

# Pressure nullspace
p_nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

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
dump_period = 50
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4
# Open file for logging diagnostic output:
f = open("params.log", "w")

# Setup problem and solver objects so we can reuse (cache) solver setup
stokes_problem = NonlinearVariationalProblem(F_stokes, z, bcs=[bcvbt, bcvfb, bcvlr])
stokes_solver = NonlinearVariationalSolver(stokes_problem, solver_parameters=stokes_solver_parameters, appctx={"mu": mu}, nullspace=p_nullspace, transpose_nullspace=p_nullspace, near_nullspace=Z_near_nullspace)
energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
energy_solver = NonlinearVariationalSolver(energy_problem, solver_parameters=energy_solver_parameters)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        output_file.write(u, p, Tnew)

    current_delta_t = delta_t
    if timestep != 0:
        delta_t.assign(compute_timestep(u, current_delta_t))  # Compute adaptive time-step
    time += float(delta_t)

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    u_rms = sqrt(assemble(dot(u, u) * dx)) * sqrt(1./domain_volume)  # RMS velocity
    u_rms_surf = sqrt(assemble(u[0] ** 2 * ds_t))  # RMS velocity at surface
    bcu = DirichletBC(u.function_space(), 0, top_id)
    ux_max = u.dat.data_ro_with_halos[bcu.nodes, 0].max(initial=0)
    ux_max = u.comm.allreduce(ux_max, MPI.MAX)  # Maximum Vx at surface
    nusselt_number_top = -1 * assemble(dot(grad(Tnew), n) * ds_t) * (1./assemble(Tnew * ds_b))
    nusselt_number_base = assemble(dot(grad(Tnew), n) * ds_b) * (1./assemble(Tnew * ds_b))
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature = assemble(Tnew * dx) / domain_volume

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((Tnew - Told)**2 * dx))

    # Log diagnostics:
    log_params(f, f"{timestep} {time} {float(delta_t)} {maxchange} {u_rms} {u_rms_surf} {ux_max} "
               f"{nusselt_number_base} {nusselt_number_top} "
               f"{energy_conservation} {average_temperature} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Set Told = Tnew - assign the values of Tnew to Told
    Told.assign(Tnew)

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

# Write final state:
final_checkpoint_data = DumbCheckpoint("Final_Temperature_State", mode=FILE_CREATE)
final_checkpoint_data.store(Tnew, name="Temperature")
final_checkpoint_data.close()

final_checkpoint_data = DumbCheckpoint("Final_Stokes_State", mode=FILE_CREATE)
final_checkpoint_data.store(z, name="Stokes")
final_checkpoint_data.close()
