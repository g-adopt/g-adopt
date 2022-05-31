from gadopt import *

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
rmin, rmax, ncells, nlayers = 1.22, 2.22, 128, 32

# Construct a circle mesh and then extrude into a cylinder:
mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type='radial')
bottom_id, top_id = "bottom", "top"
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
domain_volume = assemble(1*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)


# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Test functions and functions to hold solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
T = Function(Q, name="Temperature")
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)
T.interpolate(rmax - r + 0.02*cos(4*atan_2(X[1], X[0])) * sin((r - rmin) * pi))

Ra = Constant(1e5)  # Rayleigh number

delta_t = Constant(1e-7)  # Initial time-step
t_adapt = TimestepAdaptor(delta_t, V, maximum_timestep=0.1, increase_tolerance=1.5)

# Define time stepping parameters:
steady_state_tolerance = 1e-9
max_timesteps = 20000
time = 0.0

# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "snes_monitor": None,
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

# Nullspaces and near-nullspaces:
x_rotV = Function(V).interpolate(as_vector((-X[1], X[0])))
V_nullspace = VectorSpaceBasis([x_rotV])
V_nullspace.orthonormalize()
p_nullspace = VectorSpaceBasis(constant=True)  # Constant nullspace for pressure n
Z_nullspace = MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])  # Setting mixed nullspace

# Generating near_nullspaces for GAMG:
nns_x = Function(V).interpolate(Constant([1., 0.]))
nns_y = Function(V).interpolate(Constant([0., 1.]))
V_near_nullspace = VectorSpaceBasis([nns_x, nns_y, x_rotV])
V_near_nullspace.orthonormalize()
Z_near_nullspace = MixedVectorSpaceBasis(Z, [V_near_nullspace, Z.sub(1)])

# Write output files in VTK format:
u, p = z.split()  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output.pvd")
dump_period = 1
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4
# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)

temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}

energy_solver = EnergySolver(T, u, delta_t, ImplicitMidpoint, bcs=temp_bcs, solver_parameters=energy_solver_parameters)
Told = energy_solver.T_old
Ttheta = 0.5*T + 0.5*Told
Told.assign(T)
stokes_solver = StokesSolver(z, Ttheta, delta_t, bcs=stokes_bcs, Ra=Ra,
                             cartesian=False,
                             solver_parameters=stokes_solver_parameters,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        output_file.write(u, p, T)

    if timestep != 0:
        dt = t_adapt.update_timestep(u)
    else:
        dt = float(delta_t)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    u_rms = sqrt(assemble(dot(u, u) * dx)) * sqrt(1./domain_volume)
    f_ratio = rmin/rmax
    top_scaling = -1.3290170684486309  # log(f_ratio) / (1.- f_ratio)
    bot_scaling = -0.7303607313096079  # (f_ratio * log(f_ratio)) / (1.- f_ratio)
    nusselt_number_top = (assemble(dot(grad(T), n) * ds_t) / assemble(Constant(1.0, domain=mesh)*ds_t)) * top_scaling
    nusselt_number_base = (assemble(dot(grad(T), n) * ds_b) / assemble(Constant(1.0, domain=mesh)*ds_b)) * bot_scaling
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature = assemble(T * dx) / domain_volume

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {u_rms} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {average_temperature} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

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
