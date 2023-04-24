from gadopt import *
from mpi4py import MPI
from pyroteus import recover_hessian, hessian_metric, compute_eigendecomposition
from firedrake.meshadapt import adapt, RiemannianMetric

# Set up initial mesh
nx, ny = 100, 100
mesh = UnitSquareMesh(nx, ny, quadrilateral=False)  # Square mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

mesh_cnt = 0
timestep = 0

# Create output file and select output_frequency:
output_file = File("output.pvd", adaptive=True)
dump_period = 10

# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms u_rms_surf ux_max nu_top nu_base energy avg_t")


Q = FunctionSpace(mesh, "CG", 1)  # Temperature function space (scalar)
T_init = Function(Q)
X = SpatialCoordinate(mesh)
T_init.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))

# Frequency of checkpoint files:
checkpoint_period = dump_period * 10

def run_interval(mesh, Nt, T_init):
    global timestep, mesh_cnt

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 1)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # Set up temperature field and initialise:
    T = Function(Q, name="Temperature")
    T.project(T_init)

    delta_t = Constant(1e-6)  # Initial time-step
    t_adapt = TimestepAdaptor(delta_t, V, maximum_timestep=0.1, increase_tolerance=1.5)

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(1e6)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    time = 0.0
    steady_state_tolerance = 1e-9
    max_timesteps = 20000
    kappa = Constant(1.0)  # Thermal diffusivity

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Write output files in VTK format:
    u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
    # Next rename for output:
    u.rename("Velocity")
    p.rename("Pressure")

    gd = GeodynamicalDiagnostics(u, p, T, bottom_id, top_id)

    temp_bcs = {
        bottom_id: {'T': 1.0},
        top_id: {'T': 0.0},
    }

    stokes_bcs = {
        bottom_id: {'uy': 0},
        top_id: {'uy': 0},
        left_id: {'ux': 0},
        right_id: {'ux': 0},
    }

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    Told = energy_solver.T_old
    Ttheta = 0.5*T + 0.5*Told
    Told.assign(T)
    stokes_solver = StokesSolver(z, Ttheta, approximation, bcs=stokes_bcs,
                                 cartesian=True,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

    checkpoint_file = CheckpointFile(f"Checkpoint_State_{mesh_cnt}.h5", "w")
    checkpoint_file.save_mesh(mesh)

    # Now perform the time loop:
    for timestep in range(timestep, timestep+Nt):

        # Write output:
        if timestep % dump_period == 0:
            output_file.write(u, p, T)

        dt = t_adapt.update_timestep(u)
        time += dt

        # Solve Stokes sytem:
        stokes_solver.solve()

        # Temperature system:
        energy_solver.solve()

        # Compute diagnostics:
        bcu = DirichletBC(u.function_space(), 0, top_id)
        ux_max = u.dat.data_ro_with_halos[bcu.nodes, 0].max(initial=0)
        ux_max = u.comm.allreduce(ux_max, MPI.MAX)  # Maximum Vx at surface
        nusselt_number_top = gd.Nu_top()
        nusselt_number_base = gd.Nu_bottom()
        energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

        # Calculate L2-norm of change in temperature:
        maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

        # Log diagnostics:
        plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} {gd.u_rms_top()} {ux_max} "
                     f"{nusselt_number_top} {nusselt_number_base} "
                     f"{energy_conservation} {gd.T_avg()} ")

        # Leave if steady-state has been achieved:
        steady_state_reached = maxchange < steady_state_tolerance
        if steady_state_reached:
            log("Steady-state achieved -- exiting time-step loop")
            break

        # Checkpointing:
        if timestep % checkpoint_period == 0:
            checkpoint_file.save_function(T, name="Temperature", idx=timestep)
            checkpoint_file.save_function(z, name="Stokes", idx=timestep)

    checkpoint_file.close()
    mesh_cnt += 1

    return T, u, p, steady_state_reached

metric_file = File('metric.pvd', adaptive=True)

T = T_init
for _ in range(1000):
    T, u, p, steady_state_reached = run_interval(mesh, 10, T)
    if steady_state_reached:
        break

    Hs = [recover_hessian(u.sub(i), method='L2') for i in range(2)]
    metrics = [RiemannianMetric(hessian_metric(H)) for H in Hs]

    metric = RiemannianMetric(mesh)
    metric.rename("metric")
    import numpy as np
    metric.set_parameters(
            {'dm_plex_metric_target_complexity': 10000,
             'dm_plex_metric_p': np.inf,
             #'dm_plex_metric_verbosity': 10,
             'dm_plex_gradation_factor': 1.5,
             'dm_plex_metric_a_max': 10,
             'dm_plex_metric_h_min': 1e-5,
             'dm_plex_metric_h_max': .1})
    metric.intersect(*metrics)
    metric.normalise()
    evecs, evals = compute_eigendecomposition(metric)
    evecs.rename("Eigenvectors")
    evals.rename("Eigenvalues")
    metric_file.write(metric, evecs, evals)
    mesh = adapt(mesh, metric)

plog.close()
