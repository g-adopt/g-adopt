from gadopt import *
from animate import RiemannianMetric, adapt

# ADAPTIVITY first we set up anything mesh-independent,
# and anything associated with the initial mesh

# Set up initial mesh
nx, ny = 10, 10
mesh = UnitSquareMesh(nx, ny, quadrilateral=False)  # Square mesh generated via firedrake

timestep = 0

timesteps_per_adapt = 10

# Create output file and select output_frequency:
# ADAPTIVITY NOTE: need adaptive=True with multiple meshes in pvd series
output_file = VTKFile("output.pvd", adaptive=True)
output_frequency = 1


# Open file for logging diagnostic output:
# ADAPTIVITY: can use initial mesh here as it only uses mesh.comm
plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms u_rms_surf ux_max nu_top nu_base energy avg_t")


# ADAPTIVITY: set up initial conditions on the initial mesh
Q = FunctionSpace(mesh, "CG", 1)  # Temperature function space (scalar)
T_init = Function(Q, name="InitialTemperature")
X = SpatialCoordinate(mesh)
T_init.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))
u_init = Function(VectorFunctionSpace(mesh, "CG", 1), name="InitialVelocity")


def run_interval(mesh, timestep, Nt, T_init, u_init, p_init):
    """Run for Nt timesteps on the given mesh

    This sets up all the usual functionspace, equations, solvers, etc.
    on the given mesh, and runs the timeloop for the given n/o timesteps.
    Everything exactly like in the base case.

    args:
    mesh - mesh to solve the equations on
    timestep - starting timestep (for logging purposes)
    Nt - n/o timesteps to run
    T_init, u_init - initial conditions, or last solution from previous mesh
                     we interpolate this solution onto the current mesh
    """

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 1)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    z = Function(Z)  # A field over the mixed function space Z.

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # Set up temperature field and initialise:
    T = Function(Q, name="Temperature")
    # ADAPTIVITY: could use project here as well
    T.interpolate(T_init)
    u, p = z.subfunctions
    u.interpolate(u_init)
    # ADAPTIVITY: also interpolate pressure for initial guess
    # remind me, did we do this in Fluidity?
    p.interpolate(p_init)

    delta_t = Constant(1e-6)  # Initial time-step
    t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(1e5)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    time = 0.0
    steady_state_tolerance = 1e-9

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Next rename for output:
    u.rename("Velocity")
    p.rename("Pressure")

    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)
    gd = GeodynamicalDiagnostics(z, T, boundary.bottom, boundary.top)

    temp_bcs = {
        boundary.bottom: {'T': 1.0},
        boundary.top: {'T': 0.0},
    }

    stokes_bcs = {
        boundary.bottom: {'uy': 0},
        boundary.top: {'uy': 0},
        boundary.left: {'ux': 0},
        boundary.right: {'ux': 0},
    }

    energy_solver = EnergySolver(T, u, approximation, delta_t, BackwardEuler, bcs=temp_bcs)
    Told = energy_solver.T_old
    Ttheta = 0.5*T + 0.5*Told
    Told.assign(T)
    stokes_solver = StokesSolver(z, Ttheta, approximation, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

    # Now perform the time loop:
    for ts in range(timestep, timestep+Nt):

        # Write output:
        if ts % output_frequency == 0:
            output_file.write(u, p, T)

        dt = t_adapt.update_timestep()
        time += dt

        # Solve Stokes sytem:
        stokes_solver.solve()

        # Temperature system:
        energy_solver.solve()

        # Compute diagnostics:
        energy_conservation = abs(abs(gd.Nu_top()) - abs(gd.Nu_bottom()))

        # Calculate L2-norm of change in temperature:
        maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

        # Log diagnostics:
        plog.log_str(f"{ts} {time} {float(delta_t)} {maxchange} "
                     f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(boundary.top)} {gd.Nu_top()} "
                     f"{gd.Nu_bottom()} {energy_conservation} {gd.T_avg()} ")

        # Leave if steady-state has been achieved:
        steady_state_reached = maxchange < steady_state_tolerance
        if steady_state_reached:
            log("Steady-state achieved -- exiting time-step loop")
            break

    return T, u, p, steady_state_reached


# ADAPTIVITY: output metric tensor for diagnosing adaptivity
metric_file = VTKFile('metric.pvd', adaptive=True)

T = T_init
u = u_init
p = 0
# ADAPTIVITY loop, we set an arbitrary maximum of 1000
# but should reach steady state well before that (at Ra=1e4)
for _ in range(1000):
    T, u, p, steady_state_reached = run_interval(mesh, timestep, timesteps_per_adapt, T, u, p)
    timestep = timestep + timesteps_per_adapt
    if steady_state_reached:
        break

    metric_parameters = {
        # metric gets rescaled s.t. we always end up with ~ 1000 vertices:
        'dm_plex_metric_target_complexity': 1000,
        'dm_plex_metric_p': np.inf,  # see note (*) below
        'dm_plex_metric_gradation_factor': 1.5,  # gradation factor
        'dm_plex_metric_a_max': 10,  # maximum aspect ratio
        'dm_plex_metric_h_min': 1e-5,  # minimum edge length
        'dm_plex_metric_h_max': .1  # maximum edge length
    }
    # (*) metric_p is the order of the Lp-norm used in rescaling
    # np.inf corresponds to the default in fluidity: if we are using
    # the Hessian of the solution field, we are saying that
    # the estimated interpolation error (based on the quadratic Taylor
    # term using the Hessian) should be smaller than 1 everywhere, or
    # smaller than eps if we rescale the metric dividing it by eps. That is,
    # we are applying a infinity-norm to the estimated interpolation error.
    # Note that the rescaling here is done automatically by specifying a target
    # complexity (you could also leave out that options and indeed scale the
    # metric by eps^{-1} if you have some specific eps in mind).
    # Instead of the inf-norm we can also base the rescaling on a different p,
    # e.g. p=2 for L2. In combination with a specified target complexity
    # this means that for the given n/o vertices you get the mesh that minimizes
    # the L2 interpolation error, rather than minimizing the Linf interpolation
    # error. In practice this means it distributes that it focusses a little less
    # on the needed resolution where the jumps are the largest, to reduce the
    # global integrated error.

    TV = TensorFunctionSpace(mesh, "CG", 1)  # function space for the metric
    # first generate a metric based on the Hessian for each velocity component
    metrics = []
    for i in range(2):
        # a RiemannianMetric is just firedrake Function on a tensor function space
        # with additional functionality
        H = RiemannianMetric(TV)
        H.set_parameters(metric_parameters)
        H.compute_hessian(u.sub(i), method='L2')
        H.enforce_spd()  # this uses h_min and h_max to ensure the metric is bounded
        metrics.append(H)

    # we use the first as *the* metric
    metric = metrics[0]
    metric.rename("metric")
    # which we intersect with the others (only one other here)
    metric.intersect(*metrics[1:])

    # this applies the rescaling
    metric.normalise()

    # this decomposition in eigenvectors and eigenvalues is for diagnositc purposes only
    evecs, evals = metric.compute_eigendecomposition()
    evecs.rename("Eigenvectors")
    evals.rename("Eigenvalues")

    # we're done: write out the metric and adapt the mesh!
    metric_file.write(metric, evecs, evals)
    mesh = adapt(mesh, metric)

plog.close()
