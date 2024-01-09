from gadopt import *
from gadopt.gplates import pyGplatesConnector

rmin, rmax = 1.22, 2.22


def forward():
    # see if there are any checkpointfiles
    state_id = find_latest_id()

    # make sure the mesh is generated
    if state_id is None:
        generate_mesh()

    # Load mesh
    with CheckpointFile(f"./states_{0 if state_id is None else state_id}.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    bottom_id, top_id = "bottom", "top"

    # for accessing the coordinates
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)

    # Set up function spaces using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Test functions and functions to hold solutions:
    v, w = TestFunctions(Z)
    z = Function(Z)
    u, p = split(z)

    # Temperature field
    T = Function(Q, name="Temperature")

    # For computing layer averages
    T_avg = Function(Q, name='TemperatureProfile')  # Average temperature field
    T_dev = Function(Q, name='DeviatoricTemperature')  # Deviatoric temperature field

    # non-dimensionalise constants
    T0 = Constant(0.091)  # Non-dimensional surface temperature
    Di = Constant(0.5)  # Dissipation number.
    H_int = Constant(10.0)  # Internal heating
    Ra = Constant(5.0e6)  # Rayleigh number
    Di = Constant(0.5)  # Dissipation number.
    delta_t = Constant(1.0e-9)  # Initial time step

    # Initialise temperature field
    if state_id is None:
        old_indices, old_times = [0], [0.]
        T_initialise(T, ((1.0 - (T0*exp(Di) - T0)) * (rmax-r)))
    else:
        with CheckpointFile(f"./states_{find_latest_id()}.h5", mode="r") as f:
            old_indices, old_times = f.get_timesteps(mesh, name="Temperature")
            T.interpolate(f.load_function(mesh, name="Temperature", idx=old_indices[-1]))

    # Top velocity boundary condition
    gplates_velocities = Function(V, name="GPlates_Velocity")

    # Compressible reference state:
    rho_0 = 1.0
    alpha = 1.0
    rhobar = Function(Q, name="CompRefDensity").interpolate(
        rho_0 * exp(((1.0 - (r-rmin)) * Di) / alpha))
    Tbar = Function(Q, name="CompRefTemperature").interpolate(
        T0 * exp((1.0 - (r-rmin)) * Di) - T0)
    alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
    chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)

    approximation = TruncatedAnelasticLiquidApproximation(
        Ra, Di, rho=rhobar, Tbar=Tbar, alpha=alphabar, chi=chibar, cp=cpbar)

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False)
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2])

    # Boundary conditions
    temp_bcs = {
        bottom_id: {'T': 1.0 - (T0*exp(Di) - T0)},
        top_id: {'T': 0.0},
    }
    stokes_bcs = {
        top_id: {'u': gplates_velocities},
        bottom_id: {'un': 0},
    }

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)  # , su_advection=True)
    energy_solver.fields['source'] = rhobar * H_int
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu_constructor(T, u),
                                 cartesian=False,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)

    # Modifying default solver tolerances
    energy_solver.solver_parameters['ksp_rtol'] = 1e-4
    stokes_solver.solver_parameters['snes_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    # Initiating a plate reconstruction model
    pl_rec_model = pyGplatesConnector(
        rotation_filenames=[
            './gplates_files/Zahirovic2022_CombinedRotations_fixed_crossovers.rot'],
        topology_filenames=[
            './gplates_files/Zahirovic2022_PlateBoundaries.gpmlz',
            './gplates_files/Zahirovic2022_ActiveDeformation.gpmlz',
            './gplates_files/Zahirovic2022_InactiveDeformation.gpmlz'],
        dbc=stokes_solver.strong_bcs[0],
        geologic_zero=409,
        delta_time=1.0
    )

    # Write output files in VTK format:
    u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
    u.rename("Velocity")  # rename the fields
    p.rename("Pressure")

    # diagnostics
    gd = GeodynamicalDiagnostics(u, p, T, bottom_id, top_id)

    # adaptive time-stepper
    t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

    averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)

    # logging diagnostic
    plog = ParameterLog(f"params_{0 if state_id is None else state_id + 1}.log", mesh)
    plog.log_str("timestep time dt u_rms t_dev_avg")

    # number of timesteps
    num_timestep = old_indices[-1]

    # Period for dumping solutions
    dumping_period = 10
    pvd_period = 50

    # non-dimensionalised time for present geologic day (0)
    ndtime_now = pl_rec_model.geotime2ndtime(0)

    # what index will we use for checkpointing
    new_state_id = 0 if state_id is None else state_id + 1

    # A checkpoint file for storing data
    with CheckpointFile(f"states_{new_state_id}.h5", mode="w") as chkpoint_file:
        chkpoint_file.save_mesh(mesh)
    paraview_file = File(f"visual_{new_state_id}/output.pvd")

    time = old_times[-1]

    # Now perform the time loop:
    while time < ndtime_now:
        # Update surface velocities
        pl_rec_model.assign_plate_velocities(time)

        # Solve Stokes system:
        stokes_solver.solve()

        # Adapt time step
        if num_timestep != 0:
            dt = t_adapt.update_timestep()
        else:
            dt = float(delta_t)
        time += dt

        # Make sure we are not going past present day
        if ndtime_now - time < float(dt):
            dt = ndtime_now - time

        # Temperature system:
        energy_solver.solve()

        time += float(delta_t)
        num_timestep += 1

        # Write output:
        if num_timestep % pvd_period == 0:

            # compute radially averaged temperature profile
            averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))
            # compute deviation from layer average
            T_dev.assign(T-T_avg)
            paraview_file.write(u, p, T, T_dev)

        if num_timestep % dumping_period == 0:
            with CheckpointFile(f"states_{new_state_id}.h5", mode="a") as chkpoint_file:
                chkpoint_file.save_function(T, name="Temperature", idx=num_timestep, timetag=time)

        # Log diagnostics:
        plog.log_str(f"{num_timestep} {time} {float(dt)} "
                     f"{gd.u_rms()} {gd.T_avg()}")
    plog.close()


def generate_mesh():

    # Set up geometry:
    ref_level, nlayers = 7, 64

    # Variable radial resolution
    # Initiating layer heights with 1.
    resolution_func = np.ones((nlayers))

    # A gaussian shaped function
    def gaussian(center, c, a):
        return a * np.exp(
            -((np.linspace(rmin, rmax, nlayers) - center) ** 2) / (2 * c**2)
        )

    # building the resolution function
    for idx, r_0 in enumerate([rmin, rmax, rmax - 660 / 6370]):
        # gaussian radius
        c = 0.15
        # how different is the high res area from low res
        res_amplifier = 5.0
        resolution_func *= 1 / (1 + gaussian(center=r_0, c=c, a=res_amplifier))

    # Construct a CubedSphere mesh and then extrude into a sphere - note that unlike cylindrical case, popping is     done internally here:
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=(rmax - rmin) * resolution_func / np.sum(resolution_func),
        extrusion_type="radial",
    )

    with CheckpointFile("states_0.h5", mode="w") as fi:
        fi.save_mesh(mesh)


def T_initialise(T, average):
    import scipy
    import math
    # Initial condition for T:
    # Evaluate P_lm node-wise using scipy lpmv
    X = SpatialCoordinate(T.ufl_domain())
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    theta = atan2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
    phi = atan2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
    l, m, eps_c, eps_s = 6, 4, 0.02, 0.02
    Plm = Function(T.function_space(), name="P_lm")
    cos_phi = interpolate(cos(phi), T.function_space())
    Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)
    Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
    if m == 0:
        Plm.assign(Plm/math.sqrt(2))
    T.interpolate(average +
                  (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))


def mu_constructor(T, u):
    def step_func(r, center, mag, increasing=True, sharpness=30):
        """
        A step function designed to control viscosity jumps:
        input:
          r: is the radius array
          center: radius of the jump
          increasing: if True, the jump happens towards lower r, otherwise jump happens at higher r
          sharpness: how sharp should the jump should be (larger numbers = sharper).
        """
        if increasing:
            sign = 1
        else:
            sign = -1
        return mag * (0.5 * (1 + tanh(sign*(r-center)*sharpness)))

    # a constant mu
    mu_lin = 2.0

    # coordinates
    X = SpatialCoordinate(T.ufl_domain())
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)

    # Depth dependence: for the lower mantle increase we
    # multiply the profile with a linear function
    for line, step in zip([5.*(rmax-r), 1., 1.],
                          [step_func(r, 1.992, 30, False),
                           step_func(r, 2.078, 10, False),
                           step_func(r, 2.2, 10, True)]):
        mu_lin += line*step

    # Adding temperature dependence:
    delta_mu_T = Constant(100.)
    mu_lin *= exp(-ln(delta_mu_T) * T)
    mu_star, sigma_y = Constant(1.0), 5.0e5 + 2.5e6*(rmax-r)
    epsilon = sym(grad(u))  # strain-rate
    epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with a tolerance to ensure stability)
    mu_plast = mu_star + (sigma_y / epsii)
    mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)
    return mu


def find_latest_id():
    import glob

    file_names = glob.glob("states_[0-9]*.h5")
    if len(file_names) != 0:
        file_names = sorted(file_names, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        return int(file_names[-1].split("_")[-1].split(".")[0])
    else:
        return None


if __name__ == "__main__":
    forward()
