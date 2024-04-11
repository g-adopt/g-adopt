from gadopt import *
from gadopt.gplates import GplatesFunction, pyGplatesConnector
import scipy
import math
from firedrake.__future__ import interpolate

rmin, rmax = 1.22, 2.22


def forward():
    # make sure the mesh is generated
    generate_mesh()

    # Load mesh
    # If initialising start timestepping_history with zero
    # If restarting, then load the information from CheckpointFile
    with CheckpointFile("./simulation_states.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        try:
            timestepping_history = f.get_timestepping_history(
                mesh, name="Temperature")
        except RuntimeError:
            timestepping_history = {"index": [0], "time": [0.0], "delta_t": [1e-9]}

    bottom_id, top_id = "bottom", "top"

    # for accessing the coordinates
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)

    # Set up function spaces using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

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
    delta_t = Constant(timestepping_history.get("delta_t")[-1])  # Initial time step

    # Initialise temperature field
    if timestepping_history.get("index")[-1] == 0:
        T_initialise(T, ((1.0 - (T0*exp(Di) - T0)) * (rmax-r)))
    else:
        with CheckpointFile("./simulation_states.h5", mode="r") as f:
            T.interpolate(
                f.load_function(
                    mesh,
                    name="Temperature",
                    idx=timestepping_history.get("index")[-1]
                )
            )

    plate_reconstruction_model = pyGplatesConnector(
        rotation_filenames=[
            "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/optimisation/1000_0_rotfile_MantleOptimised.rot"
        ],
        topology_filenames=[
            "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/250-0_plate_boundaries.gpml",
            "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/410-250_plate_boundaries.gpml",
            "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Convergence.gpml",
            "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Divergence.gpml",
            "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Topologies.gpml",
            "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Transforms.gpml",
        ],
        nneighbours=4,
        nseeds=1e5,
        oldest_age=10000,
        delta_t=1.0
    )

    # Top velocity boundary condition
    gplates_velocities = GplatesFunction(
        V,
        gplates_connector=plate_reconstruction_model,
        top_boundary_marker=top_id,
        name="GPlates_Velocity"
    )

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

    # constructing viscosity
    # reference background 1d viscosiry
    mu_ref = Function(Q, name="mu2_radial")
    mu_function = Function(Q, name="Viscosity")
    assign_1d_profile(mu_ref, "mu2_radial.rad")
    # temperature and strain-rate dependence added to it
    mu = mu_constructor(mu_ref, T_avg, T, u)

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

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs, su_advection=True)
    energy_solver.fields['source'] = rhobar * H_int
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                                 cartesian=False,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)

    # Modifying default solver tolerances
    energy_solver.solver_parameters['ksp_rtol'] = 1e-4
    stokes_solver.solver_parameters['snes_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    # Write output files in VTK format:
    u_, p_ = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
    u_.rename("Velocity")  # rename the fields
    p_.rename("Pressure")

    # diagnostics
    gd = GeodynamicalDiagnostics(u_, p_, T, bottom_id, top_id)

    # adaptive time-stepper
    t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

    averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)

    # logging diagnostic
    plog = ParameterLog("params.log", mesh)
    plog.log_str("timestep time age dt u_rms t_dev_avg")

    # number of timesteps
    num_timestep = timestepping_history.get("index")[-1]

    # Period for dumping solutions
    dumping_period = 10
    pvd_period = 100

    # non-dimensionalised time for present geologic day (0)
    ndtime_now = plate_receonstion_model.age2ndtime(0)

    # paraview files
    paraview_file = File("ouput.pvd", mode='a')

    time = plate_reconstruction_model.age2ndtime(-100.0) if num_timestep == 0 else timestepping_history.get("time")[-1]

    # Now perform the time loop:
    while time < ndtime_now:
        # Update surface velocities
        gplates_velocities.update_plate_reconstruction(max(time, 0))

        # compute radially averaged temperature profile to update viscosity
        averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

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
            # compute deviation from layer average
            T_dev.assign(T-T_avg)
            mu_function.interpolate(mu)
            paraview_file.write(u_, T_dev, mu_function)

        if num_timestep % dumping_period == 0:
            with CheckpointFile("./simulation_states.h5", mode="a") as chkpoint_file:
                chkpoint_file.save_function(
                    T,
                    name="Temperature",
                    idx=num_timestep,
                    timestepping_info={"time": time, "delta_t": delta_t}
                )
                chkpoint_file.save_function(
                    z,
                    name="Stokes",
                    idx=num_timestep,
                    timestepping_info={"time": time, "delta_t": delta_t}
                )

        # Log diagnostics:
        plog.log_str(f"{num_timestep} {time} {plate_reconstruction_model.ndtime2age(time)} {float(dt)} "
                     f"{gd.u_rms()} {gd.T_avg()}")
    plog.close()


def generate_mesh():
    from pathlib import Path
    if Path("simulation_state.h5").exists():
        return

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

    with CheckpointFile("simulation_states.h5", mode="w") as fi:
        fi.save_mesh(mesh)


def T_initialise(T, average):
    # Initial condition for T:
    # Evaluate P_lm node-wise using scipy lpmv
    X = SpatialCoordinate(T.ufl_domain())
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    theta = atan2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
    phi = atan2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
    l, m, eps_c, eps_s = 6, 4, 0.02, 0.02
    Plm = Function(T.function_space(), name="P_lm")
    cos_phi = assemble(interpolate(cos(phi), T.function_space()))
    Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)
    Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
    if m == 0:
        Plm.assign(Plm/math.sqrt(2))
    T.interpolate(average +
                  (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))


def mu_constructor(mu_radial, Tave, T, u):
    """Constructing a temperature strain-rate dependent velocity

        This uses Arrhenius law for temperature dependent viscosity
        \eta = A exp(-E/R * T) = A * exp(-E/R * Tave) * exp(-E/R * (T - Tave))
                               = 1d_field * exp(-ln(delta_mu_T) * (T - Tave))

    Args:
        mu_radial (firedrake.Form): radial viscosity profile
        T (_type_): Temperature
        Tave (_type_): Average temperature
        u (_type_): velocity

    Returns:
        firedrake.BaseForm: temperature and strain-rate dependent viscosity
    """

    # coordinates
    X = SpatialCoordinate(T.ufl_domain())
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)

    # Adding temperature dependence:
    delta_mu_T = Constant(100.)
    mu_lin = mu_radial * exp(-ln(delta_mu_T) * (T - Tave))

    # Strain-Rate Dependence
    mu_star, sigma_y = Constant(1.0), 5.0e5 + 2.5e6*(rmax-r)
    epsilon = sym(grad(u))  # strain-rate
    epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with a tolerance to ensure stability)
    mu_plast = mu_star + (sigma_y / epsii)
    mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)
    return mu


def assign_1d_profile(q, one_d_filename):
    """
    Assign a one-dimensional profile to a Function `q` from a file.

    The function reads a one-dimensional radial viscosity profile from a file, broadcasts
    the read array to all processes, and then interpolates this
    array onto the function space of `q`.

    Args:
        q (firedrake.Function): The function onto which the 1D profile will be assigned.
        one_d_filename (str): The path to the file containing the 1D radial viscosity profile.

    Returns:
        None: This function does not return a value. It directly modifies the input function `q`.

    Note:
        - This function is designed to be run in parallel with MPI.
        - The input file should contain an array of viscosity values.
        - It assumes that the function space of `q` is defined on a radial mesh.
        - `rmax` and `rmin` should be defined before this function is called, representing
          the maximum and minimum radial bounds for the profile.
    """
    import numpy as np
    from firedrake.ufl_expr import extract_unique_domain
    from scipy.interpolate import interp1d
    # find the mesh
    mesh = extract_unique_domain(q)

    visc = None
    rshl = None
    # read the input file
    if mesh.comm.rank == 0:
        # The root process reads the file
        rshl, visc = np.loadtxt(one_d_filename, unpack=True, delimiter=",")

    # Broadcast the entire 'visc' array to all processes
    visc = mesh.comm.bcast(visc, root=0)
    # Similarly, broadcast 'rshl' if needed (assuming all processes need it)
    rshl = mesh.comm.bcast(rshl, root=0)

    element_family = q.function_space().ufl_element()
    X = Function(VectorFunctionSpace(mesh=mesh, family=element_family)).interpolate(SpatialCoordinate(mesh))
    rad = Function(q.function_space()).interpolate(sqrt(X**2))
    averager = LayerAveraging(mesh, cartesian=False)
    averager.extrapolate_layer_average(q, interp1d(rshl, visc, fill_value="extrapolate")(averager.get_layer_average(rad)))


if __name__ == "__main__":
    forward()
