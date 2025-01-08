from gadopt import *
from gadopt.inverse import *
from gadopt.gplates import GplatesFunction, pyGplatesConnector
import numpy as np
from firedrake.adjoint_utils import blocks
from pyadjoint import stop_annotating
from pathlib import Path
import gdrift

# Quadrature degree:
dx = dx(degree=6)

# Projection solver parameters for nullspaces:
iterative_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
}

LinearSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
NonlinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
LinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
NonlinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters

blocks.solving.Block.evaluate_adj = collect_garbage(blocks.solving.Block.evaluate_adj)
blocks.solving.Block.recompute = collect_garbage(blocks.solving.Block.recompute)

# timer decorator for fwd and derivative calls.
ReducedFunctional.__call__ = collect_garbage(
    timer_decorator(ReducedFunctional.__call__)
)
ReducedFunctional.derivative = collect_garbage(
    timer_decorator(ReducedFunctional.derivative)
)

# Set up geometry:
rmax = 2.208
rmin = 1.208
ref_level = 7
nlayers = 64


def conduct_inversion():
    Tic, reduced_functional = forward_problem()

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(Tic.function_space(), name="Lower bound temperature")
    T_ub = Function(Tic.function_space(), name="Upper bound temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 1.0e-1
    minimisation_parameters["Status Test"] = 20

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))

    # Start a LinMore Optimiser
    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )

    visualisation_path = find_last_checkpoint().resolve().parents[2] / "visual.pvd"

    vtk_file = VTKFile(str(visualisation_path))
    control_container = Function(Tic.function_space(), name="Initial Temperature")

    def callback():
        control_container.assign(Tic.block_variable.checkpoint.restore())
        vtk_file.write(control_container)

    #
    optimiser.add_callback(callback)
    # run the optimisation
    optimiser.run()


def conduct_taylor_test():
    Tic, reduced_functional = forward_problem()
    log("Reduced Functional Repeat: ", reduced_functional([Tic]))
    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    _ = taylor_test(reduced_functional, Tic, Delta_temp)


@collect_garbage
def forward_problem():
    # Enable disk checkpointing for the adjoint
    enable_disk_checkpointing()

    # Set up the base path
    base_path = Path(__file__).resolve().parent

    # Here we are managing if there is a last checkpoint from previous runs
    # to load from to restart out simulation from.
    last_checkpoint_path = find_last_checkpoint()

    # Load mesh
    with CheckpointFile(str(base_path / "input_data/REVEAL.h5"), "r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")
        Tobs = fi.load_function(mesh, name="Tobs")  # reference temperature field (seismic tomography)
        Tave = fi.load_function(mesh, name="AverageTemperature")  # Average temperature field

    # Boundary markers to top and bottom
    bottom_id, top_id = "bottom", "top"

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "DQ", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Initial Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.
    R = FunctionSpace(mesh, "R", 0)  # Function space for constants

    # Function for holding stokes results
    z = Function(Z)
    u, p = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    # Set up temperature field and initialise:
    Tic = Function(Q1, name="Tic")
    T = Function(Q, name="Temperature")

    # Viscosity is a function of temperature and radial position (i.e. axi-symmetric field)
    mu_radial = Function(W, name="Viscosity")
    assign_1d_profile(mu_radial, str(base_path.parent / "gplates_global/mu2_radial.rad"))
    mu = mu_constructor(mu_radial, Tave, T)  # Add temperature dependency

    # Initial time step
    delta_t = Function(R, name="delta_t").assign(2.0e-6)

    if last_checkpoint_path is not None:
        with CheckpointFile(str(last_checkpoint_path), "r") as fi:
            Tic.assign(fi.load_function(mesh, name="dat_0"))
    else:
        Tic.assign(Tobs)

    # Information pertaining to the plate reconstruction model
    cao_2024_files = ensure_reconstruction("Cao 2024", "../gplates_files")

    plate_reconstruction_model = pyGplatesConnector(
        rotation_filenames=cao_2024_files["rotation_filenames"],
        topology_filenames=cao_2024_files["topology_filenames"],
        nneighbours=4,
        nseeds=1e4,
        scaling_factor=1.0,
        oldest_age=1800,
        delta_t=1.0
    )

    # Top velocity boundary condition
    gplates_velocities = GplatesFunction(
        V,
        gplates_connector=plate_reconstruction_model,
        top_boundary_marker=top_id,
        name="GPlates_Velocity"
    )

    # Get a dictionary of the reference fields to be used in TALA approximation
    tala_parameters_dict = TALA_parameters(function_space=Q)
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra=Constant(5e8),  # Rayleigh number
        Di=Constant(0.9492824165791792),  # Dissipation number
        rho=tala_parameters_dict["rhobar"],  # reference density
        Tbar=tala_parameters_dict["Tbar"],  # reference temperature
        alpha=tala_parameters_dict["alphabar"],  # reference thermal expansivity
        cp=tala_parameters_dict["cpbar"],  # reference specific heat capacity
        g=tala_parameters_dict["gbar"],  # reference gravity
        H=tala_parameters_dict["H_int"],  # reference thickness
        mu=mu,  # Viscosity field (including thermal dependencies)
        kappa=tala_parameters_dict["kappa"])

    # Section: Setting up nullspaces
    # Nullspaces for stokes contains only a constant nullspace for pressure, as the top boundary is
    # imposed. The nullspace is generate with closed=True(for pressure) and rotational=False
    # as there are no rotational nullspace for velocity.
    # .. note: For compressible formulations we only provide `transpose_nullspace`
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False)
    # The near nullspaces gor gamg always include rotational and translational modes
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2])

    # Section: Setting boundary conditions
    # Temperature boundary conditions (constant)
    # for the top and bottom boundaries
    temp_bcs = {
        bottom_id: {'T': 1.0 - 930./3700.},
        top_id: {'T': 0.0},
    }
    # Velocity boundary conditions
    stokes_bcs = {
        top_id: {'u': gplates_velocities},
        bottom_id: {'un': 0},
    }

    # Constructing Energy and Stokes solver
    energy_solver = EnergySolver(
        T, u, approximation, delta_t,
        ImplicitMidpoint, bcs=temp_bcs)

    # adjusting solver parameters
    energy_solver.solver_parameters['ksp_converged_reason'] = None
    energy_solver.solver_parameters['ksp_rtol'] = 1e-4

    # Stokes solver
    stokes_solver = StokesSolver(
        z, T, approximation, bcs=stokes_bcs, mu=mu,
        constant_jacobian=False,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace)

    # tweaking solver parameters
    stokes_solver.solver_parameters['snes_rtol'] = 1e-2
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    # non-dimensionalised time for present geologic day (0)
    presentday_ndtime = plate_reconstruction_model.age2ndtime(0.)

    # non-dimensionalised time for 35 Myrs ago
    time = plate_reconstruction_model.age2ndtime(35.)

    # Defining control
    control = Control(Tic)

    # project the initial condition from Q1 to Q2, and imposing
    # boundary conditions
    project(
        Tic,
        T,
        solver_parameters=iterative_solver_parameters,
        forward_kwargs={"solver_parameters": iterative_solver_parameters},
        adj_kwargs={"solver_parameters": iterative_solver_parameters},
        bcs=energy_solver.strong_bcs,
    )

    # timestep counter
    timestep_index = 0

    # Now perform the time loop:
    while time < presentday_ndtime:
        if timestep_index % 2 == 0:
            # Update surface velocities
            gplates_velocities.update_plate_reconstruction(time)

            # Solve Stokes sytem
            stokes_solver.solve()

        # Make sure we are not going past present day
        if presentday_ndtime - time < float(delta_t):
            delta_t.assign(presentday_ndtime - time)

        # Temperature system:
        energy_solver.solve()

        # Updating time
        time += float(delta_t)
        timestep_index += 1

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    # Assembling the objective
    objective = t_misfit

    # Loggin the first objective (Make sure ROL shows the same value)
    log(f"Objective value after the first run: {objective}")

    # We want to avoid a second call to objective functional with the same value
    first_call_decorator = first_call_value(predefined_value=objective)
    ReducedFunctional.__call__ = first_call_decorator(ReducedFunctional.__call__)

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    return Tic, ReducedFunctional(objective, control)


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
    from firedrake.ufl_expr import extract_unique_domain
    from scipy.interpolate import interp1d

    with stop_annotating():
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
    q.create_block_variable()


def get_plate_reconstruction_info():
    plate_reconstruction_files = {}

    base_path = Path(__file__).resolve()

    # rotation filenames
    plate_reconstruction_files["rotation_filenames"] = [
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/optimisation/1000_0_rotfile_MantleOptimised.rot")
    ]

    # topology filenames
    plate_reconstruction_files["topology_filenames"] = [
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/250-0_plate_boundaries.gpml"),
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/410-250_plate_boundaries.gpml"),
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Convergence.gpml"),
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Divergence.gpml"),
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Topologies.gpml"),
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Transforms.gpml"),
    ]

    return plate_reconstruction_files


def generate_reference_fields():
    # get the path to the base directory
    base_path = Path(__file__).resolve().parent

    # mesh/initial guess file is comming from a long-term simulation
    mesh_path = base_path / "initial_condition_mat_prop/Final_State.h5"

    # Name of the final output
    output_path = base_path / "REVEAL.pvd"

    # Load mesh from checkpoint
    with CheckpointFile(mesh_path, mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        T_simulation = f.load_function(mesh, name="Temperature")

    mesh.cartesian = False

    # Set up function spaces for coordinates and fields
    V = VectorFunctionSpace(mesh, "CG", 1)
    X = SpatialCoordinate(mesh)
    # Compute dimensionalised coordinates
    r = Function(V, name="coordinates").interpolate(X / rmax * gdrift.R_earth)

    # Set up scalar function spaces for seismic model fields
    Q = FunctionSpace(mesh, "CG", 1)
    vsh = Function(Q, name="vsh")
    vsv = Function(Q, name="vsv")
    vs = Function(Q, name="vs")

    # Compute the depth field
    depth = Function(Q, name="depth").interpolate(
        Constant(gdrift.R_earth) - sqrt(r[0]**(2) + r[1]**(2) + r[2]**(2))
    )

    # Load the REVEAL model
    seismic_model = gdrift.SeismicModel("REVEAL")

    # Filling the vsh and vsv fields with the values from the seismic model
    reveal_vsh_vsv = seismic_model.at(
        label=["vsh", "vsv"],
        coordinates=r.dat.data_with_halos)
    vsh.dat.data_with_halos[:] = reveal_vsh_vsv[:, 0]
    vsv.dat.data_with_halos[:] = reveal_vsh_vsv[:, 1]

    # Compute the isotropic velocity field
    vs.interpolate(sqrt((2 * vsh ** 2 + vsv ** 2) / 3))

    # Average the isotropic velocity field over the layers, this will be useful for visualising devaitons from the average
    averager = LayerAveraging(mesh, quad_degree=6)

    # Define a field on Q for T_obs
    T_obs = Function(Q, name="T_obs")
    T_ave = Function(Q, name="average_temperature")

    anelastic_slb_pyrolite = buil_thermodynamic_model()

    # Convert the shear wave speed to T_obs
    T_obs.dat.data_with_halos[:] = anelastic_slb_pyrolite.vs_to_temperature(
        vs.dat.data_with_halos,
        depth.dat.data_with_halos)

    # Compute the layer-averaged T_obs (Note: T_ave is what comes from thermodynamic conversion, which is comletely off)
    averager.extrapolate_layer_average(T_ave, averager.get_layer_average(T_obs))

    # Take the average out of the T_obs field
    T_obs.interpolate(T_obs - T_ave)

    # Compute the layer-averaged temperature from the "simulation temperature"
    averager.extrapolate_layer_average(T_ave, averager.get_layer_average(T_simulation))

    # Add the mean profile to T_obs again (Note: T_ave is now from a simulation)
    T_obs.interpolate(T_obs + T_ave)

    # Boundary conditions for T_obs
    temp_bcs = {
        "bottom": {'T': 1.0 - 930./3700.},
        "top": {'T': 0.0},
    }

    # Adding Smoothing to Tobs
    smoother = DiffusiveSmoothingSolver(
        function_space=Tobs.function_space(),
        wavelength=0.05,
        bcs=temp_bcs,
    )

    # acting smoothing on Tobs
    Tobs.assign(smoother.action(Tobs))

    # Output for visualisation
    output = VTKFile(output_path.with_suffix(".pvd"))
    output.write(Tobs, T_ave)

    # Write out the file
    with CheckpointFile(output_path.with_suffix(".h5"), mode="w") as fi:
        fi.save_mesh(mesh)
        fi.save_function(Tobs, name="Tobs")
        fi.save_function(Taverage, name="AverageTemperature")


def generate_spherical_mesh(mesh_filename):
    # Set up geometry:

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

    resolution_func *= 1.0 / np.sum(resolution_func)
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)

    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=resolution_func,
        extrusion_type="radial",
    )

    with CheckpointFile(mesh_filename, "w") as fi:
        fi.save_mesh(mesh=mesh)

    return mesh_filename


def first_call_value(predefined_value):
    def decorator(func):
        has_been_called = False

        def wrapper(self, *args, **kwargs):
            nonlocal has_been_called
            if not has_been_called:
                has_been_called = True
                return predefined_value
            return func(self, *args, **kwargs)

        return wrapper
    return decorator


def find_last_checkpoint():
    try:
        checkpoint_dir = Path.cwd().resolve() / "copy_optimisation_checkpoint"
        solution_dir = sorted(list(checkpoint_dir.glob("[0-9]*")), key=lambda x: int(str(x).split("/")[-1]))[-1]
        solution_path = solution_dir / "solution_checkpoint.h5"
    except Exception:
        solution_path = None
    return solution_path


def mu_constructor(mu_radial, Tave, T):
    r"""Constructing a temperature strain-rate dependent velocity

        This uses Arrhenius law for temperature dependent viscosity
        $$\eta = A exp(-E/R * T) = A * exp(-E/R * Tave) * exp(-E/R * (T - Tave))\\
                               = 1d_field * exp(-ln(delta_mu_T) * (T - Tave))$$

    Args:
        mu_radial (firedrake.Form): radial viscosity profile
        T (_type_): Temperature
        Tave (_type_): Average temperature
        u (_type_): velocity

    Returns:
        firedrake.BaseForm: temperature and strain-rate dependent viscosity
    """

    # Adding temperature dependence:
    delta_mu_T = Constant(1000.)
    mu_lin = mu_radial * exp(-ln(delta_mu_T) * (T - Tave))

    # coordinates
    # X = SpatialCoordinate(T.ufl_domain())
    # r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)

    # Strain-Rate Dependence
    # mu_star, sigma_y = Constant(1.0), 5.0e5 + 2.5e6*(rmax-r)
    # epsilon = sym(grad(u))  # strain-rate
    # epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with a tolerance to ensure stability)
    # mu_plast = mu_star + (sigma_y / epsii)
    # mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)
    return mu_lin


def TALA_parameters(function_space):

    # radial density field
    rhobar = Function(Q, name="CompRefDensity")
    interpolate_1d_profile(function=rhobar, one_d_filename="initial_condition_mat_prop/rhobar.txt")
    rhobar.assign(rhobar / 3200.)

    # radial reference temperature field
    Tbar = Function(Q, name="CompRefTemperature")
    interpolate_1d_profile(function=Tbar, one_d_filename="initial_condition_mat_prop/Tbar.txt")
    Tbar.assign((Tbar - 1600.) / 3700.)

    # radial thermal expansivity field
    alphabar = Function(Q, name="IsobaricThermalExpansivity")
    interpolate_1d_profile(function=alphabar, one_d_filename="initial_condition_mat_prop/alphabar.txt")
    alphabar.assign(alphabar / 4.1773e-05)

    # radial specific heat capacity field
    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity")
    interpolate_1d_profile(function=cpbar, one_d_filename="initial_condition_mat_prop/CpSIbar.txt")
    cpbar.assign(cpbar / 1249.7)

    # radial gravity
    gbar = Function(Q, name="GravitationalAcceleration")
    interpolate_1d_profile(function=gbar, one_d_filename="initial_condition_mat_prop/gbar.txt")
    gbar.assign(gbar / 9.8267)

    # conductivtiy
    kappa = Constant(3.0)  # Thermal conductivity = yields a diffusivity of 7.5e-7 at surface.

    return {"rhobar": rhobar, "Tbar": Tbar, "alphabar": alphabar, "cpbar": cpbar, "gbar": gbar, "kappa": kappa}


if __name__ == "__main__":
    generate_reference_fields()
    conduct_inversion()
