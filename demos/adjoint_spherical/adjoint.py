from gadopt import *
from gadopt.inverse import *
from gadopt.gplates import *
import numpy as np
from firedrake.adjoint_utils import blocks
# from pyadjoint import stop_annotating
from pathlib import Path
import gdrift
from gdrift.profile import SplineProfile

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

blocks.solving.Block.evaluate_adj = collect_garbage(
    blocks.solving.Block.evaluate_adj)
blocks.solving.Block.recompute = collect_garbage(
    blocks.solving.Block.recompute)

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


def test_taping():
    Tic, reduced_functional = forward_problem()
    repeat_val = reduced_functional([Tic])
    log("Reduced Functional Repeat: ", repeat_val)


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

    minimisation_problem = MinimizationProblem(
        reduced_functional, bounds=(T_lb, T_ub))

    # Start a LinMore Optimiser
    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )

    visualisation_path = find_last_checkpoint(
    ).resolve().parents[2] / "visual.pvd"

    vtk_file = VTKFile(str(visualisation_path))
    control_container = Function(
        Tic.function_space(), name="Initial Temperature")

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
    with CheckpointFile(str(base_path / "REVEAL.h5"), "r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")
        # reference temperature field (seismic tomography)
        Tobs = fi.load_function(mesh, name="Tobs")
        # Average temperature field
        Tave = fi.load_function(mesh, name="AverageTemperature")

    mesh.cartesian = False

    # Boundary markers to top and bottom
    bottom_id, top_id = "bottom", "top"

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "DQ", 2)  # Temperature function space (scalar)
    # Initial Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)
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
    interpolate_1d_profile(mu_radial, str(
        base_path.parent / "gplates_global/mu2_radial.rad"))
    mu = mu_constructor(mu_radial, Tave, T)  # Add temperature dependency

    # Initial time step
    delta_t = Function(R, name="delta_t").assign(2.0e-6)

    if last_checkpoint_path is not None:
        with CheckpointFile(str(last_checkpoint_path), "r") as fi:
            Tic.assign(fi.load_function(mesh, name="dat_0"))
    else:
        Tic.assign(Tobs)

    # Information pertaining to the plate reconstruction model
    cao_2024_files = ensure_reconstruction("Cao 2024", "./gplates_files")

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
    gplates_velocities = GplatesVelocityFunction(
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
        # reference thermal expansivity
        alpha=tala_parameters_dict["alphabar"],
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
        bottom_id: {'T': 1.0 - 930. / 3700.},
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
        if timestep_index >= 5:
            break

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    # Assembling the objective
    objective = t_misfit

    # Loggin the first objective (Make sure ROL shows the same value)
    log(f"Objective value after the first run: {objective}")

    # We want to avoid a second call to objective functional with the same value
    first_call_decorator = first_call_value(predefined_value=objective)
    ReducedFunctional.__call__ = first_call_decorator(
        ReducedFunctional.__call__)

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    return Tic, ReducedFunctional(objective, control)


def generate_reference_fields():
    """
    Generates reference fields for a seismic model by loading a tomography model and converting it to temperature.

    This function performs the following steps:
    1. Loads a mesh and initial temperature from a checkpoint file.
    2. Sets up function spaces for coordinates and fields.
    3. Computes the depth field.
    4. Loads the REVEAL seismic model and fills the vsh and vsv fields with values from the model.
    5. Computes the isotropic velocity field (vs) from vsh and vsv.
    6. Averages the isotropic velocity field over the layers for visualization.
    7. Computes the layer-averaged depth and temperature to be used in a thermodynamic model.
    8. Builds a thermodynamic model and converts the shear wave speed to observed temperature (T_obs).
    9. Computes the layer-averaged T_obs and subtracts the average from T_obs.
    10. Adds the mean profile from the simulation temperature to T_obs.
    11. Applies boundary conditions and smoothing to T_obs.
    12. Outputs the results for visualization and saves the mesh and functions to a checkpoint file.

    The function uses T_average to handle the mean profile of the temperature during the conversion process.
    """
    # get the path to the base directory
    base_path = Path(__file__).resolve().parent

    # mesh/initial guess file is comming from a long-term simulation
    mesh_path = base_path / "initial_condition_mat_prop/Final_State.h5"

    # Name of the final output
    output_path = base_path / "REVEAL.pvd"

    # Load mesh from checkpoint
    with CheckpointFile(str(mesh_path), mode="r") as f:
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

    # Dimensional parameters will be used to convert simulation results to dimension full form
    nondim_parameters = get_dimensional_parameters()

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

    # finding the depth and temperature average to be passed to the thermodynamic model for linearisation
    depth_ave_array = averager.get_layer_average(depth)
    T_simulation_ave_array = averager.get_layer_average(T_simulation) * (nondim_parameters["T_CMB"] - nondim_parameters["T_surface"]) + nondim_parameters["T_surface"]

    # Define a field on Q for T_obs
    T_obs = Function(Q, name="T_obs")
    T_ave = Function(Q, name="average_temperature")

    # Building the thermodynamic model, this is a regularised version of the SLB_16 pyrolite model using the temperature profile from simulation
    anelastic_slb_pyrolite = build_thermodynamic_model(np.column_stack((depth_ave_array, T_simulation_ave_array)))

    # Convert the shear wave speed to T_obs
    T_obs.dat.data_with_halos[:] = anelastic_slb_pyrolite.vs_to_temperature(
        vs.dat.data_with_halos,
        depth.dat.data_with_halos)

    # Compute the layer-averaged T_obs (Note: T_ave is what comes from thermodynamic conversion, which is comletely off)
    averager.extrapolate_layer_average(
        T_ave, averager.get_layer_average(T_obs))

    # Take the average out of the T_obs field
    T_obs.interpolate(T_obs - T_ave)

    # Compute the layer-averaged temperature from the "simulation temperature"
    averager.extrapolate_layer_average(
        T_ave, T_simulation_ave_array)

    # Add the mean profile to T_obs again (Note: T_ave is now from a simulation)
    T_obs.interpolate(T_obs + T_ave)

    # Boundary conditions for T_obs
    # DOUBLECHECK
    temp_bcs = {
        "bottom": {'T': 1.0 - 930. / 3700.},
        "top": {'T': 0.0},
    }

    # Adding Smoothing to Tobs
    smoother = DiffusiveSmoothingSolver(
        function_space=T_obs.function_space(),
        wavelength=0.05,
        bcs=temp_bcs,
        solver_parameters=iterative_solver_parameters,
    )

    # acting smoothing on Tobs
    T_obs.assign(smoother.action(T_obs))

    # Output for visualisation
    output = VTKFile(output_path.with_suffix(".pvd"))
    output.write(T_obs, T_ave, T_simulation)

    # Write out the file
    with CheckpointFile(str(output_path.with_suffix(".h5")), mode="w") as fi:
        fi.save_mesh(mesh)
        fi.save_function(T_obs, name="Tobs")
        fi.save_function(T_ave, name="AverageTemperature")
        fi.save_function(T_simulation, name="Temperature")


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
        solution_dir = sorted(list(checkpoint_dir.glob(
            "[0-9]*")), key=lambda x: int(str(x).split("/")[-1]))[-1]
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
    nondim_parameters = get_dimensional_parameters()

    # radial density field
    rhobar = Function(Q, name="CompRefDensity")
    interpolate_1d_profile(
        function=rhobar, one_d_filename="initial_condition_mat_prop/rhobar.txt")
    rhobar.assign(rhobar / nondim_parameters["rho"])

    # radial reference temperature field
    Tbar = Function(Q, name="CompRefTemperature")
    interpolate_1d_profile(
        function=Tbar, one_d_filename="initial_condition_mat_prop/Tbar.txt")
    Tbar.assign((Tbar - nondim_parameters["T_surface"]) / (nondim_parameters["T_CMB"] - nondim_parameters["T_surface"]))

    # radial thermal expansivity field
    alphabar = Function(Q, name="IsobaricThermalExpansivity")
    interpolate_1d_profile(
        function=alphabar, one_d_filename="initial_condition_mat_prop/alphabar.txt")
    alphabar.assign(alphabar / nondim_parameters["alpha"])

    # radial specific heat capacity field
    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity")
    interpolate_1d_profile(
        function=cpbar, one_d_filename="initial_condition_mat_prop/CpSIbar.txt")
    cpbar.assign(cpbar / nondim_parameters["cp"])

    # radial gravity
    gbar = Function(Q, name="GravitationalAcceleration")
    interpolate_1d_profile(
        function=gbar, one_d_filename="initial_condition_mat_prop/gbar.txt")
    gbar.assign(gbar / nondim_parameters["g"])

    # conductivtiy
    # Thermal conductivity = yields a diffusivity of 7.5e-7 at surface.
    kappa = Constant(3.0)

    return {"rhobar": rhobar, "Tbar": Tbar, "alphabar": alphabar, "cpbar": cpbar, "gbar": gbar, "kappa": kappa}


def build_thermodynamic_model(temperature_profile_array):
    # Load the thermodynamic model
    slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")

    # Make a spline that can be passed onto regularisation
    terra_temperature_spline = gdrift.SplineProfile(
        depth=temperature_profile_array[:, 0] * 1e3,
        value=temperature_profile_array[:, 1],
        name="T_average",
        extrapolate=True
    )

    # Regularise the thermodynamic model
    regular_slb_pyrolite = gdrift.regularise_thermodynamic_table(
        slb_pyrolite, terra_temperature_spline,
        regular_range={"v_s": (-1.0, 0.0), "v_p": (-np.inf, 0.0), "rho": (-np.inf, 0.0)})

    # building solidus model
    solidus_ghelichkhan = build_solidus()
    # Using the solidus model build the anelasticity model around the solidus profile
    anelasticity = build_anelasticity_model(solidus_ghelichkhan)
    # Apply the anelasticity correction to the regularised thermodynamic model
    anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
        regular_slb_pyrolite, anelasticity)

    return anelastic_slb_pyrolite


# Compute a solidus for building anelasticity correction
def build_solidus():
    # Defining the solidus curve for manlte
    andrault_solidus = gdrift.RadialEarthModelFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")

    # Defining parameters for Cammarano style anelasticity model
    hirsch_solidus = gdrift.HirschmannSolidus()

    my_depths = []
    my_solidus = []

    for solidus_model in [hirsch_solidus, andrault_solidus]:
        d_min, d_max = solidus_model.min_max_depth("solidus temperature")
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth("solidus temperature", dpths))

    ghelichkhan_et_al = SplineProfile(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        name="Ghelichkhan et al 2021",
        extrapolate=True)

    return ghelichkhan_et_al


def build_anelasticity_model(solidus):
    def B(x):
        return np.where(x < 660e3, 1.1, 20)

    def g(x):
        return np.where(x < 660e3, 20, 10)

    def a(x):
        return 0.2

    def omega(x):
        return 1.0

    return gdrift.CammaranoAnelasticityModel(B, g, a, solidus, omega)


def get_dimensional_parameters():
    return {
        "T_CMB": 4000.0,
        "T_surface": 300.0,
        "rho": 3200.0,
        "g": 9.81,
        "cp": 1249.7,
        "alpha": 4.1773e-05,
        # "kappa": 3.0,
        # "H_int": 2900e3,
    }


if __name__ == "__main__":
    generate_reference_fields()
    conduct_inversion()
