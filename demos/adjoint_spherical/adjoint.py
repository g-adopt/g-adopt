from gadopt import *
from gadopt.inverse import *
import numpy as np
from firedrake.adjoint_utils import blocks
from pathlib import Path
import gc
from firedrake.adjoint_utils import CheckpointBase
# from memory_profiler import profile
from checkpoint_schedules import SingleDiskStorageSchedule
from pyadjoint import Block

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


def collect_garbage(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect(generation=2)
        # log(f"Number of collected objects: {gc.collect(generation=2)}")
        return result

    return wrapper


# Set up geometry:
rmax, rmin, ncells, nlayers = 2.22, 1.22, 32, 8


def just_forward_adjoint_calls(num):
    tic, rf = forward_problem()
    for i in range(num):
        rf.func_call(tic)
        rf.derivative()


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

    minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 1.0e-6
    minimisation_parameters["Status Test"] = 20

    minimisation_problem = MinimizationProblem(
        reduced_functional, bounds=(T_lb, T_ub))

    # Start a LinMore Optimiser
    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint"
    )

    # run the optimisation
    optimiser.run()


def conduct_taylor_test():
    Tic, reduced_functional = forward_problem()
    log("Reduced Functional Repeat: ", reduced_functional([Tic]))
    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    _ = taylor_test(reduced_functional, Tic, Delta_temp)


def forward_problem():
    continue_annotation()
    tape = get_working_tape()
    tape.clear_tape()

    # Enable disk checkpointing for the adjoint
    enable_disk_checkpointing()

    # setting gc collection
    tape.enable_checkpointing(
        SingleDiskStorageSchedule(), gc_timestep_frequency=1, gc_generation=2
    )

    # Set up the base path
    base_path = Path(__file__).resolve().parent

    # Load mesh and associated fields
    tala_parameters_dict = {}
    with CheckpointFile(str(base_path / "reference_fields.h5"), "r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")
        # reference temperature field (seismic tomography)
        T_obs = fi.load_function(mesh, name="Tobs")  # This is dimensional
        # Average temperature field
        T_ave = fi.load_function(mesh, name="T_ave_ref")  # Used for regularising T_ic
        T_simulation = fi.load_function(mesh, name="T_ic_0")
        # Loading adiabatic reference fields
        for key in ["rhobar", "Tbar", "alphabar", "cpbar", "gbar", "mu_radial"]:
            tala_parameters_dict[key] = fi.load_function(mesh, name=key)

    # Mesh properties
    mesh.cartesian = False
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
    T_0 = Function(Q, name="T_0")  # initial timestep
    T = Function(Q, name="Temperature")

    # Viscosity is a function of temperature and radial position (i.e. axi-symmetric field)
    mu_radial = tala_parameters_dict["mu_radial"]
    mu = mu_constructor(mu_radial, T_ave, T)  # Add temperature dependency

    # Initial time step
    delta_t = Function(R, name="delta_t").assign(5.0e-9)

    # If we are running from a checkpoint, we want to use an appropriate initial condition
    # this is normally not necessary, unless we are redefining the ReducedFunctional.__call__
    # to skip the first recall in the optimisation.
    Tic.interpolate(T_simulation)

    # Top velocity boundary condition
    gplates_velocities = Function(
        V,
        name="GPlates_Velocity"
    ).interpolate(as_vector((0.0, 0.0)))

    # Setting up the approximation
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra=Constant(1e2),  # Rayleigh number
        Di=Constant(0.9492),  # Dissipation number
        rho=tala_parameters_dict["rhobar"],  # reference density
        Tbar=tala_parameters_dict["Tbar"],  # reference temperature
        # reference thermal expansivity
        alpha=tala_parameters_dict["alphabar"],
        cp=tala_parameters_dict["cpbar"],  # reference specific heat capacity
        g=tala_parameters_dict["gbar"],  # reference gravity
        H=Constant(9.93),  # reference thickness
        mu=mu,  # viscosity
        kappa=Constant(3.0))

    # Section: Setting up nullspaces
    # Nullspaces for stokes contains only a constant nullspace for pressure, as the top boundary is
    # imposed. The nullspace is generate with closed=True(for pressure) and rotational=False
    # as there are no rotational nullspace for velocity.
    # .. note: For compressible formulations we only provide `transpose_nullspace`
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False)
    # The near nullspaces gor gamg always include rotational and translational modes
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1])

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
        z, T, approximation, bcs=stokes_bcs,
        constant_jacobian=False,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace)

    # tweaking solver parameters
    stokes_solver.solver_parameters['snes_rtol'] = 1e-2

    # stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters.pop('snes_monitor')
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    # stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1'].pop('ksp_converged_reason')
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    # Defining control
    control = Control(Tic)

    # project the initial condition from Q1 to Q2, and imposing
    # boundary conditions
    project(
        Tic,
        T_0,
        solver_parameters=iterative_solver_parameters,
        forward_kwargs={"solver_parameters": iterative_solver_parameters},
        adj_kwargs={"solver_parameters": iterative_solver_parameters},
        bcs=energy_solver.strong_bcs,
    )

    project(
        Tic,
        T,
        solver_parameters=iterative_solver_parameters,
        forward_kwargs={"solver_parameters": iterative_solver_parameters},
        adj_kwargs={"solver_parameters": iterative_solver_parameters},
        bcs=energy_solver.strong_bcs,
    )

    # while time < presentday_ndtime:
    for i in range(20):
        # Emulating what we do with GplatesVelocityFunctions
        gplates_velocities.create_block_variable()
        # Solve Stokes sytem
        stokes_solver.solve()

        # Temperature system:
        energy_solver.solve()
        break

    tape.add_block(DiagnosticBlock(T))

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - T_obs) ** 2 * dx)

    # Assembling the objective
    objective = t_misfit

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    class MyCallbackClass(object):
        def __init__(self):
            # Placeholder for control and FullT
            self.cb_control = Function(Tic.function_space(), name="control")
            self.cb_state = Function(T.function_space(), name="state")

            # Initial index
            self.idx = 0
            self.block_variable = T.block_variable

        def __call__(self, f, c):
            self.cb_control.interpolate(c)
            # Increasing index
            self.idx += 1

    callback = MyCallbackClass()
    return Tic, ReducedFunctional(objective, control, eval_cb_post=MyCallbackClass())

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

    mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)  # construct a circle mesh
    mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type='radial')  # extrude into a cylinder
    mesh.cartesian = False

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    depth = rmax - r

    # Set up scalar function spaces for seismic model fields
    Q = FunctionSpace(mesh, "CG", 1)

    # Define a field on q for t_obs: THESE ARE ALL WITH DIMENSION
    T_obs = Function(Q, name="T_obs").interpolate(depth + 0.1 * exp(-(((X[0] - 1.5)**2 + (X[1] - 0.0) ** 2)) / 0.01))  # This will be the "tomography temperature field"
    T_simulation = Function(Q, name="Temperature").interpolate(depth + 0.1 * exp(-(((X[0] - 0.0)**2 + (X[1] - 1.5) ** 2)) / 0.01))  # This will be the "tomography temperature field"
    T_ave = Function(Q, name="T_ave").interpolate(depth)

    TALAdict = TALA_parameters(Q)
    # Write out the file
    with CheckpointFile("reference_fields.h5", mode="w") as fi:
        fi.save_mesh(mesh)
        fi.save_function(T_obs, name="Tobs")
        fi.save_function(T_ave, name="T_ave_ref")
        fi.save_function(T_simulation, name="T_ic_0")
        # Storing all reference fields
        for item in TALAdict.keys():
            fi.save_function(TALAdict[item], name=item)

    ref_fi = VTKFile("reference_fields.pvd")
    ref_fi.write(T_obs, T_ave, T_simulation, *TALAdict.values())


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

    return mu_lin


def TALA_parameters(function_space):
    nondim_parameters = get_dimensional_parameters()

    # radial density field
    rhobar = Function(function_space, name="CompRefDensity")
    interpolate_1d_profile(
        function=rhobar, one_d_filename="initial_condition_mat_prop/rhobar_cyl.txt")
    rhobar.assign(rhobar / nondim_parameters["rho"])

    # radial reference temperature field
    Tbar = Function(function_space, name="CompRefTemperature")
    interpolate_1d_profile(
        function=Tbar, one_d_filename="initial_condition_mat_prop/Tbar_cyl.txt")
    # We trust the increase in the adiabatic temperature
    Tbar.assign((Tbar - 1600.) / (nondim_parameters["T_CMB"] - nondim_parameters["T_surface"]))

    # radial thermal expansivity field
    alphabar = Function(function_space, name="IsobaricThermalExpansivity")
    interpolate_1d_profile(
        function=alphabar, one_d_filename="initial_condition_mat_prop/alphabar_cyl.txt")
    alphabar.assign(alphabar / nondim_parameters["alpha"])

    # radial specific heat capacity field
    cpbar = Function(function_space, name="IsobaricSpecificHeatCapacity")
    interpolate_1d_profile(
        function=cpbar, one_d_filename="initial_condition_mat_prop/CpSIbar_cyl.txt")
    cpbar.assign(cpbar / nondim_parameters["cp"])

    # radial gravity
    gbar = Function(function_space, name="GravitationalAcceleration")
    interpolate_1d_profile(
        function=gbar, one_d_filename="initial_condition_mat_prop/gbar_cyl.txt")
    gbar.assign(gbar / nondim_parameters["g"])

    mu_radial = Function(function_space, name="mu_radial")

    base_path = Path(__file__).resolve().parent
    interpolate_1d_profile(mu_radial, "initial_condition_mat_prop/mu2_radial_cyl.txt")

    return {"rhobar": rhobar, "Tbar": Tbar, "alphabar": alphabar, "cpbar": cpbar, "gbar": gbar, "mu_radial": mu_radial}


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


def return_block_variable_checkpoint(a_block_variable):
    print(type(a_block_variable))
    return (
        a_block_variable.checkpoint.restore()
        if isinstance(a_block_variable.checkpoint, CheckpointBase)
        else a_block_variable.checkpoint
    )


class DiagnosticBlock(Block):
    """
    Useful for outputting gradients in time dependent simulations or inversions
    """

    def __init__(self, function):
        """Initialises the Diagnostic block.

        Args:
          function:
            Calculate gradient of reduced functional wrt to this function.
            Dictionary specifying riesz represenation (defaults to L2).
        """
        super().__init__()
        self.add_dependency(function)
        self.add_output(function.block_variable)
        self.f_name = function.name()
        self.idx = 0

    def recompute_component(self, inputs, block_variable, idx, prepared):
        fi = CheckpointFile(f"FinalState{self.idx}.h5", mode="w")
        fi.save_mesh(block_variable.checkpoint.ufl_domain())
        fi.save_function(block_variable.checkpoint, name="Temperature")
        self.idx += 1
        return

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return


if __name__ == "__main__":
    # conduct_inversion()
    forward_problem()
