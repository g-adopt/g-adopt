from gadopt import *
from gadopt.gplates import *
from gadopt.stokes_integrators import iterative_stokes_solver_parameters
from pathlib import Path


def main():
    # Get all checkpoint files first
    checkpoint_file = Path("/scratch/xd2/sg8812/untar_simulations/C51/C51/Final_State.h5")

    # Load mesh from the first checkpoint
    with CheckpointFile(checkpoint_file.as_posix(), mode="r") as f:
        log(f"Loading mesh from {checkpoint_file}")
        mesh = f.load_mesh("firedrake_default_extruded")

    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    # Set up function spaces - currently using the bilinear Q2Q1 element pair for Stokes and DQ2 T:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "DQ", 2)  # Temperature function space (DG scalar)
    Q_CG = FunctionSpace(mesh, "CG", 2)  # CG Temperature function space for visualisation.
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    T = Function(Q, name="Temperature")
    z = Function(Z, name="Stokes")

    # Test functions and functions to hold solutions:
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    # X = SpatialCoordinate(mesh)
    # r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)

    # Get approximation profiles with the correct directory
    approximation_sources = get_approximation_profiles(checkpoint_file.parent)

    approximation_profiles = {}
    for func, details in approximation_sources.items():
        f = Function(Q, name=details["name"])
        interpolate_1d_profile(function=f, one_d_filename=details["filename"])
        f.assign(details["scaling"](f))
        approximation_profiles[func] = f

    # We next prepare our viscosity, starting with a radial profile.
    mu_rad = Function(Q, name="Viscosity_Radial")  # Depth dependent component of viscosity
    radial_viscosity_filename = (checkpoint_file.parent / "initial_condition_mat_prop/visc/mu_1e20_asthenosphere_linear_increase_7e22_LM.visc").as_posix()
    interpolate_1d_profile(function=mu_rad, one_d_filename=radial_viscosity_filename)

    # Visualisation Fields
    FullT = Function(Q_CG, name="FullTemperature")
    T_avg = Function(Q, name='Layer_Averaged_Temp')
    T_dev = Function(Q, name='Temperature_Deviation')

    # The averager tool
    averager = LayerAveraging(mesh, quad_degree=6)

    #  building viscosity field
    mu = mu_rad  # * exp(-ln(activation_energy(r)) * T_dev)

    # These fields are used to set up our Truncated Anelastic Liquid Approximation.
    # Pass the Function objects explicitly instead of using **approximation_profiles
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra=Constant(8.7668e8),
        Di=Constant(0.9492824),
        H=Constant(9.93),
        mu=mu,
        kappa=Constant(3.9904),
        rho=approximation_profiles["rho"],
        Tbar=approximation_profiles["Tbar"],
        alpha=approximation_profiles["alpha"],
        cp=approximation_profiles["cp"],
        g=approximation_profiles["g"]
    )

    # Nullspaces and near-nullspace objects are next set up,
    # For free slip, we need rotational=True to handle the additional rotational modes
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False
    )
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2]
    )
    # Free slip boundary conditions: normal component zero at both boundaries
    stokes_bcs = {
        boundary.bottom: {'un': 0},
        boundary.top: {'u': 0},
    }

    my_solver_parameters = iterative_stokes_solver_parameters
    # my_solver_parameters['snes_rtol'] = 1e-2
    # my_solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    # my_solver_parameters['fieldsplit_0']['ksp_rtol'] = 2.5e-5
    # my_solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    # my_solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    # my_solver_parameters['fieldsplit_1']['ksp_rtol'] = 2.0e-4

    # Set up Stokes Solver with my iterative solver parameters
    stokes_solver = StokesSolver(
        z, approximation, T, bcs=stokes_bcs,
        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace, solver_parameters=my_solver_parameters
    )

    # We now initiate the time loop:
    checkpoint = checkpoint_file[-1]
    log(f"Processing checkpoint: {checkpoint}")

    with CheckpointFile(checkpoint.as_posix(), mode="r") as f:
        T.assign(f.load_function(mesh, "Temperature"))

    # Assigning FullT
    FullT.interpolate(T + approximation_profiles["Tbar"])
    # Average temperature field
    averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))

    # Compute deviation from layer average
    T_dev.interpolate(FullT-T_avg)

    # Solve Stokes sytem:
    stokes_solver.solve()


def get_approximation_profiles(directory_to_params: Path):
    """ Return the approximation profiles. Together with how to non-dimensionalise them."""
    return {
        "rho": {
            "name": "CompRefDensity",
            "filename": (directory_to_params / "initial_condition_mat_prop/rhobar.txt").as_posix(),
            "scaling": lambda x: x / 3200.,
        },
        "Tbar": {
            "name": "CompRefTemperature",
            "filename": (directory_to_params / "initial_condition_mat_prop/Tbar.txt").as_posix(),
            "scaling": lambda x: (x - 1600.) / 3700.,  # Assumes surface T of 300, CMB T = 4000.
        },
        "alpha": {
            "name": "IsobaricThermalExpansivity",
            "filename": (directory_to_params / "initial_condition_mat_prop/alphabar.txt").as_posix(),
            "scaling": lambda x: x / 4.1773e-05,
        },
        "cp": {
            "name": "IsobaricSpecificHeatCapacity",
            "filename": (directory_to_params / "initial_condition_mat_prop/CpSIbar.txt").as_posix(),
            "scaling": lambda x: x / 1249.7,
        },
        "g": {
            "name": "GravitationalAcceleration",
            "filename": (directory_to_params / "initial_condition_mat_prop/gbar.txt").as_posix(),
            "scaling": lambda x: x / 9.8267,
        },
    }


def activation_energy(r):
    # Now that we have the average T profile, we add lateral viscosity variation due to temperature variations.
    # These variations are stronger in the upper mantle than the lower mantle.
    UM_val = 100000.
    LM_val = 10000.
    r0 = 1.9916262975778547
    taper_thickness = 0.07
    r_mid = r0 - taper_thickness / 2
    smoothing_width = taper_thickness / 10

    # Define delta_mu_T as a pure UFL expression
    delta_mu_T = LM_val + (UM_val - LM_val) * 0.5 * (1 + tanh((r - r_mid) / smoothing_width))
    return delta_mu_T


def get_mesh_parameters():
    return {
        "rmin": 1.208,
        "rmax": 2.208,
        "ref_level": 7,
        "nlayers": 64,
    }


if __name__ == "__main__":
    main()
