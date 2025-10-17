from gadopt import *
from gadopt.gplates import *
from gadopt.stokes_integrators import iterative_stokes_solver_parameters
from pathlib import Path
import argparse


def main(free_slip=False):
    # Get all checkpoint files first
    all_checkpoints = get_all_snapshot_checkpoints("/scratch/xd2/sg8812/untar_simulations")
    if not all_checkpoints:
        raise ValueError("No checkpoint files found!")

    # Load mesh from the first checkpoint
    first_checkpoint = all_checkpoints[-1]
    with CheckpointFile(first_checkpoint.as_posix(), mode="r") as f:
        log(f"Loading mesh from {first_checkpoint}")
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

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)

    # Get approximation profiles with the correct directory
    approximation_sources = get_approximation_profiles(first_checkpoint.parent)

    approximation_profiles = {}
    for func, details in approximation_sources.items():
        f = Function(Q, name=details["name"])
        interpolate_1d_profile(function=f, one_d_filename=details["filename"])
        f.assign(details["scaling"](f))
        approximation_profiles[func] = f

    # We next prepare our viscosity, starting with a radial profile.
    mu_rad = Function(Q, name="Viscosity_Radial")  # Depth dependent component of viscosity
    radial_viscosity_filename = (first_checkpoint.parent / "initial_condition_mat_prop/visc/mu_1e20_asthenosphere_linear_increase_7e22_LM.visc").as_posix()
    interpolate_1d_profile(function=mu_rad, one_d_filename=radial_viscosity_filename)

    # Visualisation Fields
    FullT = Function(Q_CG, name="FullTemperature")
    T_avg = Function(Q, name='Layer_Averaged_Temp')
    T_dev = Function(Q, name='Temperature_Deviation')

    # The averager tool
    averager = LayerAveraging(mesh, quad_degree=6)

    #  building viscosity field
    mu = mu_rad # * exp(-ln(activation_energy(r)) * T_dev)

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
        Z, closed=True, rotational=free_slip
    )
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2]
    )

    # Set up GPlates reconstruction
    zahirovic_2022_files = ensure_reconstruction("Zahirovic 2022", "./")
    plate_reconstruction_model = pyGplatesConnector(
        rotation_filenames=zahirovic_2022_files["rotation_filenames"],
        topology_filenames=zahirovic_2022_files["topology_filenames"],
        oldest_age=410,
        nseeds=1e4,
        nneighbours=4,
        scaling_factor=1.0,
        kappa=1e-6,
    )

    # Top velocity boundary condition
    gplates_velocities = GplatesVelocityFunction(
        V,
        gplates_connector=plate_reconstruction_model,
        top_boundary_marker=boundary.top,
        name="GPlates_Velocity"
    )

    # Followed by boundary conditions for velocity and temperature.
    if free_slip:
        # Free slip boundary conditions: normal component zero at both boundaries
        stokes_bcs = {
            boundary.bottom: {'un': 0},
            boundary.top: {'un': 0},
        }
    else:
        # Plate velocity boundary conditions
        stokes_bcs = {
            boundary.bottom: {'un': 0},
            boundary.top: {'u': gplates_velocities},
        }

    my_solver_parameters = iterative_stokes_solver_parameters
    my_solver_parameters['snes_rtol'] = 1e-2
    my_solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    my_solver_parameters['fieldsplit_0']['ksp_rtol'] = 2.5e-4
    my_solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    my_solver_parameters['fieldsplit_1']['ksp_rtol'] = 2.0e-3

    # Set up Stokes Solver with my iterative solver parameters
    stokes_solver = StokesSolver(
        z, approximation, T, bcs=stokes_bcs,
        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace, solver_parameters=my_solver_parameters,
        constant_jacobian=True,
    )

    # Set up fields for visualisation on CG meshes - DG is overkill for output.
    FullT_CG = Function(Q_CG, name="FullTemperature_CG")
    T_CG = Function(Q_CG, name='Temperature_CG')
    T_dev_CG = Function(Q_CG, name='Temperature_Deviation_CG')
    mu_field_CG = Function(Q_CG, name="Viscosity_CG")

    # Create Function objects for dynamic topography
    # i.e., (Thermal Expansivity) x (Background Earth's Mantle Density) x (Thickness of mantle) / (Ra number)
    dimensionalisation_factor = Constant(3e-5 * 4e3 * 3e6) / approximation.Ra
    dynamic_topography_top = Function(W, name="Dynamic_Topography_Top")
    dynamic_topography_bottom = Function(W, name="Dynamic_Topography_Bottom")

    # Compute dynamic topography values
    delta_rho_top = Constant(1.0)  # i.e., \Delta \rho_top = 1.0 \times \rho_mantle
    g_top = Constant(1.0)
    delta_rho_bottom = Constant(-2.5)  # i.e., \Delta \rho_CMB = (\rho_mantle - \rho_outer_core) / \rho_mantle
    g_bottom = Constant(1.0)

    # Set output filename based on boundary condition type
    output_filename = "output_free_slip.pvd" if free_slip else "output_plates.pvd"
    output_file = VTKFile(output_filename)

    log(f"Calculating dynamic topography with {'free slip' if free_slip else 'plate velocity'} boundary conditions")

    # We now initiate the time loop:
    for checkpoint in all_checkpoints[-5:-1]:  # Process last 2 checkpoints for testing
        log(f"Processing checkpoint: {checkpoint}")

        with CheckpointFile(checkpoint.as_posix(), mode="r") as f:
            T.assign(f.load_function(mesh, "Temperature"))
            # z.assign(f.load_function(mesh, "Stokes"))


        # Get the time of the checkpoint
        time = get_time(checkpoint.parent / "params.log")
        log(f"Time: {time:.2e}, {plate_reconstruction_model.ndtime2age(time):.2f} Ma")

        # Assigning FullT
        FullT.interpolate(T + approximation_profiles["Tbar"])
        # Average temperature field
        averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))

        # Compute deviation from layer average
        T_dev.interpolate(FullT-T_avg)

        # Update plate velocities (only needed for plate boundary conditions, also for visualisation):
        gplates_velocities.update_plate_reconstruction(time)

        # Solve Stokes sytem:
        stokes_solver.solve()

        # Compute normal stresses at boundaries
        ns_top = stokes_solver.force_on_boundary(boundary.top)
        ns_bottom = stokes_solver.force_on_boundary(boundary.bottom)

        # Compute dynamic topography
        dynamic_topography_top.interpolate(ns_top / (delta_rho_top * g_top) * dimensionalisation_factor)
        dynamic_topography_bottom.interpolate(ns_bottom / (delta_rho_bottom * g_bottom) * dimensionalisation_factor)

        # Write output and interpolate to CG:
        mu_field_CG.interpolate(mu)
        FullT_CG.interpolate(FullT)
        T_CG.interpolate(T)
        T_dev_CG.interpolate(T_dev)

        # Write all fields to output
        output_file.write(
            *z.subfunctions, FullT_CG, T_CG, T_dev_CG, mu_field_CG,
            dynamic_topography_top, dynamic_topography_bottom, gplates_velocities)


def get_all_snapshot_checkpoints(directory_to_params: Path):
    """ Return the list of all snapshot checkpoints."""
    directory_to_params = Path(directory_to_params)
    all_checkpoints = [f for f in directory_to_params.glob("C*/C*/Final_State.h5")]
    all_checkpoints.sort()
    return all_checkpoints


def get_time(filename):
    """ Return the last time in the parameter log file. """
    with open(filename, "r") as f:
        f.readline()
        for line in f:
            try:
                last_ndtime = line.split()[1]
            except (IndexError, ValueError):
                pass

    return float(last_ndtime)


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
    UM_val = 1000.
    LM_val = 100.
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
    parser = argparse.ArgumentParser(
        description="Spherical dynamic topography calculation with G-ADOPT",
    )
    parser.add_argument(
        "--freeslip",
        action="store_true",
        help="Use free slip boundary conditions instead of plate velocity boundary conditions"
    )

    args = parser.parse_args()
    main(free_slip=args.freeslip)
