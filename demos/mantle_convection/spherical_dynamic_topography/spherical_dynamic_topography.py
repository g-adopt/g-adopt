from gadopt import *
from gadopt.gplates import *
from mpi4py import MPI
import pandas as pd
from pathlib import Path


def main():
    # Set up geometry:
    mesh_parameters = get_mesh_parameters()
    path_to_mesh = Path("./initial_condition_mat_prop/Final_State.h5")

    # Construct a CubedSphere mesh and then extrude into a sphere (or load from checkpoint):
    with CheckpointFile(path_to_mesh.as_posix(), mode="r") as f:
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

    approximation_sources = get_approximation_profiles()

    approximation_profiles = {}
    for func, details in approximation_sources.items():
        f = Function(Q, name=details["name"])
        interpolate_1d_profile(function=f, one_d_filename=details["filename"])
        f.assign(details["scaling"](f))

        approximation_profiles[func] = f

    Tbar = approximation_profiles["Tbar"]
    # We next prepare our viscosity, starting with a radial profile.
    mu_rad = Function(Q, name="Viscosity_Radial")  # Depth dependent component of viscosity
    radial_viscosity_filename = "initial_condition_mat_prop/visc/mu_1e20_asthenosphere_linear_increase_7e22_LM.visc"
    interpolate_1d_profile(function=mu_rad, one_d_filename=radial_viscosity_filename)

    # Visualisation Fields
    FullT = Function(Q_CG, name="FullTemperature")
    T_avg = Function(Q, name='Layer_Averaged_Temp')
    T_dev = Function(Q, name='Temperature_Deviation')

    # The averager tool
    averager = LayerAveraging(mesh, quad_degree=6)

    #  building viscosity field
    mu = mu_rad * exp(-ln(activation_energy(r)) * T_dev)

    # These fields are used to set up our Truncated Anelastic Liquid Approximation.
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra=Constant(8.7668e8),
        Di=Constant(0.9492824),
        H=Constant(9.93),
        mu=mu,
        kappa=Constant(3.9904),
        **approximation_profiles
    )

    # Nullspaces and near-nullspace objects are next set up,
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False
    )
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2]
    )

    zahirovic_2022_files = ensure_reconstruction("Zahirovic 2022", "../gplates_files")
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
    stokes_bcs = {
        boundary.bottom: {'un': 0},
        boundary.top: {'u': gplates_velocities},
    }

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)
    stokes_solver.solver_parameters['snes_rtol'] = 1e-2
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 2.5e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 2.0e-2

    # Set up fields for visualisation on CG meshes - DG is overkill for output.
    FullT_CG = Function(Q_CG, name="FullTemperature_CG").interpolate(FullT)
    T_CG = Function(Q_CG, name='Temperature_CG').interpolate(T)
    T_dev_CG = Function(Q_CG, name='Temperature_Deviation_CG').interpolate(T_dev)
    mu_field_CG = Function(Q_CG, name="Viscosity_CG").interpolate(mu)

    all_checkpoints = get_all_snapshot_checkpoints("/scratch/xd2/sg8812/untar_simulations")

    # Create Function objects for dynamic topography
    # i.e., (Thermal Expansivity) x (Background Earth's Mantle Density) x (Thickness of mantle) / (Ra number)
    dimensionalisation_factor = Constant(3e-5 * 4e3 * 3e6) / approximation.Ra
    dynamic_topography_top = Function(W, name="Dynamic_Topography_Top")
    dynamic_topography_bottom = Function(W, name="Dynamic_Topography_Bottom")

    # Compute dynamic topography values
    delta_rho_top = Constant(1.0)  # i.e., \Delta \rho_top = 1.0 \times \rho_mantle
    g_top = Constant(1.0)



    output_file = VTKFile("output.pvd")

    # We now initiate the time loop:
    for checkpoint in all_checkpoints[-2:-1]:
        with CheckpointFile(checkpoint.as_posix(), mode="r") as f:
            T.assign(f.load_function(mesh, "Temperature"))
            z.assign(f.load_function(mesh, "Stokes"))

        # Get the time of the checkpoint
        time = get_time(checkpoint.parent / "params.log")

        # Assigning FullT
        FullT.assign(T + approximation_profiles["Tbar"])
        # Average temperature field
        averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))

        # Compute deviation from layer average
        T_dev.assign(FullT-T_avg)

        # Update plate velocities:
        gplates_velocities.update_plate_reconstruction(time)

        # Solve Stokes sytem:
        stokes_solver.solve()

        ns_top = stokes_solver.force_on_boundary(boundary.top)
        ns_bottom = stokes_solver.force_on_boundary(boundary.bottom)

dynamic_topography_top.interpolate(ns_top / (delta_rho_top * g_top) * dimensionalisation_factor)



        # Write output and interpolate to CG:
        mu_field_CG.interpolate(mu)
        FullT_CG.interpolate(FullT)
        T_CG.interpolate(T)
        T_dev_CG.interpolate(T_dev)
        output_file.write(*z.subfunctions, vr, FullT_CG, T_CG, T_dev_CG, mu_field_CG)


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
            except:
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
