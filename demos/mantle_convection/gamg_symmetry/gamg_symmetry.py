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
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    T = Function(Q, name="Temperature")
    z = Function(Z, name="Stokes")

    # Test functions and functions to hold solutions:
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

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

    #  building viscosity field
    mu = mu_rad

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
    my_solver_parameters['snes_rtol'] = 1e-2
    my_solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    my_solver_parameters['fieldsplit_0']['ksp_rtol'] = 2.5e-5
    # my_solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    my_solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    my_solver_parameters['fieldsplit_1']['ksp_rtol'] = 2.0e-4

    # Set up Stokes Solver with my iterative solver parameters
    stokes_solver = StokesSolver(
        z, approximation, T, bcs=stokes_bcs,
        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace, solver_parameters=my_solver_parameters
    )

    with CheckpointFile(checkpoint_file.as_posix(), mode="r") as f:
        T.assign(f.load_function(mesh, "Temperature"))

    # Solve Stokes sytem:
    stokes_solver.solve()

    log("Done!")


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


if __name__ == "__main__":
    main()
