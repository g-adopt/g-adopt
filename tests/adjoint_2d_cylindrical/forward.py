"""
This runs the forward portion of the adjoint test case, to generate the reference
final condition, and synthetic forcing (surface velocity observations).
"""

from gadopt import *
import numpy as np


def run_forward(visualise=False):
    # Set up geometry:
    geo_constants = get_geometry_parameters()
    ref_values = get_reference_values()
    # Start with a previously-initialised temperature field
    with CheckpointFile("Checkpoint230.h5", mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        mesh.cartesian = False

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    DG = FunctionSpace(mesh, "DG", 2)  # Temperature function space (scalar, DG2)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W, DG])

    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    z.assign(0)
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    T = Function(DG, name="Temperature")

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    Ra = Constant(ref_values["Ra"])  # Rayleigh number

    # Define time stepping parameters:
    max_timesteps = ref_values["max_timesteps"]
    delta_t = Constant(ref_values["delta_t"])  # Constant time step

    with CheckpointFile("Checkpoint230.h5", mode="r") as f:
        T = f.load_function(mesh, "Temperature")

    # Build the viscosity using u, T, and the geometry parameters
    mu = get_viscosity(r, T, u)

    Taverage = Function(Q1, name="Average_Temperature")

    mu_function = Function(W, name="Viscosity")

    # Configure approximation
    approximation = BoussinesqApproximation(Ra, mu=mu)

    # Calculate the layer average of the initial state
    averager = LayerAveraging(
        mesh, np.linspace(geo_constants["rmin"], geo_constants["rmax"], geo_constants["nlayers"] * 2), quad_degree=6
    )
    averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
    checkpoint_file.save_mesh(mesh)
    checkpoint_file.save_function(Taverage, name="Average_Temperature", idx=0)
    checkpoint_file.save_function(T, name="Temperature", idx=0)

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1]
    )

    # Free-slip velocity boundary condition on all sides
    stokes_bcs = {
        "bottom": {"un": 0},
        "top": {"un": 0},
    }
    temp_bcs = {
        "bottom": {"T": 1.0},
        "top": {"T": 0.0},
    }

    energy_solver = EnergySolver(
        T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs
    )

    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        solver_parameters="direct",
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
    )

    if visualise:
        # Create output file and select output_frequency
        output_file = VTKFile("vtu-files/output.pvd")
        dump_period = 10

    # Now perform the time loop:
    for timestep in range(0, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()

        # Storing velocity to be used in the objective F
        checkpoint_file.save_function(z.subfunctions[0], name="Velocity", idx=timestep)

        if (timestep % dump_period == 0 or timestep == max_timesteps - 1) and visualise:
            mu_function.interpolate(mu)
            output_file.write(*z.subfunctions, T, mu_function)

    # Save the reference final temperature
    checkpoint_file.save_function(T, name="Temperature", idx=max_timesteps - 1)
    checkpoint_file.close()


def get_geometry_parameters():
    """
    Returns the geometry parameters for the 2D cylindrical adjoint test case.
    """
    rmin, rmax, nlayers = 1.22, 2.22, 128
    rmax_earth = 6370  # Radius of Earth [km]
    rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
    r_410_earth = rmax_earth - 410  # 410 radius [km]
    r_660_earth = rmax_earth - 660  # 660 raidus [km]
    r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
    r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

    return {
        "rmin": rmin,
        "rmax": rmax,
        "nlayers": nlayers,
        "rmax_earth": rmax_earth,
        "rmin_earth": rmin_earth,
        "r_410": r_410,
        "r_660": r_660,
        "r_410_earth": r_410_earth,
        "r_660_earth": r_660_earth,
    }


def get_reference_values():
    """
    Return reference parameters for the 2D cylindrical adjoint test case.

    Provides physical and numerical parameters used throughout the simulation,
    including rheological constants, time-stepping parameters, and yield stress
    values. These parameters are used by get_viscosity() and other functions.

    Returns:
        dict: Dictionary containing simulation parameters:
            - Ra: Rayleigh number (1e7)
            - max_timesteps: Maximum number of timesteps (250)
            - delta_t: Time-step length (3e-6)
            - mu_0: Background viscosity (2.0)
            - mu_plast: Minimum plastic viscosity (0.1)
            - mu_min: Minimum effective viscosity (0.4)
            - sigma_y: Surface yield stress (2e4)
            - sigma_y_depth: Depth dependence of yield stress (4e5)
            - mu_T: Temperature dependence of viscosity (80)
    """
    return {
        "Ra": 1e7,  # Rayleigh number
        "max_timesteps": 125,  # Maximum number of timesteps
        "delta_t": 6e-6,  # Time-step length
        "mu_0": 2.0,  # Background viscosity
        "mu_plast": 0.1,  # minimum plastic viscosity: mu_plast = 0.1 + sigma_y / epsii
        "mu_min": 0.4,  # Miminimum amount of effective viscosity: mu = min(mu_eff, 0.4)
        "sigma_y": 2e4,  # yield stress at the surface: sigma_y = 2e4 + 4e5 * (rmax - r)
        "sigma_y_depth": 4e5,  # depth dependence of yield stress: sigma_y = 2e4 + 4e5 * (rmax - r)
        "mu_T": 80,  # Temperature dependence of viscosity: mu_lin *= exp(-ln(Constant(80)) * T)
    }


def get_viscosity(r, T, u):
    """
    Compute effective viscosity as a function of radius and temperature.

    Implements a depth-dependent Arrhenius rheology with temperature dependence
    and strain-rate dependent plastic yielding. Rheological parameters are
    defined in reference_values().

    Args:
        r: Radial coordinate
        T: Temperature field
        u: Velocity field

    Returns:
        Effective viscosity field incorporating depth, temperature, and strain-rate effects
    """

    geometry_parameters = get_geometry_parameters()
    reference_values = get_reference_values()

    # A step function designed to design viscosity jumps
    # Build a step centred at "centre" with given magnitude
    # Increase with radius if "increasing" is True
    def step_func(centre, mag, increasing=True, sharpness=50):
        return mag * (
            0.5 * (1 + tanh((1 if increasing else -1) * (r - centre) * sharpness))
        )

    # From this point, we define a depth-dependent viscosity mu_lin
    mu_lin = reference_values["mu_0"]

    # Assemble the depth dependence
    for line, step in zip(
        [5.0 * (geometry_parameters["rmax"] - r), 1.0, 1.0],
        [
            step_func(geometry_parameters["r_660"], 30, False),
            step_func(geometry_parameters["r_410"], 10, False),
            step_func(geometry_parameters["rmax"], 10, True),
        ],
    ):
        mu_lin += line * step

    # Add temperature dependence of viscosity
    mu_lin *= exp(-ln(Constant(reference_values["mu_T"])) * T)

    # Assemble the viscosity expression in terms of velocity u
    eps = sym(grad(u))
    epsii = sqrt(inner(eps, eps) + 1e-10)

    # yield stress and its depth dependence
    # consistent with values used in Coltice et al. 2017
    sigma_y = reference_values["sigma_y"] + reference_values["sigma_y_depth"] * (geometry_parameters["rmax"] - r)
    mu_plast = 0.1 + (sigma_y / epsii)
    mu_eff = 2 * (mu_lin * mu_plast) / (mu_lin + mu_plast)
    mu = conditional(mu_eff > reference_values["mu_min"], mu_eff, reference_values["mu_min"])
    return mu


if __name__ == "__main__":
    run_forward()
