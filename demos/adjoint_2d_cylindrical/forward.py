"""
This runs the forward portion of the adjoint test case, to generate the reference
final condition, and synthetic forcing (surface velocity observations).
"""

from gadopt import *
import numpy as np

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
    "snes_stol": 0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_converged_reason": None,
    "fieldsplit_0": {
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_converged_reason": None,
    },
}


def run_forward():
    # Set up geometry:
    rmin, rmax, nlayers = 1.22, 2.22, 128
    rmax_earth = 6370  # Radius of Earth [km]
    rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
    r_410_earth = rmax_earth - 410  # 410 radius [km]
    r_660_earth = rmax_earth - 660  # 660 raidus [km]
    r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
    r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

    # Start with a previously-initialised temperature field
    with CheckpointFile("Checkpoint230.h5", mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W])

    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    Ra = Constant(1e7)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra, cartesian=False)

    # Define time stepping parameters:
    max_timesteps = 200
    delta_t = Constant(5e-6)  # Constant time step

    with CheckpointFile("Checkpoint230.h5", mode="r") as f:
        T = f.load_function(mesh, "Temperature")

    Taverage = Function(Q1, name="Average Temperature")

    # A step function designed to design viscosity jumps
    # Build a step centred at "centre" with given magnitude
    # Increase with radius if "increasing" is True
    def step_func(centre, mag, increasing=True, sharpness=50):
        return mag * (
            0.5 * (1 + tanh((1 if increasing else -1) * (r - centre) * sharpness))
        )

    # From this point, we define a depth-dependent viscosity mu_lin
    mu_lin = 2.0

    # Assemble the depth dependence
    for line, step in zip(
        [5.0 * (rmax - r), 1.0, 1.0],
        [
            step_func(r_660, 30, False),
            step_func(r_410, 10, False),
            step_func(2.2, 10, True),
        ],
    ):
        mu_lin += line * step

    # Add temperature dependence of viscosity
    mu_lin *= exp(-ln(Constant(80)) * T)

    # Assemble the viscosity expression in terms of velocity u
    eps = sym(grad(u))
    epsii = sqrt(inner(eps, eps) + 1e-10)
    # yield stress and its depth dependence
    # consistent with values used in Coltice et al. 2017
    sigma_y = 2e4 + 4e5 * (rmax - r)
    mu_plast = 0.1 + (sigma_y / epsii)
    mu_eff = 2 * (mu_lin * mu_plast) / (mu_lin + mu_plast)
    mu = conditional(mu_eff > 0.4, mu_eff, 0.4)

    mu_function = Function(W, name="Viscosity")

    # Calculate the layer average of the initial state
    averager = LayerAveraging(
        mesh, np.linspace(rmin, rmax, nlayers * 2), cartesian=False, quad_degree=6
    )
    averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
    checkpoint_file.save_mesh(mesh)
    checkpoint_file.save_function(Taverage, name="Average Temperature", idx=0)
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
        mu=mu,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
        solver_parameters=newton_stokes_solver_parameters,
    )

    # Create output file and select output_frequency
    output_file = VTKFile("vtu-files/output.pvd")
    dump_period = 10

    # Split and rename the velocity and pressure functions
    # so that they can be used for visualisation
    u, p = z.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")

    # Now perform the time loop:
    for timestep in range(0, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()

        # Storing velocity to be used in the objective F
        checkpoint_file.save_function(u, name="Velocity", idx=timestep)

        if timestep % dump_period == 0 or timestep == max_timesteps - 1:
            mu_function.interpolate(mu)
            output_file.write(u, p, T, mu_function)

    # Save the reference final temperature
    checkpoint_file.save_function(T, name="Temperature", idx=max_timesteps - 1)
    checkpoint_file.close()


if __name__ == "__main__":
    run_forward()
