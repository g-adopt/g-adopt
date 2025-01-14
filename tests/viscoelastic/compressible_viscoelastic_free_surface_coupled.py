#!/usr/bin/env python3
import argparse

import numpy as np

from gadopt import *

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--case",
    default="viscoelastic-compressible",
    help="""Test case to run:
- viscoelastic-compressible
- elastic-compressible
- viscoelastic-incompressible-1e15
- elastic-incompressible
- viscous-incompressible""",
)
args = parser.parse_args()


def viscoelastic_model(
    nx=80, dt_factor=0.1, sim_time="long", shear_modulus=1e11, bulk_modulus=2e11
):
    # Set up geometry:
    nz = nx  # Number of vertical cells
    D = 3e6  # Depth of the domain in m
    L = D / 2  # Length of domain in m
    mesh = RectangleMesh(nx, nz, L, D)  # Square mesh generated via Firedrake
    mesh.cartesian = True
    mesh.coordinates.dat.data[:, -1] -= D

    # Squash mesh to refine near top boundary
    # Modified from the ocean model Roms
    # See, for example, https://www.myroms.org/wiki/Vertical_S-coordinate
    x, y = SpatialCoordinate(mesh)
    a, b = 4.0, 0.0
    z_scaled = y / D
    Cs = (1.0 - b) * sinh(a * z_scaled) / sinh(a) + b * (
        tanh(a * (z_scaled + 0.5)) / 2 / tanh(a / 2) - 0.5
    )

    depth_c = 500.0
    scaled_z_coordinates = depth_c * z_scaled + (D - depth_c) * Cs
    mesh.coordinates.interpolate(as_vector([x, scaled_z_coordinates]))
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

    # Set up function spaces - currently using P2P1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
    # Stress tensor function space (tensor, discontinuous)
    S = TensorFunctionSpace(mesh, "DG", 1)
    Z = MixedFunctionSpace([V, S])  # Mixed function space
    Q = FunctionSpace(mesh, "CG", 2)  # Analytical function space (scalar)
    R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

    # Output function space information:
    log("Number of Displacement DOF:", V.dim())
    log("Number of Stress DOF:", S.dim())
    log("Number of Velocity and Stress DOF:", V.dim() + S.dim())
    log("Number of Analytical DOF:", Q.dim())

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    # Next rename for output:
    z.subfunctions[0].rename("Displacement")
    z.subfunctions[1].rename("Stress")

    # Physical fields
    rho = Function(R).assign(4500.0)  # Density (kg/m^3)
    g = 10.0  # Gravitational acceleration (m/s^2)
    mu = 1e21  # Viscosity (Pa s)
    K = bulk_modulus  # Bulk modulus (Pa)
    G = shear_modulus  # Shear modulus (Pa)

    maxwell_time = mu / G  # Maxwell time (s)

    approximation = Approximation(
        "CIVA",
        dimensional=True,
        parameters={"G": G, "g": g, "K": K, "mu": mu, "rho": rho},
    )

    # Set up surface load
    load_wavelegnth = D / 8
    load_wavenumber = 2 * pi / load_wavelegnth  # Angular wavenumber
    load_amplitude = 1000.0
    load_height = -load_amplitude * cos(load_wavenumber * x)

    # Timestepping parameters
    year_in_seconds = 8.64e4 * 365.25
    time = 0.0
    tau = 2 * load_wavenumber * mu / float(rho) / g
    dt = dt_factor * tau  # Initial time-step
    max_timesteps = round(2 * tau / dt) if sim_time == "long" else 1

    log("tau in years", tau / year_in_seconds)
    log("max timesteps", max_timesteps)
    log("dt in years", dt / year_in_seconds)
    log("maxwell time in years", maxwell_time / year_in_seconds)

    # Setup boundary conditions
    stokes_bcs = {
        bottom_id: {"uy": 0},
        top_id: {
            "normal_stress": rho * g * load_height,
            "free_surface": {"rho_ext": 0},
        },
        left_id: {"ux": 0},
        right_id: {"ux": 0},
    }

    solver = InternalVariableSolver(
        z,
        approximation,
        coupled_tstep=dt,
        theta=0.5,
        bcs=stokes_bcs,
        solver_parameters="direct",
    )

    # Setup analytical solution for the free surface from Cathles et al. (2024)
    lambda_lame = K - 2 / 3 * G  # Lame's first parameter
    nu = lambda_lame / 2 / (lambda_lame + G)  # Poisson's ratio
    elastic_factor = 2 * (1 - nu)
    log(f"elastic_factor: {elastic_factor}")

    norm_factor = 1 + elastic_factor * maxwell_time / tau
    h_elastic_ini = (
        load_amplitude * float(rho) * g / 2 / load_wavenumber / G / norm_factor
    )
    h_elastic = load_amplitude * (1 - 1 / norm_factor)
    log(f"Maximum initial elastic displacement: {h_elastic_ini}")
    log(f"Maximum elastic displacement: {h_elastic}")

    load_height_analytical = Function(Q, name="Analytical height").interpolate(
        (
            (load_amplitude - h_elastic) * (1 - exp(-time / tau / norm_factor))
            + h_elastic
        )
        * cos(load_wavenumber * x)
    )

    error = 0  # Initial error

    output_file = VTKFile(f"output_dt{dt / year_in_seconds:.2f}.pvd")
    output_file.write(*z.subfunctions, load_height_analytical, time=time)

    # Now perform the time loop
    vert_displ = z.subfunctions[0][1]
    for _ in range(max_timesteps):
        # Solve coupled system
        solver.solve()

        time += dt
        load_height_analytical.interpolate(
            (
                (load_amplitude - h_elastic) * (1 - exp(-time / tau / norm_factor))
                + h_elastic
            )
            * cos(load_wavenumber * x)
        )

        output_file.write(*z.subfunctions, load_height_analytical, time=time)

        # Calculate error
        local_error = assemble(pow(vert_displ - load_height_analytical, 2) * ds(top_id))
        log(f"local_error: {local_error}")

        if sim_time == "long":
            error += local_error * dt
        else:
            # For elastic solve only one timestep so don't scale error by timestep
            # length (get 0.5x convergence rate as artificially making error smaller
            # when in reality displacement formulation shouldn't depend on time)
            error += local_error

    final_error = pow(error, 0.5) / L
    return final_error


params = {
    "viscoelastic-compressible": {
        "dtf_start": 0.1,
        "nx": 160,
        "sim_time": "long",
        "shear_modulus": 1e11,
        "bulk_modulus": 2e11,
    },
    "elastic-compressible": {
        "dtf_start": 0.001,
        "nx": 160,
        "sim_time": "short",
        "shear_modulus": 1e11,
        "bulk_modulus": 2e11,
    },
    "viscoelastic-incompressible-1e15": {
        "dtf_start": 0.1,
        "nx": 320,
        "sim_time": "long",
        "shear_modulus": 1e11,
        "bulk_modulus": 1e15,
    },
    "elastic-incompressible": {
        "dtf_start": 0.001,
        "nx": 160,
        "sim_time": "short",
        "shear_modulus": 1e11,
        "bulk_modulus": 1e16,
    },
    "viscous-incompressible": {
        "dtf_start": 0.1,
        "nx": 160,
        "sim_time": "long",
        "shear_modulus": 1e14,
        "bulk_modulus": 1e14,
    },
}


def run_benchmark(case_name):
    """Runs default case for four values of dt_factor."""
    dtf_start = params[case_name].pop("dtf_start")
    dt_factors = dtf_start / (2 ** np.arange(4))
    errors = [
        viscoelastic_model(dt_factor=dtf, **params[case_name]) for dtf in dt_factors
    ]

    nx = params[case_name]["nx"]
    prefix = f"errors-{case_name}-internalvariable-coupled-{nx}cells"
    np.savetxt(f"{prefix}-free-surface-1e16.dat", errors)

    relative_errors = errors / errors[-1]
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    print(convergence)


if __name__ == "__main__":
    run_benchmark(args.case)
