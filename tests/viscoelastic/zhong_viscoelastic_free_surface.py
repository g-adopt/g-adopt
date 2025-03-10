# Test case based on simple harmonic loading and unloading problems from `Viscosity of
# the Earth's Mantle' by Cathles (1975). The specific analytic solution is actually
# based off of Equation 2 in `On calculating glacial isostatic adjustment', Cathles
# (2024). The decay time is the viscous relaxation timescale plus the Maxwell time,
# including the elastic buoyancy effects. Note that we are solving a loading problem not
# an unloading problem.

# There are three default tests:
# 1) elastic case limit (dt << Maxwell time, 1 step)
# 2) viscoelastic (dt ~ Maxwell time)
# 3) viscous limit (dt >> Maxwell time)

import argparse

import numpy as np

from gadopt import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case",
    default="viscoelastic",
    required=False,
    help="""Test case to run:
- elastic limit (dt << Maxwell time, 1 step)
- viscoelastic (dt ~ Maxwell time)
- viscous limit (dt >> Maxwell time)""",
)
parser.add_argument("--output", action="store_true", help="Output VTK files")
parser.add_argument(
    "--output_directory",
    default="2d_analytic_zhong_viscoelastic_freesurface/",
    required=False,
    help="Output directory",
)
args = parser.parse_args()


def viscoelastic_model(nx=80, dt_factor=0.1, sim_time="long", G=1e11):
    # Set up geometry:
    nz = nx  # Number of vertical cells
    D = 3e6  # Depth of the domain in m
    L = D / 2  # Length of domain in m
    mesh = RectangleMesh(nx, nz, L, D)  # Rectangle mesh generated via Firedrake
    mesh.cartesian = True
    mesh.coordinates.dat.data[:, -1] -= D

    # Squash mesh to refine near top boundary modified from the ocean model
    # Roms e.g. https://www.myroms.org/wiki/Vertical_S-coordinate
    x, z = SpatialCoordinate(mesh)
    a, b = 4.0, 0.0
    z_scaled = z / D
    Cs = (1.0 - b) * sinh(a * z_scaled) / sinh(a) + b * (
        tanh(a * (z_scaled + 0.5)) / (2 * tanh(0.5 * a)) - 0.5
    )

    depth_c = 500.0
    scaled_z_coordinates = depth_c * z_scaled + (D - depth_c) * Cs
    mesh.coordinates.interpolate(as_vector([x, scaled_z_coordinates]))
    boundary = get_boundary_ids(mesh)

    # Set up function spaces - currently using P2P1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.
    Q = FunctionSpace(mesh, "CG", 2)  # Analytical function space (scalar)
    # (Discontinuous) Stress tensor function space (tensor)
    S = TensorFunctionSpace(mesh, "DG", 1)
    R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())
    log("Number of Analytical DOF:", Q.dim())

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    # Next rename for output:
    z.subfunctions[0].rename("Incremental displacement")
    z.subfunctions[1].rename("Pressure")

    displ = Function(V, name="Displacement")
    tau_old = Function(S, name="Deviatoric stress (old)")

    # Physical fields
    rho = Function(R).assign(4500.0)  # density in kg/m^3
    g = 10.0  # gravitational acceleration in m/s^2
    mu = 1e21  # Viscosity Pa s

    maxwell_time = mu / G

    # Set up surface load
    lam = D / 8  # wavelength of load in m
    kk = 2 * pi / lam  # wavenumber in m^-1
    F0 = 1000.0  # initial free surface amplitude in m
    eta = F0 * (1 - cos(kk * x))

    # Timestepping parameters
    year_in_seconds = 8.64e4 * 365.25
    tau = 2 * kk * mu / float(rho) / g
    log("tau in years", tau / year_in_seconds)
    time = 0.0
    dt = dt_factor * tau  # Initial time-step
    if sim_time == "long":
        max_timesteps = round(2 * tau / dt)
    else:
        max_timesteps = 1

    log("max timesteps", max_timesteps)
    dump_period = 1
    log("dump_period", dump_period)
    log("dt in years", dt / year_in_seconds)
    log("maxwell time in years", maxwell_time / year_in_seconds)

    approximation = Approximation(
        "SDVA", dimensional=True, parameters={"G": G, "g": g, "mu": mu, "rho": rho}
    )

    # Create output file
    if args.output:
        output_directory = args.output_directory
        output_file = VTKFile(
            f"{output_directory}viscoelastic_freesurface_maxwelltime"
            f"{maxwell_time / year_in_seconds:.0f}a_nx{nx}_dt{dt / year_in_seconds:.0f}"
            f"a_tau{tau / year_in_seconds:.0f}.pvd"
        )

    # Setup boundary conditions
    stokes_bcs = {
        boundary.bottom: {"uy": 0},
        boundary.top: {"normal_stress": rho * g * eta, "free_surface": {"rho_ext": 0}},
        boundary.left: {"ux": 0},
        boundary.right: {"ux": 0},
    }

    # Setup analytical solution for the free surface from Cathles et al. 2024
    eta_analytical = Function(Q, name="eta analytical")
    h_elastic = (F0 * float(rho) * g / (2 * kk * G)) / (1 + maxwell_time / tau)
    log("Maximum initial elastic displacement:", h_elastic)
    eta_analytical.interpolate(
        ((F0 - h_elastic) * (1 - exp(-(time) / (tau + maxwell_time))) + h_elastic)
        * cos(kk * x)
    )
    error = 0  # Initialise error

    viscoelastic_solver = ViscoelasticSolver(
        z, displ, tau_old, approximation, dt, bcs=stokes_bcs, solver_parameters="direct"
    )

    if args.output:
        output_file.write(*z.subfunctions, displ, tau_old)

    # Now perform the time loop:
    for timestep in range(max_timesteps):
        # Solve viscoelastic system
        viscoelastic_solver.solve()
        time += dt

        # Update analytical solution
        eta_analytical.interpolate(
            ((F0 - h_elastic) * (1 - exp(-(time) / (tau + maxwell_time))) + h_elastic)
            * cos(kk * x)
        )

        # Calculate error
        local_error = assemble(pow(displ[1] - eta_analytical, 2) * ds(boundary.top))
        error += local_error * dt

        # Write output:
        if (timestep + 1) % dump_period == 0:
            log("timestep", timestep)
            log("time", time)
            if args.output:
                output_file.write(*z.subfunctions, displ, tau_old)

    final_error = pow(error, 0.5) / L
    return final_error


params = {
    "viscoelastic": {"dtf_start": 0.1, "nx": 80, "sim_time": "long", "G": 1e11},
    "elastic": {"dtf_start": 0.001, "nx": 80, "sim_time": "short", "G": 1e11},
    "viscous": {"dtf_start": 0.1, "nx": 80, "sim_time": "long", "G": 1e14},
}


def run_benchmark(case_name):
    # Run default case run for four dt factors
    dtf_start = params[case_name]["dtf_start"]
    params[case_name].pop("dtf_start")  # Do not pass this to viscoelastic_model
    dt_factors = dtf_start / (2 ** np.arange(4))
    prefix = f"errors-{case_name}-zhong"
    errors = np.array(
        [viscoelastic_model(dt_factor=dtf, **params[case_name]) for dtf in dt_factors]
    )
    np.savetxt(f"{prefix}-free-surface.dat", errors)
    ref = errors[-1]
    relative_errors = errors / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    print(convergence)


if __name__ == "__main__":
    run_benchmark(args.case)
