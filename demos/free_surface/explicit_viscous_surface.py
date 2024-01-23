from gadopt import *
import numpy as np


def viscous_freesurface_model(nx, dt_factor, do_write=False):

    # Set up geometry:
    D = 3e6  # Depth of domain in m
    L = D  # Length of the domain in m
    lam_dimensional = D/2  # wavelength of load in m
    L0 = D  # characteristic length scale for scaling the equations
    lam = lam_dimensional/L0  # dimensionless lambda

    ny = nx
    mesh = RectangleMesh(nx, ny, L/L0, D/L0)  # Rectangle mesh generated via firedrake
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    u_, p_ = z.subfunctions  # Do this first to extract individual velocity and pressure fields

    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")

    eta = Function(W, name="eta")  # Define a free surface function

    T = Function(Q, name="Temperature").assign(0)  # Setup a dummy function for temperature

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    eta_eq = FreeSurfaceEquation(W, W, free_surface_id=top_id)  # Initialise the separate free surface equation for explicit coupling

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(0)  # Rayleigh number, here we set this to zero as there are no bouyancy terms
    approximation = BoussinesqApproximation(Ra)

    rho0 = approximation.rho  # This defaults to rho0 = 1 (dimensionless)
    g = approximation.g  # This defaults to g = 1 (dimensionless)

    kk = Constant(2 * pi / lam)  # wavenumber (dimensionless)
    F0 = Constant(1000 / L0)  # initial free surface amplitude (dimensionless)
    X = SpatialCoordinate(mesh)
    eta.interpolate(F0 * cos(kk * X[0]))  # initial free surface condition

    # timestepping
    mu = Constant(1)  # Dynamic viscosity (dimensionless)
    tau0 = Constant(2 * kk * mu / (rho0 * g))  # Characteristic time scale (dimensionless)
    log("tau0", tau0)

    dt = Constant(0.5*dt_factor*tau0)  # Timestep (dimensionless)
    log("dt (dimensionless)", dt)

    time = Constant(0.0)
    max_timesteps = round(10*tau0/dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
    log("max_timesteps", max_timesteps)

    # No normal flow except on the free surface
    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'free_surface_stress': rho0 * g * eta},  # Apply stress on free surface
        left_id: {'un': 0},
        right_id: {'un': 0},
    }

    eta_fields = {'velocity': u_}

    eta_bcs = {}
    eta_strong_bcs = [InteriorBC(W, 0., top_id)]  # Apply strong homogenous boundary to interior DOFs to prevent a singular matrix when only integrating the free surface equation over the top surface.

    mumps_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
        'snes_atol': 1e-6,
        'mat_mumps_icntl_14': 200
    }
    # Set up the stokes solver
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, cartesian=True, solver_parameters=mumps_solver_parameters)

    # Set up a timestepper for the free surface, here we use a first order backward Euler method following Kramer et al. 2012
    eta_timestepper = BackwardEuler(eta_eq, eta, eta_fields, dt, bnd_conditions=eta_bcs, strong_bcs=eta_strong_bcs, solver_parameters=mumps_solver_parameters)

    # analytical function
    eta_analytical = Function(W, name="eta analytical")
    eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

    if do_write:
        # Create output file and select output_frequency:
        filename = "explicit_viscous_freesurface"
        output_file = File(filename+"_D"+str(float(D/L0))+"_mu"+str(float(mu))+"_nx"+str(nx)+"_dt"+str(float(dt/tau0))+"tau.pvd")
        dump_period = round(tau0/dt)
        log(dump_period)
        output_file.write(u_, eta, p_, eta_analytical)

    error = 0
    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve Stokes sytem:
        stokes_solver.solve()
        eta_timestepper.advance(time)

        time.assign(time + dt)
        eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

        local_error = assemble(pow(eta-eta_analytical, 2)*ds(top_id))
        error += local_error*dt
        # Write output:
        if do_write:
            if timestep % dump_period == 0:
                log("timestep", timestep)
                log("time", time)
                output_file.write(u_, eta, p_, eta_analytical)

    final_error = pow(error, 0.5)/L
    return final_error


if __name__ == "__main__":
    # default case run with nx = 80 for four dt factors
    dt_factors = 2 / (2**np.arange(4))
    errors = np.array([viscous_freesurface_model(80, dtf) for dtf in dt_factors])
    np.savetxt("errors-explicit-free-surface-coupling.dat", errors)
