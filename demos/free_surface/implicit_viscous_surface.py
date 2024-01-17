from gadopt import *
import numpy as np


def implicit_viscous_freesurface_model(nx, dt_factor, do_write=False):

    # Set up geometry:
    D = 3e6  # Depth of domain in m
    lam = D/2  # wavelength of load in m
    L = D  # Length of the domain in m
    ny = nx
    mesh = RectangleMesh(nx, ny, L, D)  # Rectangle mesh generated via firedrake
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W, W])  # Mixed function space.

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p, eta = split(z)  # Returns symbolic UFL expression for u, p and eta
    u_, p_, eta_ = z.subfunctions  # Returns functions for u, p and eta

    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")
    eta_.rename("eta")

    T = Function(Q, name="Temperature").assign(0)
    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(0)  # Rayleigh number
    rho0 = 4500  # density in kg/m^3
    g = 10  # gravitational acceleration in m/s^2
    approximation = BoussinesqApproximation(Ra, g=g, rho=rho0)

    kk = 2 * pi / lam  # wavenumber in m^-1
    F0 = 1000  # initial free surface amplitude in m
    X = SpatialCoordinate(mesh)
    eta_.interpolate(F0 * cos(kk * X[0]))

    # timestepping
    mu = 1e21  # Shear modulus in Pa
    tau0 = 2 * kk * mu / (rho0 * g)
    log("tau0", tau0)
    dt = dt_factor*tau0/round(D/lam)
    log("dt", dt)
    time = 0.0
    max_timesteps = round(10*tau0/dt)
    log("max_timesteps", max_timesteps)

    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'eta_interior': 0, 'eta': None},  # stress from free surface in momentum_equation.py
        left_id: {'un': 0},
        right_id: {'un': 0},
    }

    stokes_fields = {
        'surface_id': 4,  # VERY HACKY!
        'rhog': 45000,
        'dt': dt}  # Incredibly hacky! rho*g

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
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, cartesian=True, solver_parameters=mumps_solver_parameters, equations=FreeSurfaceStokesEquations, additional_fields=stokes_fields)

    if do_write:
        eta_midpoint = []
        eta_midpoint.append(eta_.at(L/2, D-0.001))

    # analytical function
    eta_analytical = Function(W, name="eta analytical")
    eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

    # Create output file and select output_frequency:
    if do_write:
        # Write output files in VTK format:
        dump_period = 1
        log("dump_period ", dump_period)
        filename = "implicit_viscous_freesurface"
        output_file = File(filename+"_D"+str(D)+"_mu"+str(mu)+"_nx"+str(nx)+"_dt"+str(dt/tau0)+"tau.pvd")
        output_file.write(u_, eta_, p_, eta_analytical)

    error = 0
    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve Stokes sytem:
        stokes_solver.solve()
        time += dt
        eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

        local_error = assemble(pow(eta-eta_analytical, 2)*ds(top_id))
        error += local_error*dt

        # Write output:
        if do_write:
            eta_midpoint.append(eta_.at(L/2, D-0.001))

            if timestep % dump_period == 0:
                log("timestep", timestep)
                log("time", time)
                output_file.write(u_, eta_, p_, eta_analytical)
    if do_write:
        with open(filename+"_D3e6_visc1e21_impliciteta_nx"+str(nx)+"_dt"+str(dt_factor)+".txt", 'w') as file:
            for line in eta_midpoint:
                file.write(f"{line}\n")

    final_error = pow(error, 0.5)/L
    return final_error


if __name__ == "__main__":
    # default case run with nx = 80 for four dt factors
    dt_factors = 4 / (2**np.arange(4))
    errors = np.array([implicit_viscous_freesurface_model(80, dtf) for dtf in dt_factors])
    np.savetxt("errors-implicit-free-surface-coupling.dat", errors)
