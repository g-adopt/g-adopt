from gadopt import *
import numpy as np


def implicit_viscous_freesurface_model(nx, dt_factor, do_write=False):

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
    Z = MixedFunctionSpace([V, W, W])  # Mixed function space.

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p, eta = split(z)  # Returns symbolic UFL expression for u, p and eta
    u_, p_, eta_ = z.subfunctions  # Returns functions for u, p and eta

    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")
    eta_.rename("eta")

    T = Function(Q, name="Temperature").assign(0)  # Setup a dummy function for temperature
    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(0)  # Rayleigh number, here we set this to zero as there are no bouyancy terms
    approximation = BoussinesqApproximation(Ra)

    rho0 = approximation.rho  # This defaults to rho0 = 1 (dimensionless)
    g = approximation.g  # This defaults to g = 1 (dimensionless)

    kk = Constant(2 * pi / lam)  # wavenumber (dimensionless)
    F0 = Constant(1000 / L0)  # initial free surface amplitude (dimensionless)
    X = SpatialCoordinate(mesh)
    eta_.interpolate(F0 * cos(kk * X[0]))  # Initial free surface condition

    # timestepping
    mu = Constant(1)  # Shear modulus (dimensionless)
    tau0 = Constant(2 * kk * mu / (rho0 * g))  # Characteristic time scale (dimensionless)
    log("tau0", tau0)

    dt = Constant(dt_factor*tau0)  # timestep (dimensionless)
    log("dt (dimensionless)", dt)

    time = Constant(0.0)
    max_timesteps = round(10*tau0/dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
    log("max_timesteps", max_timesteps)

    # No normal flow except on the free surface
    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {},  # Free surface boundary conditions are applied automatically in stokes_integrators and momentum_equation for implicit free surface coupling
        left_id: {'un': 0},
        right_id: {'un': 0},
    }

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, cartesian=True, equations=FreeSurfaceStokesEquations, free_surface_dt=dt, free_surface_id=top_id)
    stokes_solver.solver_parameters.update(
            {'pc_fieldsplit_0_fields': '0',
             'pc_fieldsplit_1_fields': '1,2',
             })

    if do_write:
        eta_midpoint = []
        eta_midpoint.append(eta_.at((L/L0)/2, (D/L0)-0.001/L0))

    # analytical function
    eta_analytical = Function(W, name="eta analytical")
    eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

    # Create output file and select output_frequency:
    if do_write:
        # Write output files in VTK format:
        dump_period = 1
        log("dump_period ", dump_period)
        filename = "implicit_viscous_freesurface"
        output_file = File(filename+"_D"+str(float(D/L0))+"_mu"+str(float(mu))+"_nx"+str(nx)+"_dt"+str(float(dt/tau0))+"tau.pvd")
        output_file.write(u_, eta_, p_, eta_analytical)

    error = 0
    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve Stokes sytem:
        stokes_solver.solve()
        time.assign(time + dt)
        eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))

        local_error = assemble(pow(eta-eta_analytical, 2)*ds(top_id))
        error += local_error*dt

        # Write output:
        if do_write:
            eta_midpoint.append(eta_.at((L/L0)/2, (D/L0)-0.001/L0))

            if timestep % dump_period == 0:
                log("timestep", timestep)
                log("time", float(time))
                output_file.write(u_, eta_, p_, eta_analytical)
    if do_write:
        with open(filename+"_D"+str(float(D/L0))+"_mu"+str(float(mu))+"_nx"+str(nx)+"_dt"+str(float(dt/tau0))+"tau.txt", 'w') as file:
            for line in eta_midpoint:
                file.write(f"{line}\n")

    final_error = pow(error, 0.5)/L
    return final_error


if __name__ == "__main__":
    # default case run with nx = 80 for four dt factors
    dt_factors = 2 / (2**np.arange(4))
    errors = np.array([implicit_viscous_freesurface_model(80, dtf) for dtf in dt_factors])
    np.savetxt("errors-implicit-free-surface-coupling.dat", errors)
