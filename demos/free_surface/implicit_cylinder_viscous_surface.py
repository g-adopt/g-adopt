from gadopt import *
import numpy as np
from gadopt.utility import CombinedSurfaceMeasure
dx = dx(degree=6)
from mpi4py import MPI


def implicit_viscous_freesurface_model(nx, dt_factor, do_write=True, iterative_2d=False):
    # Free surface relaxation test in a cylindrical domain.
    

    # Set up geometry:
    rmin, rmax, ncells, nlayers = 1.22, 2.22, 512, 64
    
    # Construct a circle mesh and then extrude into a cylinder:
    mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
    mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type='radial')
    bottom_id, top_id = "bottom", "top"
    n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
    domain_volume = assemble(1*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)

    # Set up geometry:
    D = 3e6  # Depth of domain in m
    L = D  # Length of the domain in m
    lam_dimensional = D/2  # wavelength of load in m
    L0 = D  # characteristic length scale for scaling the equations
    lam = lam_dimensional/L0  # dimensionless lambda

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W, W])  # Mixed function space.
    

    print("Z.extruded", W.extruded)
    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])
    
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

    number_of_lam = 4*round(2 * pi * rmax / (rmax - rmin)) # wavenumber (dimensionless)
    lam = (2*pi*rmax) / number_of_lam

    kk = 2*pi / lam
    F0 = Constant(1000 / L0)  # initial free surface amplitude (dimensionless)
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    eta_.interpolate(F0 * cos(number_of_lam * atan2(X[1], X[0])))  # Initial free surface condition

    # timestepping
    mu = Constant(1)  # Shear modulus (dimensionless)
    tau0 = Constant(2 * (kk) * mu / (rho0 * g))  # Characteristic time scale (dimensionless)
    log("tau0", tau0)

    dt = Constant(dt_factor*tau0)  # timestep (dimensionless)
    log("dt (dimensionless)", dt)

    time = Constant(0.0)
    max_timesteps = round(10*tau0/dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
    log("max_timesteps", max_timesteps)

    # No normal flow except on the free surface
    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'free_surface': {}},  # Free surface boundary conditions are applied automatically in stokes_integrators and momentum_equation for implicit free surface coupling
    }

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, cartesian=False, free_surface_dt=dt, iterative_2d=iterative_2d, nullspace=Z_nullspace, transpose_nullspace=Z_nullspace, near_nullspace=Z_near_nullspace)


    # analytical function
    eta_analytical = Function(W, name="eta analytical")
    eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(number_of_lam * atan2(X[1], X[0])))  # Initial free surface condition

    ds = CombinedSurfaceMeasure(mesh, degree=6)

    # Create output file and select output_frequency:
    if do_write:
        # Write output files in VTK format:
        dump_period = 1
        log("dump_period ", dump_period)
        filename = "implicit_cylinder_viscous_freesurface"
        output_file = File(f"{filename}_D{float(D/L0)}_mu{float(mu)}_nx{nx}_dt{float(dt/tau0)}tau.pvd")
        output_file.write(u_, eta_, p_, eta_analytical)

    error = 0
    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve Stokes sytem:
        stokes_solver.solve()
        time.assign(time + dt)
        eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(number_of_lam * atan2(X[1], X[0])))  # Initial free surface condition
        
        local_error = assemble(pow(eta-eta_analytical, 2)*ds(top_id))
        error += local_error*dt

        # Write output:
        if do_write:

            if timestep % dump_period == 0:
                log("timestep", timestep)
                log("time", float(time))
                output_file.write(u_, eta_, p_, eta_analytical)

    final_error = pow(error, 0.5)/L
    return final_error


if __name__ == "__main__":
    # default case run with nx = 80 for four dt factors
    dt_factors =  2 / (2**np.arange(4))

    # Rerun with iterative solvers
    errors_iterative = np.array([implicit_viscous_freesurface_model(80, dtf) for dtf in dt_factors])
    
    if MPI.COMM_WORLD.rank == 0:
        np.savetxt("errors-implicit-cylindrical-iterative-free-surface-coupling.dat", errors_iterative)
