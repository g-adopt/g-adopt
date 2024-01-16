from gadopt import *
from mpi4py import MPI
import os
import numpy as np
OUTPUT=True
output_directory="."

def viscous_freesurface_model(nx, dt_factor):

    # Set up geometry:
    D = 3e6 # length of domain in m
    lam = D/2 # wavelength of load in m
    L = D #lam # Depth of the domain in m
    ny = nx
    mesh = RectangleMesh(nx, ny, L, D)  # Rectangle mesh generated via firedrake
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Function to store the solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p


    eta = Function(W, name="eta")

    T = Function(Q, name="Temperature").assign(0)
    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())



    eta_eq = FreeSurfaceEquation(W, W, surface_id=top_id)

    steady_state_tolerance = 1e-9

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
    eta_nullspace=VectorSpaceBasis(constant=True)

    # Write output files in VTK format:
    u_, p_ = z.subfunctions #subfunctions  # Do this first to extract individual velocity and pressure fields.
    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    Ra = Constant(0)  # Rayleigh number
    rho0 = 4500 # density in kg/m^3
    g = 10 # gravitational acceleration in m/s^2
    approximation = BoussinesqApproximation(Ra,g=g,rho=rho0)

    kk = 2 * pi / lam # wavenumber in m^-1
    F0 = 1000 # initial free surface amplitude in m
    X = SpatialCoordinate(mesh)
    eta.interpolate(F0 * cos(kk * X[0]))
    n = FacetNormal(mesh)
    # timestepping 

    mu = 1e21 # Shear modulus in Pa
    tau0 = 2 * kk * mu / (rho0 * g) 
    print("tau0", tau0)
    dt = dt_factor*tau0/round(D/lam)  # Initial time-step for explicit free surface to be stable you need the 
    #timestep to be smaller than the maximum possible wavelength which is the domain width. Otherwise at later times an instability can grow. 
    dump_period = round(tau0/dt)
    print(dump_period)
    time = 0.0
    max_timesteps = round(10*tau0/dt)
    print("max_timesteps", max_timesteps)
    # Create output file and select output_frequency:
    filename=os.path.join(output_directory, "viscous_freesurface")
    if OUTPUT:
        output_file = File(filename+"_D"+str(D)+"_mu"+str(mu)+"_nx"+str(nx)+"_dt"+str(dt/tau0)+"tau.pvd")
    

    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'stress': -rho0 * g * eta * n},
        left_id: {'un': 0},
        right_id: {'un': 0},
    }

    eta_fields = {'velocity': u_,
                    'surface_id': top_id}

    eta_bcs = {} 

    eta_strong_bcs = [InteriorBC(W, 0., top_id)]

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
     #   'ksp_monitor': None,
    }
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, cartesian=True, solver_parameters=mumps_solver_parameters)


    eta_timestepper = BackwardEuler(eta_eq, eta, eta_fields, dt, bnd_conditions=eta_bcs, strong_bcs=eta_strong_bcs,solver_parameters=mumps_solver_parameters) 
 # bnd_conditons is different to thwaites (it is just passed in order...)
    # analytical function
    eta_analytical = Function(W, name="eta analytical")
    eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))
    
    if OUTPUT:
        output_file.write(u_, eta, p_, eta_analytical)
    
    error = 0
    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve Stokes sytem:
        stokes_solver.solve()
        eta_timestepper.advance(time)
        
        time += dt
        eta_analytical.interpolate(exp(-time/tau0)*F0 * cos(kk * X[0]))
        
        local_error = assemble(pow(eta-eta_analytical,2)*ds(top_id))
        error += local_error*dt
        # Write output:
        
        if timestep == dump_period:
            error_tau = pow(local_error,0.5)/L

        if timestep % dump_period == 0:
            print("timestep", timestep)
            print("time", time)
            if OUTPUT:
                output_file.write(u_, eta, p_, eta_analytical)
    
    final_error = pow(error,0.5)/L
    return final_error, error_tau 


dt_factors = [2,1,0.5 , 0.25]#, 0.125, 0.0625, 0.03125]
errors = np.array([viscous_freesurface_model(80, dtf) for dtf in dt_factors]) 
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('time surface displacement errors: ', errors[:,0])
print('time surface displacement conv: ', conv[:, 0])

#assert all(conv[:,0]> 0.95)

cells = [5,10]
dt_factors_spatial = [1/128, 1/512]  # use smaller timesteps for spatial test to make sure spatial error dominates
#errors = np.array([viscous_freesurface_model(c, dtf) for (c,dtf) in zip(cells, dt_factors_spatial)]) # I think the timestep needs to be really small to make sure this error does not dominate for finer meshes
#conv = np.log(errors[:-1]/errors[1:])/np.log(2)
#print('spatial surface displacement errors at t=tau: ', errors[:, 1])
#print('spatial surface displacement conv at t=tau: ', conv[:, 1])


#assert all(conv[:,1]> polynomial_order+0.85)
