"""
This runs the optimisation portion of the adjoint test case. A forward run first sets up
the tape with the adjoint information, then a misfit functional is constructed to be
used as the goal condition for nonlinear optimisation using ROL.

annulus_taylor_test is also added to this script for testing the correctness of the gradient for the inverse problem.
    taylor_test(alpha_T, alpha_u, alpha_d, alpha_s):
            alpha_T (float): The coefficient of the temperature misfit term.
            alpha_u (float): The coefficient of the velocity misfit term.
            alpha_m (float): The coefficient of the melt region term.
            alpha_d (float): The coefficient of the initial condition damping term.
            alpha_s (float): The coefficient of the smoothing term.
            float: The minimum convergence rate from the Taylor test. (Should be close to 2)
"""
from gadopt import *
from gadopt.inverse import *
#from checkpoint_schedules import SingleMemoryStorageSchedule
import numpy as np
import sys, os
from mpi4py import MPI

melt_approx = False

def inverse(alpha_T=1e0, alpha_u=1e-1, alpha_m=1e-1, alpha_d=1e-2, alpha_s=1e-1, checkpoint_restore=0, restore_iteration=0,
            max_iteration=1, runID=0):

    # Checkpoint directory for ROL:
    rol_checkpoint_directory = f"../optimisation_checkpoint_{runID}"
    
    # For solving the inverse problem we need the reduced functional, any callback functions,
    # and the initial guess for the control variable. 
    inverse_problem = generate_inverse_problem(alpha_u=alpha_u, alpha_m=alpha_m, alpha_d=alpha_d, alpha_s=alpha_s,
                                               rol_checkpoint_dir=rol_checkpoint_directory, restore_iteration=restore_iteration)

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(inverse_problem["control"].function_space(), name="Lower_bound_temperature")
    T_ub = Function(inverse_problem["control"].function_space(), name="Upper_bound_temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    # Set up to use Newton Krylov:
    minimisation_problem = MinimizationProblem(inverse_problem["reduced_functional"], bounds=(T_lb, T_ub))    
    minimisation_parameters["Status Test"]["Iteration Limit"] = max_iteration
    minimisation_parameters["Step"]["Line Search"] = {"Descent Method": {"Type": "Newton-Krylov"}}
    
    rol_solver = ROLSolver(minimisation_problem, minimisation_parameters, inner_product="L2")
    rol_params = ROL.ParameterList(minimisation_parameters, "Parameters")
    rol_algorithm = ROL.LineSearchAlgorithm(rol_params)
    rol_algorithm.setStatusTest(inverse_problem["status_test"](rol_params), False)
    rol_algorithm.run(rol_solver.rolvector, rol_solver.rolobjective)

    # If we're performing multiple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them.
    continue_annotation()


def annulus_taylor_test(alpha_T, alpha_u, alpha_m, alpha_d, alpha_s):
    """
    Perform a Taylor test to verify the correctness of the gradient for the inverse problem.

    This function calls a main function to populate the tape for the inverse problem
    with specified regularization parameters, generates a random perturbation for the control variable,
    and performs a Taylor test to ensure the gradient is correct. Finally, it ensures that annotations
    are switched back on for any subsequent tests.

    Returns:
        minconv (float): The minimum convergence rate from the Taylor test.
    """

    # For solving the inverse problem we the reduced functional, any callback functions,
    # and the initial guess for the control variable
    inverse_problem = generate_inverse_problem(alpha_T, alpha_u, alpha_m, alpha_d, alpha_s)

    # generate perturbation for the control variable
    delta_temp = Function(inverse_problem["control"].function_space(), name="Delta_Temperature")
    delta_temp.dat.data[:] = np.random.random(delta_temp.dat.data.shape)

    # Perform a taylor test to ensure the gradient is correct
    minconv = taylor_test(
        inverse_problem["reduced_functional"],
        inverse_problem["control"],
        delta_temp
    )

    # If we're performing mulitple successive tests we want
    # to ensure the annotations are switched back on for the next code to use them
    continue_annotation()

    return minconv


def generate_inverse_problem(alpha_T=1.0, alpha_u=-1, alpha_m=-1, alpha_d=-1, alpha_s=-1, rol_checkpoint_dir=None, restore_iteration=0):
    """
    Use adjoint-based optimisation to solve for the initial condition of the cylindrical
    problem.

    Parameters:
        alpha_u: The coefficient of the velocity misfit term
        alpha_m: The coefficient of the melt region term
        alpha_d: The coefficient of the initial condition damping term
        alpha_s: The coefficient of the smoothing term
    """

    # Get working tape
    tape = get_working_tape()
    tape.clear_tape()

    # If we are not annotating, let's switch on taping
    if not annotate_tape():
        continue_annotation()

    # # Writing to disk for block variables
    # enable_disk_checkpointing()

    # Using SingleMemoryStorageSchedule
#    if any([alpha_T > 0, alpha_u > 0, alpha_m > 0]):
#        tape.enable_checkpointing(SingleMemoryStorageSchedule())
        
    # Set up geometry:
    rmax = 2.22
    rmax_earth = 6370  # Radius of Earth [km]
    rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
    r_410_earth = rmax_earth - 410  # 410 radius [km]
    r_660_earth = rmax_earth - 660  # 660 raidus [km]
    r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
    r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

    with CheckpointFile("../../Forward/Checkpoint_State.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        mesh.cartesian = False

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W])
    R = FunctionSpace(mesh, "R", 0)  # Real number function space

    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    z.assign(0)
    u, p = split(z)  # Returns symbolic UFL expression for u and p    
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")

    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2)
    Ra = Constant(1e7)  # Rayleigh number

    # Define time stepping parameters:
    max_timesteps = 250
    delta_t = Function(R, name="delta_t").assign(3.0e-06)  # Constant time step

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")
    T_0 = Function(Q, name="T_0")  # Temperature for zeroth time-step
    Taverage = Function(Q1, name="Average Temperature")

    # Initialise the control. If we are not restarting from a pre-existing ROL state,
    # we set this to the final temperature state of our synthetic forward model. If we are,
    # we load it from the ROL checkpoint:
    forward_checkpoint_file = CheckpointFile("../../Forward/Checkpoint_State.h5", "r")
    if restore_iteration == 0:
        # Load from the final iteration of the forward checkpoint
        Tic.project(
            forward_checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
        )
    else:
        # Load from a specific ROL restore checkpoint
        rol_path = f"{rol_checkpoint_dir}/{restore_iteration}/solution_checkpoint.h5"
        rol_checkpoint_file = CheckpointFile(rol_path, "r")
        Tic.project(
            rol_checkpoint_file.load_function(mesh, "dat_0")
        )

    # Load layer average in all cases. 
    Taverage.project(forward_checkpoint_file.load_function(mesh, "Average Temperature", idx=0))

    # Temperature function in Q2, where we solve the equations
    T = Function(Q, name="Temperature")

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

    # Configure approximation
    approximation = BoussinesqApproximation(Ra, mu=mu)

    # Nullspaces and near-nullspaces:
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
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
        solver_parameters="direct"
    )

    # Control variable for optimisation
    control = Control(Tic)

    # If we are using surface veolocit misfit in the functional
    if alpha_u > 0:
        u_surface_misfit = 0.0

    # Likewise for melt misfit:
    if alpha_m > 0:
        m_misfit = 0.0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T_0.project(Tic, bcs=energy_solver.strong_bcs)
    T.assign(T_0)

    # if the weighting for misfit terms non-positive, then no need to integrate in time
    min_timesteps = 0 if any([w > 0 for w in [alpha_T, alpha_u, alpha_m]]) else max_timesteps

    # Generate a surface velocity reference
    uobs = Function(V, name="uobs")

    # Populate the tape by running the forward simulation
    for timestep in tape.timestepper(iter(range(min_timesteps, max_timesteps))):

        # Update the accumulated melt region misfit using observed temperature and melt region values
        if alpha_m > 0:
            if melt_approx == False:  # Full melting info
                M_mask = forward_checkpoint_file.load_function(mesh, name="Melt_Indicator", idx=timestep)
                T_melt = forward_checkpoint_file.load_function(mesh, name="Temperature", idx=timestep)
                m_misfit += assemble(Function(R, name="alpha_m").assign(float(alpha_m)) * M_mask * (T - T_melt) ** 2 * dx)
            elif melt_approx == True:  # Approximation of melting info
                M_mask = forward_checkpoint_file.load_function(mesh, name="Melt_Indicator_30", idx=timestep)
                T_melt = forward_checkpoint_file.load_function(mesh, name="Melt_T_Average", idx=timestep)
                M_clusters = forward_checkpoint_file.load_function(mesh, name="Cluster_ID", idx=timestep)
                max_clusters = M_clusters.dat.data.max()
                max_clusters = int(M_clusters.comm.allreduce(max_clusters, MPI.MAX))
                eps = 1e-12
                # Take misfit in an average sense, per cluster:
                for c in range(max_clusters):
                    M_c = float(c + 1)
                    m_misfit += assemble(Function(R, name="alpha_m").assign(float(alpha_m)) * conditional(abs(M_clusters - M_c) < eps, T - T_melt, 0.0) * dx) ** 2
#                    m_misfit += assemble(conditional(abs(M_clusters - M_c) < eps, T - T_melt, 0.0) * dx) ** 2
            log(f"M_misfit: {m_misfit}")

        stokes_solver.solve()
        energy_solver.solve()

        if alpha_u > 0:
            # Update the accumulated surface velocity misfit using the observed value
            uobs.assign(forward_checkpoint_file.load_function(mesh, name="Velocity", idx=timestep))
            u_surface_misfit += assemble(Function(R, name="alpha_u").assign(float(alpha_u)) * dot(u - uobs, u - uobs) * ds_t)
            log(f"U_surface_misfit: {u_surface_misfit}")

    # Load observed final state.
    Tobs = forward_checkpoint_file.load_function(mesh, "Smooth_Temperature", idx=max_timesteps - 1)
    Tobs.rename("Observed Temperature")

    # Load true final state.
    T_final_true = forward_checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    T_final_true.rename("True Final Temperature")
    
    # Load reference initial state (needed to measure performance).
    Tic_ref = forward_checkpoint_file.load_function(mesh, "Temperature", idx=0)
    Tic_ref.rename("Reference Initial Temperature")

    # Load average temperature profile.
    Taverage = forward_checkpoint_file.load_function(mesh, "Average Temperature", idx=0)

    # Load final state melt info for normalisation:
    if alpha_m > 0:
        if melt_approx == False:  # Full melting info        
            M_mask_final = forward_checkpoint_file.load_function(mesh, name="Melt_Indicator", idx=max_timesteps-1)
            T_melt_final = forward_checkpoint_file.load_function(mesh, name="Temperature", idx=max_timesteps-1)
        elif melt_approx == True:  # Approximation of melting info            
            M_mask_final = forward_checkpoint_file.load_function(mesh, name="Melt_Indicator_30", idx=max_timesteps-1)
            T_melt_final = forward_checkpoint_file.load_function(mesh, name="Melt_T_Average", idx=max_timesteps-1)
            
    # All fields loaded. Close checkpoint.
    forward_checkpoint_file.close()

    # Initiate objective functional.
    objective = 0.0

    if any([w > 0 for w in [alpha_u, alpha_m, alpha_d, alpha_s]]):
        # Calculate norms of the observed temperature, it will be used in multiple spots later
        norm_obs = assemble(Tobs**2 * dx)
        log(f"Norm Obs: {norm_obs}")        

    # Define the component terms of the overall objective functional

    # Temperature term
    if alpha_T > 0:
        # Temperature misfit between solution and observation
        T_misfit = assemble(Function(R, name="alpha_T").assign(float(alpha_T)) * (T - Tobs) ** 2 * dx)
        log(f"T_misfit: {T_misfit}")
        objective += T_misfit
        log(f"Objective (T): {objective}")        

    # Velocity misfit term
    if alpha_u > 0:
        norm_u_surface = assemble((max_timesteps - min_timesteps) * dot(uobs, uobs) * ds_t)  # measure of u_obs from the last timestep
        u_contribution = norm_obs * u_surface_misfit / norm_u_surface
        objective += u_contribution
        log(f"Norm Obs: {norm_obs}")
        log(f"Norm Us: {norm_u_surface}")                                        
        log(f"U_contribution: {u_contribution}")                        
        log(f"Objective (TU): {objective}")                

    # Melt region misfit term
    if alpha_m > 0:
        # This assumes that melting at the present day is roughly representative of melting through the geological past
        # in an integral sense. It also scales volumes between mask and entire domain.
        ratio = assemble(M_mask_final * dx) / assemble(Constant(1) * dx(domain=M_mask_final.function_space().mesh()))
        norm_m = assemble(ratio * (max_timesteps - min_timesteps) * dot(M_mask_final * T_melt_final, M_mask_final * T_melt_final) * dx)
        m_contribution = norm_obs * m_misfit / norm_m
        objective += m_contribution
        log(f"Ratio: {ratio}")                
        log(f"Norm Obs: {norm_obs}")        
        log(f"Norm M: {norm_m}")                    
        log(f"M_contribution: {m_contribution}")
        log(f"Objective (TUM): {objective}")

    # Damping term
    if alpha_d > 0:
        damping = assemble(Function(R, name="alpha_d").assign(float(alpha_d)) * (T_0 - Taverage) ** 2 * dx)
        log(f"Damping: {damping}")        
        norm_damping = assemble((Tobs - Taverage)**2 * dx)
        d_contribution = norm_obs * damping / norm_damping
        objective += d_contribution
        log(f"D_contribution: {d_contribution}")
        log(f"Objective (TUMD): {objective}")                

    # Smoothing term
    if alpha_s > 0:
        smoothing = assemble(Function(R, name="alpha_s").assign(float(alpha_s)) * dot(grad(T_0 - Taverage), grad(T_0 - Taverage)) * dx)
        log(f"Smoothing: {smoothing}")                
        norm_smoothing = assemble(dot(grad(Tobs - Taverage), grad(Tobs - Taverage)) * dx)
        s_contribution = norm_obs * smoothing / norm_smoothing
        objective += s_contribution
        log(f"S_contribution: {s_contribution}")        
        log(f"Objective (TUMDS): {objective}")                

    # All done with the forward run, stop annotating anything else to the tape.
    pause_annotation()

    inverse_problem = {}

    # Specify control.
    inverse_problem["control"] = Tic

    # ReducedFunctional that is to be minimised.
    inverse_problem["reduced_functional"] = ReducedFunctional(objective, control)
    
    # Callback function to print misfit at start and end of each optimisation iteration.
    # For sake of book-keeping the simulation, we have also implemented a user-defined way of
    # recording information that might be used to check the optimisation performance. This
    # callback function will be executed at the end of each iteration. Here, we write out
    # the control field, i.e., the reconstructed intial temperature field, at the end of
    # each iteration and final stage T. To access the last value of *an overloaded object* we
    # should access the `.block_variable.checkpoint` method as below. We also record the values
    # of the reduced functional and misfits directly in order to produce a plot of the convergence.
    
    solutions_vtk = VTKFile("solutions.pvd")
    solution_IC = Function(Tic.function_space(), name="Initial_Temperature")
    solution_final = Function(T.function_space(), name="Final_Temperature")
    functional_values = []
    initial_misfit_values = []
    final_misfit_values = []

    # Log value of reduced functional:
    def record_functional_values(func_value, *args):
        functional_values.append(func_value)

    # Log values of initial and final misfit:
    def record_misfit_values(init_misfit, final_misfit):
        initial_misfit_values.append(init_misfit)
        final_misfit_values.append(final_misfit)

    class StatusTest(ROL.StatusTest):
        def check(self, status):
            initial_misfit = assemble(
                (Tic.block_variable.checkpoint - Tic_ref) ** 2 * dx
            )
            final_misfit = assemble(
                (T.block_variable.checkpoint - T_final_true) ** 2 * dx
            )
            observed_misfit = assemble(
                (T.block_variable.checkpoint - Tobs) ** 2 * dx
            )        

            inverse_problem["reduced_functional"].eval_cb_post = record_functional_values
            record_misfit_values(initial_misfit, final_misfit)

            # Print output for ease of tracking simulation progress:
            if functional_values:
                log(f"Functional: {functional_values[-1]};  Misfit (IC): {initial_misfit};  Misfit (Final): {final_misfit}; Misfit (Observed): {observed_misfit}")
            else:
                log(f"Functional value not recorded; Misfit (IC): {initial_misfit}; Misfit (Final): {final_misfit}; Misfit (Observed): {observed_misfit}")
            
            # Write functional and misfit values to a file (appending to avoid overwriting)
            if MPI.COMM_WORLD.Get_rank() == 0:        
                with open("optimisation_values.txt", "a") as f:
                    if functional_values:            
                        f.write(f"{functional_values[-1]}, {initial_misfit}, {final_misfit}, {observed_misfit}\n")
                    else:
                        f.write(f"0.0, {initial_misfit}, {final_misfit}, {observed_misfit}\n")
            
            # Write VTK output:
            solution_IC.assign(Tic.block_variable.checkpoint)
            solution_final.assign(T.block_variable.checkpoint)
            solutions_vtk.write(solution_IC, solution_final)
            
            return super().check(status)

    inverse_problem["status_test"] = StatusTest
    
    return inverse_problem


if __name__ == "__main__":
    inverse(alpha_T=float(os.environ['ALPHA_T']), alpha_u=float(os.environ['ALPHA_U']), alpha_m=float(os.environ['ALPHA_M']), alpha_d=float(os.environ['ALPHA_D']), alpha_s=float(os.environ['ALPHA_S']), checkpoint_restore=float(os.environ['CHECKPOINT_RESTORE']), restore_iteration=int(os.environ['RESTORE_IT']), max_iteration=int(os.environ['MAX_IT']), runID=int(os.environ["ID"]))
