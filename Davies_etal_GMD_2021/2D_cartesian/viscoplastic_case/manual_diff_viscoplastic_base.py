# Stokes solver dictionary:
stokes_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,    
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Energy solver dictionary:
energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

---------------------------------------------------------------------------------------------
# Viscosity calculation and Rayleigh number:
Ra = Constant(100.) # Rayleigh number
gamma_T, gamma_Z = Constant(ln(10**5)), Constant(ln(10))
mu_star, sigma_y = Constant(0.001), Constant(1.0)
epsilon = 0.5 * (grad(u)+transpose(grad(u))) # strain-rate
epsii = sqrt(inner(epsilon,epsilon) + 1e-20) # 2nd invariant (with tolerance to ensure stability)
mu_lin = exp(-gamma_T*Tnew + gamma_Z*(1 - X[1]))
mu_plast = mu_star + (sigma_y / epsii)
mu = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast)

---------------------------------------------------------------------------------------------
# Updated solver:
stokes_solver = NonlinearVariationalSolver(stokes_problem, solver_parameters=stokes_solver_parameters, nullspace=p_nullspace, transpose_nullspace=p_nullspace)
energy_solver = NonlinearVariationalSolver(energy_problem, solver_parameters=energy_solver_parameters)

