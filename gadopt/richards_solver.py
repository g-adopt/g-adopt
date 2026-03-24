import firedrake as fd
from RichardsSolver.utilities import interior_penalty_factor
import numpy as np
import ufl

"""
Richards equation solver in mixed form:

    ∂θ/∂t = ∇·(K(h) ∇(h + z))

where:
    h  = pressure head [L]
    S  = effective saturation [-]
    K  = hydraulic conductivity [L/T]
    z  = vertical coordinate [L]

The solver constructs the weak residual F(h, v) = 0 for a given test function v.
"""


def RichardsSolver(h: fd.Function,  
                   h_old: fd.Function, 
                   time: fd.Constant, 
                   time_step: fd.Constant, 
                   eq):

    # Determine the evaluation state for spatial terms
    match eq.time_integrator:
        case 'BackwardEuler':
            h_eval = h
        case 'ImplicitMidpoint':
            h_eval = 0.5 * (h + h_old)
        case 'CrankNicolson':
            h_eval = None # ImplicitMidpoint is generally better than CN
        case _:
            raise ValueError(f"Unknown time integrator '{eq.time_integrator}'")

    # Mass and Source are usually consistent across methods
    F = richards_mass_term(h, h_old, time_step, eq) + richards_source_term(eq)

    # Add spatial terms
    if h_eval is not None:
        F += richards_gravity_advection(h_eval, eq) + richards_diffusion_term(h_eval, eq)
    else:
        # True Crank-Nicolson (Average of fluxes)
        F += 0.5 * (richards_gravity_advection(h, eq) + richards_gravity_advection(h_old, eq))
        F += 0.5 * (richards_diffusion_term(h, eq) + richards_diffusion_term(h_old, eq))

    problem = fd.NonlinearVariationalProblem(F, h)

    solver_parameters = eq.solver_parameters

    solverRichardsNonlinear = fd.NonlinearVariationalSolver(
                                problem,
                                solver_parameters=solver_parameters
                                )

    return solverRichardsNonlinear


def richards_mass_term(h: fd.Function, h_old: fd.Function, time_step: fd.Constant, eq):

    theta = eq.soil_curves.moisture_content
    F_mixed = fd.inner((theta(h) - theta(h_old))/time_step, eq.test_function) * eq.dx

    # Use this if wanting to solve in pressure-head form
    water_retention = eq.soil_curves.water_retention
    match eq.time_integrator:
        case "ImplicitMidpoint":
            C = water_retention(0.5*(h+h_old)) 
        case 'BackwardEuler':
            C = water_retention(h)
        case 'CrankNicolson':
            C = 0.5*(water_retention(h)+water_retention(h_old))

    F_head = fd.inner(C*(h - h_old)/time_step, eq.test_function) * eq.dx

    if eq.equation_form == 'MixedForm':
        return F_mixed
    elif eq.equation_form == 'PressureHeadForm':
        return F_head


def richards_source_term(eq):

    F = -fd.inner(eq.source_term, eq.test_function) * eq.dx

    return F


def richards_diffusion_term(h: fd.Function, eq):

    v = eq.test_function
    grad_v = fd.grad(v)
    bcs = eq.bcs

    relative_conductivity = eq.soil_curves.relative_conductivity
    K = relative_conductivity(h)

    # Volume integral
    F = fd.inner(grad_v, K * fd.grad(h)) * eq.dx

    # SIPG
    sigma = interior_penalty_factor(eq, shift=0)
    sigma_int = sigma * fd.avg(fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh))

    jump_v = fd.jump(v, eq.n)
    jump_h = fd.jump(h, eq.n)
    avg_K  = fd.avg(K)

    F += sigma_int * fd.inner(jump_v, avg_K * jump_h) * eq.dS
    F -= fd.inner(fd.avg(K * grad_v), jump_h) * eq.dS
    F -= fd.inner(jump_v, fd.avg(K * fd.grad(h))) * eq.dS

    # Impose bcs within the weak formulation
    for bc_idx, bc_info in bcs.items():
        boundaryInfo = bc_info
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]
        if boundaryType == 'h':
            sigma_ext = sigma * fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh)
            diff = h - boundaryValue

            F += 2 * sigma_ext * v * K * diff * eq.ds(bc_idx)
            F -= fd.inner(K * grad_v, eq.n) * diff * eq.ds(bc_idx)
            F -= fd.inner(v * eq.n, K * fd.grad(h)) * eq.ds(bc_idx)
        elif boundaryType == 'flux':
            F -= boundaryValue * eq.test_function * eq.ds(bc_idx)
        else:
            raise ValueError("Unknown boundary type, must be 'h' or 'flux'")

    return F


def richards_gravity_advection(h: fd.Function, eq):

    v = eq.test_function
    x = fd.SpatialCoordinate(eq.mesh)

    K = eq.soil_curves.relative_conductivity(h)
    e_z = fd.grad(x[eq.dim - 1])
    q = K * e_z

    # Conservative split: - ∫Ω q · ∇v
    F = -fd.inner(q, fd.grad(v)) * eq.dx

    # Interior upwind flux:  ∫F (q̂·n) [v]
    qn = 0.5 * (fd.dot(q, eq.n) + abs(fd.dot(q, eq.n)))  # one-sided “+”
    F += fd.jump(v) * (qn('+') - qn('-')) * eq.dS

    return -F
