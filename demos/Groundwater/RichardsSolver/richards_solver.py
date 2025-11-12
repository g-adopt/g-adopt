import firedrake as fd
import numpy as np
from gadopt.equations import interior_penalty_factor
import ufl

"""
This model provides a solver solving solving Richards equation in pressure-head form:
$$(C + S_sS) \frac{\parial h}{\partial t} = \nabla (K \grad(h + z))
where:
- $h$ is the pressure head
- $S_s$ is the specific storage coefficient
- $S(h) = (\theta - \theta_r)/(\theta_s - \theta_r)$ is the effective saturation
- $C(h) = d\theta/dh$ is the specific moisture capacity
- $K(h)$ is the hydraulic conductivity
- $z$ is the vertical coordinate (gravity term)
This equation is expressed in variational (weak) form and expressied in the residual form $0 = F(h, v)$ where $v$ is the 
INPUTS:
    h : solution at current time step to be solved for
    h_hold : solution at previous time step
    time : curent time
    time_step : size of timestep
    time_integrator : choice of method to integrate equation in time
    eq : dumb
    soil_curves : models the hydrological properties
"""


def richardsSolver(h, h_old, time, time_step, time_integrator, eq, soil_curves, bcs):

    water_retention = soil_curves.water_retention
    moisture_content = soil_curves.moisture_content
    relative_permeability = soil_curves.relative_permeability

    # Choose the time integration method
    match time_integrator:
        case "SemiImplicit":
            hDiff = h
            K = relative_permeability(h_old)
            C = water_retention(h_old)
            theta = moisture_content(h_old)
        case "ImplicitMidpoint":
            hDiff = (h + h_old)/2
            K = relative_permeability(hDiff)
            C = water_retention(hDiff)
            theta = moisture_content(hDiff)
        case 'BackwardEuler':
            hDiff = h
            K = relative_permeability(h)
            C = water_retention(h)
            theta = moisture_content(h)
        case 'CrankNicolson':
            hDiff = (h + h_old)/2
            K = (relative_permeability(h)+relative_permeability(h_old))/2
            C = (water_retention(h)+water_retention(h_old))/2
            theta = (moisture_content(h)+moisture_content(h_old))/2
        case _:
            raise ValueError("Temporal integration method not recognised")
    K_old = relative_permeability(h_old)

    # Richards equation in variational residual form
    F = mass_term(h, h_old, time_step, soil_curves, C, theta, eq) \
        + gravity_advection(K, eq) \
        + diffusion_term(hDiff, K, K_old, eq, bcs)

    problem = fd.NonlinearVariationalProblem(F, h)

    # Jacobian not needed for SemiImplicit
    if time_integrator == "SemiImplicit":
        snesType = 'ksponly'
    else:
        snesType = 'newtonls'

    # Use direct solvers for 2d and iterative for 3d
    if eq.dimen <= 2:
        ksp_type = 'preonly'
        pc_type = 'lu'
    else:
        ksp_type = 'bcgs'
        pc_type = 'hypre'

    solverRichardsNonlinear = fd.NonlinearVariationalSolver(problem,
                                solver_parameters={
                                    'mat_type': 'aij',
                                    'snes_type': snesType,
                                    'ksp_type': ksp_type,
                                    "pc_type": pc_type,
                                })

    return solverRichardsNonlinear


def mass_term(h, hOld, time_step, soil_curves, C, theta, eq):

    Ss      = soil_curves.parameters["Ss"]
    theta_s = soil_curves.parameters['theta_s']
    theta_r = soil_curves.parameters['theta_r']
    S = (theta - theta_r)/(theta_s - theta_r)

    F = fd.inner((C + Ss*S) * (h - hOld)/time_step, eq.v) * eq.dx

    return F


def diffusion_term(hDiff, K, K_old, eq, bcs):

    # Volume integral
    F = fd.inner(fd.grad(eq.v), K * fd.grad(hDiff)) * eq.dx

    # SIPG
    sigma = interior_penalty_factor(eq, shift=0)
    sigma_int = sigma * fd.avg(fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh))
    F += sigma_int*fd.inner(fd.jump(eq.v, eq.n), fd.avg(K_old) * fd.jump(hDiff, eq.n)) * eq.dS
    F -= fd.inner(fd.avg(K_old * fd.grad(eq.v)), fd.jump(hDiff, eq.n)) * eq.dS
    F -= fd.inner(fd.jump(eq.v, eq.n), fd.avg(K_old * fd.grad(hDiff))) * eq.dS

    # Impose bcs within the weak formulation
    for bc_idx, bc_info in bcs.items():
        boundaryInfo = bc_info
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]
        if boundaryType == 'h':
            sigma_ext = sigma * fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh)
            F += 2 * sigma_ext * eq.v * K * (hDiff - boundaryValue) * eq.ds(bc_idx)
            F -= fd.inner(K * fd.grad(eq.v), eq.n) * (hDiff - boundaryValue) * eq.ds(bc_idx)
            F -= fd.inner(eq.v * eq.n, K * fd.grad(hDiff)) * eq.ds(bc_idx)
        elif boundaryType == 'flux':
            F -= boundaryValue * eq.v * eq.ds(bc_idx)
        else:
            raise ValueError("Unknown boundary type, must be 'h' or 'flux'")

    return F


def gravity_advection(K, eq):

    # Gravity driven volumetric flux

    v = eq.v
    x = fd.SpatialCoordinate(eq.mesh)

    nDown = fd.grad(x[eq.dimen-1])

    q  = K * nDown
    qn = 0.5*(fd.dot(q, eq.n) + abs(fd.dot(q, eq.n)))

    # Main volume integral
    F = fd.inner(q, fd.grad(eq.v))*eq.dx
    
    # Upwinding
    F -= fd.jump(v)*(qn('+') - qn('-'))*eq.dS

    return F
