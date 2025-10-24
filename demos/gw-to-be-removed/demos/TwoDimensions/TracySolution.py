import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate
from RichardsSolver import *
from utilities import *
from modelTypes import *
import time

# Steady state solution
timeParameters = {
    "finalTime": 1e06,
    "timeStepType": "constant",
    "timeStepSize": 5000,
    "theta_diff": 1.0,
    "theta_nonlin": 1.00,
    "epsilon": 0,
    "quadratureDegree": 3,
    "Cf": 50
}

modelParameters = {
    "modelType": "exponential",
    "thetaR": 0.15,
    "thetaS": 0.45,
    "alpha": 0.328,
    "Ks": 1.0e-05,
    "Ss": 0,
}

solverParameters = {
    "fileName": "Tracy2D",
    "numberPlots": 0
}

for nodes in [36]:

    L = 15.24
    mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=False)

    mesh.cartesian = True
    x = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "Lagrange", 2)  # Function space for pressure head

    alpha = modelParameters["alpha"]
    hr = -L
    h0 = 1 - exp(alpha*hr)

    def setBoundaryConditions(timeConstant, x, K, theta):

        leftBC, rightBC, bottomBC, topBC = 1, 2, 3, 4
        # Boundary conditions no side flux (exactSolutionNoFlux)
        boundaryCondition = {
            leftBC: {'q': 0*hr},
            rightBC: {'q': 0*hr},
            bottomBC: {'h': hr},
            topBC: {'h': (1/alpha)*ln(exp(alpha*hr) + (h0/2)*(1 - cos(2*pi*x[0]/L)))},
        }

        # Boundary conditions prescribed pressure head (exactSolutionSpecifiedHead)
        boundaryCondition = {
            leftBC: {'h': hr},
            rightBC: {'h': hr},
            bottomBC: {'h': hr},
            topBC: {'h': (1/alpha)*ln(exp(alpha*hr) + (h0)*(sin(pi*x[0]/15.24)))},
        }

        return boundaryCondition

    def exactSolutionSpecifiedHead(alpha, x, t):

        # Exact solution from Tracy 2006 (https://doi.org/10.1029/2005WR004638, page 4)
        h0 = 1 - exp(alpha * hr)
        beta = sqrt(alpha**2/4 + (pi/L)**2)
        hss = h0*sin(pi*x[0]/L)*exp((alpha/2)*(L - x[1]))*sinh(beta*x[1])/sinh(beta*L)
        c = alpha*(modelParameters["thetaS"] - modelParameters["thetaR"])/modelParameters["Ks"]

        phi = 0
        for k in range(1, 200):
            lambdak = k*pi/L
            gamma = (beta**2 + lambdak**2)/c
            phi = phi + ((-1)**k)*(lambdak/gamma)*sin(lambdak*x[1])*exp(-gamma*t)
        phi = phi*((2*h0)/(L*c))*sin(pi*x[0]/L)*exp(alpha*(L-x[1])/2)

        hBar = hss + phi

        hExact = ((1/alpha)*ln(exp(alpha*hr) + hBar))

        return hExact


    def exactSolutionNoFlux(alpha, x, t):

        # Exact solution from Tracy 2006 (https://doi.org/10.1029/2005WR004638, page 5)
        h0 = 1 - exp(alpha * hr)
        beta = sqrt(alpha**2/4 + (2*pi/L)**2)
        hss = (h0/2)*exp((alpha/2)*(L - x[1]))*(sinh(alpha*x[1]/2)/sinh(alpha*L/2) - cos(2*pi*x[0]/L)*sinh(beta*x[1])/sinh(beta*L))
        c = alpha*(modelParameters["thetaS"] - modelParameters["thetaR"])/modelParameters["Ks"]

        phi = 0
        for k in range(1, 200):
            lambdak = k*pi/L
            gamma1 = (lambdak**2 + alpha**2/4)/c
            gamma2 = ((2*pi/L)**2 + lambdak**2 + alpha**2/4)/c
            phi = phi + ((-1)**k)*lambdak*((1/gamma1)*exp(-gamma1*t) - (1/gamma2)*cos(2*pi*x[0]/L)*exp(-gamma2*t))*(sin(lambdak*x[1]))
        phi = phi*((h0)/(L*c))*exp(alpha*(L-x[1])/2)

        hBar = hss + phi

        hExact = ((1/alpha)*ln(exp(alpha*hr) + hBar))

        return hExact

    offset = 2000
    hInitial = Function(V, name="InitialCondition")
    hInitial.interpolate(exactSolutionSpecifiedHead(alpha, x, offset))

    start = time.time()
    VTKFile("output.pvd").write(hInitial)
    h, theta, q, K = RichardsSolver(hInitial, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions)
    VTKFile("output_f.pvd").write(h, theta, K, q)
    end = time.time()
    PETSc.Sys.Print(f"Simulation time: {end - start}")

    quadratureDegree = timeParameters["quadratureDegree"]
    dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadratureDegree})
    hExact = exactSolutionSpecifiedHead(alpha, x, timeParameters["finalTime"]+offset)
    thetaExact = moistureContent(modelParameters, hExact, x, 0)
    PETSc.Sys.Print("L2 error", assemble(sqrt(dot((h - hExact), (h - hExact)))*dx))
    PETSc.Sys.Print()