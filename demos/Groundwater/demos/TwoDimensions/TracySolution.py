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
from scipy.io import savemat, loadmat

# Steady state solution
timeParameters = {
    "finalTime": 5e07,
    "timeStepType": "constant",
    "timeStepSize": 1e05,
    "theta_diff": 1,
    "theta_nonlin": 0,
    "epsilon": 0,
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
    "numberPlots": 1
}

for nodes in [501]:

    L = 15.24
    mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=False)

    mesh.cartesian = True
    x = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)  # Function space for volumetric flux

    alpha = modelParameters["alpha"]
    hr = -L
    H0 = 1 - exp(alpha*hr)

    h0 = Function(V, name="InitialCondition")
    h0.interpolate(conditional(x[1] >= L, (1/alpha)*ln(exp(alpha*hr) + (H0)*(sin(pi*x[0]/15.24))), hr))

    def setBoundaryConditions(timeConstant, x, K, theta):

        leftBC, rightBC, bottomBC, topBC = 1, 2, 3, 4
        boundaryCondition = {
            leftBC: {'h': hr},
            rightBC: {'h': hr},
            bottomBC: {'h': hr},
            topBC: {'h': (1/alpha)*ln(exp(alpha*hr) + (H0)*(sin(pi*x[0]/15.24)))},
        }

        return boundaryCondition


    start = time.time()
    h, theta, q, K = RichardsSolver(h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions)
    end = time.time()
    print(f"Simulation time: {end - start}")

    # Exact solution from Tracy 2006
    t = timeParameters["finalTime"]
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
    PETSc.Sys.Print("L2 error", assemble(sqrt(dot((h - hExact), (h - hExact)))*dx))
