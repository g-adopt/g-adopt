import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate
from RichardsSolver import *
import time

# Transient solution
timeParameters = {
    "finalTime": 2**17,
    "timeStepType": "constant",
    "timeStepSize": 2**(7),
    "theta_diff": 0.5,
    "theta_nonlin": 0.5,
}

# Steady state solution
#timeParameters = {
#    "finalTime": 5e07,
#    "timeStepType": "constant",
#    "timeStepSize": 5e07/1000,
#    "timeIntegration": 'modifiedEuler'
#}
#

solverParameters = {
    "fileName": "Tracy2D.pvd",
    "numberPlots": 0
}

nodes, L = 201, 15.24
mesh = RectangleMesh(nodes, nodes, L, L, name="mesh")

mesh.cartesian = True
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)


modelParameters = {
    "modelType": "exponential",
    "thetaR": 0.15,
    "thetaS": 0.45,
    "alpha": 0.492,
    "Ks": 1.0e-05,
    "Ss": 0,
}

alpha = modelParameters["alpha"]
hr = -L
H0 = 1 - exp(alpha*hr)

beta = 10
hssBar = H0*sin(pi*x[0]/L)*exp((alpha/2)*(15.24-x[1]))*sinh(beta*x[1])/sinh(beta*15.24)

h0 = Function(V, name="InitialCondition")
h0.interpolate((1/alpha)*ln(exp(alpha*hr) + hssBar)),


def setBoundaryConditions(timeConstant, x):

    leftBC, rightBC, bottomBC, topBC = 1, 2, 3, 4
    boundaryCondition = {
        leftBC: {'h': hr},
        rightBC: {'h': hr},
        bottomBC: {'h': hr},
        topBC: {'h': (1/alpha)*ln(exp(alpha*hr) + (H0)*(sin(pi*x[0]/15.24)))},
    }

    return boundaryCondition


start = time.time()
RichardsSolver(h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions)
end = time.time()
print(end - start)
