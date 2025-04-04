import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate
from RichardsSolver import *
import time

timeParameters = {
    "finalTime"    : 86400,
    "timeStepType" : "constant",
    "timeStepSize" : 5,
    "timeIntegration" : 'IMEX'
}

solverParameters = {
  "fileName"         : "multiMaterial.pvd",
  "numberPlots"      : 30
}

nodesX = 151; nodesZ = 101;
mesh = RectangleMesh( nodesX, nodesZ, 3, 2, name = "mesh" )
mesh.cartesian = True
x     = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

'''
modelParameters = {
   "modelType" : "VanGenuchten",
   "thetaR"    : 0.10,
   "thetaS"    : 0.37,
   "alpha"     : 3.35,
   "n"         : 2.00,
   "Ks"        : 5e-05,
   "Ss"        : 1e-05,
   "gridSpacing" : 0,
}

'''
modelParameters = {
   "modelType" : "Haverkamp",
   "thetaR"    : 0.10,
   "thetaS"    : 0.37,
   "alpha"     : 0.44,
   "beta"      : 1.2924,
   "A"         : 0.0104,
   "gamma"     : 1.5722,
   "Ks"        : 5e-05,
   "Ss"        : 1e-05,
   "gridSpacing" : 0,
}


h0   = Function(V, name="InitialCondition")
h0.assign( -10. )

def setBoundaryConditions(timeConstant, x):

    leftBC, rightBC, bottomBC, topBC = 1,2,3,4
    boundaryCondition = {
    leftBC   : {'q' : 0 },
    rightBC  : {'q' : 0 },
    bottomBC : {'q' : 0 },    # cm/s
    topBC    : {'q' : tanh(0.0005*timeConstant)*2e-05*(0.5*(1 + tanh(10*(x[0]+0.5))) - 0.5*(1 + tanh(10*(x[0]-0.5))))},
    }

    return boundaryCondition

start = time.time()
RichardsSolver( h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions )
end = time.time()
print(end - start)
