## Reproduces the two-dimensional test problem by Kirkland et al. 

import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
from firedrake import *
from RichardsSolver import *
import time

timeParameters = {
    "finalTime"    : 86400,
    "timeStepType" : "constant",
    "timeStepSize" : 15,
    "timeIntegration" : 'crankNicolson'
}

solverParameters = {
  "domainDepth"     : 1.00,   # depth of domain (cms)
  "nodesDepth"      : 1000,
  "modelFormulation" : 'mixedForm',
  "smoothingFactor" : 1.0e-8, 
  "fileName"        : "Celia2.mat",
  "numberPlots"     : 30
}

mesh = IntervalMesh( solverParameters["nodesDepth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

modelParameters = {
   "modelType" : "VanGenuchten",
   "thetaR"    : 0.102,
   "thetaS"    : 0.368,
   "alpha"     : 3.35,
   "n"         : 2.00,
   "Ks"        : 0.0000922,
}

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

# Initial condition
h0   = Function(V, name="InitialCondition")
h0.interpolate( -10.0*(x[0]<0.9) + (x[0]>=0.9)*(92.5*x[0] - 93.25) )

def setBoundaryConditions(timeConstant, x):

    bottomBC, topBC = 1,2
    boundaryCondition = {
    bottomBC : {'h' : -10.0 },  
    topBC    : {'h' : -0.75},
    }

    return boundaryCondition

start = time.time()
RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions )
end = time.time()
print(end - start)