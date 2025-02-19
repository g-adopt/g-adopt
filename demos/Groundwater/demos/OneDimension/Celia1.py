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
    "finalTime"    : 360,
    "timeStepType" : "constant",
    "timeStepSize" : 1.0,
    "timeIntegration" : 'forwardEuler'
}

solverParameters = {
  "domainDepth"  : 0.4,   # depth of domain (cms)
  "nodesDepth"   : 200,
  "modelFormulation" : 'mixedForm',
  "smoothingFactor" : 0.00, 
  "fileName"   : "Celia1.mat",
  "numberPlots"   : 10
}

mesh = IntervalMesh( solverParameters["nodesDepth"], solverParameters["domainDepth"])
mesh.cartesian = True
x     = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

cellSize = Function(V, name="InitialCondition")
cellSize.interpolate(CellSize(mesh)); 
gridSpacing = cellSize.at(0); print(gridSpacing)


modelParameters = {
   "modelType" : "Haverkamp",
   "thetaR"    : 0.075,
   "thetaS"    : 0.287,
   "A"         : 1.175e06,
   "gamma"     : 4.74,
   "alpha"     : 1.611e06,
   "beta"      : 3.96,
   "Ks"        : 9.44e-05,
   "gridSpacing" : gridSpacing,
}

# Initial condition
h0   = Function(V, name="InitialCondition")
h0.interpolate( -61.5 * (x[0] < 35) + (7.96*x[0] - 340.1) * (x[0] >= 35))
h0.interpolate( -0.615 )

def setBoundaryConditions(timeConstant, x):

    bottomBC, topBC = 1,2
    boundaryCondition = {
    bottomBC : {'h' : -0.615 },    # cm/s
    topBC    : {'h' : -0.207},
    }

    return boundaryCondition


start = time.time()
RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions )
end = time.time()
print("Simulation time ", end - start)