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
    "finalTime"    : 150000,
    "timeStepType" : "adaptive",
    "timeStepTolerance" : 0.1,
    "timeIntegration" : 'crankNicolson'
}

solverParameters = {
  "modelFormulation" : 'mixedForm',
  "smoothingFactor"  : 0e-08, 
  "fileName"         : "multiMaterial.pvd",
  "numberPlots"      : 30
}

mesh = RectangleMesh( 250, 150, 5, 3)
mesh.cartesian = True
x     = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

cellSize = Function(V, name="InitialCondition")
cellSize.interpolate(CellSize(mesh)); 
gridSpacing = cellSize.at(0, 0); print(gridSpacing)

eps = 25; I = 1
xPoints = [2.5, 1.5, 4.0, 3.25, 0.75]
zPoints = [1.5, 2, 1.75, 0.5, 0.5]
for index in range(len(xPoints)):
  R = sqrt( (x[0] - xPoints[index])**2 + (x[1] - zPoints[index])**2 ) - 0.50
  I = 0.5*I*(1 + tanh(eps*R) );


modelParameters = {
   "modelType" : "VanGenuchten",
   "thetaR"    : 0.10,
   "thetaS"    : 0.38,
   "alpha"     : 3.35,
   "n"         : 2.00,
   "Ks"        : 5.00e-05*I + 5.00e-07*(1 - I),
    "gridSpacing" : gridSpacing,
}

h0   = Function(V, name="InitialCondition")
h0.interpolate( -2.5 )

def setBoundaryConditions(timeConstant, x):

    leftBC, rightBC, bottomBC, topBC = 1,2,3,4
    boundaryCondition = {
    leftBC   : {'q' : 0 },
    rightBC  : {'q' : 0 },
    bottomBC : {'q' : 0 },    # cm/s
    topBC    : {'q' : 2.5e-06},
    }

    return boundaryCondition

start = time.time()
RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions )
end = time.time()
print(end - start)
