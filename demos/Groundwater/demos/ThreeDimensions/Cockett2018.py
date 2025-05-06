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
    "finalTime": 259200,
    "timeStepType": "constant",
    "timeStepSize": 600,
    "theta_diff": 0.5,
    "theta_nonlin": 0.5,
}

solverParameters = {
    "fileName": "Cockett2018.pvd",
    "numberPlots": 1
}

nodesX, nodesY, nodesZ, Lx, Ly, Lz = 10, 10, 13, 2, 2, 2.6
mesh = BoxMesh(nodesX, nodesY, nodesZ, Lx, Ly, Lz, name="mesh")

hierarchy = MeshHierarchy(mesh, 3)
mesh = hierarchy[-1]

mesh.cartesian = True
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837, 0.1321, 0.7227, 0.1104, 0.1175, 0.6407]
I = sin(3*(x[0]-r[0])) + sin(3*(x[1]-r[1])) + sin(3*(x[2]-r[2])) + sin(3*(x[0]-r[3])) + sin(3*(x[1]-r[4])) + sin(3*(x[2]-r[5]))+ sin(3*(x[0]-r[6])) + sin(3*(x[1]-r[7])) + sin(3*(x[2]-r[8]))
I = 0.5*(1 + tanh(5*I))

modelParameters = {
    "modelType": "VanGenuchten",
    "thetaR": 0.02*I + 0.035*(1-I),
    "thetaS": 0.417*I + 0.401*(1-I),
    "n": 1.592*I + 1.474*(1-I),
    "alpha": 13.8*I + 11.5*(1-I),
    "Ks": 5.82e-05*I + 1.69e-05*(1-I),
    "Ss": 0,
}

h0 = Function(V, name="InitialCondition")
h0.interpolate(-3 + 2.9*exp(5*(x[2]-Lz)))


def setBoundaryConditions(timeConstant, x):

    leftBC, rightBC, frontBC, backBC, bottomBC, topBC = 1, 2, 3, 4, 5, 6
    boundaryCondition = {
        leftBC: {'q': 0},
        rightBC: {'q': 0},
        frontBC: {'q': 0},
        backBC: {'q': 0},
        bottomBC: {'q': 0},
        topBC: {'h': -0.1},
    }

    return boundaryCondition


start = time.time()
RichardsSolver(h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions)
end = time.time()
print(end - start)
