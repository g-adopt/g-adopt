import os, sys
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate
from ProblemDefinition import ProblemDefinitionNonlinear
from modelTypes import *
import time

# Steady state solution
timeParameters = {
    "finalTime": 20000,
    "timeStepType": "constant",
    "timeStepSize": 50,
    "theta_diff": 0.5,
    "theta_nonlin": 0.5,
    "epsilon": 0,
}

modelParameters = {
    "modelType": "exponential",
    "thetaR": 0.15,
    "thetaS": 0.45,
    "alpha": 0.30,
    "Ks": 1.0e-05,
    "Ss": 0,
}

solverParameters = {
    "fileName": "Tracy2D",
    "numberPlots": 1
}

L, nodes = 15.24, 51
mesh = Mesh('squareMesh.msh')

mesh.cartesian = True
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "Raviart-Thomas", 2)  # Function space for volumetric flux
W = FunctionSpace(mesh, "DG", 1)              # Function space for pressure head
Z = MixedFunctionSpace([V, W])             # Mixed function space.

# Initial condition
hInitial = Function(W, name="InitialCondition")
hInitial.interpolate(-5)

z    = Function(Z)  # Mixed solution (current)
zOld = Function(Z)  # Mixed solution (previous)

q, h = split(z)       # Split mixed functions
q0, h0 = split(zOld)  # Split mixed functions

z.sub(1).interpolate(hInitial)
zOld.sub(1).interpolate(hInitial)


timeStep = timeParameters["timeStepSize"]
solverRichardsEqn = ProblemDefinitionNonlinear(z, zOld, timeStep, Z, timeParameters, modelParameters, mesh)

outfile = VTKFile(solverParameters["fileName"]+".pvd")  # Save the solution

for ii in range(1, int(timeParameters["finalTime"]/timeParameters["timeStepSize"])):

    zOld.assign(z)
    solverRichardsEqn.solve()
    q, h = fd.split(z)

PressureHead = Function(W, name="PressureHead")
theta = Function(W, name="Moisture Conent")

PressureHead.interpolate(h)
theta.interpolate(modelParameters["thetaR"] + (modelParameters["thetaS"]-modelParameters["thetaR"])*exp(PressureHead*modelParameters["alpha"]))
theta.interpolate(conditional(PressureHead <= 0, theta, modelParameters["thetaS"]))
outfile.write(PressureHead, theta, time=float(timeParameters["finalTime"]))