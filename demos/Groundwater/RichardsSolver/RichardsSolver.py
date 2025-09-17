import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np
from scipy.io import savemat
from ProblemDefinition import ProblemDefinitionNonlinear
from modelTypes import relativePermeability, moistureContent


def RichardsSolver(h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions):

    VecSpace = fd.VectorFunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, 'R', 0)

    dimen = mesh.topological_dimension()
    x = fd.SpatialCoordinate(mesh)

    # Do some unpacking
    finalTime = timeParameters["finalTime"]

    h = fd.Function(V, name="PressureHead")
    hOld = fd.Function(V, name="Previous Solution")
    hStar = fd.Function(V, name="Approximate Solution")
    h.assign(h0)

    currentTime = fd.Function(R, name="time").assign(0)
    timeStep = fd.Function(R, name="delta_t").assign(timeParameters["timeStepSize"])
    iterations = 0

    theta = fd.Function(V, name="Moisture Conent")
    theta.interpolate(moistureContent(modelParameters, h, x, currentTime))
    K = fd.Function(V, name="Relative Permeability")
    K.interpolate(relativePermeability(modelParameters, h, x, currentTime))
    q = fd.Function(VecSpace, name='volumetricFlux')
    q.interpolate(fd.project(fd.as_vector(-K*fd.grad(h + x[dimen-1])), VecSpace))

    solverRichardsEqn = ProblemDefinitionNonlinear(h, hOld, hStar, currentTime, timeStep, V, modelParameters, timeParameters, setBoundaryConditions, mesh)

    # Save the solution
    outfile = fd.VTKFile(solverParameters["fileName"]+".pvd")

    if solverParameters['numberPlots'] == 0:
        tNext = 2*finalTime
    else:
        tNext = 0
        tInterval = finalTime / solverParameters["numberPlots"]
        plotIdx = 0

    totalIterations = 0

    # Main time loop
    while float(currentTime) <= finalTime:

        # Save the solution
        if float(currentTime) >= tNext:

            theta.interpolate(moistureContent(modelParameters, h, x, currentTime))
            K.interpolate(relativePermeability(modelParameters, h, x, currentTime))
            q.interpolate(fd.project(fd.as_vector(-K*fd.grad(h + x[dimen-1])), VecSpace))

            PETSc.Sys.Print("Time: " + str(float(currentTime)))
            outfile.write(h, theta, K, q, time=float(currentTime))

            plotIdx += 1
            tNext += tInterval

        hOld.assign(h)
        solverRichardsEqn.solve()

        totalIterations += solverRichardsEqn.snes.ksp.getIterationNumber()
        currentTime.assign(currentTime + timeStep)
        iterations += 1

        if float(currentTime + timeStep) > finalTime:
            timeStepNew = finalTime - float(currentTime)
            timeStepNew = np.maximum(timeStepNew, 1)
            timeStep.assign(timeStepNew)

    PETSc.Sys.Print("Total number of timesteps", iterations)
    PETSc.Sys.Print("Total number of linear iterations", totalIterations)

    theta.interpolate(moistureContent(modelParameters, h, x, currentTime))
    K.interpolate(relativePermeability(modelParameters, h, x, currentTime))
    q.assign(fd.project(fd.as_vector(-K*fd.grad(h + x[dimen-1])), VecSpace))

    return h, theta, q, K
