import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np
from ProblemDefinition import ProblemDefinitionNonlinear
from modelTypes import relativePermeability, moistureContent


def RichardsSolver(h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions):

    VecSpace = fd.VectorFunctionSpace(mesh, "CG", 1)
    R = fd.FunctionSpace(mesh, 'R', 0)

    dx = fd.Measure("dx", domain=mesh, metadata={"quadrature_degree": 3})
    ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": 3})
    dimen = mesh.topological_dimension()
    x = fd.SpatialCoordinate(mesh)

    # Do some unpacking
    finalTime = timeParameters["finalTime"]

    h = fd.Function(V, name="Pressure Head")
    hOld = fd.Function(V, name="PressureHeadOld")
    hStar = fd.Function(V, name="ApproximateSolution")
    h.assign(h0)

    q = fd.Function(VecSpace, name='volumetricFlux')
    q.assign(fd.project(fd.as_vector(fd.grad(h)), VecSpace))
    theta = fd.Function(V, name="Moisture Conent")
    K = fd.Function(V, name="Relative Permeability")

    # get IMEX parameters
    theta_diff = fd.Constant(timeParameters["theta_diff"])
    theta_nonlin = fd.Constant(timeParameters["theta_nonlin"])

    currentTime = fd.Function(R, name="time").assign(0)
    if timeParameters["timeStepType"] == 'constant':
        timeStep = fd.Function(R, name="delta_t").assign(timeParameters["timeStepSize"])
    else:
        timeStep = fd.Function(R, name="delta_t").assign(1)
    iterations = 0

    solverRichardsEqn = ProblemDefinitionNonlinear(h, hOld, hStar, currentTime, timeStep, theta_diff, theta_nonlin, V, modelParameters, setBoundaryConditions, mesh, dx, ds)

    # Save the solution
    outfile = fd.VTKFile(solverParameters["fileName"])

    if solverParameters['numberPlots'] == 0:
        tNext = 2*finalTime
    else:
        tNext = 0
        tInterval = finalTime / solverParameters["numberPlots"]
        plotIdx = 0

    # Main time loop
    while float(currentTime) <= finalTime:

        # Save the solution
        if float(currentTime) >= tNext:

            theta.interpolate(moistureContent(modelParameters, h, x, currentTime))
            K.interpolate(relativePermeability(modelParameters, h, x, currentTime))
            q.assign(fd.project(fd.as_vector(-K*fd.grad(h + x[dimen-1])), VecSpace))

            PETSc.Sys.Print("Time: " + str(float(currentTime)))

            outfile.write(h, theta, K, q, time=float(currentTime))

            plotIdx += 1
            tNext += tInterval

        hOld.assign(h)
        solverRichardsEqn.solve()

        currentTime.assign(currentTime + timeStep)
        iterations += 1

        # Update timestep
        # if iterations % 1 == 0:
        #    timeStep = updateTimeStep( h, hOld, timeStep, timeParameters, V )

        if float(currentTime + timeStep) > finalTime:
            timeStepNew = finalTime - float(currentTime)
            timeStepNew = np.maximum(timeStepNew, 1)
            timeStep.assign(timeStepNew)

    with fd.CheckpointFile("example.h5", 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(h)
        afile.save_function(theta)

    PETSc.Sys.Print("Total number of timesteps", iterations)
