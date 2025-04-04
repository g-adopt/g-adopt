import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np
from ProblemDefinition import ProblemDefinitionNonlinear
from modelTypes import relativePermeability, moistureContent
from utilities import updateTimeStep
from scipy.io import savemat

def RichardsSolver( h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions ):

    VecSpace = fd.VectorFunctionSpace(mesh, "CG", 1)
    R        = fd.FunctionSpace(mesh, 'R', 0)

    dx = fd.Measure("dx", domain=mesh, metadata={"quadrature_degree": 3})
    ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": 3})
    dimen = mesh.topological_dimension();
    x     = fd.SpatialCoordinate(mesh)

    # Do some unpacking
    finalTime     = timeParameters["finalTime"]

    h     = fd.Function(V, name="Pressure Head");          h.assign(h0)
    hOld  = fd.Function(V, name="PressureHeadOld");        hOld.assign(h0)
    hStar = fd.Function(V, name="ApproximateSolution");    hStar.assign(h0)

    q     = fd.Function(VecSpace, name='volumetricFlux');  q.assign( fd.project(fd.as_vector(fd.grad( h )), VecSpace) )
    theta = fd.Function(V, name = "Moisture Conent")
    K     = fd.Function(V, name = "Relative Permeability")

    timeIntegrator  = timeParameters["timeIntegration"]

    currentTime        = fd.Function(R, name="time").assign( 0 )
    if timeParameters["timeStepType"] == 'constant':
        timeStep           = fd.Function(R, name="delta_t").assign( timeParameters["timeStepSize"] )
    else:
        timeStep           = fd.Function(R, name="delta_t").assign( 1 )
    iterations = 0

    solverRichardsNonlinear = ProblemDefinitionNonlinear( h, hOld, hStar, currentTime, timeStep, V, timeIntegrator, modelParameters, setBoundaryConditions, mesh, dx, ds )
    solverRichardsLinear    = ProblemDefinitionNonlinear( h, hOld, hStar, currentTime, timeStep, V, "IMEX", modelParameters, setBoundaryConditions, mesh, dx, ds )

    # Save the solution
    outfile = fd.VTKFile(solverParameters["fileName"])

    if solverParameters['numberPlots'] == 0:
        tNext = 2*finalTime
    else:
        tNext = 0
        tInterval = finalTime / solverParameters["numberPlots"];
        plotIdx = 0

    # Main time loop
    while float(currentTime) <= finalTime:

        # Save the solution
        if float(currentTime) >= tNext:

            theta.interpolate(moistureContent(modelParameters, h, x, currentTime))
            K.interpolate(relativePermeability(modelParameters, h, x, currentTime))
            q.assign( fd.project(fd.as_vector(-K*fd.grad( h + x[dimen-1])), VecSpace) )

            PETSc.Sys.Print(float(currentTime))

            outfile.write(h, theta, K, q, time=float(currentTime))

            plotIdx += 1
            tNext += tInterval

        hOld.assign(h)
        converged = 0

        if timeIntegrator == 'IMEX':

            solverRichardsLinear.solve()
            converged = 0

        # First take a linear step
        if (timeIntegrator == "modifiedEuler"):

            solverRichardsLinear.solve()
            hStar.assign(hOld)
            hOld.assign(h)
            solverRichardsLinear.solve()
            h.assign(0.5*(h + hStar))

            converged = 1

        # Implicit method
        elif (timeIntegrator == "backwardEuler") or (timeIntegrator == "crankNicolson"):
            try:
                solverRichardsNonlinear.solve()
            except:
                converged = 0
            else:
                converged = 1

        # Picard iteration
        else:

            picardIterations = 0
            relativeError = 100

            while (relativeError > 1e-04) and (picardIterations < 50):

                hStar.assign( h )
                solverRichardsNonlinear.solve()
                picardIterations +=  1

                relativeErrorFunc = fd.Function(V).interpolate(abs((h - hStar)/ (1)))
                with relativeErrorFunc.dat.vec_ro as v:
                    relativeErrorNew = v.max()[1]

                relativeError = relativeErrorNew

            if (picardIterations >= 48):
                converged = 0
            else:
                converged = 1


        if converged == 1:
            currentTime.assign( currentTime + timeStep )
            iterations += 1
        else:
            PETSc.Sys.Print("Solver not converging, trying a smaller timestep")
            timeStep.assign( float(timeStep) - 1 )
            PETSc.Sys.Print(float(timeStep)) 
            h.assign(hOld); hStar.assign(hOld)

        # Update timestep
        #if iterations % 1 == 0:
        #    timeStep = updateTimeStep( h, hOld, timeStep, timeParameters, V )

        if float(currentTime + timeStep) > finalTime:
            timeStepNew = finalTime - float(currentTime)
            timeStepNew = np.maximum(timeStepNew, 1)
            timeStep.assign( timeStepNew )

    with fd.CheckpointFile("example.h5", 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(h)
        afile.save_function(theta)

    PETSc.Sys.Print("Total number of timesteps", iterations)