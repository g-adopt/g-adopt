import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np
from scipy.io import savemat
from ProblemDefinition import ProblemDefinitionNonlinear, incompressibility
from modelTypes import relativePermeability, moistureContent


def RichardsSolver(h0, V, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions):

    ufl_elem = V.ufl_element()
    R = fd.FunctionSpace(mesh, 'R', 0)
    W = fd.VectorFunctionSpace(mesh, "CG", 1)  # Function space for vectors
    Z = fd.FunctionSpace(mesh, 'CG', 1)       # Continuous function space to project h onto

    dimen = mesh.topological_dimension()
    x = fd.SpatialCoordinate(mesh)
    n = -fd.FacetNormal(mesh)

    # Do some unpacking
    finalTime = timeParameters["finalTime"]

    h = fd.Function(V, name="Solution")
    hOld = fd.Function(V, name="Previous Solution")
    PressureHead = fd.Function(Z, name="PressureHead")
    h.assign(h0)
    PressureHead.interpolate(h)

    timeParameters.setdefault('quadratureDegree', 4)
    quadratureDegree = timeParameters["quadratureDegree"]
    dx = fd.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadratureDegree})
    ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": quadratureDegree})

    currentTime = fd.Function(R, name="time").assign(0)
    timeStep = fd.Function(R, name="delta_t").assign(timeParameters["timeStepSize"])
    iterations = 0

    theta = fd.Function(Z, name="Moisture Conent")
    theta.interpolate(moistureContent(modelParameters, PressureHead, x, currentTime))
    K = fd.Function(Z, name="Relative Permeability")
    K.interpolate(relativePermeability(modelParameters, PressureHead, x, currentTime))
    q = fd.Function(W, name='volumetricFlux')
    q.interpolate(fd.project(fd.as_vector(-K*fd.grad(h + x[dimen-1])), W))

    div_vel = fd.Function(V, name="Mass conservation")

    solverRichardsEqn = ProblemDefinitionNonlinear(h, hOld, currentTime, timeStep, V, modelParameters, timeParameters, setBoundaryConditions, mesh)
    solverMassConservation = incompressibility(div_vel, h, hOld, currentTime, timeStep, V, modelParameters, timeParameters, mesh)

    initialMass = fd.assemble(theta * dx)
    totalFlux   = 0

    # Save the solution
    outfile = fd.VTKFile(solverParameters["fileName"]+".pvd")

    if solverParameters['numberPlots'] == 0:
        tNext = 2*finalTime
    else:
        tNext = 0
        tInterval = finalTime / solverParameters["numberPlots"]
        plotIdx = 0

    totalIterations = 0
    massError = 0

    # Main time loop
    while float(currentTime) <= finalTime:

        K.interpolate(relativePermeability(modelParameters, PressureHead, x, currentTime))
        theta.interpolate(moistureContent(modelParameters, PressureHead, x, currentTime))
        qOld = q
        thetaOld = theta
        massOld = fd.assemble(theta*dx)

        # Save the solution
        if float(currentTime) >= tNext:

            PETSc.Sys.Print("Time: " + str(float(currentTime)))
            outfile.write(h, theta, K, q, div_vel, time=float(currentTime))

            plotIdx += 1
            tNext += tInterval

        hOld.assign(h)
        solverRichardsEqn.solve()
        PressureHead.interpolate(h)
        #solverMassConservation.solve()

        hBar = timeParameters["theta_nonlin"]*h + (1 - timeParameters["theta_nonlin"])*hOld
        hDiff = timeParameters["theta_diff"]*h + (1 - timeParameters["theta_diff"])*hOld
        K.interpolate(relativePermeability(modelParameters, hBar, x, currentTime))
        theta.interpolate(moistureContent(modelParameters, hBar, x, currentTime))
        q.interpolate(fd.project(fd.as_vector(-K*fd.grad(hDiff + x[dimen-1])), W))
        massNew = fd.assemble(theta*dx)

        flux = fd.assemble(timeStep*fd.dot(q, n)*ds)
        massError += ((massNew - massOld) - flux)**2
        totalFlux += flux

        #PETSc.Sys.Print("Mass change", massNew - massOld)
        #qn = fd.dot(q, n)
        #PETSc.Sys.Print("Interior surface integrals", fd.assemble(timeStep*(qn('-') + 0*qn('+')) * fd.dS))
        #PETSc.Sys.Print("Boundary surface integrals", fd.assemble(timeStep*qn * fd.ds))

        totalIterations += solverRichardsEqn.snes.ksp.getIterationNumber()
        currentTime.assign(currentTime + timeStep)
        iterations += 1

        if float(currentTime + timeStep) > finalTime:
            timeStepNew = finalTime - float(currentTime)
            timeStepNew = np.maximum(timeStepNew, 1)
            timeStep.assign(timeStepNew)

    theta.interpolate(moistureContent(modelParameters, h, x, currentTime))
    finalMass = fd.assemble(theta*dx)

    PETSc.Sys.Print("Total number of timesteps", iterations)
    PETSc.Sys.Print("Total number of linear iterations", totalIterations)
    PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())
    PETSc.Sys.Print()
    PETSc.Sys.Print("Initial water mass", initialMass)
    PETSc.Sys.Print("Final water mass", finalMass)
    PETSc.Sys.Print("MSE", fd.sqrt(massError/iterations))
    PETSc.Sys.Print("Total flux", totalFlux)
    PETSc.Sys.Print("Global mass loss", (finalMass - initialMass)/totalFlux)

    theta.interpolate(moistureContent(modelParameters, PressureHead, x, currentTime))
    K.interpolate(relativePermeability(modelParameters, PressureHead, x, currentTime))

    return h, theta, q, K
