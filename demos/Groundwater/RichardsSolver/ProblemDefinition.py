import firedrake as fd
from modelTypes import relativePermeability, waterRetention, moistureContent


def ProblemDefinitionNonlinear(h, hOld, hStar, timeConstant, timeStep, theta_diff, theta_mid, V, modelParameters, setBoundaryConditions, mesh, dx, ds):

    # Returns the variational problem for solving Richards equation

    dimen = mesh.topological_dimension()
    x = fd.SpatialCoordinate(mesh)
    v = fd.TestFunction(V)

    boundaryCondition = setBoundaryConditions(timeConstant, x)

    hBar = theta_mid*h + (1 - theta_mid)*hOld
    hDiff = theta_diff*h + (1 - theta_diff)*hOld

    C = waterRetention(modelParameters, hBar, x, timeConstant)
    K = relativePermeability(modelParameters, hBar, x, timeConstant)

    gravity = K*fd.as_vector(fd.grad(x[dimen-1]))
    normalVector = fd.FacetNormal(mesh)

    Ss = modelParameters["Ss"]
    thetaR = modelParameters["thetaR"]
    thetaS = modelParameters["thetaS"]
    theta = moistureContent(modelParameters, hBar, x, timeConstant)
    S = (theta - thetaR) / (thetaS - thetaR)

    # Define problem
    F = (fd.inner((Ss*S + C)*(h - hOld)/timeStep, v) + fd.inner(K*fd.grad(hDiff + x[dimen-1]), fd.grad(v)))*dx

    # Impose boundary conditions
    strongBCS = []

    if "top" in boundaryCondition:
        boundaryInfo = boundaryCondition.get('top')
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]

        if boundaryType == "h":
            strongBCS.append(fd.DirichletBC(V, boundaryValue, "top"))
        else:
            F = F - (-(fd.dot(normalVector, gravity) - boundaryValue)) * v * fd.ds_t

    if "bottom" in boundaryCondition:
        boundaryInfo = boundaryCondition.get('bottom')
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]

        if boundaryType == "h":
            strongBCS.append(fd.DirichletBC(V, boundaryValue, "bottom"))
        else:
            F = F - (-(fd.dot(normalVector, gravity) - boundaryValue)) * v * fd.ds_b

    for index in range(10):

        if index in boundaryCondition:

            boundaryInfo = boundaryCondition.get(index)
            boundaryType = next(iter(boundaryInfo))
            boundaryValue = boundaryInfo[boundaryType]

            if boundaryType == "h":
                strongBCS.append(fd.DirichletBC(V, boundaryValue, index))
            else:
                F = F - boundaryValue * v * ds(index)

    problem = fd.NonlinearVariationalProblem(F, h, bcs=strongBCS)

    # Use direct solvers
    if dimen <= 2:

        solverRichardsNonlinear = fd.NonlinearVariationalSolver(problem,
            solver_parameters={
                'mat_type': 'aij',
                'snes_type': 'newtonls',
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                })

    # Use iterative solvers
    else:

        solverRichardsNonlinear = fd.NonlinearVariationalSolver(problem,
                                    solver_parameters={
                                        'mat_type': 'aij',
                                        'snes_type': 'newtonls',
                                        'ksp_type': 'gmres',
                                        "ksp_rtol": 1e-5,
                                        'pc_type': 'sor',
                                    })

    return solverRichardsNonlinear
