import firedrake as fd
from modelTypes import relativePermeability, waterRetention, moistureContent


def ProblemDefinitionNonlinear(h, hOld, hStar, timeConstant, timeStep, V, modelParameters, timeParameters, setBoundaryConditions, mesh):

    # Returns the variational problem for solving Richards equation

    dimen = mesh.topological_dimension()
    x = fd.SpatialCoordinate(mesh)
    v = fd.TestFunction(V)

    # Unpack some parameters
    Ss = modelParameters["Ss"]
    thetaR = modelParameters["thetaR"]
    thetaS = modelParameters["thetaS"]
    epsilon = timeParameters["epsilon"]

    # get IMEX parameters
    theta_diff = float(timeParameters["theta_diff"])
    theta_mid = float(timeParameters["theta_nonlin"])
    hBar = theta_mid*h + (1 - theta_mid)*hOld
    hDiff = theta_diff*h + (1 - theta_diff)*hOld

    timeParameters.setdefault('quadratureDegree', 4)
    quadratureDegree = timeParameters["quadratureDegree"]
    dx = fd.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadratureDegree})
    if dimen == 2:
        ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": quadratureDegree})
    else:
        ds_t = fd.Measure("ds_t", domain=mesh, metadata={"quadrature_degree": quadratureDegree})
        ds_b = fd.Measure("ds_b", domain=mesh, metadata={"quadrature_degree": quadratureDegree})
        ds_v = fd.Measure("ds_v", domain=mesh, metadata={"quadrature_degree": quadratureDegree})

    theta = moistureContent(modelParameters, hBar, x, timeConstant)
    S = (theta - thetaR) / (thetaS - thetaR)
    C = waterRetention(modelParameters, hBar, x, timeConstant)
    K = relativePermeability(modelParameters, hBar, x, timeConstant)

    boundaryCondition = setBoundaryConditions(timeConstant, x, K, theta)

    # Define problem
    F = (fd.inner((Ss*S + C)*(h - hOld)/timeStep, v) + fd.inner(K*fd.grad(hDiff + x[dimen-1]), fd.grad(v)) + epsilon*fd.inner(fd.grad(h - hOld), fd.grad(v)))*dx

    # Impose boundary conditions
    strongBCS = []
    if "top" in boundaryCondition:
        boundaryInfo = boundaryCondition.get('top')
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]

        if boundaryType == "h":
            strongBCS.append(fd.DirichletBC(V, boundaryValue, "top"))
        else:
            F = F - boundaryValue * v * ds_t

    if "bottom" in boundaryCondition:
        boundaryInfo = boundaryCondition.get('bottom')
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]

        if boundaryType == "h":
            strongBCS.append(fd.DirichletBC(V, boundaryValue, "bottom"))
        else:
            F = F - boundaryValue * v * ds_b

    for index in range(20):

        if index in boundaryCondition:

            boundaryInfo = boundaryCondition.get(index)
            boundaryType = next(iter(boundaryInfo))
            boundaryValue = boundaryInfo[boundaryType]

            if boundaryType == "h":
                strongBCS.append(fd.DirichletBC(V, boundaryValue, index))
            else:
                if dimen == 2:
                    F = F - boundaryValue * v * ds(index)
                else:
                    F = F - boundaryValue * v * ds_v(index)

    problem = fd.NonlinearVariationalProblem(F, h, bcs=strongBCS)

    if float(theta_mid) == 0 and float(theta_diff) == 1:
        snesType = 'ksponly'
    else:
        snesType = 'newtonls'

    # Use direct solvers for 2D and iterative for 3d
    if dimen <= 2:
        ksp_type = 'preonly'
        pc_type = 'lu'
    else:
        ksp_type = 'bcgs'
        pc_type = 'bjacobi'

    solverRichardsNonlinear = fd.NonlinearVariationalSolver(problem,
                                solver_parameters={
                                    'mat_type': 'aij',
                                    'snes_type': snesType,
                                    'ksp_type': ksp_type,
                                    "pc_type": pc_type,
                                })
#                                    'ksp_converged_reason': None,

    return solverRichardsNonlinear
