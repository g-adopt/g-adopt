import firedrake as fd
from modelTypes import relativePermeability, waterRetention, moistureContent


def ProblemDefinitionNonlinear(z, zOld, timeStep, Z, timeParameters, modelParameters, mesh):

    thetaR = modelParameters["thetaR"]
    thetaS = modelParameters["thetaS"]
    alpha  = modelParameters["alpha"]
    Ks     = modelParameters["Ks"]

    v, w = fd.TestFunctions(Z)
    q, h = fd.split(z)
    qOld, hOld = fd.split(zOld)

    # Returns the variational problem for solving Richards equation
    dimen = mesh.topological_dimension()
    x = fd.SpatialCoordinate(mesh)

    # Unpack some parameters
    Ss = modelParameters["Ss"]

    # get IMEX parameters
    theta_diff = float(timeParameters["theta_diff"])
    theta_mid = float(timeParameters["theta_nonlin"])
    hBar = theta_mid*h + (1 - theta_mid)*hOld
    hDiff = theta_diff*h + (1 - theta_diff)*hOld

    dx = fd.Measure("dx", domain=mesh)
    dS = fd.Measure("dS", domain=mesh)

    theta = fd.conditional(h <= 0, thetaR + (thetaS - thetaR) * fd.exp(hBar * alpha), thetaS)
    S = (theta - thetaR) / (thetaS - thetaR)
    K = fd.conditional(h < 0, Ks*fd.exp(hBar*alpha), Ks)
    C = fd.conditional(h < 0, (thetaS - thetaR) * fd.exp(hBar * alpha) * alpha, 0)

    n = fd.FacetNormal(mesh)
    qn = 0.5*(fd.dot(q, n) + abs(fd.dot(q, n)))

    # Define problem
    F0 = (fd.inner((Ss*S + C)*(h - hOld)/timeStep, w) - fd.inner(q, fd.grad(w))) * dx + (w('+') - w('-'))*(qn('+') - qn('-'))*dS
    F1 = (fd.inner(q, v) + fd.inner(K*fd.grad(hDiff + x[dimen-1]), v)) * dx
    F = F0 + F1

    problem = fd.NonlinearVariationalProblem(F, z)

    solverRichardsNonlinear = fd.NonlinearVariationalSolver(problem)

    return solverRichardsNonlinear
