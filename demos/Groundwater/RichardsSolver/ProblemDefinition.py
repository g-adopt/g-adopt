def ProblemDefinitionNonlinear( h, hOld, smoothingParameter, timeConstant, timeStep, v, V, timeIntegrator, modelFormulation, modelParameters, setBoundaryConditions, mesh, dx, ds ):
    
    # Returns the variational problem for solving Richards equation
    import firedrake as fd
    import numpy as np
    from modelTypes import relativePermeability, waterRetention, moistureContent

    dimen = mesh.topological_dimension()
    x     = fd.SpatialCoordinate(mesh)

    #timeIntegrator   = timeParameters['timeIntegration']
    #modelFormulation = timeParameters['modelFormulation']

    boundaryCondition = setBoundaryConditions(timeConstant, x)


    if timeIntegrator == 'backwardEuler':
        hBar = h; hDiff = h
    elif timeIntegrator == 'crankNicolson':
        hBar = 0.5*(h + hOld); hDiff = hBar
    elif timeIntegrator == 'modifiedEuler':
        hBar = hOld; hDiff = h
    else:
        hBar = hOld; hDiff = h


    C = waterRetention( modelParameters, hBar, x, timeConstant )
    K = relativePermeability( modelParameters, hBar, x, timeConstant )

    if dimen == 1:
        gravity = fd.as_vector([ K ])
    elif dimen == 2:
        gravity = fd.as_vector([0,  K ]);
    else:
        gravity = fd.as_vector([0, 0, K ])

    normalVector = fd.FacetNormal(mesh)

    # Define problem
    if modelFormulation == 'mixedForm':

        thetaOld = moistureContent( modelParameters, hOld, x, timeConstant)
        thetaNew = moistureContent( modelParameters, h, x, timeConstant)

        F = ( fd.inner( (thetaNew - thetaOld )/timeStep, v) +
            fd.inner( K*fd.grad( hDiff ), fd.grad(v) )  -
        fd.inner( K.dx(dimen-1), v )
        + fd.inner( smoothingParameter*fd.grad( h), fd.grad(v) ) )*dx

    else:

        F = ( fd.inner( C*(h - hOld)/timeStep, v) + fd.inner( K*fd.grad( hDiff ), fd.grad(v) )  -
        fd.inner( K.dx(dimen-1), v ) + fd.inner( smoothingParameter*fd.grad( h), fd.grad(v) ) )*dx

    strongBCS = [];

    for index in range(len(boundaryCondition)):

        boundaryInfo  = boundaryCondition[index+1]
        boundaryType  = next(iter(boundaryInfo));
        boundaryValue = boundaryInfo[boundaryType]; 

        if boundaryType == "h":
            strongBCS.append(fd.DirichletBC(V, boundaryValue, index+1))
        else:
            F = F - ( -( fd.dot( normalVector , gravity ) - boundaryValue) ) * v * ds(index+1)

    problem = fd.NonlinearVariationalProblem(F, h, bcs = strongBCS)
   
    # Use direct solvers
    if dimen <= 2:

        solverRichardsNonlinear  = fd.NonlinearVariationalSolver(problem,
                                            solver_parameters={
                                            'mat_type': 'aij',
                                            'snes_type': 'newtonls',
                                            'ksp_type': 'preonly',
                                            'pc_type': 'lu',
                                            })
        
    # Use iterative solvers
    else:

        solverRichardsNonlinear  = fd.NonlinearVariationalSolver(problem,
                                    solver_parameters={
                                    'mat_type': 'aij',
                                    'snes_type': 'ksponly',
                                    'ksp_type': 'gmres',
                                    "ksp_rtol": 1e-5,
                                    'pc_type': 'sor',
                                    })

    
    return solverRichardsNonlinear
    




