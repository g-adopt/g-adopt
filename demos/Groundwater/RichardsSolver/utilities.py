from types import SimpleNamespace

def updateSmoothingParameter( smoothingParameter, smoothingFactor, q, h, dimen, V ):
    
    import firedrake as fd
    from firedrake.petsc import PETSc

    qNorm = fd.Function(V).interpolate( fd.sqrt(fd.dot(q, q)) ) 
    with qNorm.dat.vec_ro as v:
        qMax = v.max()[1]


    laplacian = fd.Function(V).interpolate(abs( h.dx(dimen-1) )) 
    with laplacian.dat.vec_ro as v:
        lapMax = v.max()[1]

    smoothingParameter.assign( smoothingFactor*lapMax * qMax)

 #   smoothingParameter.assign( smoothingFactor)

    return smoothingParameter

def updateTimeStep( h, hOld, timeStep, timeParameters, V ):
   
    import numpy as np
    import firedrake as fd
    from firedrake.petsc import PETSc

    if timeParameters["timeStepType"] == 'constant':

        timeStep.assign( timeParameters["timeStepSize"] ); 

    elif timeParameters["timeStepType"] == 'adaptive':

        relativeErrorFunc = fd.Function(V).interpolate(abs((h - hOld)/ (h)))
        with relativeErrorFunc.dat.vec_ro as v:
            relativeError = v.max()[1]
        #PETSc.Sys.Print(relativeError)

        if float(timeStep) <= 100:
            base  = 5
        else:
            base = round(timeStep / 100) * 10
        timeStepNew = float( timeStep ) * timeParameters['timeStepTolerance'] / (relativeError + 1e-06)
        timeStepNew = round( int(base * round(float(timeStepNew/base)) ))

        timeStepNew = np.maximum(timeStepNew, 1e-0)
        PETSc.Sys.Print("dt ", float(timeStep))

        timeStep.assign( timeStepNew ); 
    
    else:

        PETSc.Sys.Print("Time stepping method not recognised")

    return timeStep
