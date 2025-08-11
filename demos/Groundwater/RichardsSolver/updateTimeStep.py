def updateTimeStep( h, hOld, timeStep, solverParameters, V ):
   
    import numpy as np
    import firedrake as fd
    from firedrake.petsc import PETSc

    if solverParameters["timeStepType"] == 'constant':

        timeStep.assign( solverParameters["timeStepSize"] ); 

    elif solverParameters["timeStepType"] == 'adaptive':

        relativeErrorFunc = fd.Function(V).interpolate(abs((h - hOld)/ (h)))
        with relativeErrorFunc.dat.vec_ro as v:
            relativeError = v.max()[1]
        #PETSc.Sys.Print(relativeError)

        if float(timeStep) <= 100:
            base  = 5
        else:
            base = round(timeStep / 100) * 10
        timeStepNew = float( timeStep ) * 0.25 / (relativeError + 1e-06)
        timeStepNew = round( int(base * round(float(timeStepNew/base)) ))

        timeStepNew = np.maximum(timeStepNew, 1)
        #PETSc.Sys.Print("dt ", float(timeStep))

        timeStep.assign( timeStepNew ); 
    
    else:

        PETSc.Sys.Print("Time stepping method not recognised")

    return timeStep