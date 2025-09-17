import numpy as np
import scipy.io
import firedrake as fd
from firedrake.petsc import PETSc
from scipy.interpolate import griddata

def data_2_function(mesh_coords, file_name):
    # Takes a data set that defines a value defined at the surface of the mesh and defines a firedrake function from this data

    x_coord = mesh_coords[:, 0]
    y_coord = mesh_coords[:, 1]
    elevation = x_coord*0
    distance = elevation + 100000

    mat = scipy.io.loadmat(file_name)
    x = mat.get('x')
    x_surface = x.flatten()
    y = mat.get('y')
    y_surface = y.flatten()
    z = mat.get('z')
    z_surface = z.flatten()

    points = np.vstack((x_surface, y_surface))
    points = points.T

    elevation = griddata(points, z_surface, (x_coord, y_coord), method='linear')

    return elevation


def updateTimeStep(h, hOld, timeStep, timeParameters, V):

    if timeParameters["timeStepType"] == 'constant':

        timeStep.assign(timeParameters["timeStepSize"]); 

    elif timeParameters["timeStepType"] == 'adaptive':

        relativeErrorFunc = fd.Function(V).interpolate(abs((h - hOld)/(h)))
        with relativeErrorFunc.dat.vec_ro as v:
            relativeError = v.max()[1]
        PETSc.Sys.Print(relativeError)

        timeStepNew = float(timeStep) * timeParameters['timeStepTolerance'] / (relativeError + 1e-06)
        timeStepNew = round(timeStepNew)

        timeStepNew = np.maximum(timeStepNew, 1e-1)
        timeStepNew = np.minimum(timeStepNew, timeParameters["maximumTimeStep"])
        timeStepNew = np.maximum(timeStepNew, timeParameters["minimumTimeStep"])
        PETSc.Sys.Print("dt ", float(timeStep))

        timeStep.assign(timeStepNew)

    else:

        PETSc.Sys.Print("Time stepping method not recognised")

    return timeStep
