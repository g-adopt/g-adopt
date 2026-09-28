import firedrake as fd
import numpy as np
from mpi4py import MPI


def generate_superparametric_spherical_shell(mesh, degree=2):
    P1 = fd.FunctionSpace(mesh, "CG", 1)
    VPn = fd.VectorFunctionSpace(mesh, "CG", degree)
    Pn_coordinates = fd.Function(VPn)
    P1_r = fd.Function(P1)
    x = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(x, x))
    P1_r.interpolate(r)
    Pn_coordinates.interpolate(x/r*P1_r)
    return fd.Mesh(Pn_coordinates)


def min_max_coordinates(mesh):
    min_max = []
    for id in (1, 2):
        bc = fd.DirichletBC(mesh.coordinates.function_space(), 0, id)
        xy = mesh.coordinates.dat.data_ro_with_halos[bc.nodes, :]
        r = np.sqrt((xy**2).sum(axis=1))
        if len(r) > 0:
            min_max.append(r.min())
            min_max.append(r.max())
        else:
            min_max.append(np.finfo('d').max)
            min_max.append(0.)

    min_max[0::2] = mesh.comm.allreduce(min_max[0::2], MPI.MIN)
    min_max[1::2] = mesh.comm.allreduce(min_max[1::2], MPI.MAX)

    return min_max


def correct_surface_radius(mesh, surface_id, r):
    V = mesh.coordinates.function_space()
    x = fd.SpatialCoordinate(mesh)
    bc = fd.DirichletBC(V, x / fd.sqrt(fd.dot(x,x)) * fd.Constant(r), surface_id)
    bc.apply(mesh.coordinates)
