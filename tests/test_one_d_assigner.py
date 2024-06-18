

from gadopt import *


def test_oned_average_assignment():
    # a 10 x 10 square mesh
    n = 10
    mesh = UnitSquareMesh(n, n)
    # because this is not an extruded mesh we have to provide r1d for layer averaging
    y_disc = np.linspace(0.0, 1.0, n+1)
    # spatial coordinates and computing vertical coordinate
    X = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    q = Function(Q).interpolate(X[1])
    # computing the average
    averager = LayerAveraging(mesh, y_disc if mesh.layers is None else None, cartesian=True)
    q_profile = averager.get_layer_average(q)
    # write the average out
    output_fi = ParameterLog("one_d_test.output", mesh)
    output_fi.log_str("\n".join([f"{y}, {val}" for y, val in zip(y_disc, q_profile)]))
    output_fi.close()
    # assign the average to a new function
    p = Function(Q)
    interpolate_1d_profile(p, "one_d_test.output", cartesian=True)

    assert assemble((p-q) ** 2 * dx) < 1e-10


def test_layer_averaging():
    # a 10 x 10 square mesh
    n = 10
    mesh = UnitSquareMesh(n, n)
    # because this is not an extruded mesh we have to provide r1d for layer averaging
    y_disc = np.linspace(0.0, 1.0, n+1)
    # spatial coordinates and computing vertical coordinate
    X = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 2)
    q = Function(Q).interpolate(X[1])
    # computing the average
    averager = LayerAveraging(mesh, y_disc if mesh.layers is None else None, cartesian=True)
    q_profile = averager.get_layer_average(q)

    assert np.allclose(q_profile, y_disc, atol=1e-10)
