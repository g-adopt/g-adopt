from gadopt import *
from gadopt.utility import vertical_component


def test_oned_average_assignment_spherical(tmp_path):
    """ A test case for `interpolate_1d_profile` for 3d spherical mesh

    The test calculates a radius profile, writes it out, and then assigns it
    and tests if the values end up being the same
    """

    rmin = 1.208
    ref_level = 5
    nlayers = 4

    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type='radial')
    mesh.cartesian = False

    X = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    q = Function(Q).interpolate(vertical_component(X))

    averager = LayerAveraging(mesh, None)
    q_profile = averager.get_layer_average(q)

    output_fi = ParameterLog(tmp_path / "one_d_test_spherical.output", mesh)
    output_fi.log_str("\n".join([f"{val}, {val}" for val in q_profile]))
    output_fi.close()

    p = Function(Q)
    interpolate_1d_profile(p, tmp_path / "one_d_test_spherical.output")
    log(assemble((p-q) ** 2 * dx))
    assert assemble((p-q) ** 2 * dx) < 1e-10


def test_oned_average_assignment_cartesian(tmp_path):
    """ A test case for `interpolate_1d_profile` for 2d square case

    The test calculates a vertical profile, writes it out, and then assigns it
    and tests if the values end up being the same
    """
    n = 10
    mesh = UnitSquareMesh(n, n)
    mesh.cartesian = True

    # because this is not an extruded mesh we have to provide r1d for layer averaging
    y_disc = np.linspace(0.0, 1.0, n+1)

    X = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    q = Function(Q).interpolate(X[1])

    averager = LayerAveraging(mesh, y_disc if mesh.layers is None else None)
    q_profile = averager.get_layer_average(q)

    output_fi = ParameterLog(tmp_path / "one_d_test.output", mesh)
    output_fi.log_str("\n".join([f"{y}, {val}" for y, val in zip(y_disc, q_profile)]))
    output_fi.close()

    p = Function(Q)
    interpolate_1d_profile(p, tmp_path / "one_d_test.output")

    assert assemble((p-q) ** 2 * dx) < 1e-10
