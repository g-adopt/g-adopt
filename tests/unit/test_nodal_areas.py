import firedrake as fd
from mpi4py import MPI
import numpy as np
import pytest

from gadopt import layerwise_nodal_control_volumes, nodal_control_volumes


def _global_sum(function):
    with function.dat.vec_ro as vec:
        return vec.sum()


def _global_min(function):
    local_min = np.min(function.dat.data_ro, initial=np.inf)
    return function.comm.allreduce(local_min, op=MPI.MIN)


def _non_extruded_mesh(geometry):
    if geometry == "2d-cartesian-triangles":
        return fd.UnitSquareMesh(2, 2)
    if geometry == "2d-cartesian-quadrilaterals":
        return fd.UnitSquareMesh(2, 2, quadrilateral=True)
    if geometry == "3d-cartesian-tetrahedra":
        return fd.UnitCubeMesh(1, 1, 1)
    if geometry == "3d-cartesian-hexahedra":
        return fd.UnitCubeMesh(1, 1, 1, hexahedral=True)
    if geometry == "2d-cylindrical":
        return fd.AnnulusMesh(2.0, 1.0, nr=3, nt=12)
    if geometry == "3d-spherical":
        return fd.UnitBallMesh(1)
    if geometry == "circular-manifold":
        return fd.CircleManifoldMesh(8, radius=1.2, degree=2)
    if geometry == "spherical-manifold":
        return fd.CubedSphereMesh(1.2, refinement_level=1, degree=2)
    raise ValueError(f"Unknown geometry: {geometry}")


@pytest.mark.parametrize(
    "geometry",
    [
        "2d-cartesian-triangles",
        "2d-cartesian-quadrilaterals",
        "3d-cartesian-tetrahedra",
        "3d-cartesian-hexahedra",
        "2d-cylindrical",
        "3d-spherical",
        "circular-manifold",
        "spherical-manifold",
    ],
)
def test_nodal_control_volumes_sum_to_domain_measure(geometry):
    mesh = _non_extruded_mesh(geometry)
    volumes = nodal_control_volumes(mesh, name="Control-volume measure")

    expected = fd.assemble(fd.Constant(1) * fd.dx(domain=mesh))
    assert np.isclose(_global_sum(volumes), expected, rtol=1.0e-5)
    assert _global_min(volumes) > 0.0
    assert volumes.name() == "Control-volume measure"


@pytest.mark.parametrize("degree", [1, 2])
def test_nodal_control_volume_degree(degree):
    mesh = fd.UnitSquareMesh(2, 2)
    volumes = nodal_control_volumes(mesh, degree=degree)

    expected_space = fd.FunctionSpace(mesh, "CG", degree)
    assert volumes.function_space().ufl_element() == expected_space.ufl_element()
    assert np.isclose(_global_sum(volumes), 1.0)


def _extruded_mesh(geometry):
    if geometry == "2d-cartesian":
        base_mesh = fd.IntervalMesh(3, 2.0)
        mesh = fd.ExtrudedMesh(base_mesh, layers=2, layer_height=0.5)
        expected = np.full(5, 2.0)
        tolerance = 1.0e-13
    elif geometry == "3d-cartesian":
        base_mesh = fd.RectangleMesh(2, 3, 2.0, 3.0, quadrilateral=True)
        mesh = fd.ExtrudedMesh(base_mesh, layers=2, layer_height=0.5)
        expected = np.full(5, 6.0)
        tolerance = 1.0e-13
    elif geometry == "2d-cylindrical":
        base_mesh = fd.CircleManifoldMesh(16, radius=1.2, degree=2)
        mesh = fd.ExtrudedMesh(
            base_mesh, layers=2, layer_height=0.5, extrusion_type="radial"
        )
        radii = np.linspace(1.2, 2.2, 5)
        expected = 2 * np.pi * radii
        tolerance = 5.0e-5
    elif geometry == "3d-spherical":
        base_mesh = fd.CubedSphereMesh(1.2, refinement_level=1, degree=2)
        mesh = fd.ExtrudedMesh(
            base_mesh, layers=2, layer_height=0.5, extrusion_type="radial"
        )
        radii = np.linspace(1.2, 2.2, 5)
        expected = 4 * np.pi * radii**2
        tolerance = 2.0e-3
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    mesh.cartesian = "cartesian" in geometry
    return mesh, expected, tolerance


def _layer_sums(function, degree=2):
    mesh = function.function_space().mesh()
    level_count = degree * (mesh.layers - 1) + 1
    local_sums = function.dat.data_ro.reshape((-1, level_count)).sum(axis=0)
    return mesh.comm.allreduce(local_sums, op=MPI.SUM)


@pytest.mark.parametrize(
    "geometry", ["2d-cartesian", "3d-cartesian", "2d-cylindrical", "3d-spherical"]
)
def test_layerwise_nodal_control_volume_sums(geometry):
    mesh, expected, tolerance = _extruded_mesh(geometry)
    areas = layerwise_nodal_control_volumes(mesh)
    sums = _layer_sums(areas)

    np.testing.assert_allclose(sums, expected, rtol=tolerance)
    assert _global_min(areas) > 0.0

    if not mesh.cartesian:
        # All layers are homothetic. This checks the circumference/radius or
        # area/radius-squared scaling independently of geometric approximation.
        np.testing.assert_allclose(sums / sums[0], expected / expected[0], rtol=1.0e-12)


def test_linear_layerwise_nodal_control_volumes():
    mesh, _, _ = _extruded_mesh("2d-cartesian")
    areas = layerwise_nodal_control_volumes(mesh, degree=1)

    np.testing.assert_allclose(_layer_sums(areas, degree=1), 2.0)


def test_layerwise_nodal_control_volumes_after_checkpoint(tmp_path):
    mesh, expected, tolerance = _extruded_mesh("3d-spherical")
    checkpoint_path = tmp_path / "mesh.h5"

    with fd.CheckpointFile(str(checkpoint_path), "w") as checkpoint:
        checkpoint.save_mesh(mesh)

    with fd.CheckpointFile(str(checkpoint_path), "r") as checkpoint:
        loaded_mesh = checkpoint.load_mesh(mesh.name)

    areas = layerwise_nodal_control_volumes(loaded_mesh, name="cv_area_3d")
    np.testing.assert_allclose(_layer_sums(areas), expected, rtol=tolerance)

    output_path = tmp_path / "cv_area_3d.pvd"
    fd.VTKFile(str(output_path)).write(areas)
    assert output_path.is_file()


def test_nodal_control_volume_validation():
    mesh = fd.UnitSquareMesh(1, 1)
    with pytest.raises(ValueError, match="degree must be either 1 or 2"):
        nodal_control_volumes(mesh, degree=3)
    with pytest.raises(ValueError, match="require an extruded mesh"):
        layerwise_nodal_control_volumes(mesh)

    periodic_mesh = fd.ExtrudedMesh(fd.UnitIntervalMesh(2), layers=3, periodic=True)
    with pytest.raises(ValueError, match="require non-periodic extrusion"):
        layerwise_nodal_control_volumes(periodic_mesh)
