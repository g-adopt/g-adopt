from pathlib import Path

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
    elif geometry in ("3d-spherical", "3d-spherical-decreasing-radius"):
        decreasing_radius = geometry.endswith("decreasing-radius")
        base_radius = 2.2 if decreasing_radius else 1.2
        layer_height = -0.5 if decreasing_radius else 0.5
        base_mesh = fd.CubedSphereMesh(
            base_radius, refinement_level=1, degree=2
        )
        mesh = fd.ExtrudedMesh(
            base_mesh,
            layers=2,
            layer_height=layer_height,
            extrusion_type="radial",
        )
        radii = np.linspace(base_radius, 1.2 if decreasing_radius else 2.2, 5)
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


def _independently_assembled_layer_values(mesh, degree=2):
    """Assemble every layer separately, without the production fast path."""
    coordinate_degree = mesh.coordinates.ufl_element().degree()
    horizontal_coordinate_degree = (
        coordinate_degree[0]
        if isinstance(coordinate_degree, tuple)
        else coordinate_degree
    )
    coordinate_space = fd.VectorFunctionSpace(
        mesh,
        "CG",
        horizontal_coordinate_degree,
        vfamily="CG",
        vdegree=degree,
    )
    coordinates = fd.Function(coordinate_space).interpolate(mesh.coordinates)
    level_count = degree * (mesh.layers - 1) + 1
    coordinate_data = coordinates.dat.data_ro.reshape(
        (-1, level_count, mesh.geometric_dimension)
    )

    base_mesh = mesh._base_mesh
    base_topology = base_mesh if base_mesh is not None else mesh.topology._base_mesh
    base_coordinate_space = fd.VectorFunctionSpace(
        base_topology,
        "CG",
        horizontal_coordinate_degree,
        dim=mesh.geometric_dimension,
    )
    if base_mesh is None:
        topological_coordinates = fd.CoordinatelessFunction(base_coordinate_space)
        topological_coordinates.dat.data[:] = coordinate_data[:, 0, :]
        layer_mesh = fd.Mesh(topological_coordinates)
        layer_coordinates = layer_mesh.coordinates
    else:
        layer_coordinates = fd.Function(base_coordinate_space)
        layer_coordinates.dat.data[:] = coordinate_data[:, 0, :]
        layer_mesh = fd.Mesh(layer_coordinates)
    if degree == 1:
        control_space = fd.FunctionSpace(layer_mesh, "CG", 1)
    else:
        control_space = fd.FunctionSpace(
            layer_mesh, "CG", 1, variant="equispaced,iso(2)"
        )
    form = fd.TestFunction(control_space) * fd.dx

    values = None
    for level in range(level_count):
        layer_coordinates.dat.data[:] = coordinate_data[:, level, :]
        layer_values = fd.assemble(form).dat.data_ro
        if values is None:
            values = np.empty((layer_values.size, level_count))
        values[:, level] = layer_values
    return values


def _record_control_volume_assemblies(monkeypatch):
    original_assemble = fd.assemble
    assemblies = []

    def counted_assemble(*args, **kwargs):
        result = original_assemble(*args, **kwargs)
        if isinstance(result, fd.Cofunction):
            assemblies.append(result)
        return result

    monkeypatch.setattr(fd, "assemble", counted_assemble)
    return assemblies


@pytest.mark.parametrize(
    "geometry",
    [
        "2d-cartesian",
        "3d-cartesian",
        "2d-cylindrical",
        "3d-spherical",
        "3d-spherical-decreasing-radius",
    ],
)
def test_layerwise_nodal_control_volume_sums(geometry, monkeypatch):
    mesh, expected, tolerance = _extruded_mesh(geometry)
    assemblies = _record_control_volume_assemblies(monkeypatch)
    areas = layerwise_nodal_control_volumes(mesh)
    sums = _layer_sums(areas)

    assert len(assemblies) == 1
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


def test_linear_areas_preserve_quadratic_spherical_geometry():
    mesh, _, _ = _extruded_mesh("3d-spherical")
    expected_values = _independently_assembled_layer_values(mesh, degree=1)

    areas = layerwise_nodal_control_volumes(mesh, degree=1)
    actual_values = areas.dat.data_ro.reshape(expected_values.shape)

    np.testing.assert_allclose(actual_values, expected_values, rtol=1.0e-12)
    discrete_base_area = fd.assemble(fd.Constant(1) * fd.dx(domain=mesh._base_mesh))
    radii = np.linspace(1.2, 2.2, 3)
    np.testing.assert_allclose(
        _layer_sums(areas, degree=1),
        discrete_base_area * (radii / radii[0]) ** 2,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize(
    "geometry",
    ["2d-cylindrical", "3d-spherical", "3d-spherical-decreasing-radius"],
)
def test_similarity_fast_path_matches_independent_assemblies(geometry):
    mesh, _, _ = _extruded_mesh(geometry)
    expected_values = _independently_assembled_layer_values(mesh)

    areas = layerwise_nodal_control_volumes(mesh)
    actual_values = areas.dat.data_ro.reshape(expected_values.shape)

    np.testing.assert_allclose(actual_values, expected_values, rtol=1.0e-12)


@pytest.mark.parametrize(
    ("length", "warp_amplitude"), [(1.0, 0.2), (1.0e-7, 1.0e-6)]
)
def test_warped_layers_are_assembled_independently(
    length, warp_amplitude, monkeypatch
):
    mesh = fd.ExtrudedMesh(
        fd.IntervalMesh(4, length), layers=2, layer_height=0.5
    )
    original_coordinates = mesh.coordinates.copy(deepcopy=True)
    mesh.coordinates.interpolate(
        fd.as_vector(
            [
                original_coordinates[0]
                * (
                    1
                    + warp_amplitude
                    * original_coordinates[1]
                    * original_coordinates[0]
                    / length
                ),
                original_coordinates[1],
            ]
        )
    )

    expected_values = _independently_assembled_layer_values(mesh)
    assemblies = _record_control_volume_assemblies(monkeypatch)
    areas = layerwise_nodal_control_volumes(mesh)
    actual_values = areas.dat.data_ro.reshape(expected_values.shape)
    expected_sums = length * (
        1 + warp_amplitude * np.linspace(0.0, 1.0, 5)
    )

    assert len(assemblies) == 5
    assert len({id(assembly) for assembly in assemblies}) == 1
    np.testing.assert_allclose(actual_values, expected_values, rtol=1.0e-12)
    np.testing.assert_allclose(_layer_sums(areas), expected_sums, rtol=1.0e-12)


@pytest.mark.parametrize("warped", [False, True])
def test_large_coordinate_offset_cannot_hide_layer_deformation(warped, monkeypatch):
    mesh = fd.ExtrudedMesh(fd.UnitIntervalMesh(4), layers=2, layer_height=0.5)
    original_coordinates = mesh.coordinates.copy(deepcopy=True)
    x, z = original_coordinates
    if warped:
        horizontal_coordinate = 1.0e12 + x * (1 + 0.01 * z * x)
    else:
        horizontal_coordinate = 1.0e12 + x * (1 + 0.1 * z)
    mesh.coordinates.interpolate(fd.as_vector([horizontal_coordinate, z]))

    expected_values = _independently_assembled_layer_values(mesh)
    assemblies = _record_control_volume_assemblies(monkeypatch)
    areas = layerwise_nodal_control_volumes(mesh)
    actual_values = areas.dat.data_ro.reshape(expected_values.shape)

    np.testing.assert_allclose(actual_values, expected_values, rtol=1.0e-12)
    if warped:
        assert len(assemblies) == 5


def test_layerwise_areas_on_mesh_with_only_base_topology():
    original_mesh, _, _ = _extruded_mesh("3d-spherical")
    mesh = fd.Mesh(original_mesh.coordinates.copy(deepcopy=True))
    assert mesh._base_mesh is None

    areas = layerwise_nodal_control_volumes(mesh)

    np.testing.assert_allclose(
        areas.dat.data_ro,
        layerwise_nodal_control_volumes(original_mesh).dat.data_ro,
        rtol=1.0e-12,
    )


def test_layerwise_nodal_control_volumes_after_checkpoint(tmp_path):
    mesh, expected, tolerance = _extruded_mesh("3d-spherical")
    checkpoint_path = mesh.comm.bcast(str(tmp_path / "mesh.h5"), root=0)

    with fd.CheckpointFile(checkpoint_path, "w") as checkpoint:
        checkpoint.save_mesh(mesh)

    with fd.CheckpointFile(checkpoint_path, "r") as checkpoint:
        loaded_mesh = checkpoint.load_mesh(mesh.name)

    areas = layerwise_nodal_control_volumes(loaded_mesh, name="cv_area_3d")
    np.testing.assert_allclose(_layer_sums(areas), expected, rtol=tolerance)

    output_path = mesh.comm.bcast(str(tmp_path / "cv_area_3d.pvd"), root=0)
    fd.VTKFile(output_path).write(areas)
    assert Path(output_path).is_file()


def test_nodal_control_volume_validation():
    mesh = fd.UnitSquareMesh(1, 1)
    with pytest.raises(ValueError, match="degree must be either 1 or 2"):
        nodal_control_volumes(mesh, degree=3)
    with pytest.raises(ValueError, match="degree must be either 1 or 2"):
        nodal_control_volumes(mesh, degree=1.0)
    with pytest.raises(ValueError, match="degree must be either 1 or 2"):
        nodal_control_volumes(mesh, degree=True)
    with pytest.raises(ValueError, match="require an extruded mesh"):
        layerwise_nodal_control_volumes(mesh)

    periodic_mesh = fd.ExtrudedMesh(fd.UnitIntervalMesh(2), layers=3, periodic=True)
    with pytest.raises(ValueError, match="require non-periodic extrusion"):
        layerwise_nodal_control_volumes(periodic_mesh)
