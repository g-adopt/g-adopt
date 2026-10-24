"""Real curved-mesh surface traces and VTK round trips; no GIA solve.

Run in a Firedrake environment, also under mpiexec to exercise empty ranks.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import firedrake as fd
import numpy as np
import pytest
import selfgrav_common as common
from convert_spada_surface import convert_surface
from mpi4py import MPI
from spada import ProfileRateHistory, parse_args, run_settings
from spada_surface_io import GROUP, read_frames, read_metadata
from spada_surface_output import NodalRestriction, SurfaceOutput


def curved_cube(nx=2, ny=2, nz=2):
    mesh = fd.UnitCubeMesh(nx, ny, nz)
    mesh.coordinates.dat.data[:] += .3
    return common.curve_mesh(mesh)


def matching_submesh(parent, dimension, tag):
    submesh = fd.Submesh(parent, dimension, tag)
    coordinates = fd.Function(fd.VectorFunctionSpace(submesh, "CG", 2))
    NodalRestriction(parent, submesh, 2)(parent.coordinates, coordinates)
    return fd.Mesh(coordinates)


def output_directory(tmp_path):
    # Every MPI rank must write pieces of the same PVD series.
    return Path(fd.COMM_WORLD.bcast(str(tmp_path), root=0))


def test_surface_option_is_independent_of_volume_and_profiles():
    defaults = parse_args([])
    surface = parse_args(["--write_surface_output"])
    assert surface.write_surface_output
    assert not surface.write_output
    assert not surface.write_profiles
    assert run_settings(defaults) == run_settings(surface)
    assert surface.surface_output_format == "hdf5"
    direct = parse_args(["--write_surface_output", "--surface_output_format", "vtk"])
    assert direct.surface_output_format == "vtk"
    assert run_settings(defaults) == run_settings(direct)
    both = parse_args(["--write_output", "--write_surface_output"])
    assert both.write_output and both.write_surface_output


@pytest.mark.parametrize("degree", [2, 3])
def test_nodal_trace_preserves_curved_geometry_and_cubic_fields(degree):
    parent = curved_cube()
    marker = fd.Function(fd.FunctionSpace(parent, "DG", 0)).assign(1.)
    parent.mark_entities(marker, 101)
    mantle = matching_submesh(parent, 3, 101)
    surface = matching_submesh(mantle, 2, 6)
    x = fd.SpatialCoordinate(mantle)
    source = fd.Function(fd.VectorFunctionSpace(mantle, "CG", degree)).interpolate(
        fd.as_vector((x[0]**degree, x[1] + x[2], x[2]**degree)))
    result = fd.Function(fd.VectorFunctionSpace(surface, "CG", degree))
    NodalRestriction(mantle, surface, degree)(source, result)
    y = fd.SpatialCoordinate(surface)
    expected = fd.Function(result.function_space()).interpolate(
        fd.as_vector((y[0]**degree, y[1] + y[2], y[2]**degree)))
    np.testing.assert_allclose(result.dat.data_ro, expected.dat.data_ro, atol=3e-13)
    np.testing.assert_allclose(fd.assemble(fd.Constant(1.) * fd.dx(domain=surface)),
                               fd.assemble(fd.Constant(1.) * fd.ds(6, domain=mantle)),
                               rtol=1e-9)


@pytest.mark.parametrize("rotating", [False, True])
@pytest.mark.parametrize("output_format", ["hdf5", "vtk"])
def test_surface_fields_rates_and_vtu_roundtrip(tmp_path, monkeypatch, rotating, output_format):
    monkeypatch.setattr(common, "D_SCALE", 2.)
    monkeypatch.setattr(common, "gravity_exact_ufl", lambda r: fd.Constant(4.))
    parent = curved_cube()
    marker = fd.Function(fd.FunctionSpace(parent, "DG", 0)).assign(1.)
    parent.mark_entities(marker, 101)
    mantle = matching_submesh(parent, 3, 101)
    original_coordinates = mantle.coordinates.dat.data_ro.copy()
    x, xp = fd.SpatialCoordinate(mantle), fd.SpatialCoordinate(parent)
    base_u = fd.Function(fd.VectorFunctionSpace(mantle, "CG", 3)).interpolate(
        fd.as_vector((x[0]**3, x[1] + x[2], x[2]**3)))
    base_phi = fd.Function(fd.FunctionSpace(parent, "CG", 2)).interpolate(
        4 + xp[0] + 2*xp[1] - 3*xp[2])
    u, phi = fd.Function(base_u.function_space()), fd.Function(base_phi.function_space())
    u.assign(base_u)
    phi.assign(base_phi)
    rotation = {name: 0. for name in ("m1", "m2", "m3")} if rotating else {}
    solver = SimpleNamespace(displacement=u, potential=phi, rotation_values=lambda: rotation)
    rates = ProfileRateHistory(solver)
    directory = output_directory(tmp_path)
    output = SurfaceOutput(directory, "polar-motion" if rotating else "cap", parent, mantle,
                           boundary_id=6, output_format=output_format)
    assert output.mesh.topological_dimension == 2
    assert output.mesh.geometric_dimension == 3
    assert output.cell_count == 8
    rates.after_step(solver, 0.)
    output.write(solver, rates, 0.)
    assert np.all(output.fields["rate_valid"].dat.data_ro == 0)
    for name, field in output.fields.items():
        if name.endswith("rate_m_per_yr"):
            assert np.all(np.isnan(field.dat.data_ro))

    rates.before_step(solver, .02)
    u.assign(base_u * 3.)
    phi.assign(base_phi * 3.)
    if rotating:
        rotation.update(m1=1e-4, m2=-2e-4, m3=3e-5)
    rates.after_step(solver, .02)
    output.write(solver, rates, .02)
    np.testing.assert_array_equal(mantle.coordinates.dat.data_ro, original_coordinates)
    np.testing.assert_allclose(output.mesh.coordinates.dat.data_ro,
                               output.surface.coordinates.dat.data_ro * 2., atol=1e-14)
    assert np.all(output.fields["rate_valid"].dat.data_ro == 1)
    assert np.all(output.fields["rate_dt_yr"].dat.data_ro == 20)
    points = fd.Function(fd.VectorFunctionSpace(output.mesh, "CG", 3)).interpolate(
        fd.SpatialCoordinate(output.mesh)).dat.data_ro / 2.
    expected_u = 6*np.column_stack((points[:, 0]**3, points[:, 1]+points[:, 2], points[:, 2]**3))
    np.testing.assert_allclose(output.fields["displacement_m"].dat.data_ro, expected_u, atol=3e-13)
    np.testing.assert_allclose(output.fields["displacement_rate_m_per_yr"].dat.data_ro,
                               expected_u / 30., atol=3e-13)
    radial = points / np.linalg.norm(points, axis=1)[:, None]
    rho = np.linalg.norm(points[:, :2], axis=1)
    east = np.column_stack((-points[:, 1]/rho, points[:, 0]/rho, np.zeros_like(rho)))
    north = np.cross(radial, east)
    for name, basis in (("radial", radial), ("northward", north), ("eastward", east)):
        np.testing.assert_allclose(output.fields[f"{name}_displacement_m"].dat.data_ro,
                                   np.sum(expected_u*basis, axis=1), atol=3e-13)
    rot = common.OMEGA_SQ * (rotation.get("m3", 0.)*(points[:, 0]**2 + points[:, 1]**2)
                             - rotation.get("m1", 0.)*points[:, 0]*points[:, 2]
                             - rotation.get("m2", 0.)*points[:, 1]*points[:, 2])
    grav_base = 4 + points[:, 0] + 2*points[:, 1] - 3*points[:, 2]
    np.testing.assert_allclose(output.fields["geoid_total_m"].dat.data_ro,
                               (3*grav_base + rot) / 2., atol=3e-13)
    np.testing.assert_allclose(output.fields["geoid_total_rate_m_per_yr"].dat.data_ro,
                               (2*grav_base + rot) / 40., atol=3e-13)
    assert output.metadata()["coordinates_unit"] == "m"
    assert output.metadata()["boundary_id"] == 6

    vtk_path = output.path
    if output_format == "hdf5":
        if fd.COMM_WORLD.rank == 0:
            # Read an MPI-written file on ONE rank: checks redistribution,
            # P2 geometry, CG2/CG3 fields and no dependency on parent meshes.
            assert len(list(directory.glob("*.h5"))) == 1
            assert not list(directory.rglob("*.vtu"))
            with fd.CheckpointFile(output.path, "r", comm=fd.COMM_SELF) as checkpoint:
                metadata = read_metadata(checkpoint)
                assert [f["t_kyr"] for f in read_frames(checkpoint)] == [0., .02]
                mesh = checkpoint.load_mesh(metadata["mesh_name"])
                assert mesh.coordinates.function_space().ufl_element().degree() == 2
                displacement = checkpoint.load_function(mesh, "displacement_m", idx=1)
                geoid = checkpoint.load_function(mesh, "geoid_gravitational_m", idx=1)
                assert displacement.function_space().ufl_element().degree() == 3
                assert geoid.function_space().ufl_element().degree() == 2
                p = fd.Function(displacement.function_space()).interpolate(
                    fd.SpatialCoordinate(mesh)).dat.data_ro / 2.
                np.testing.assert_allclose(displacement.dat.data_ro,
                                           6*np.column_stack((p[:, 0]**3, p[:, 1]+p[:, 2], p[:, 2]**3)),
                                           atol=3e-13)
                rate0 = checkpoint.load_function(mesh, "displacement_rate_m_per_yr", idx=0)
                rate1 = checkpoint.load_function(mesh, "displacement_rate_m_per_yr", idx=1)
                assert np.all(np.isnan(rate0.dat.data_ro))
                np.testing.assert_allclose(rate1.dat.data_ro, displacement.dat.data_ro/30., atol=3e-13)
        fd.COMM_WORLD.barrier()
        # Upstream runs collectively on COMM_WORLD, just as its CLI does.
        vtk_path = str(directory / "converted.pvd")
        convert_surface(output.path, vtk_path)
        selected = str(directory / "selected.pvd")
        convert_surface(output.path, selected, functions="geoid_total_m", timesteps="last")
        with pytest.raises(ValueError, match="Unknown fields"):
            convert_surface(output.path, selected, functions="missing")
        with pytest.raises(ValueError, match="Unknown frames"):
            convert_surface(output.path, selected, timesteps="99")
    fd.COMM_WORLD.barrier()
    if fd.COMM_WORLD.rank == 0:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        if output_format == "hdf5":
            selected_frames = ET.parse(selected).findall(".//DataSet")
            assert [float(e.attrib["timestep"]) for e in selected_frames] == [.02]
        datasets = ET.parse(vtk_path).findall(".//DataSet")
        assert [float(e.attrib["timestep"]) for e in datasets] == [0., .02]
        for index, dataset in enumerate(datasets):
            path = directory / dataset.attrib["file"]
            reader = (vtk.vtkXMLPUnstructuredGridReader() if path.suffix == ".pvtu"
                      else vtk.vtkXMLUnstructuredGridReader())
            reader.SetFileName(str(path))
            reader.Update()
            grid = reader.GetOutput()
            assert grid.GetNumberOfCells() == output.cell_count
            assert all(grid.GetCell(i).GetCellDimension() == 2
                       for i in range(grid.GetNumberOfCells()))
            assert all(grid.GetCellType(i) == 69 for i in range(grid.GetNumberOfCells()))
            assert grid.GetPointData().GetArray("geoid_total_m") is not None
            assert grid.GetPointData().GetArray("displacement_rate_m_per_yr") is not None
            valid = vtk_to_numpy(grid.GetPointData().GetArray("rate_valid"))
            np.testing.assert_allclose(valid, float(index), atol=1e-12)
    fd.COMM_WORLD.barrier()


@pytest.mark.parametrize("output_format", ["hdf5", "vtk"])
def test_surface_writer_handles_ranks_without_surface_cells(tmp_path, monkeypatch, output_format):
    monkeypatch.setattr(common, "gravity_exact_ufl", lambda r: fd.Constant(1.))
    mesh = curved_cube(1, 1, 4)
    solver = SimpleNamespace(
        displacement=fd.Function(fd.VectorFunctionSpace(mesh, "CG", 3)).assign(fd.Constant((1, 2, 3))),
        potential=fd.Function(fd.FunctionSpace(mesh, "CG", 2)).assign(4.),
        rotation_values=dict)
    rates = ProfileRateHistory(solver)
    rates.after_step(solver, 0.)
    output = SurfaceOutput(output_directory(tmp_path), "empty-ranks", mesh, mesh, boundary_id=6,
                           output_format=output_format)
    assert output.cell_count == 2
    empty = mesh.comm.allreduce(int(output.surface.cell_set.size == 0), op=MPI.SUM)
    if mesh.comm.size > 2:
        assert empty >= mesh.comm.size - 2
    output.write(solver, rates, 0.)


def test_surface_hdf5_restart_and_incomplete_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "D_SCALE", 2.)
    monkeypatch.setattr(common, "gravity_exact_ufl", lambda r: fd.Constant(1.))
    mesh = curved_cube()
    solver = SimpleNamespace(
        displacement=fd.Function(fd.VectorFunctionSpace(mesh, "CG", 3)).assign(fd.Constant((1, 2, 3))),
        potential=fd.Function(fd.FunctionSpace(mesh, "CG", 2)).assign(4.), rotation_values=dict)
    rates = ProfileRateHistory(solver)
    rates.after_step(solver, 0.)
    directory = output_directory(tmp_path)
    volume_path = str(directory / "restart_state.h5")
    with fd.CheckpointFile(volume_path, "w") as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(solver.displacement, name="u")
        checkpoint.save_function(solver.potential, name="psi")
    output = SurfaceOutput(directory, "restart", mesh, mesh, boundary_id=6)
    output.write(solver, rates, 0.)
    rates.before_step(solver, .02)
    solver.displacement.assign(fd.Constant((2, 4, 6)))
    rates.after_step(solver, .02)
    output.write(solver, rates, .02)
    with fd.CheckpointFile(output.path, "a") as checkpoint:
        checkpoint.h5pyfile[GROUP]["frames"].create_group("2")  # Interrupted save.
    # Load the volume checkpoint and recreate its surface, as in the driver.
    with fd.CheckpointFile(volume_path, "r") as checkpoint:
        mesh = checkpoint.load_mesh(mesh.name)
        solver.displacement = checkpoint.load_function(mesh, "u")
        solver.potential = checkpoint.load_function(mesh, "psi")
    rates = ProfileRateHistory(solver, initial_time=0.)
    rates.before_step(solver, .01)
    x = fd.SpatialCoordinate(mesh)
    solver.displacement.interpolate(fd.as_vector((x[0]**3, x[1]+x[2], x[2]**3)))
    rates.after_step(solver, .01)
    output = SurfaceOutput(directory, "restart", mesh, mesh, boundary_id=6, restart_time=0.)
    assert output.series.next_index == 3
    output.write(solver, rates, .01)
    with fd.CheckpointFile(output.path, "r") as checkpoint:
        assert read_frames(checkpoint) == [{"index": 0, "t_kyr": 0.}, {"index": 3, "t_kyr": .01}]
    fd.COMM_WORLD.barrier()
    if fd.COMM_WORLD.rank == 0:
        with fd.CheckpointFile(output.path, "r", comm=fd.COMM_SELF) as checkpoint:
            stored_mesh = checkpoint.load_mesh(output.mesh.name)
            displacement = checkpoint.load_function(stored_mesh, "displacement_m", idx=3)
            rate = checkpoint.load_function(stored_mesh, "displacement_rate_m_per_yr", idx=3)
            p = fd.Function(displacement.function_space()).interpolate(
                fd.SpatialCoordinate(stored_mesh)).dat.data_ro / 2.
            expected = 2*np.column_stack((p[:, 0]**3, p[:, 1]+p[:, 2], p[:, 2]**3))
            np.testing.assert_allclose(displacement.dat.data_ro, expected, atol=3e-13)
            np.testing.assert_allclose(rate.dat.data_ro, (expected-np.array([2, 4, 6]))/10., atol=3e-13)
    fd.COMM_WORLD.barrier()
    converted = directory / "resumed.pvd"
    convert_surface(output.path, converted, timesteps="0,3")
    if fd.COMM_WORLD.rank == 0:
        assert [float(e.attrib["timestep"]) for e in ET.parse(converted).findall(".//DataSet")] == [0., .01]
    with pytest.raises(ValueError, match="Unknown frames"):
        convert_surface(output.path, converted, timesteps="1")
    fd.COMM_WORLD.barrier()
    with pytest.raises(ValueError, match="strictly increasing"):
        output.write(solver, rates, .01)
    # A fresh run preserves one previous archive instead of appending to it.
    output = SurfaceOutput(directory, "restart", mesh, mesh, boundary_id=6)
    assert Path(output.path).with_suffix(".previous.h5").exists()
    with fd.CheckpointFile(output.path, "r") as checkpoint:
        assert not read_frames(checkpoint)
    mesh.coordinates.dat.data[:] *= 1.01
    with pytest.raises(ValueError, match="surface mesh nodes"):
        SurfaceOutput(directory, "restart", mesh, mesh, boundary_id=6, restart_time=0.)
