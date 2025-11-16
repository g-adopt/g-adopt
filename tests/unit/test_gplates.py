import pickle
from pathlib import Path
import numpy as np

from gadopt import *
from gadopt.gplates import GplatesVelocityFunction, pyGplatesConnector, ensure_reconstruction


def test_obtain_muller_2022_se():
    gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
    plate_reconstruction_files_with_path = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

    # Check if the files are downloaded and accessible
    for file_list in plate_reconstruction_files_with_path.values():
        for file_path in file_list:
            assert Path(file_path).exists(), f"{file_path} does not exist."


def test_gplates(write_pvd=False):
    gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"

    # Set up geometry:
    rmin, rmax, ref_level, nlayers = 1.22, 2.22, 5, 16

    # Construct a CubedSphere mesh and then extrude into a sphere
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=(rmax - rmin)/(nlayers-1),
        extrusion_type="radial",
    )
    mesh.cartesian = False  # I don't think we need this in the tests, but for clarity

    V = VectorFunctionSpace(mesh, "CG", 2)
    mueller_2022_se = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

    # compute surface velocities
    rec_model = pyGplatesConnector(
        rotation_filenames=mueller_2022_se["rotation_filenames"],
        topology_filenames=mueller_2022_se["topology_filenames"],
        nseeds=1e5,
        nneighbours=4,
        oldest_age=409,
        delta_t=1.0
    )

    gplates_function = GplatesVelocityFunction(V, gplates_connector=rec_model, top_boundary_marker="top")

    surface_rms = []

    # Create a VTK file if needed
    if write_pvd:
        vtkfile = VTKFile("gplates_velocity.pvd")

    for t in np.arange(409, 0, -50):
        gplates_function.update_plate_reconstruction(rec_model.age2ndtime(t))

        # Visualise the velocity field
        if write_pvd:
            vtkfile.write(gplates_function)

        # Calculate and test radial component
        radial_component = assemble(inner(gplates_function, FacetNormal(mesh)) * ds_t)

        # Assert that radial component is essentially zero
        assert abs(radial_component) < 5e-9, f"Radial component at time {t} Ma is {radial_component}, should be 0"

        surface_rms.append(sqrt(assemble(inner(gplates_function, gplates_function) * ds_t)))

    # Loading reference plate velocities
    test_data_path = Path(__file__).resolve().parent / "data"

    with open(test_data_path / "test_gplates.pkl", "rb") as file:
        ref_surface_rms = pickle.load(file)

    np.testing.assert_allclose(surface_rms, ref_surface_rms)
