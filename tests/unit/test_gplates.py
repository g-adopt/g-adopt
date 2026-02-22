import pickle
from pathlib import Path
import numpy as np
import pytest

from gadopt import *
from gadopt.gplates import (
    GplatesVelocityFunction,
    pyGplatesConnector,
    ensure_reconstruction,
    LithosphereConnector,
    CratonConnector,
)


def test_obtain_muller_2022_se():
    gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
    plate_reconstruction_files_with_path = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

    # Check if the files are downloaded and accessible
    # Values can be lists (rotation/topology files) or strings (polygon files)
    for files in plate_reconstruction_files_with_path.values():
        file_list = files if isinstance(files, list) else [files]
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


# =============================================================================
# Age Validation Tests
# =============================================================================

def half_space_cooling(age_myr):
    """Convert seafloor age (Myr) to lithospheric thickness (km)."""
    age_sec = np.maximum(age_myr, 0) * 3.15576e13
    return np.minimum(2.32 * np.sqrt(1e-6 * age_sec) / 1e3, 150.0)


@pytest.fixture
def plate_model_with_polygons():
    """Create pyGplatesConnector with polygon files for testing."""
    gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
    muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

    return pyGplatesConnector(
        rotation_filenames=muller_files["rotation_filenames"],
        topology_filenames=muller_files["topology_filenames"],
        oldest_age=200,
        continental_polygons=muller_files.get("continental_polygons"),
        static_polygons=muller_files.get("static_polygons"),
    )


@pytest.fixture
def synthetic_data():
    """Create synthetic thickness data for testing."""
    latlon = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
    values = np.array([100, 150, 200, 180])
    return (latlon, values)


@pytest.fixture
def test_coords():
    """Create test coordinates."""
    return np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])


class TestLithosphereConnectorAgeValidation:
    """Test age validation in LithosphereConnector."""

    def test_valid_age_works(self, plate_model_with_polygons, synthetic_data, test_coords):
        """Test that valid ages within bounds work correctly."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
        )

        # Valid age within bounds
        ndtime = plate_model_with_polygons.age2ndtime(100)
        result = connector.get_indicator(test_coords, ndtime)

        assert result.shape == (len(test_coords),)
        assert np.all(np.isfinite(result))

    def test_age_older_than_oldest_raises_error(self, plate_model_with_polygons, synthetic_data, test_coords):
        """Test that requesting age > oldest_age raises ValueError."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
        )

        # Age older than oldest_age (200 Ma)
        ndtime = plate_model_with_polygons.age2ndtime(250)

        with pytest.raises(ValueError, match="older than the plate model's oldest age"):
            connector.get_indicator(test_coords, ndtime)

    def test_negative_age_raises_error(self, plate_model_with_polygons, synthetic_data, test_coords):
        """Test that requesting negative age (future) raises ValueError."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
        )

        # Negative age (in the future)
        ndtime = plate_model_with_polygons.age2ndtime(-10)

        with pytest.raises(ValueError, match="negative.*future"):
            connector.get_indicator(test_coords, ndtime)

    def test_backward_step_raises_error(self, plate_model_with_polygons, synthetic_data, test_coords):
        """Test that going backward in ocean tracker raises ValueError."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
        )

        # First step forward to 50 Ma
        ndtime_50 = plate_model_with_polygons.age2ndtime(50)
        connector.get_indicator(test_coords, ndtime_50)

        # Now try to go backward to 150 Ma (should fail)
        ndtime_150 = plate_model_with_polygons.age2ndtime(150)

        with pytest.raises(ValueError, match="can only evolve forward"):
            connector.get_indicator(test_coords, ndtime_150)

    def test_forward_steps_work(self, plate_model_with_polygons, synthetic_data, test_coords):
        """Test that sequential forward steps (decreasing age) work."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
        )

        # Sequential forward steps (decreasing age)
        for age in [150, 100, 50, 0]:
            ndtime = plate_model_with_polygons.age2ndtime(age)
            result = connector.get_indicator(test_coords, ndtime)
            assert result.shape == (len(test_coords),)


class TestCratonConnectorAgeValidation:
    """Test age validation in CratonConnector."""

    @pytest.fixture
    def craton_connector(self, plate_model_with_polygons, synthetic_data):
        """Create CratonConnector for testing."""
        craton_shapefile = (
            Path(__file__).resolve().parents[2]
            / "demos/mantle_convection/gplates_lithosphere"
            / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2/shapes_cratons.shp"
        )

        if not craton_shapefile.exists():
            pytest.skip("Craton shapefile not available")

        return CratonConnector(
            gplates_connector=plate_model_with_polygons,
            craton_polygons=str(craton_shapefile),
            craton_thickness_data=synthetic_data,
        )

    def test_valid_age_works(self, craton_connector, test_coords, plate_model_with_polygons):
        """Test that valid ages within bounds work correctly."""
        ndtime = plate_model_with_polygons.age2ndtime(100)
        result = craton_connector.get_indicator(test_coords, ndtime)

        assert result.shape == (len(test_coords),)
        assert np.all(np.isfinite(result))

    def test_age_older_than_oldest_raises_error(self, craton_connector, test_coords, plate_model_with_polygons):
        """Test that requesting age > oldest_age raises ValueError."""
        ndtime = plate_model_with_polygons.age2ndtime(250)

        with pytest.raises(ValueError, match="older than the plate model's oldest age"):
            craton_connector.get_indicator(test_coords, ndtime)

    def test_negative_age_raises_error(self, craton_connector, test_coords, plate_model_with_polygons):
        """Test that requesting negative age (future) raises ValueError."""
        ndtime = plate_model_with_polygons.age2ndtime(-10)

        with pytest.raises(ValueError, match="negative.*future"):
            craton_connector.get_indicator(test_coords, ndtime)

    def test_any_order_works(self, craton_connector, test_coords, plate_model_with_polygons):
        """Test that CratonConnector allows any age order (unlike LithosphereConnector)."""
        # CratonConnector uses rotation only, so any order should work

        # First go to 50 Ma
        ndtime_50 = plate_model_with_polygons.age2ndtime(50)
        craton_connector.get_indicator(test_coords, ndtime_50)

        # Then go back to 150 Ma (should work for CratonConnector)
        ndtime_150 = plate_model_with_polygons.age2ndtime(150)
        result = craton_connector.get_indicator(test_coords, ndtime_150)

        assert result.shape == (len(test_coords),)
        assert np.all(np.isfinite(result))
