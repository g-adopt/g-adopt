import pickle
from pathlib import Path
import numpy as np
import pytest

from gadopt import *
from gadopt.gplates import (
    GplatesVelocityFunction,
    PlateModelFiles,
    pyGplatesConnector,
    ensure_reconstruction,
    ConnectorFactory,
    LithosphereIndicator,
    LithosphereSource,
    TanhOutput,
)


def test_connector_no_longer_carries_polygon_kwargs():
    """pyGplatesConnector is velocity-only now: the polygon paths moved to
    PlateModelFiles. The constructor must not accept the old kwargs, and the
    new dataclass must carry them. No reconstruction data needed."""
    import inspect

    params = inspect.signature(pyGplatesConnector.__init__).parameters
    assert "continental_polygons" not in params
    assert "static_polygons" not in params

    pf = PlateModelFiles(
        continental_polygons="cont.gpml", static_polygons="static.gpml"
    )
    assert pf.continental_polygons == "cont.gpml"
    assert pf.static_polygons == "static.gpml"
    # Defaults are None so the source None-validation can fire.
    assert PlateModelFiles().continental_polygons is None
    assert PlateModelFiles().static_polygons is None


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
    """Create pyGplatesConnector for testing the indicator/geotherm path."""
    gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
    muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

    return pyGplatesConnector(
        rotation_filenames=muller_files["rotation_filenames"],
        topology_filenames=muller_files["topology_filenames"],
        oldest_age=200,
    )


@pytest.fixture
def plate_files():
    """Polygon file paths for the indicator/geotherm sources."""
    gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
    muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)
    return PlateModelFiles(
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


class TestConnectorFactory:
    def test_cannot_construct_source_without_class(self):
        factory = ConnectorFactory()
        with pytest.raises(
            TypeError, match="Do not know what kind of Source to construct!"
        ):
            factory.construct_source()

    def test_cannot_construct_indicator_without_source_class(self):
        factory = ConnectorFactory()
        with pytest.raises(
            TypeError, match="Do not know what kind of Source to construct!"
        ):
            with pytest.warns(UserWarning):
                _ = factory.indicator

    def test_warn_on_default_source(self):
        factory = ConnectorFactory(source_class=LithosphereSource)
        # LithosphereSource cannot be initialised with default args
        with pytest.raises(
            TypeError,
            match=r"LithosphereSource.__init__.*missing.*required positional arguments:.*",
        ):
            with pytest.warns(UserWarning):
                _ = factory.indicator

    def test_constructed_source(
        self, plate_model_with_polygons, plate_files, synthetic_data
    ):
        factory = ConnectorFactory(source_class=LithosphereSource)
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )

        assert isinstance(factory.source, LithosphereSource)

    def test_inherited_source(
        self, plate_model_with_polygons, plate_files, synthetic_data
    ):
        factory1 = ConnectorFactory(source_class=LithosphereSource)
        factory2 = ConnectorFactory()
        factory1.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )
        factory2.source = factory1.source
        assert factory1.source is factory2.source

    def test_strictly_single_source(
        self, plate_model_with_polygons, plate_files, synthetic_data
    ):
        factory = ConnectorFactory(source_class=LithosphereSource)
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )
        source = LithosphereSource(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )
        with pytest.raises(RuntimeError, match="This indicator already has a Source!"):
            factory.source = source

    def test_constructed_output(self):
        factory = ConnectorFactory(output_class=TanhOutput)
        factory.construct_output()

        assert isinstance(factory.output, TanhOutput)

    def test_inherited_output(self):
        factory1 = ConnectorFactory(output_class=TanhOutput)
        factory2 = ConnectorFactory()
        factory1.construct_output()
        factory2.output = factory1.output
        assert factory1.output is factory2.output

    def test_strictly_single_output(self):
        factory = ConnectorFactory(output_class=TanhOutput)
        factory.construct_output()
        output = TanhOutput()
        with pytest.raises(RuntimeError, match="This indicator already has an Output!"):
            factory.output = output

    def test_default_output_raises_warning(
        self, plate_model_with_polygons, plate_files, synthetic_data
    ):
        factory = ConnectorFactory(source_class=LithosphereSource)
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )
        with pytest.warns(UserWarning):
            _ = factory.indicator


class TestLithosphereConnectorAgeValidation:
    """Test age validation in LithosphereConnector."""

    def test_valid_age_works(self, plate_model_with_polygons, plate_files, synthetic_data, test_coords):
        """Test that valid ages within bounds work correctly."""
        factory = LithosphereIndicator()
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )

        # Valid age within bounds
        ndtime = plate_model_with_polygons.age2ndtime(100)
        result = factory.indicator.get_indicator(test_coords, ndtime)

        assert result.shape == (len(test_coords),)
        assert np.all(np.isfinite(result))

    def test_age_older_than_oldest_raises_error(self, plate_model_with_polygons, plate_files, synthetic_data, test_coords):
        """Test that requesting age > oldest_age raises ValueError."""
        factory = LithosphereIndicator()
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )

        # Age older than oldest_age (200 Ma)
        ndtime = plate_model_with_polygons.age2ndtime(250)

        with pytest.raises(ValueError, match="older than the plate model's oldest age"):
            factory.indicator.get_indicator(test_coords, ndtime)

    def test_negative_age_raises_error(self, plate_model_with_polygons, plate_files, synthetic_data, test_coords):
        """Test that requesting negative age (future) raises ValueError."""
        factory = LithosphereIndicator()
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )

        # Negative age (in the future)
        ndtime = plate_model_with_polygons.age2ndtime(-10)

        with pytest.raises(ValueError, match="negative.*future"):
            factory.indicator.get_indicator(test_coords, ndtime)

    def test_backward_step_raises_error(self, plate_model_with_polygons, plate_files, synthetic_data, test_coords):
        """Test that going backward in ocean tracker raises ValueError."""
        factory = LithosphereIndicator()
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )

        # First step forward to 50 Ma
        ndtime_50 = plate_model_with_polygons.age2ndtime(50)
        factory.indicator.get_indicator(test_coords, ndtime_50)

        # Now try to go backward to 150 Ma (should fail)
        ndtime_150 = plate_model_with_polygons.age2ndtime(150)

        with pytest.raises(ValueError, match="can only evolve forward"):
            factory.indicator.get_indicator(test_coords, ndtime_150)

    def test_forward_steps_work(self, plate_model_with_polygons, plate_files, synthetic_data, test_coords):
        """Test that sequential forward steps (decreasing age) work."""
        factory = LithosphereIndicator()
        factory.construct_source(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_data,
            age_to_property=half_space_cooling,
            plate_files=plate_files,
        )

        # Sequential forward steps (decreasing age)
        for age in [150, 100, 50, 0]:
            ndtime = plate_model_with_polygons.age2ndtime(age)
            result = factory.indicator.get_indicator(test_coords, ndtime)
            assert result.shape == (len(test_coords),)
