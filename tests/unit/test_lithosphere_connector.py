"""Tests for LithosphereConnector, CratonConnector, and GplatesScalarFunction.

These tests verify the indicator connector functionality:
- LithosphereConnector: oceanic (age-tracked) + continental (back-rotated) data
- CratonConnector: craton polygon filtering and back-rotation
Both produce smooth 3D indicator fields (~1 in region, ~0 outside).
"""

from pathlib import Path
import numpy as np
import pytest

from gadopt import *
from gadopt.gplates import (
    GplatesScalarFunction,
    IndicatorConnector,
    LithosphereConnector,
    LithosphereConfig,
    CratonConnector,
    CratonConfig,
    pyGplatesConnector,
    ensure_reconstruction
)


# Age-to-thickness conversion function for tests
def half_space_cooling(age_myr, kappa=1e-6):
    """Convert seafloor age (Myr) to lithospheric thickness (km)."""
    age_sec = np.maximum(age_myr, 0) * 3.15576e13
    thickness_m = 2.32 * np.sqrt(kappa * age_sec)
    return np.minimum(thickness_m / 1e3, 150.0)


class TestPyGplatesConnectorExtensions:
    """Test extensions to pyGplatesConnector for lithosphere tracking."""

    def test_connector_stores_filenames(self):
        """Test that pyGplatesConnector stores original filenames."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        connector = pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
        )

        # Check filenames are stored
        assert connector.rotation_filenames == muller_files["rotation_filenames"]
        assert connector.topology_filenames == muller_files["topology_filenames"]

    def test_connector_accepts_polygon_files(self):
        """Test that pyGplatesConnector accepts polygon file parameters."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        connector = pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
            continental_polygons=muller_files.get("continental_polygons"),
            static_polygons=muller_files.get("static_polygons"),
        )

        # Check polygon files are stored
        assert connector.continental_polygons == muller_files.get("continental_polygons")
        assert connector.static_polygons == muller_files.get("static_polygons")

    def test_connector_backward_compatible(self):
        """Test that existing code without polygon files still works."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        # Old-style creation should still work
        connector = pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
        )

        # Polygon files default to None
        assert connector.continental_polygons is None
        assert connector.static_polygons is None


class TestLithosphereConnectorValidation:
    """Test LithosphereConnector validation and error handling."""

    def test_requires_continental_polygons(self):
        """Test that LithosphereConnector raises error without continental_polygons."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        # Connector without polygon files
        connector = pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
            # No continental_polygons or static_polygons
        )

        # Create synthetic data
        latlon = np.array([[0, 0], [10, 10]])
        values = np.array([100, 150])

        with pytest.raises(ValueError, match="continental_polygons"):
            LithosphereConnector(
                gplates_connector=connector,
                continental_data=(latlon, values),
                age_to_property=half_space_cooling,
            )

    def test_requires_static_polygons(self):
        """Test that LithosphereConnector raises error without static_polygons."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        # Connector with continental but not static polygons
        connector = pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
            continental_polygons=muller_files.get("continental_polygons"),
            # No static_polygons
        )

        latlon = np.array([[0, 0], [10, 10]])
        values = np.array([100, 150])

        with pytest.raises(ValueError, match="static_polygons"):
            LithosphereConnector(
                gplates_connector=connector,
                continental_data=(latlon, values),
                age_to_property=half_space_cooling,
            )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists(),
    reason="Plate reconstruction files not downloaded"
)
class TestLithosphereConnectorFunctional:
    """Functional tests for LithosphereConnector (require plate model data)."""

    @pytest.fixture
    def plate_model_with_polygons(self):
        """Create pyGplatesConnector with polygon files."""
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
    def synthetic_continental_data(self):
        """Create synthetic continental thickness data."""
        np.random.seed(42)
        n_points = 1000

        # Generate random points on sphere
        phi = np.random.uniform(0, 2*np.pi, n_points)
        theta = np.arccos(np.random.uniform(-1, 1, n_points))
        lat = 90 - np.degrees(theta)
        lon = np.degrees(phi) - 180

        # Synthetic thickness: 150-250 km
        thickness = 150 + 100 * np.random.rand(n_points)

        return (np.column_stack([lat, lon]), thickness)

    def test_connector_creation(self, plate_model_with_polygons, synthetic_continental_data):
        """Test LithosphereConnector can be created successfully."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_continental_data,
            age_to_property=half_space_cooling,
        )

        # Check config is populated with defaults
        assert connector.config.property_name == "thickness"
        assert connector.config.k_neighbors == 50  # default
        assert connector.config.default_thickness == 100.0  # default
        assert connector.config.r_outer == 2.208  # default
        assert connector.config.depth_scale == 2890.0  # default
        assert connector.config.transition_width == 10.0  # default

    def test_connector_time_conversion(self, plate_model_with_polygons, synthetic_continental_data):
        """Test time conversion delegation to gplates_connector."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_continental_data,
            age_to_property=half_space_cooling,
        )

        # Test delegation
        age = 100.0
        ndtime = connector.age2ndtime(age)
        recovered_age = connector.ndtime2age(ndtime)

        np.testing.assert_allclose(recovered_age, age, rtol=1e-10)

    def test_get_indicator_returns_correct_shape(self, plate_model_with_polygons, synthetic_continental_data):
        """Test that get_indicator returns array of correct shape."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_continental_data,
            age_to_property=half_space_cooling,
            config_extra={"k_neighbors": 10},
        )

        # Create some target coordinates at different radii
        n_targets = 100
        np.random.seed(123)
        phi = np.random.uniform(0, 2*np.pi, n_targets)
        theta = np.arccos(np.random.uniform(-1, 1, n_targets))
        # Vary radius from inner to outer
        r = np.linspace(1.208, 2.208, n_targets)
        target_coords = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

        # Get indicator at some time
        ndtime = connector.age2ndtime(100.0)
        result = connector.get_indicator(target_coords, ndtime)

        assert result.shape == (n_targets,)
        assert np.all(np.isfinite(result))
        # Indicator should be between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_indicator_varies_with_radius(self, plate_model_with_polygons, synthetic_continental_data):
        """Test that indicator is higher at surface than at depth."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_continental_data,
            age_to_property=half_space_cooling,
            config_extra={"k_neighbors": 10, "r_outer": 2.208, "depth_scale": 2890.0},
        )

        # Create points at same (lon, lat) but different radii
        phi = 0.5  # Fixed longitude
        theta = 1.0  # Fixed colatitude

        # Surface points (r = r_outer)
        r_surface = 2.208
        surface_coords = np.array([[
            r_surface * np.sin(theta) * np.cos(phi),
            r_surface * np.sin(theta) * np.sin(phi),
            r_surface * np.cos(theta)
        ]])

        # Deep mantle points (r = r_inner)
        r_deep = 1.208
        deep_coords = np.array([[
            r_deep * np.sin(theta) * np.cos(phi),
            r_deep * np.sin(theta) * np.sin(phi),
            r_deep * np.cos(theta)
        ]])

        ndtime = connector.age2ndtime(100.0)

        surface_indicator = connector.get_indicator(surface_coords, ndtime)
        deep_indicator = connector.get_indicator(deep_coords, ndtime)

        # Surface should have higher indicator (closer to 1)
        assert surface_indicator[0] > deep_indicator[0]
        # Surface should be close to 1 (inside lithosphere)
        assert surface_indicator[0] > 0.9
        # Deep mantle should be close to 0
        assert deep_indicator[0] < 0.1

    def test_get_indicator_caching(self, plate_model_with_polygons, synthetic_continental_data):
        """Test that get_indicator uses caching for repeated calls."""
        connector = LithosphereConnector(
            gplates_connector=plate_model_with_polygons,
            continental_data=synthetic_continental_data,
            age_to_property=half_space_cooling,
            config_extra={"k_neighbors": 10},
        )

        # Create target coordinates
        n_targets = 50
        np.random.seed(456)
        phi = np.random.uniform(0, 2*np.pi, n_targets)
        theta = np.arccos(np.random.uniform(-1, 1, n_targets))
        r = 2.0  # Fixed radius
        target_coords = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

        ndtime = connector.age2ndtime(100.0)

        # First call
        result1 = connector.get_indicator(target_coords, ndtime)

        # Second call with same time should use cache
        result2 = connector.get_indicator(target_coords, ndtime)

        np.testing.assert_array_equal(result1, result2)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists(),
    reason="Plate reconstruction files not downloaded"
)
class TestGplatesScalarFunction:
    """Tests for GplatesScalarFunction Firedrake integration."""

    @pytest.fixture
    def mesh_and_function_space(self):
        """Create a simple spherical mesh and scalar function space."""
        rmin, rmax, ref_level, nlayers = 1.208, 2.208, 3, 4

        mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
        mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial")
        mesh.cartesian = False

        Q = FunctionSpace(mesh, "CG", 2)
        return mesh, Q, rmax

    @pytest.fixture
    def lithosphere_connector(self):
        """Create LithosphereConnector for testing."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        connector = pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
            continental_polygons=muller_files.get("continental_polygons"),
            static_polygons=muller_files.get("static_polygons"),
        )

        # Synthetic continental data
        np.random.seed(42)
        n_points = 500
        phi = np.random.uniform(0, 2*np.pi, n_points)
        theta = np.arccos(np.random.uniform(-1, 1, n_points))
        lat = 90 - np.degrees(theta)
        lon = np.degrees(phi) - 180
        thickness = 150 + 100 * np.random.rand(n_points)

        return LithosphereConnector(
            gplates_connector=connector,
            continental_data=(np.column_stack([lat, lon]), thickness),
            age_to_property=half_space_cooling,
            config_extra={
                "r_outer": 2.208,
                "depth_scale": 2890.0,
                "transition_width": 10.0,
                "k_neighbors": 10,
            },
        )

    def test_scalar_function_creation(self, mesh_and_function_space, lithosphere_connector):
        """Test GplatesScalarFunction can be created."""
        mesh, Q, rmax = mesh_and_function_space

        scalar_func = GplatesScalarFunction(
            Q,
            indicator_connector=lithosphere_connector,
            name="Lithosphere_Indicator"
        )

        assert scalar_func.name() == "Lithosphere_Indicator"
        assert scalar_func.function_space() == Q

    def test_scalar_function_update(self, mesh_and_function_space, lithosphere_connector):
        """Test GplatesScalarFunction.update_plate_reconstruction()."""
        mesh, Q, rmax = mesh_and_function_space

        scalar_func = GplatesScalarFunction(
            Q,
            indicator_connector=lithosphere_connector,
            name="Lithosphere_Indicator"
        )

        # Update to some time
        ndtime = lithosphere_connector.age2ndtime(100.0)
        scalar_func.update_plate_reconstruction(ndtime)

        # Check that values are populated
        data = scalar_func.dat.data_ro
        assert np.all(np.isfinite(data))
        # Indicator should be between 0 and 1
        assert data.min() >= 0
        assert data.max() <= 1

    def test_scalar_function_values_change_with_time(self, mesh_and_function_space, lithosphere_connector):
        """Test that values change when updating to different times."""
        mesh, Q, rmax = mesh_and_function_space

        scalar_func = GplatesScalarFunction(
            Q,
            indicator_connector=lithosphere_connector,
            name="Lithosphere_Indicator"
        )

        # Update to 150 Ma
        ndtime1 = lithosphere_connector.age2ndtime(150.0)
        scalar_func.update_plate_reconstruction(ndtime1)
        data1 = scalar_func.dat.data_ro.copy()

        # Update to 50 Ma
        ndtime2 = lithosphere_connector.age2ndtime(50.0)
        scalar_func.update_plate_reconstruction(ndtime2)
        data2 = scalar_func.dat.data_ro.copy()

        # Values should be different (plates have moved, ages changed)
        assert not np.allclose(data1, data2)


class TestIndicatorComputation:
    """Test the indicator computation logic."""

    def test_tanh_transition(self):
        """Test that tanh transition gives expected values."""
        # At the boundary (r = lith_base), indicator should be 0.5
        r = 2.0
        lith_base = 2.0
        transition_width = 0.01

        indicator = 0.5 * (1 + np.tanh((r - lith_base) / transition_width))
        np.testing.assert_allclose(indicator, 0.5, rtol=1e-10)

    def test_tanh_inside_lithosphere(self):
        """Test indicator is ~1 well inside lithosphere."""
        r = 2.2  # Above lithosphere base
        lith_base = 2.0
        transition_width = 0.01

        indicator = 0.5 * (1 + np.tanh((r - lith_base) / transition_width))
        assert indicator > 0.99

    def test_tanh_in_mantle(self):
        """Test indicator is ~0 well below lithosphere."""
        r = 1.8  # Below lithosphere base
        lith_base = 2.0
        transition_width = 0.01

        indicator = 0.5 * (1 + np.tanh((r - lith_base) / transition_width))
        assert indicator < 0.01

    def test_transition_width_effect(self):
        """Test that larger transition width gives smoother transition."""
        r_values = np.linspace(1.9, 2.1, 100)
        lith_base = 2.0

        # Narrow transition
        narrow_width = 0.01
        narrow_indicator = 0.5 * (1 + np.tanh((r_values - lith_base) / narrow_width))

        # Wide transition
        wide_width = 0.1
        wide_indicator = 0.5 * (1 + np.tanh((r_values - lith_base) / wide_width))

        # Narrow should have steeper gradient
        narrow_gradient = np.abs(np.gradient(narrow_indicator)).max()
        wide_gradient = np.abs(np.gradient(wide_indicator)).max()

        assert narrow_gradient > wide_gradient


class TestInterpolation:
    """Test the interpolation functionality."""

    def test_inverse_distance_weighting(self):
        """Test inverse distance weighted interpolation."""
        from scipy.spatial import cKDTree

        # Create simple source points
        source_xyz = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ], dtype=float)
        source_values = np.array([10, 20, 30, 40], dtype=float)

        # Target point at origin (equidistant from all)
        target = np.array([[0, 0, 0]])

        tree = cKDTree(source_xyz)
        dists, idx = tree.query(target, k=4)

        # Inverse distance weights (all equal since equidistant)
        weights = 1.0 / dists
        weights /= weights.sum(axis=1, keepdims=True)

        result = np.sum(weights * source_values[idx], axis=1)

        # Should be mean of all values
        expected = source_values.mean()
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_exact_match_handling(self):
        """Test that exact matches return source value directly."""
        source_xyz = np.array([[1, 0, 0], [0, 1, 0]])
        source_values = np.array([100.0, 200.0])

        # Target exactly on first source point
        target = np.array([[1, 0, 0]])

        from scipy.spatial import cKDTree
        tree = cKDTree(source_xyz)
        dists, idx = tree.query(target, k=2)

        # If distance is essentially zero, use nearest value
        epsilon = 1e-10
        exact_match = dists[:, 0] < epsilon

        result = np.zeros(len(target))
        result[exact_match] = source_values[idx[exact_match, 0]]

        np.testing.assert_allclose(result, [100.0])


class TestLithosphereConfig:
    """Test LithosphereConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = LithosphereConfig()

        assert config.time_step == 1.0
        assert config.n_points == 10000
        assert config.reinit_interval_myr == 50.0
        assert config.k_neighbors == 50
        assert config.distance_threshold == 0.1
        assert config.default_thickness == 100.0
        assert config.r_outer == 2.208
        assert config.depth_scale == 2890.0
        assert config.transition_width == 10.0
        assert config.property_name == "thickness"

    def test_custom_values(self):
        """Test setting custom values."""
        config = LithosphereConfig(
            n_points=40000,
            r_outer=2.5,
            transition_width=5.0,
        )

        assert config.n_points == 40000
        assert config.r_outer == 2.5
        assert config.transition_width == 5.0
        # Other values should be defaults
        assert config.k_neighbors == 50

    def test_to_dict(self):
        """Test config serialization to dict."""
        config = LithosphereConfig(n_points=20000)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["n_points"] == 20000
        assert d["r_outer"] == 2.208
        assert "time_step" in d

    def test_from_dict(self):
        """Test config creation from dict."""
        d = {"n_points": 30000, "r_outer": 2.0}
        config = LithosphereConfig.from_dict(d)

        assert config.n_points == 30000
        assert config.r_outer == 2.0
        # Defaults for unspecified values
        assert config.k_neighbors == 50

    def test_from_dict_ignores_unknown_keys(self):
        """Test that from_dict ignores unknown keys."""
        d = {"n_points": 30000, "unknown_key": 999}
        config = LithosphereConfig.from_dict(d)

        assert config.n_points == 30000
        assert not hasattr(config, "unknown_key")

    def test_with_overrides(self):
        """Test creating new config with overrides."""
        base = LithosphereConfig(n_points=10000, r_outer=2.208)
        modified = base.with_overrides({"n_points": 40000, "transition_width": 5.0})

        # Modified values
        assert modified.n_points == 40000
        assert modified.transition_width == 5.0
        # Preserved values
        assert modified.r_outer == 2.208
        # Original unchanged
        assert base.n_points == 10000
        assert base.transition_width == 10.0

    def test_validation_time_step(self):
        """Test validation of time_step."""
        with pytest.raises(ValueError, match="time_step must be positive"):
            LithosphereConfig(time_step=-1.0)

        with pytest.raises(ValueError, match="time_step must be positive"):
            LithosphereConfig(time_step=0.0)

    def test_validation_n_points(self):
        """Test validation of n_points."""
        with pytest.raises(ValueError, match="n_points must be at least 100"):
            LithosphereConfig(n_points=50)

    def test_validation_r_outer(self):
        """Test validation of r_outer."""
        with pytest.raises(ValueError, match="r_outer must be positive"):
            LithosphereConfig(r_outer=-1.0)

    def test_validation_depth_scale(self):
        """Test validation of depth_scale."""
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            LithosphereConfig(depth_scale=0.0)

    def test_validation_transition_width(self):
        """Test validation of transition_width."""
        with pytest.raises(ValueError, match="transition_width must be positive"):
            LithosphereConfig(transition_width=-5.0)

    def test_validation_k_neighbors(self):
        """Test validation of k_neighbors."""
        with pytest.raises(ValueError, match="k_neighbors must be at least 1"):
            LithosphereConfig(k_neighbors=0)

    def test_validation_distance_threshold(self):
        """Test validation of distance_threshold."""
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            LithosphereConfig(distance_threshold=-0.1)

    def test_validation_default_thickness(self):
        """Test validation of default_thickness."""
        with pytest.raises(ValueError, match="default_thickness must be non-negative"):
            LithosphereConfig(default_thickness=-10.0)

    def test_validation_reinit_interval_myr(self):
        """Test validation of reinit_interval_myr."""
        with pytest.raises(ValueError, match="reinit_interval_myr must be positive"):
            LithosphereConfig(reinit_interval_myr=0.0)


class TestAgeToThicknessFunction:
    """Test the age-to-thickness conversion function."""

    def test_zero_age_gives_zero_thickness(self):
        """Test that age=0 gives thickness=0."""
        result = half_space_cooling(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0])

    def test_positive_age_gives_positive_thickness(self):
        """Test that positive age gives positive thickness."""
        ages = np.array([10, 50, 100])
        result = half_space_cooling(ages)
        assert np.all(result > 0)

    def test_thickness_increases_with_age(self):
        """Test that thickness increases with age (sqrt relationship)."""
        ages = np.array([10, 40, 90])  # Chosen so sqrt gives nice ratios
        result = half_space_cooling(ages)

        # sqrt relationship: t2/t1 should equal sqrt(age2/age1)
        ratio_10_40 = result[1] / result[0]
        expected_ratio = np.sqrt(40 / 10)
        np.testing.assert_allclose(ratio_10_40, expected_ratio, rtol=1e-10)

    def test_thickness_caps_at_maximum(self):
        """Test that thickness is capped at 150 km."""
        # Very old age should give max thickness
        ages = np.array([1000.0])  # 1 billion years
        result = half_space_cooling(ages)
        assert result[0] == 150.0

    def test_negative_age_treated_as_zero(self):
        """Test that negative ages are treated as zero."""
        ages = np.array([-10.0])
        result = half_space_cooling(ages)
        np.testing.assert_allclose(result, [0.0])


class TestIndicatorConnectorInheritance:
    """Test that connectors properly inherit from IndicatorConnector."""

    def test_lithosphere_connector_is_indicator_connector(self):
        """Test LithosphereConnector inherits from IndicatorConnector."""
        assert issubclass(LithosphereConnector, IndicatorConnector)

    def test_craton_connector_is_indicator_connector(self):
        """Test CratonConnector inherits from IndicatorConnector."""
        assert issubclass(CratonConnector, IndicatorConnector)


class TestCratonConfig:
    """Test CratonConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = CratonConfig()

        assert config.n_points == 20000
        assert config.k_neighbors == 50
        assert config.distance_threshold == 0.1
        assert config.default_thickness == 200.0
        assert config.r_outer == 2.208
        assert config.depth_scale == 2890.0
        assert config.transition_width == 10.0
        assert config.property_name == "thickness"

    def test_custom_values(self):
        """Test setting custom values."""
        config = CratonConfig(
            n_points=50000,
            transition_width=5.0,
        )

        assert config.n_points == 50000
        assert config.transition_width == 5.0
        # Other values should be defaults
        assert config.k_neighbors == 50

    def test_to_dict(self):
        """Test config serialization to dict."""
        config = CratonConfig(n_points=30000)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["n_points"] == 30000
        assert d["r_outer"] == 2.208
        assert d["depth_scale"] == 2890.0

    def test_from_dict(self):
        """Test config creation from dict."""
        d = {"n_points": 40000, "transition_width": 5.0}
        config = CratonConfig.from_dict(d)

        assert config.n_points == 40000
        assert config.transition_width == 5.0
        # Defaults for unspecified values
        assert config.k_neighbors == 50

    def test_with_overrides(self):
        """Test creating new config with overrides."""
        base = CratonConfig(n_points=20000)
        modified = base.with_overrides({"n_points": 50000, "transition_width": 5.0})

        assert modified.n_points == 50000
        assert modified.transition_width == 5.0
        # Original unchanged
        assert base.n_points == 20000
        assert base.transition_width == 10.0

    def test_validation_n_points(self):
        """Test validation of n_points."""
        with pytest.raises(ValueError, match="n_points must be at least 100"):
            CratonConfig(n_points=50)

    def test_validation_k_neighbors(self):
        """Test validation of k_neighbors."""
        with pytest.raises(ValueError, match="k_neighbors must be at least 1"):
            CratonConfig(k_neighbors=0)

    def test_validation_distance_threshold(self):
        """Test validation of distance_threshold."""
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            CratonConfig(distance_threshold=-0.1)

    def test_validation_default_thickness(self):
        """Test validation of default_thickness."""
        with pytest.raises(ValueError, match="default_thickness must be non-negative"):
            CratonConfig(default_thickness=-10.0)

    def test_validation_r_outer(self):
        """Test validation of r_outer."""
        with pytest.raises(ValueError, match="r_outer must be positive"):
            CratonConfig(r_outer=-1.0)

    def test_validation_depth_scale(self):
        """Test validation of depth_scale."""
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            CratonConfig(depth_scale=0.0)

    def test_validation_transition_width(self):
        """Test validation of transition_width."""
        with pytest.raises(ValueError, match="transition_width must be positive"):
            CratonConfig(transition_width=-5.0)


class TestCratonConnectorFunctional:
    """Functional tests for CratonConnector with real gtrack operations."""

    @pytest.fixture
    def gplates_connector(self):
        """Create pyGplatesConnector with polygon files."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"

        if not (gplates_data_path / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists():
            pytest.skip("Plate reconstruction data not available")

        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        return pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
            continental_polygons=muller_files.get("continental_polygons"),
            static_polygons=muller_files.get("static_polygons"),
        )

    @pytest.fixture
    def craton_shapefile(self):
        """Path to craton shapefile."""
        # The craton shapefile is in the gplates_lithosphere demo
        craton_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_lithosphere/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2/shapes_cratons.shp"
        if not craton_path.exists():
            pytest.skip("Craton shapefile not available")
        return str(craton_path)

    def test_craton_connector_creation(self, gplates_connector, craton_shapefile):
        """Test CratonConnector can be created."""
        connector = CratonConnector(
            gplates_connector=gplates_connector,
            craton_polygons=craton_shapefile,
        )

        assert connector.gplates_connector is gplates_connector
        assert connector.config is not None
        assert isinstance(connector, IndicatorConnector)

    def test_craton_connector_with_config(self, gplates_connector, craton_shapefile):
        """Test CratonConnector with custom config."""
        config = CratonConfig(n_points=10000, smooth_width=0.03)

        connector = CratonConnector(
            gplates_connector=gplates_connector,
            craton_polygons=craton_shapefile,
            config=config,
        )

        assert connector.config.n_points == 10000
        assert connector.config.smooth_width == 0.03

    def test_craton_connector_time_conversion(self, gplates_connector, craton_shapefile):
        """Test time conversion delegates to gplates_connector."""
        connector = CratonConnector(
            gplates_connector=gplates_connector,
            craton_polygons=craton_shapefile,
        )

        # Test round-trip conversion
        age = 100.0
        ndtime = connector.age2ndtime(age)
        age_back = connector.ndtime2age(ndtime)

        np.testing.assert_allclose(age, age_back, rtol=1e-10)

    def test_craton_get_indicator_returns_correct_shape(self, gplates_connector, craton_shapefile):
        """Test that get_indicator returns array of correct shape."""
        connector = CratonConnector(
            gplates_connector=gplates_connector,
            craton_polygons=craton_shapefile,
            config_extra={"n_points": 5000},  # Smaller for faster test
        )

        # Create some test coordinates (unit sphere)
        n_points = 100
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = np.random.uniform(1.5, 2.2, n_points)

        coords = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])

        ndtime = connector.age2ndtime(100.0)
        indicator = connector.get_indicator(coords, ndtime)

        assert indicator.shape == (n_points,)
        assert np.all(np.isfinite(indicator))
        # Indicator should be between 0 and 1
        assert indicator.min() >= 0
        assert indicator.max() <= 1

    def test_craton_indicator_caching(self, gplates_connector, craton_shapefile):
        """Test that results are cached when time doesn't change."""
        connector = CratonConnector(
            gplates_connector=gplates_connector,
            craton_polygons=craton_shapefile,
            config_extra={"n_points": 5000},
        )

        coords = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        ndtime = connector.age2ndtime(100.0)

        # First call
        result1 = connector.get_indicator(coords, ndtime)

        # Second call with same time should use cache
        result2 = connector.get_indicator(coords, ndtime)

        np.testing.assert_array_equal(result1, result2)


class TestCratonIndicatorComputation:
    """Test the craton indicator computation logic."""

    def test_tanh_transition_at_boundary(self):
        """Test that tanh gives 0.5 at distance_threshold."""
        distance_threshold = 0.05
        smooth_width = 0.02
        dist = distance_threshold  # At boundary

        indicator = 0.5 * (1.0 - np.tanh((dist - distance_threshold) / smooth_width))
        np.testing.assert_allclose(indicator, 0.5, rtol=1e-10)

    def test_tanh_inside_craton(self):
        """Test indicator is ~1 well inside craton (dist=0)."""
        distance_threshold = 0.05
        smooth_width = 0.02
        dist = 0.0  # Inside craton

        indicator = 0.5 * (1.0 - np.tanh((dist - distance_threshold) / smooth_width))
        # Should be close to 1
        assert indicator > 0.9

    def test_tanh_outside_craton(self):
        """Test indicator is ~0 well outside craton."""
        distance_threshold = 0.05
        smooth_width = 0.02
        dist = 0.2  # Far outside

        indicator = 0.5 * (1.0 - np.tanh((dist - distance_threshold) / smooth_width))
        # Should be close to 0
        assert indicator < 0.01

    def test_smooth_width_effect(self):
        """Test that smooth_width controls transition sharpness."""
        distance_threshold = 0.05
        dist = 0.06  # Just outside boundary

        # Narrow transition
        narrow = 0.5 * (1.0 - np.tanh((dist - distance_threshold) / 0.005))

        # Wide transition
        wide = 0.5 * (1.0 - np.tanh((dist - distance_threshold) / 0.05))

        # Narrow should drop faster (lower value at same distance)
        assert narrow < wide


class TestCratonGplatesScalarFunction:
    """Test GplatesScalarFunction with CratonConnector."""

    @pytest.fixture
    def mesh_and_function_space(self):
        """Create a simple spherical shell mesh for testing."""
        rmin, rmax = 1.208, 2.208
        mesh2d = CubedSphereMesh(rmin, refinement_level=2, degree=2)
        mesh = ExtrudedMesh(mesh2d, layers=4, extrusion_type="radial")
        mesh.cartesian = False

        Q = FunctionSpace(mesh, "CG", 2)
        return mesh, Q, rmax

    @pytest.fixture
    def craton_connector(self):
        """Create CratonConnector for tests."""
        gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"

        if not (gplates_data_path / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists():
            pytest.skip("Plate reconstruction data not available")

        craton_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_lithosphere/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2/shapes_cratons.shp"
        if not craton_path.exists():
            pytest.skip("Craton shapefile not available")

        muller_files = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

        gplates_connector = pyGplatesConnector(
            rotation_filenames=muller_files["rotation_filenames"],
            topology_filenames=muller_files["topology_filenames"],
            oldest_age=200,
            continental_polygons=muller_files.get("continental_polygons"),
            static_polygons=muller_files.get("static_polygons"),
        )

        return CratonConnector(
            gplates_connector=gplates_connector,
            craton_polygons=str(craton_path),
            config_extra={"n_points": 5000},  # Smaller for faster tests
        )

    def test_scalar_function_with_craton_connector(self, mesh_and_function_space, craton_connector):
        """Test GplatesScalarFunction works with CratonConnector."""
        mesh, Q, rmax = mesh_and_function_space

        scalar_func = GplatesScalarFunction(
            Q,
            indicator_connector=craton_connector,
            name="Craton_Indicator"
        )

        assert scalar_func.name() == "Craton_Indicator"
        assert isinstance(scalar_func.indicator_connector, CratonConnector)

    def test_craton_scalar_function_update(self, mesh_and_function_space, craton_connector):
        """Test GplatesScalarFunction.update_plate_reconstruction() with cratons."""
        mesh, Q, rmax = mesh_and_function_space

        scalar_func = GplatesScalarFunction(
            Q,
            indicator_connector=craton_connector,
            name="Craton_Indicator"
        )

        ndtime = craton_connector.age2ndtime(100.0)
        scalar_func.update_plate_reconstruction(ndtime)

        data = scalar_func.dat.data_ro
        assert np.all(np.isfinite(data))
        assert data.min() >= 0
        assert data.max() <= 1
