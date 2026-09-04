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
    LithosphereConnectorFactory,
    PointCloudSource,
    GlobalLayerIndicator,
    HalfSpaceCoolingGeotherm,
    SourceLateralWeight,
)
from gtrack.age_sources import AgeCloudSource


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


def test_ensure_reconstruction_downloads_muller_2022_se():
    gplates_data_path = Path(__file__).resolve().parents[2] / "demos/mantle_convection/gplates_global"
    plate_reconstruction_files_with_path = ensure_reconstruction("Muller 2022 SE v1.2", gplates_data_path)

    # Check if the files are downloaded and accessible
    # Values can be lists (rotation/topology files) or strings (polygon files)
    for files in plate_reconstruction_files_with_path.values():
        file_list = files if isinstance(files, list) else [files]
        for file_path in file_list:
            assert Path(file_path).exists(), f"{file_path} does not exist."


def test_velocity_reconstruction_regression(write_pvd=False):
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
        assert abs(radial_component) < 5e-9, f"Radial component at time {t} Ma is {radial_component}; expected 0"

        surface_rms.append(sqrt(assemble(inner(gplates_function, gplates_function) * ds_t)))

    # Loading reference plate velocities
    test_data_path = Path(__file__).resolve().parent / "data"

    with open(test_data_path / "test_gplates.pkl", "rb") as file:
        ref_surface_rms = pickle.load(file)

    np.testing.assert_allclose(surface_rms, ref_surface_rms)


# =============================================================================
# Factory mechanics
# =============================================================================
#
# ConnectorFactory only does bookkeeping — the single-source/single-output
# guards, inheritance through the setters, and forwarding of the connector-level
# parameters. That logic is producer-agnostic, so these tests use a fake
# producer and a stand-in connector rather than a real gtrack
# LithosphereCloudSource: the factory never drives the producer, so there is no
# reconstruction to download. Age validation is covered data-free in
# test_sources.py, and the end-to-end path with a REAL producer is the
# reconstruction-backed regression in test_connectors.py (indicator/geotherm)
# plus test_velocity_reconstruction_regression above (velocity).
#
# Checkpoint discovery moved into gtrack (CheckpointPolicy) with the ocean
# tracker; it is exercised by gtrack's own suite. gadopt no longer owns it.


class _FakeProducer:
    """Minimal gtrack ``AgeCloudSource`` for the factory tests: satisfies the
    protocol (provides / monotonic_backward / at_age / validate_age) so
    PointCloudSource accepts it, with no reconstruction data. ``provides``
    carries both ``thickness`` and ``age`` so a GlobalLayerIndicator indicator and a
    HalfSpaceCoolingGeotherm geotherm both pass the connector's requires<=provides
    check. ``at_age`` raises: the factory-mechanics tests must never drive it."""

    provides = frozenset({"thickness", "age"})
    monotonic_backward = False

    def at_age(self, age):
        raise AssertionError("factory-mechanics tests must not drive the producer")

    def validate_age(self, age):
        pass


class _DummyGplates:
    """Stand-in for pyGplatesConnector: the factory tests store it on the source
    but never touch its time API."""

    oldest_age = 100.0
    delta_t = 1.0

    def ndtime2age(self, ndtime):
        return float(ndtime) * 100.0

    def age2ndtime(self, age):
        return age / 100.0


class TestConnectorFactory:
    def test_fake_producer_satisfies_the_protocol(self):
        # If this fails, every factory test that wraps it is testing nothing.
        assert isinstance(_FakeProducer(), AgeCloudSource)

    def test_cannot_create_source_without_class(self):
        factory = ConnectorFactory()
        with pytest.raises(
            TypeError, match="The source class is not configured."
        ):
            factory.create_source()

    def test_cannot_construct_indicator_without_source_class(self):
        factory = ConnectorFactory()
        with pytest.raises(
            RuntimeError,
            match="A source must be created or assigned before you access the indicator",
        ):
            _ = factory.indicator

    def test_constructed_source(self):
        factory = ConnectorFactory(source_class=PointCloudSource)
        factory.create_source(_FakeProducer(), _DummyGplates())

        assert isinstance(factory.source, PointCloudSource)

    def test_inherited_source(self):
        factory1 = ConnectorFactory(source_class=PointCloudSource)
        factory2 = ConnectorFactory()
        factory1.create_source(_FakeProducer(), _DummyGplates())
        factory2.source = factory1.source
        assert factory1.source is factory2.source

    def test_strictly_single_source(self):
        factory = ConnectorFactory(source_class=PointCloudSource)
        factory.create_source(_FakeProducer(), _DummyGplates())
        source = PointCloudSource(_FakeProducer(), _DummyGplates())
        with pytest.raises(RuntimeError, match=r"This factory already has a source\."):
            factory.source = source

    def test_constructed_output(self):
        factory = ConnectorFactory(output_class=GlobalLayerIndicator)
        factory.create_indicator()

        assert isinstance(factory.output, GlobalLayerIndicator)

    def test_lithosphere_factory_forwards_lateral_weight_strategy(self):
        factory = LithosphereConnectorFactory()
        lateral_weight = SourceLateralWeight()
        factory.create_indicator(lateral_weight=lateral_weight)

        assert factory.output.lateral_weight is lateral_weight
        assert factory.output.requires == frozenset(
            {"thickness", "lateral_weight"}
        )

    def test_inherited_output(self):
        factory1 = ConnectorFactory(output_class=GlobalLayerIndicator)
        factory2 = ConnectorFactory()
        factory1.create_indicator()
        factory2.output = factory1.output
        assert factory1.output is factory2.output

    def test_strictly_single_output(self):
        factory = ConnectorFactory(output_class=GlobalLayerIndicator)
        factory.create_indicator()
        output = GlobalLayerIndicator()
        with pytest.raises(
            RuntimeError, match=r"This factory already has an indicator output\."
        ):
            factory.output = output

    def test_strictly_single_geotherm_output(self):
        factory = ConnectorFactory(geotherm_output_class=HalfSpaceCoolingGeotherm)
        factory.create_geotherm()
        with pytest.raises(
            RuntimeError, match=r"This factory already has a geotherm output\."
        ):
            factory.geotherm_output = HalfSpaceCoolingGeotherm()

    def test_constructed_geotherm_output(self):
        factory = ConnectorFactory(geotherm_output_class=HalfSpaceCoolingGeotherm)
        factory.create_geotherm(thermal_diffusivity_m2_per_s=2e-6)

        assert isinstance(factory.geotherm_output, HalfSpaceCoolingGeotherm)
        assert factory.geotherm_output.thermal_diffusivity_m2_per_s == 2e-6

    def test_geotherm_requires_geotherm_output(self):
        factory = ConnectorFactory(source_class=PointCloudSource)
        factory.create_source(_FakeProducer(), _DummyGplates())
        with pytest.raises(
            RuntimeError,
            match="A geotherm output must be created or assigned before you access the geotherm",
        ):
            _ = factory.geotherm

    def test_indicator_and_geotherm_share_source(self):
        """The whole point of the factory: both connectors hold the same
        Source instance, so the forward-only ocean tracker advances once per
        age no matter which connector updates first."""
        factory = LithosphereConnectorFactory()
        factory.create_source(_FakeProducer(), _DummyGplates())
        factory.create_indicator()
        factory.create_geotherm()

        assert factory.indicator.source is factory.geotherm.source
        assert factory.indicator is not factory.geotherm
        assert isinstance(factory.indicator.output, GlobalLayerIndicator)
        assert isinstance(factory.geotherm.output, HalfSpaceCoolingGeotherm)

    def test_connector_params_forwarded(self):
        """Typed connector-level parameters reach every connector the
        factory creates."""
        from gadopt.gplates import MeshConfig, InterpolationConfig

        mesh_cfg = MeshConfig(r_outer=2.22, depth_scale=2890.0)
        interp_cfg = InterpolationConfig(neighbor_count=8)
        factory = LithosphereConnectorFactory(
            mesh=mesh_cfg, interpolation=interp_cfg, gc_collect_frequency=3
        )
        factory.create_source(_FakeProducer(), _DummyGplates())
        factory.create_indicator()
        factory.create_geotherm()

        assert factory.indicator.gc_collect_frequency == 3
        assert factory.geotherm.gc_collect_frequency == 3
        assert factory.indicator.mesh is mesh_cfg
        assert factory.geotherm.mesh is mesh_cfg
        assert factory.indicator.interpolation is interp_cfg
        assert factory.geotherm.interpolation is interp_cfg

    def test_indicator_raises_when_output_not_constructed(self):
        factory = ConnectorFactory(source_class=PointCloudSource)
        factory.create_source(_FakeProducer(), _DummyGplates())
        with pytest.raises(
            RuntimeError,
            match="An output must be created or assigned before you access the indicator",
        ):
            _ = factory.indicator
