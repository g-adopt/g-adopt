"""Tests for ScalarFieldConnector and its factory functions.

Three groups:

  * **Validation contracts** — no reconstruction data; verify that bad
    Source/Output pairings raise at construction, that bad function spaces
    are caught up front by GplatesScalarFunction, and that
    InterpolationConfig rejects invalid kernels.

  * **Connector regression** — for each of the four factory pairings
    (lithosphere indicator, lithosphere geotherm, polygon indicator, polygon
    geotherm) plus the GplatesScalarFunction wrapper, evaluate on a small
    extruded cubed-sphere mesh at a fixed series of geological ages, record
    volume/surface integrals, and compare against a pickled reference. Built
    once against the Muller 2022 SE v1.2 reconstruction.

  * **Shared-source consistency** — confirms that a single LithosphereSource
    can drive both an indicator and a geotherm connector simultaneously
    without producing duplicate state.
"""

import gc
from pathlib import Path
import pickle

import firedrake as fd
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from gadopt.gplates import (
    GeothermERFOutput,
    GeothermLinearOutput,
    GplatesScalarFunction,
    ScalarFieldConnector,
    InterpolationConfig,
    SphericalKNNInterpolator,
    MaskedGeothermLinearOutput,
    MaskedQuinticOutput,
    MeshConfig,
    PlateModelFiles,
    PointCloudSource,
    Source,
    QuinticOutput,
    ensure_reconstruction,
    ConnectorFactory,
    LithosphereConnectorFactory,
    PolygonConnectorFactory,
    pyGplatesConnector,
)
from gtrack import (
    LithosphereCloudConfig,
    LithosphereCloudSource,
    PolygonIndicatorConfig,
    PolygonIndicatorSource,
)
from gtrack.config import TracerConfig
# Internal to the deblend rather than public API, so imported from the module
# it lives in rather than widening gadopt.gplates.
from gadopt.gplates.outputs import MEMBERSHIP_FLOOR


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GPLATES_GLOBAL = REPO_ROOT / "demos/mantle_convection/gplates_global"
GPLATES_FIELDS = REPO_ROOT / "demos/mantle_convection/gplates_fields"
CONTINENTAL_DATA = GPLATES_FIELDS / "continental_lithospheric_thickness_mesh.h5"
CRATON_SHAPEFILE = GPLATES_FIELDS / "Craton_Boundaries_Inferred.shp"
DATA_DIR = Path(__file__).resolve().parent / "data"

OLDEST_AGE = 120
LITH_N_POINTS = 2000
POLYGON_N_POINTS = 3000
TEST_AGES = (100, 50, 0)


def _require_data():
    if not (GPLATES_GLOBAL / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2").exists():
        pytest.skip("Muller 2022 SE reconstruction not downloaded; run `make data`.")
    if not CONTINENTAL_DATA.exists():
        pytest.skip(f"Continental thickness data missing at {CONTINENTAL_DATA}.")


def _require_craton():
    if not CRATON_SHAPEFILE.exists():
        pytest.skip(f"Craton shapefile missing at {CRATON_SHAPEFILE}.")


def half_space_cooling(age_myr):
    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    return np.minimum(2.32 * np.sqrt(1e-6 * age_sec) / 1e3, 150.0)


def _load_continental_data():
    with h5py.File(CONTINENTAL_DATA, "r") as f:
        lonlat = f["lonlat"][:]
        values = f["values"][:]
    return np.column_stack([lonlat[:, 1], lonlat[:, 0]]), values


# ---------------------------------------------------------------------------
# Lightweight mock source for validation contract tests (no I/O)
# ---------------------------------------------------------------------------

class _DummySource(Source):
    """Source that does no work — exists only so the validation contract
    can be exercised without paying for a real reconstruction load."""

    def __init__(self, provides, comm=MPI.COMM_WORLD):
        self._provides = frozenset(provides)
        self.comm = comm
        self._is_root = (comm.rank == 0)
        self.gplates_connector = _DummyGplates()

    @property
    def provides(self) -> frozenset[str]:
        return self._provides

    def _compute_sources(self, age):
        raise RuntimeError("dummy source should never reach _compute_sources")


class _DummyGplates:
    """Minimal stand-in for pyGplatesConnector — only the time-conversion
    helpers and oldest_age are touched by the validation path."""
    oldest_age = 100.0
    delta_t = 1.0

    def ndtime2age(self, ndtime):
        return float(ndtime) * 100.0

    def age2ndtime(self, age):
        return age / 100.0


# ---------------------------------------------------------------------------
# Validation contracts (no reconstruction data)
# ---------------------------------------------------------------------------

class TestRequiresProvidesContract:
    """An OutputStrategy declares the source-property keys it needs; the
    connector validates this against the Source's provides set at
    construction time."""

    @pytest.mark.parametrize(
        "provides, output",
        [
            # A lithosphere source provides {"thickness","age"} (xyz is implicit
            # and never listed); QuinticOutput requires {"thickness"}.
            (frozenset({"thickness", "age"}), QuinticOutput()),
            # GeothermLinearOutput requires only {"thickness"}.
            (frozenset({"thickness", "age"}), GeothermLinearOutput()),
            # A source that genuinely provides a ``thickness`` channel pairs
            # with QuinticOutput; the guard against the polygon case is the
            # channel NAME (masked_thickness), exercised in TestChannelNameGuard.
            (frozenset({"thickness", "membership"}),
             QuinticOutput(default_thickness_km=0.0)),
            (frozenset({"thickness", "membership"}), GeothermLinearOutput()),
        ],
    )
    def test_pairing_allowed(self, provides, output):
        ScalarFieldConnector(_DummySource(provides), output)  # must not raise

    def test_missing_channel_raises(self):
        # GeothermERFOutput needs "age"; this source provides only
        # {"thickness","membership"}, so requires<=provides rejects the pairing.
        src = _DummySource({"thickness", "membership"})
        with pytest.raises(ValueError, match="age"):
            ScalarFieldConnector(src, GeothermERFOutput())


class TestConnectorConstruction:
    def test_gc_collect_frequency_validated(self):
        src = _DummySource({"thickness"})
        with pytest.raises(ValueError, match="gc_collect_frequency"):
            ScalarFieldConnector(src, QuinticOutput(), gc_collect_frequency=0)

    def test_defaults_use_module_level_configs(self):
        src = _DummySource({"thickness"})
        conn = ScalarFieldConnector(src, QuinticOutput())
        assert isinstance(conn.mesh, MeshConfig)
        assert isinstance(conn.interpolation, InterpolationConfig)


class TestGcCollectDefault:
    """The gc_collect_frequency default is 10 (matching gtrack's own internal
    GC cadence), not 1 — a full gc.collect() every call was an undefended
    default that wasted ~130s over a 5000-step loop while gtrack already
    breaks the pygplates C++ cycles internally. The default must be 10 on
    every construction path, including the factories (which forward their own
    default to the connector, so a stale factory default would silently
    override the connector's)."""

    def test_default_is_ten_direct(self):
        conn = ScalarFieldConnector(_DummySource({"thickness"}), QuinticOutput())
        assert conn.gc_collect_frequency == 10

    def test_default_is_ten_lithosphere_factory(self):
        # Assigning a pre-built source skips construct_source -> no
        # reconstruction I/O.
        factory = LithosphereConnectorFactory()
        factory.source = _DataSource()
        factory.construct_output()
        assert factory.indicator.gc_collect_frequency == 10

    def test_default_is_ten_polygon_factory(self):
        # PolygonConnectorFactory's indicator is MaskedQuinticOutput, which
        # requires masked_thickness — use the polygon-style _CapSource.
        factory = PolygonConnectorFactory()
        factory.source = _CapSource()
        factory.construct_output()
        assert factory.indicator.gc_collect_frequency == 10

    def _drive(self, monkeypatch, frequency, n_calls):
        calls = {"n": 0}
        monkeypatch.setattr(
            "gadopt.gplates.connectors.gc.collect",
            lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
        )
        conn = ScalarFieldConnector(
            _DataSource(), QuinticOutput(), gc_collect_frequency=frequency
        )
        target = _target_coords()
        # Distinct ages (spaced > delta_t=1.0, all <= oldest_age=100) so every
        # call is a cache miss and runs _compute (where the gc counter lives).
        for age in range(90, 90 - 10 * n_calls, -10):
            conn.get_indicator(target, conn.source.age2ndtime(float(age)))
        return calls["n"]

    @pytest.mark.parametrize(
        "frequency, n_calls, expected_collects",
        [
            (3, 9, 3),      # collect every Nth call
            (None, 5, 0),   # None disables the connector-level collect entirely
            (1, 4, 4),      # collect on every call
        ],
    )
    def test_collect_cadence(self, monkeypatch, frequency, n_calls, expected_collects):
        assert self._drive(monkeypatch, frequency, n_calls) == expected_collects


class TestResultCacheKey:
    """The result cache is keyed on the *identity* of the target_coords
    buffer (via a weakref), not on its contents. GplatesScalarFunction holds
    one mesh_coords array for its lifetime, so identity is a sound O(1) key
    and avoids hashing the whole coordinate buffer on every call.
    """

    @staticmethod
    def _conn():
        return ScalarFieldConnector(_DummySource({"thickness"}), QuinticOutput())

    def test_same_array_same_age_hits(self):
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = np.zeros(4)
        conn._update_cache(10.0, arr, result)
        assert conn._check_cache(10.0, arr) is True

    def test_byte_equal_distinct_array_misses(self):
        # arr2 is byte-for-byte identical but a different object — a content
        # hash would (wrongly) hit; identity correctly misses.
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = np.zeros(4)
        conn._update_cache(10.0, arr, result)
        arr2 = arr.copy()
        assert np.array_equal(arr, arr2)
        assert conn._check_cache(10.0, arr2) is False

    def test_dead_referent_misses_without_raising(self):
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        conn._update_cache(10.0, arr, np.zeros(4))
        # Drop every binding to the cached array, then collect.
        del arr
        gc.collect()
        assert conn._cached_coords_ref() is None
        new_array = np.arange(12, dtype=float).reshape(4, 3)
        assert conn._check_cache(10.0, new_array) is False

    def test_age_guard_independent_of_identity(self):
        conn = self._conn()
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = np.zeros(4)
        # delta_t is 1.0 (from _DummyGplates).
        conn._update_cache(10.0, arr, result)
        # Same buffer, but the age has moved a full delta_t away -> miss.
        assert conn._check_cache(11.0, arr) is False
        # Re-cache at the new age; a sub-delta_t age with the same buffer hits.
        conn._update_cache(11.0, arr, result)
        assert conn._check_cache(11.5, arr) is True


# ---------------------------------------------------------------------------
# Shared interpolation geometry (P10)
# ---------------------------------------------------------------------------

class _DataSource(Source):
    """Dummy source returning a fixed small cloud — enough to exercise the real
    kNN interpolation path without any reconstruction I/O.

    Provides ``thickness`` (a genuine depth everywhere) plus ``membership``, so
    it stands in for a total-coverage source — pairs with ``QuinticOutput`` and
    the geotherms. A polygon-style fixture (``masked_thickness``) is a separate
    subclass.
    """

    provides = frozenset({"thickness", "membership"})

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm
        self._is_root = (comm.rank == 0)
        self.gplates_connector = _DummyGplates()
        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(20, 3))
        xyz = 6.371e6 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        self._fixed = {
            "xyz": xyz,
            "thickness": rng.uniform(50.0, 200.0, size=20),
            "membership": rng.uniform(0.0, 1.0, size=20),
        }

    def _compute_sources(self, age):
        # Same cloud at every age (good enough for the geometry-sharing tests).
        return {k: v.copy() for k, v in self._fixed.items()}


def _fibonacci_directions(n, offset=0.0):
    """``n`` quasi-uniform unit vectors, offset so two clouds never coincide."""
    i = np.arange(n, dtype=float) + 0.5 + offset
    polar = np.arccos(np.clip(1.0 - 2.0 * i / n, -1.0, 1.0))
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([
        np.cos(azimuth) * np.sin(polar),
        np.sin(azimuth) * np.sin(polar),
        np.cos(polar),
    ])


class _CapSource(_DataSource):
    """A polar cap of CONSTANT depth on a quasi-uniform cloud.

    Polygon-style: provides ``masked_thickness`` (depth times membership) and
    ``membership``, so it pairs with the masked outputs. Inside the cap every
    seed carries ``DEPTH_KM`` and membership 1; outside, masked_thickness 0 and
    membership 0. So pointwise ``masked_thickness == DEPTH_KM * membership``,
    and because interpolation is linear, that identity survives blending with
    ANY kernel: the blended ratio is exactly ``DEPTH_KM`` everywhere the
    coverage is nonzero, at every bandwidth.

    That is what makes this fixture worth having. Any bandwidth dependence a
    test sees in the recovered depth is contamination by coverage, not honest
    smoothing of a varying depth field — there is nothing to smooth.

    The same construction is its blind spot, and it is worth stating so nobody
    over-trusts it. Making the two channels exactly proportional makes them
    linearly dependent, so this fixture CANNOT detect channel confusion: an
    implementation that derived membership from ``thickness / DEPTH_KM`` would
    pass every test built on it. What catches that is
    ``test_recovers_a_laterally_varying_depth_exactly`` in test_outputs.py,
    which drives non-proportional depth and coverage. Do not delete that test
    on the grounds that these ones cover the same ground.
    """

    provides = frozenset({"masked_thickness", "membership"})
    CAP_RADIUS_RAD = 0.5
    DEPTH_KM = 200.0

    def __init__(self, n_points=20000, comm=MPI.COMM_WORLD):
        super().__init__(comm=comm)
        directions = _fibonacci_directions(n_points)
        inside = np.arccos(np.clip(directions[:, 2], -1.0, 1.0)) < self.CAP_RADIUS_RAD
        self._fixed = {
            "xyz": 6.371e6 * directions,
            "membership": inside.astype(float),
            "masked_thickness": self.DEPTH_KM * inside.astype(float),
        }


class _CountingCKDTree:
    """Wraps cKDTree, counting constructions so a test can assert how many
    interpolation geometries were built."""

    count = 0

    def __init__(self, *args, **kwargs):
        type(self).count += 1
        from scipy.spatial import cKDTree as _real
        self._tree = _real(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self._tree.query(*args, **kwargs)


def _target_coords():
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(15, 3))
    return 2.0 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


class _MaskedSource(_CapSource):
    """Stands in for a polygon-style source: provides ``masked_thickness`` and
    ``membership`` rather than a plain ``thickness`` channel."""


class TestChannelNameGuard:
    """A polygon-style source publishes ``masked_thickness`` (depth times
    membership), NOT ``thickness``. That single naming choice is now the whole
    guard against the known-bad pairing that once needed a dedicated
    ``zero_outside`` / ``isinstance(QuinticOutput)`` cross-check: a
    ``QuinticOutput`` (or a geotherm) that requires plain ``thickness`` fails
    the connector's ``requires <= provides`` subset check outright, and so would
    any user output reading ``thickness`` as a depth off such a source.

    The check lives in ScalarFieldConnector.__init__ rather than on the factory,
    so it holds on every route to a connector. Both routes are exercised here;
    the direct-construction case is the one a factory-level check would miss."""

    def test_quintic_on_masked_source_raises_via_factory(self):
        factory = ConnectorFactory()
        factory.source = _MaskedSource()
        factory.output = QuinticOutput()
        with pytest.raises(ValueError, match="thickness"):
            _ = factory.indicator

    def test_quintic_on_masked_source_raises_on_direct_construction(self):
        """The guard must not be reachable only through the factory. Demos and
        user scripts build ScalarFieldConnector directly, which is exactly how
        the bad pairing survived in the demo while a factory-only check was in
        place."""
        with pytest.raises(ValueError, match="thickness"):
            ScalarFieldConnector(_MaskedSource(), QuinticOutput())

    def test_masked_quintic_on_masked_source_passes(self):
        factory = ConnectorFactory()
        factory.source = _MaskedSource()
        factory.output = MaskedQuinticOutput(width_km=50.0)
        assert factory.indicator is not None

    def test_masked_geotherm_on_masked_source_passes(self):
        """The masked geotherm reads the same masked_thickness + membership
        channels, so it pairs with a polygon source; the plain
        GeothermLinearOutput (which requires ``thickness``) would not."""
        assert ScalarFieldConnector(
            _MaskedSource(), MaskedGeothermLinearOutput()
        ) is not None

    def test_plain_geotherm_on_masked_source_raises(self):
        with pytest.raises(ValueError, match="thickness"):
            ScalarFieldConnector(_MaskedSource(), GeothermLinearOutput())

    def test_quintic_on_regular_source_passes(self):
        # Lithosphere-style sources cover the whole sphere and never have
        # vanishing thickness, so the plain step stays legitimate there.
        factory = ConnectorFactory()
        factory.source = _DataSource()
        factory.output = QuinticOutput()
        assert factory.indicator is not None


class TestCoverageDoesNotContaminateDepth:
    """The point of carrying membership separately: the interpolation
    bandwidth stops being a physics parameter.

    With one channel, widening the kernel changes the recovered depth and the
    interior amplitude, so the bandwidth cannot be chosen on numerical grounds
    — it is carrying modelling meaning. With two, it only controls how many
    nodes the transition is spread over.

    Read the claim precisely. It is NOT that a recovered depth is independent
    of bandwidth in general — over a laterally varying depth map a wider kernel
    averages more of it, and that is honest low-pass filtering which should
    change the answer. The claim is that COVERAGE no longer leaks into the
    depth. The fixture below has a constant depth precisely so the two cannot
    be confused.

    Evaluated on a cloud INDEPENDENT of the source points. Evaluating on the
    seeds themselves is invalid here: the interpolator snaps an exact hit to
    the raw property value, so every diagnostic would report a perfect answer
    that no real mesh node ever sees.
    """

    # 2.5x apart, both comfortably above the ~0.025 rad seed spacing so
    # neither degenerates into nearest-neighbour lookup.
    SIGMAS = (0.03, 0.075)
    # Large enough that the Gaussian is not truncated by the kNN cutoff:
    # sqrt(400/pi) * 0.025 ~ 0.28 rad, about 3.7 sigma at the wider setting.
    K = 400

    @staticmethod
    def _targets(polar_angles, mesh, depth_km):
        """Directions at the given polar angles, at one depth below surface."""
        azimuth = np.linspace(0.0, 2.0 * np.pi, len(polar_angles), endpoint=False)
        directions = np.column_stack([
            np.cos(azimuth) * np.sin(polar_angles),
            np.sin(azimuth) * np.sin(polar_angles),
            np.cos(polar_angles),
        ])
        return directions * (mesh.r_outer - depth_km / mesh.depth_scale)

    def _field(self, src, sigma, polar_angles, depth_km, mesh):
        cfg = InterpolationConfig(
            kernel="gaussian",
            gaussian_sigma=sigma,
            k_neighbors=self.K,
            distance_threshold=1.0,
        )
        conn = ScalarFieldConnector(
            src, MaskedQuinticOutput(width_km=20.0), mesh=mesh, interpolation=cfg
        )
        target = self._targets(polar_angles, mesh, depth_km)
        return conn.get_indicator(target, src.age2ndtime(0.0))

    def test_recovered_depth_is_uncontaminated_across_the_edge(self):
        """The sharp test. Sample straight across the cap boundary, where
        coverage runs from ~1 to ~0, and compare the field just ABOVE the
        region's base against the field at the surface.

        If the base depth is honest, a node inside the region at 190 km is
        still above its 200 km base, so it reads the same coverage as its
        surface node — at every bandwidth. If the base were the blended
        product instead, it would sit at ``coverage * 200`` km, so a node at
        coverage 0.5 would have its base at 100 km, and the 190 km sample
        would read zero instead of 0.5. Widening the kernel would then change
        which nodes were affected AND by how much.
        """
        mesh = MeshConfig()
        src = _CapSource()
        polar = np.linspace(0.30, 0.70, 40)  # cap edge is at 0.5 rad

        for sigma in self.SIGMAS:
            surface = self._field(src, sigma, polar, 0.0, mesh)
            above_base = self._field(src, sigma, polar, 190.0, mesh)

            # Where the region is present at all, the two agree to machine
            # precision — measured max difference 7e-16, not a loose bound.
            covered = surface > MEMBERSHIP_FLOOR
            assert covered.sum() > 10, "sweep must straddle the boundary"
            np.testing.assert_allclose(
                above_base[covered], surface[covered], rtol=1e-12, atol=1e-14,
                err_msg=f"base depth contaminated by coverage at sigma={sigma}",
            )

            # Below the floor the deblend declares the node outside rather
            # than dividing two vanishing numbers, so the column drops to zero
            # instead of carrying a depth recovered from noise. The surface
            # still shows the residual coverage, which is bounded by the floor.
            np.testing.assert_array_equal(above_base[~covered], 0.0)
            assert np.all(surface[~covered] <= MEMBERSHIP_FLOOR)

    def test_interior_amplitude_is_uncontaminated(self):
        """Deep inside the region the field is 1 regardless of bandwidth and
        regardless of how deep the region is. This is the property
        `fade_ref_km` could not have: it capped interior amplitude at
        ``depth / fade_ref_km`` thousands of km from any boundary, an error
        no choice of kernel could reduce."""
        mesh = MeshConfig()
        src = _CapSource()
        polar = np.linspace(0.0, 0.15, 12)  # >= 4.7 sigma clear of the edge

        for sigma in self.SIGMAS:
            surface = self._field(src, sigma, polar, 0.0, mesh)
            np.testing.assert_allclose(surface, 1.0, atol=1e-6)

    def test_bandwidth_still_controls_the_transition_width(self):
        """Guards the two tests above against passing vacuously. The kernel
        must still do its job — a wider one spreads the coverage roll-off over
        a wider band — otherwise 'independent of bandwidth' would just mean
        'nothing depends on anything'."""
        mesh = MeshConfig()
        src = _CapSource()
        polar = np.linspace(0.30, 0.70, 40)

        narrow = self._field(src, self.SIGMAS[0], polar, 0.0, mesh)
        wide = self._field(src, self.SIGMAS[1], polar, 0.0, mesh)

        assert not np.allclose(narrow, wide, atol=1e-3)

        # Roll-off width measured as the angular span over which coverage
        # crosses from 0.9 to 0.1; wider kernel, wider span.
        def span(field):
            inside = polar[field >= 0.9]
            outside = polar[field <= 0.1]
            return outside.min() - inside.max()

        assert span(wide) > span(narrow)


class TestGeometrySharing:
    """The interpolation geometry (cKDTree indices + weights) depends only on
    (source cloud, target coords, cfg), so connectors sharing those should
    build it once and reuse the cached bundle."""

    @pytest.fixture(autouse=True)
    def _patch_tree(self, monkeypatch):
        _CountingCKDTree.count = 0
        monkeypatch.setattr(
            "gadopt.gplates.interpolation.cKDTree", _CountingCKDTree
        )

    def test_siblings_share_one_build_and_agree(self):
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        # Two indicator connectors sharing the same source, target and cfg.
        conn_a = ScalarFieldConnector(src, QuinticOutput(), interpolation=cfg)
        conn_b = ScalarFieldConnector(src, QuinticOutput(), interpolation=cfg)

        ndtime = src.age2ndtime(50.0)
        out_a = conn_a.get_indicator(target, ndtime)
        out_b = conn_b.get_indicator(target, ndtime)

        # Geometry built exactly once, shared by both.
        assert _CountingCKDTree.count == 1
        # Both QuinticOutputs on the same source/geometry must agree byte-for-byte.
        np.testing.assert_array_equal(out_a, out_b)

    def test_distinct_config_values_build_separately(self):
        src = _DataSource()
        target = _target_coords()
        conn_a = ScalarFieldConnector(
            src, QuinticOutput(), interpolation=InterpolationConfig(k_neighbors=10)
        )
        conn_b = ScalarFieldConnector(
            src, QuinticOutput(), interpolation=InterpolationConfig(k_neighbors=15)
        )

        ndtime = src.age2ndtime(50.0)
        conn_a.get_indicator(target, ndtime)
        conn_b.get_indicator(target, ndtime)

        # Different cfg values -> different cache keys -> two builds.
        assert _CountingCKDTree.count == 2

    def test_age_advance_rebuilds_geometry(self):
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, QuinticOutput(), interpolation=cfg)

        conn.get_indicator(target, src.age2ndtime(50.0))
        assert _CountingCKDTree.count == 1
        # Advancing a full delta_t misses the source cache, which clears the
        # geometry cache -> rebuild.
        conn.get_indicator(target, src.age2ndtime(20.0))
        assert _CountingCKDTree.count == 2

    def test_gather_matches_hand_computed(self):
        # The gathered property must equal the explicit weighted sum.
        src = _DataSource()
        cfg = InterpolationConfig()
        target = _target_coords()
        conn = ScalarFieldConnector(src, QuinticOutput(), interpolation=cfg)

        source_dict = src.prepare(50.0)
        bundle = conn._interpolator.geometry(source_dict["xyz"], target)
        prop = source_dict["thickness"]
        gathered = SphericalKNNInterpolator.gather(bundle, prop)

        idx = bundle["idx"]
        weights = bundle["weights"]
        expected = np.sum(weights * prop[idx], axis=1)
        expected[bundle["exact_match"]] = prop[idx[bundle["exact_match"], 0]]
        np.testing.assert_array_equal(gathered, expected)


# ---------------------------------------------------------------------------
# GplatesScalarFunction rejects non-scalar spaces
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_mesh():
    """A small extruded icosahedral sphere just big enough to exercise the
    Firedrake path. Used by the function-space-rejection test and reused
    later by the connector regression tests."""
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=1, degree=1)
    return fd.ExtrudedMesh(mesh2d, layers=2, layer_height=0.5, extrusion_type="radial")


class TestGplatesScalarFunctionSpaceCheck:
    def test_vector_space_rejected(self, tiny_mesh):
        V = fd.VectorFunctionSpace(tiny_mesh, "CG", 1)
        src = _DummySource({"thickness"})
        conn = ScalarFieldConnector(src, QuinticOutput())
        with pytest.raises(TypeError, match="scalar function space"):
            GplatesScalarFunction(V, indicator_connector=conn)

    def test_scalar_space_accepted(self, tiny_mesh):
        Q = fd.FunctionSpace(tiny_mesh, "CG", 1)
        src = _DummySource({"thickness"})
        conn = ScalarFieldConnector(src, QuinticOutput())
        # Should construct without raising; the mock source is never asked
        # for data because we never call update_plate_reconstruction.
        GplatesScalarFunction(Q, indicator_connector=conn)


# ---------------------------------------------------------------------------
# Reconstruction-backed fixtures for regression tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plate_model():
    _require_data()
    files = ensure_reconstruction("Muller 2022 SE v1.2", GPLATES_GLOBAL)
    return pyGplatesConnector(
        rotation_filenames=files["rotation_filenames"],
        topology_filenames=files["topology_filenames"],
        oldest_age=OLDEST_AGE,
    )


@pytest.fixture(scope="module")
def plate_files():
    _require_data()
    files = ensure_reconstruction("Muller 2022 SE v1.2", GPLATES_GLOBAL)
    return PlateModelFiles(
        continental_polygons=files.get("continental_polygons"),
        static_polygons=files.get("static_polygons"),
    )


def _make_lith_source(plate_model, plate_files):
    """Wrap a gtrack LithosphereCloudSource in a gadopt PointCloudSource."""
    producer = LithosphereCloudSource(
        rotation_files=plate_model.rotation_filenames,
        topology_files=plate_model.topology_filenames,
        continental_polygons=plate_files.continental_polygons,
        static_polygons=plate_files.static_polygons,
        continental_data=_load_continental_data(),
        age_to_property=half_space_cooling,
        oldest_age=OLDEST_AGE,
        config=LithosphereCloudConfig(
            tracer=TracerConfig(default_mesh_points=LITH_N_POINTS),
        ),
    )
    return PointCloudSource(producer, plate_model)


@pytest.fixture(scope="module")
def lith_source(plate_model, plate_files):
    return _make_lith_source(plate_model, plate_files)


@pytest.fixture(scope="module")
def poly_source(plate_model, plate_files):
    _require_craton()
    producer = PolygonIndicatorSource(
        rotation_files=plate_model.rotation_filenames,
        topology_files=plate_model.topology_filenames,
        polygons=str(CRATON_SHAPEFILE),
        static_polygons=plate_files.static_polygons,
        thickness_data=200.0,
        # Scalar thickness -> seed_fallback_n sizes the seeds; pin both so the
        # regression matches the old single-n_points seeding.
        config=PolygonIndicatorConfig(
            background_n=POLYGON_N_POINTS, seed_fallback_n=POLYGON_N_POINTS
        ),
    )
    return PointCloudSource(producer, plate_model)


# Must match generate_expected_connectors.py — see the note there for why the
# radial spacing is graded rather than uniform (a uniform coarse mesh puts no
# node inside the lithosphere, and the regression stops depending on age).
REGRESSION_LAYER_HEIGHTS = (0.60, 0.25, 0.08, 0.04, 0.02, 0.01)


@pytest.fixture(scope="module")
def regression_mesh():
    """Bigger than `tiny_mesh` to give the indicator field somewhere to
    show structure, but still cheap. Refinement level 2 + 6 graded layers ~
    2700 DoFs at CG1."""
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=2, degree=1)
    mesh = fd.ExtrudedMesh(
        mesh2d,
        layers=len(REGRESSION_LAYER_HEIGHTS),
        layer_height=np.array(REGRESSION_LAYER_HEIGHTS, dtype=float),
        extrusion_type="radial",
    )
    mesh.cartesian = False
    return mesh


@pytest.fixture(scope="module")
def Q(regression_mesh):
    return fd.FunctionSpace(regression_mesh, "CG", 1)


# ---------------------------------------------------------------------------
# Regression: four factory pairings + GplatesScalarFunction wrapper
# ---------------------------------------------------------------------------

def _reduced_quantities(values, mesh):
    """Volume and surface integrals of a scalar field on the regression mesh,
    plus mean / std / min / max of the DoF values."""
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    f.dat.data_with_halos[:] = values
    return {
        "volume": float(fd.assemble(f * fd.dx)),
        "surface": float(fd.assemble(f * fd.ds_t)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _evaluate_connectors_lockstep(connectors_by_name, mesh, Q, ages):
    """Drive all connectors in lockstep through the same age series.

    Connectors sharing a forward-only source (LithosphereSource) must
    advance together: once the tracker passes an age, it can't be revisited
    by any sibling connector. The lockstep walk mirrors a real time loop.
    """
    sfs = {
        name: GplatesScalarFunction(Q, indicator_connector=c, name=name)
        for name, c in connectors_by_name.items()
    }
    out = {name: {} for name in connectors_by_name}
    sample = next(iter(connectors_by_name.values()))
    for age in ages:
        ndtime = sample.source.age2ndtime(float(age))
        for name in connectors_by_name:
            sfs[name].update_plate_reconstruction(ndtime)
            out[name][age] = _reduced_quantities(
                sfs[name].dat.data_ro_with_halos.copy(), mesh
            )
    return out


def _load_reference():
    ref_path = DATA_DIR / "test_connectors.pkl"
    if not ref_path.exists():
        pytest.skip(
            f"Reference fixture missing: {ref_path}. "
            "Generate via tests/unit/data/generate_expected_connectors.py."
        )
    with open(ref_path, "rb") as f:
        return pickle.load(f)


def _check_reduced(observed, expected, label):
    # atol floors the round-off on quantities that are analytically zero
    # (e.g. a geotherm's surface integral) so they aren't held to rtol.
    for key, expected_value in expected.items():
        np.testing.assert_allclose(
            observed[key], expected_value, rtol=1e-3, atol=1e-9,
            err_msg=f"{label}: {key}",
        )


class TestConnectorRegression:
    """Drive the four factory connectors in lockstep (siblings sharing a
    Source advance together) and assert volume/surface integrals match the
    pickled reference."""

    def test_lithosphere_pair(self, lith_source, regression_mesh, Q):
        ref = _load_reference()
        factory = LithosphereConnectorFactory()
        factory.source = lith_source
        # Must match generate_expected_connectors.py. Age sensitivity comes
        # from the radial grading of `regression_mesh`: with the plain
        # one-sided step, nodes have to sit inside the lithosphere for a
        # moving base depth to change the reduced integrals at all.
        factory.construct_output()
        factory.construct_geotherm()
        observed = _evaluate_connectors_lockstep({
            "lith_indicator": factory.indicator,
            "lith_geotherm": factory.geotherm,
        }, regression_mesh, Q, TEST_AGES)
        for name in ("lith_indicator", "lith_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")

    def test_polygon_pair(self, poly_source, regression_mesh, Q):
        ref = _load_reference()
        factory = PolygonConnectorFactory()
        factory.source = poly_source
        factory.construct_output()
        factory.construct_geotherm()
        observed = _evaluate_connectors_lockstep({
            "polygon_indicator": factory.indicator,
            "polygon_geotherm": factory.geotherm,
        }, regression_mesh, Q, TEST_AGES)
        for name in ("polygon_indicator", "polygon_geotherm"):
            for age in TEST_AGES:
                _check_reduced(observed[name][age], ref[name][age],
                               f"{name} age {age}")

    def test_reference_is_age_sensitive(self):
        """The fixture must actually depend on the reconstruction.

        A regression that matches a reference at three ages proves nothing if
        the field is identical at all three. That is not hypothetical: with a
        plain one-sided step on a uniform 4-layer mesh, no node falls inside
        the lithosphere, so the indicator reads 1 at the surface node and 0 at
        the next node down at every age and the reduced integrals are constant
        to the last bit. Measured on that mesh the lithosphere volume was
        6.856154 at every age; the graded REGRESSION_LAYER_HEIGHTS is what
        restores the dependence. This test fails if anyone flattens the mesh
        again, whereas the comparisons above would keep passing.

        Volumes specifically, not "the recorded quantities". Three of the six
        recorded for lith_indicator are age-invariant BY CONSTRUCTION and it
        would be wrong to widen this: with no lateral term the indicator is
        exactly 1 at every surface node at every age, so `surface` is the
        sphere's area (60.107436 in the fixture, bit-identical across ages),
        `max` is 1, and `min` is 0 because some node always sits below every
        base. Only volume, mean and std carry the base-depth signal. That the
        surface integral is constant is also what proves the volumes are clean:
        under CG1 the surface DoFs contribute a fixed amount to the volume
        integral, so the ridge-axis surface skin inflates the absolute number
        and cancels exactly in every age-to-age difference.
        """
        ref = _load_reference()
        for name in ("lith_indicator", "polygon_indicator"):
            volumes = [ref[name][age]["volume"] for age in TEST_AGES]
            assert len(set(volumes)) == len(TEST_AGES), (
                f"{name} has repeated volumes across {TEST_AGES}: {volumes}. "
                "The reference does not depend on the reconstruction, so the "
                "regression cannot detect a change in it."
            )

    def test_geotherm_surface_is_zero(self):
        """The lithosphere geotherm (erf) is zero at the surface by
        construction: erf(0) = 0 at every covered node, and the background grid
        makes every target node reachable, so a nonzero surface integral means
        the source cloud has developed coverage holes.

        The polygon geotherm is deliberately excluded. Since A5 it is a
        `MaskedGeothermLinearOutput`, which reads mantle (T_norm = 1) everywhere
        outside the region — including the exterior surface — so its surface
        integral is the exterior area, not zero. That is the two-facts fix, not
        a coverage hole; the old GeothermLinearOutput read 0 there only because
        it conflated "outside the region" with "surface temperature"."""
        ref = _load_reference()
        for age in TEST_AGES:
            name = "lith_geotherm"
            assert abs(ref[name][age]["surface"]) < 1e-9, (
                f"{name} at {age} Ma has surface integral "
                f"{ref[name][age]['surface']:.3e}, not ~0 — some surface "
                "node was flagged too_far."
            )


class TestSharedSourceConsistency:
    """Two connectors holding the same LithosphereSource should produce the
    same field as a connector built standalone (same source instance, just a
    different consumer).

    Uses a fresh source (class-scoped) so the forward-only tracker hasn't
    been walked past the test age by sibling regression tests.
    """

    @pytest.fixture(scope="class")
    def fresh_lith_source(self, plate_model, plate_files):
        return _make_lith_source(plate_model, plate_files)

    def test_two_indicators_sharing_source_agree(
        self, fresh_lith_source, regression_mesh, Q
    ):
        # Build two independent indicator connectors that share the same
        # source. Same source ⇒ same prepared dict ⇒ same kNN interpolation
        # ⇒ identical DoF values at the same ndtime. A factory only hands
        # out one indicator, so two factories share the assigned source.
        factory_a = LithosphereConnectorFactory()
        factory_a.source = fresh_lith_source
        factory_a.construct_output()
        factory_b = LithosphereConnectorFactory()
        factory_b.source = fresh_lith_source
        factory_b.construct_output()
        ind_a = factory_a.indicator
        ind_b = factory_b.indicator

        sf_a = GplatesScalarFunction(Q, indicator_connector=ind_a, name="ind_a")
        sf_b = GplatesScalarFunction(Q, indicator_connector=ind_b, name="ind_b")

        ndtime = fresh_lith_source.age2ndtime(50.0)
        sf_a.update_plate_reconstruction(ndtime)
        sf_b.update_plate_reconstruction(ndtime)

        # DoF arrays must match to machine precision — the kNN interpolation
        # is deterministic given identical inputs.
        np.testing.assert_array_equal(
            sf_a.dat.data_ro_with_halos,
            sf_b.dat.data_ro_with_halos,
        )

    def test_indicator_and_geotherm_share_one_tracker_step(
        self, fresh_lith_source, regression_mesh, Q
    ):
        # Pair an indicator with a geotherm on the same source. Both produce
        # different scalar fields (quintic step vs. erf geotherm), but they must
        # agree on what the underlying source dict is — verifiable by
        # checking that the source's per-age cache hits on the second call.
        factory = LithosphereConnectorFactory()
        factory.source = fresh_lith_source
        factory.construct_output()
        factory.construct_geotherm()
        ind = factory.indicator
        geo = factory.geotherm

        sf_ind = GplatesScalarFunction(Q, indicator_connector=ind, name="ind")
        sf_geo = GplatesScalarFunction(Q, indicator_connector=geo, name="geo")

        # Use a fresh age that lies forward of the previous test's stop
        # (we share the class-scoped source, which is forward-only).
        ndtime = fresh_lith_source.age2ndtime(40.0)
        sf_ind.update_plate_reconstruction(ndtime)
        d_first = fresh_lith_source._cached_dict
        sf_geo.update_plate_reconstruction(ndtime)
        d_second = fresh_lith_source._cached_dict

        # The geotherm call should have hit the source cache rather than
        # rebuilding the dict.
        assert d_first is d_second
