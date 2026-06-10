# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Worktree Overview

This worktree (`sghelichkhani/gtrack`) adds **indicator field** capabilities to G-ADOPT by integrating with [gtrack](https://pypi.org/project/gtrack/) for time-dependent 3D scalar fields (lithosphere, cratons). It is the integration ("mother") branch for the stacked sub-PRs `gtrack-02-indicator-connector-abc` (#516) → `gtrack-03-lithosphere-connector` (#517) → `gtrack-04-polygon-connector` (#518) → `gtrack-05-gplates-fields-demo` (#519); `gtrack-01-infra-fixes` (#515) already merged to main. Content flows by merge-up, and this branch carries the union of the stack.

## Build & Test Commands

```bash
# Run tests (requires plate reconstruction data)
cd demos/mantle_convection/gplates_global && make data  # Download test data first
pytest tests/unit/test_outputs.py -v        # pure math, no data needed
pytest tests/unit/test_connectors.py -v     # includes pkl regression (needs data)
pytest tests/unit/test_sources.py tests/unit/test_gplates.py -v

# Run a single test
pytest "tests/unit/test_outputs.py::TestQuinticOutputVariableBase" -v

# Run demo
cd demos/mantle_convection/gplates_fields
make data  # Download plate reconstruction files
PYTHONPATH=/path/to/this/worktree:$PYTHONPATH mpiexec -n 4 python gplates_fields.py
```

Use the Firedrake env: `~/Workplace/firedrake-2026-03-03/venv-firedrake/bin/python3`.

## Architecture

### Source / Output / Connector split

```
Source (ABC, sources.py)               OutputStrategy (ABC, outputs.py)
├── LithosphereSource                  ├── QuinticOutput          (indicator)
│     SeafloorAgeTracker (ocean)       ├── GeothermERFOutput      (oceanic geotherm)
│     + PointRotator (continental)     ├── GeothermLinearOutput   (continental geotherm)
└── PolygonSource                      └── LateralFractionOutput  (pure lateral fraction)
      PolygonFilter + PointRotator

ScalarFieldConnector(source, output)   — kNN Gaussian interpolation onto mesh nodes
GplatesScalarFunction(Q, indicator_connector=...) — Firedrake Function wrapper
ConnectorFactory / LithosphereConnectorFactory / PolygonConnectorFactory (factories.py)
```

Sources answer "where do source points live and what properties do they carry at age X?"; Outputs answer "given interpolated arrays at target nodes, what scalar field do we want?". Two consumers (indicator + geotherm) share one Source; a per-age cache means whichever updates first pays for the reconstruction step.

Indicator fields use a **one-sided quintic smoothstep** at the region base (`QuinticOutput`): exactly 1 from the surface down to the base, decaying to exactly 0 over `width_km` below it (C² junctions; 0.5-crossing at base + width/2 depth). Because zero thickness puts the base at the surface, where the surface node would read 1, zero-outside polygon sources MUST pair the step with a lateral fade (`fade_ref_km`); `PolygonConnectorFactory.construct_output` makes it a required argument, and `ConnectorFactory.indicator` cross-checks `Source.zero_outside` against the output for the generic setter route.

### Factory usage

```python
factory = PolygonConnectorFactory(mesh=mesh_cfg, interpolation=interp_cfg)
factory.construct_source(
    gplates_connector=plate_model, polygons="cratons.shp",
    thickness_data=continental_data, plate_files=plate_files, comm=mesh.comm,
)
factory.construct_output(fade_ref_km=150.0, width_km=10.0)  # fade REQUIRED for polygons
I_craton = GplatesScalarFunction(Q, indicator_connector=factory.indicator, name="I_craton")

# Shared-source pairs can also be wired directly:
lith_source = LithosphereSource(...)
I_lith = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    lith_source, QuinticOutput(width_km=10.0, default_thickness_km=100.0)))
T_erf = GplatesScalarFunction(Q, indicator_connector=ScalarFieldConnector(
    lith_source, GeothermERFOutput()))

# Update in time loop — order-independent thanks to the per-age source cache.
I_lith.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
```

### Config pattern: direct kwargs vs. wrapped-dict extra

Most configs are leaf dataclasses over our own parameters and take direct
kwargs with no extra hook: `InterpolationConfig` (`connectors.py`),
`MeshConfig` (`outputs.py`), `PolygonSourceConfig` (`sources.py`).

The wrapped-dict "extra" pattern is reserved for a config that fronts a
third-party config object whose surface we don't want to re-declare. The
sole case is `LithosphereSourceConfig.gtrack_config`, spliced into
gtrack's `TracerConfig` at construction. Keys in `gtrack_config` pass
straight through to `gtrack.config.TracerConfig`.

Rule: add an extra hook only when a config fronts a third-party config
object you don't want to re-declare; configs over our own parameters stay
flat.

```python
# Leaf config — direct kwargs
interp = InterpolationConfig(k_neighbors=80, kernel="gaussian")

# Wrapped-dict extra — forward arbitrary TracerConfig fields to gtrack
lith_cfg = LithosphereSourceConfig(
    n_points=40000,
    gtrack_config={"ridge_sampling_degrees": 0.5},  # passed to gtrack TracerConfig
)
```

### MPI Parallelization

Pass `comm=mesh.comm` to sources. Rank 0 handles I/O and gtrack computations, broadcasts numpy arrays to other ranks. Each rank interpolates to its local mesh points via KDTree.

## Key Config Parameters

**LithosphereSourceConfig** (`sources.py`): `n_points` (ocean tracker resolution, default 10000), `time_step`, `reinit_interval_myr`, `checkpoint_interval_myr`/`checkpoint_dir`, `gtrack_config` pass-through.

**PolygonSourceConfig** (`sources.py`): `n_points` (sphere-mesh sample resolution, default 20000).

**InterpolationConfig** (`connectors.py`): `k_neighbors` (default 50), `distance_threshold` (radians; 0.1 ≈ 640 km — controls horizontal reach of the kNN), `kernel`, `gaussian_sigma` (sets the lateral roll-off across polygon boundaries).

**MeshConfig** (`outputs.py`): `r_outer` (default 2.208), `depth_scale` (default 2890 km).

**QuinticOutput** (`outputs.py`): `width_km` (radial transition, default 10), `base_depth_km` (None = per-node from thickness; number = fixed base), `fade_ref_km` (None = no fade; required in practice for polygon sources), `default_thickness_km` (fill at `too_far` nodes: 100 lithosphere-style, 0 polygon-style).

## Coordinate Systems

- **gtrack**: Meters at Earth radius (~6.38e6 m)
- **Firedrake mesh**: Non-dimensional (typically r_inner=1.208, r_outer=2.208)
- **KDTree lookups**: Both normalized to unit sphere for angular distance calculations
