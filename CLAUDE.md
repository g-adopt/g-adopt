# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Worktree Overview

This worktree (`sghelichkhani/gtrack`) adds **indicator field** capabilities to G-ADOPT by integrating with [gtrack](https://pypi.org/project/gtrack/) for time-dependent 3D scalar fields (lithosphere, cratons).

## Build & Test Commands

```bash
# Run tests (requires plate reconstruction data)
cd demos/mantle_convection/gplates_global && make data  # Download test data first
pytest tests/unit/test_lithosphere_connector.py -v

# Run a single test
pytest tests/unit/test_lithosphere_connector.py::TestLithosphereConfig::test_default_values -v

# Run demo
cd demos/mantle_convection/gplates_lithosphere
make data  # Download plate reconstruction files
PYTHONPATH=/path/to/this/worktree:$PYTHONPATH mpiexec -n 4 python gplates_lithosphere.py
```

## Architecture

### IndicatorConnector Hierarchy

```
IndicatorConnector (ABC in connectors.py)
├── LithosphereConnector  - Oceanic (age-tracked) + continental thickness
├── PolygonConnector      - Polygon boundaries + thickness data
├── LithosphereGeotherm   - Wraps LithosphereConnector (composition, shared tracker)
└── PolygonGeotherm       - Wraps PolygonConnector (composition, shared rotator)
```

Indicator connectors produce smooth 3D indicator fields (~1 in region, ~0 outside) via tanh transitions. Geotherm connectors wrap an existing indicator connector and produce normalized temperature profiles [0, 1] instead.

### Key Classes

| Class | Purpose |
|-------|---------|
| `IndicatorConnector` | Abstract base class defining `get_indicator(target_coords, ndtime)` interface |
| `LithosphereConnector` | Combines gtrack's `SeafloorAgeTracker` (ocean) + `PointRotator` (continental) |
| `PolygonConnector` | Uses gtrack's `PolygonFilter` to filter thickness data to polygon-defined regions |
| `LithosphereGeotherm` | Wraps `LithosphereConnector`, shares its tracker, produces erf geotherm |
| `PolygonGeotherm` | Wraps `PolygonConnector`, shares its rotator, produces linear geotherm |
| `GplatesScalarFunction` | Firedrake `Function` that updates from any `IndicatorConnector` |
| `LithosphereConfig` / `PolygonConfig` | Dataclasses with tunable parameters |

### GplatesScalarFunction Usage

```python
# Indicator connectors
lith_connector = LithosphereConnector(gplates, data, half_space, comm=mesh.comm)
I_lith = GplatesScalarFunction(Q, indicator_connector=lith_connector)

# Geotherm wraps the same connector (shared tracker, no duplication)
T_erf = GplatesScalarFunction(Q, indicator_connector=LithosphereGeotherm(lith_connector))

# Update in time loop. Call order is immaterial: the per-age source cache
# (sources.py) means whichever field updates first pays for the
# reconstruction step and the other reuses the cached result.
I_lith.update_plate_reconstruction(ndtime)
T_erf.update_plate_reconstruction(ndtime)
```

*The class names in this snippet predate the Source/Output refactor and are updated in a later change.*

> Updating these fields is order-independent: a shared source caches its reconstruction per geological age (`sources.py:261-270`, asserted by `test_connectors.py:413-419`), so whichever field updates first does the work and the other reuses the result. Either may be updated first.

### Config + Extra Pattern

Following G-ADOPT's `solver_parameters` pattern:

```python
# Use defaults
connector = LithosphereConnector(gplates, data, func)

# Override specific values
connector = LithosphereConnector(gplates, data, func,
    config_extra={"n_points": 40000, "transition_width": 5.0})

# Full custom config
config = LithosphereConfig(n_points=40000, r_outer=2.5)
connector = LithosphereConnector(gplates, data, func, config=config)
```

### MPI Parallelization

Pass `comm=mesh.comm` to connectors. Rank 0 handles I/O and gtrack computations, broadcasts numpy arrays to other ranks. Each rank interpolates to its local mesh points via KDTree.

## Files Modified in This Worktree

| File | Changes |
|------|---------|
| `gadopt/gplates/connectors.py` | NEW: `IndicatorConnector` abstract base class |
| `gadopt/gplates/gplates.py` | `LithosphereConnector`, `PolygonConnector`, `GplatesScalarFunction`, configs |
| `gadopt/gplates/__init__.py` | Exports for new classes |
| `gadopt/gplates/gplatesfiles.py` | Fixed string/list handling |
| `pyproject.toml` | Added `gtrack` dependency |
| `demos/mantle_convection/gplates_lithosphere/` | Demo with both indicators |
| `tests/unit/test_lithosphere_connector.py` | Comprehensive test suite |

## Key Config Parameters

**LithosphereConfig** (ocean tracking + interpolation):
- `n_points`: Ocean tracker mesh resolution (default: 10000)
- `k_neighbors`: KDTree neighbors for interpolation (default: 50)
- `distance_threshold`: Max angular distance in radians (default: 0.1)
- `transition_width`: Tanh transition width in km (default: 10.0)
- `r_outer`: Mesh outer radius, non-dimensional (default: 2.208)

**PolygonConfig** (polygon filtering + interpolation):
- `n_points`: Sample points for polygon coverage (default: 20000)
- `k_neighbors`: KDTree neighbors (default: 50)
- `distance_threshold`: Max angular distance in radians - **controls horizontal extent** (default: 0.1)
- `transition_width`: Tanh transition width in km (default: 10.0)

**Note**: `distance_threshold` in radians: 0.1 rad ≈ 640 km, 0.02 rad ≈ 127 km. Smaller values give sharper boundaries.

## Coordinate Systems

- **gtrack**: Meters at Earth radius (~6.38e6 m)
- **Firedrake mesh**: Non-dimensional (typically r_inner=1.208, r_outer=2.208)
- **KDTree lookups**: Both normalized to unit sphere for angular distance calculations
