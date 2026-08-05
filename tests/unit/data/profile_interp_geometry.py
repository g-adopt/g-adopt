"""Profiling harness for the kNN interpolation geometry (P11).

Standalone tooling — NOT imported by the test suite, NOT wired into pytest.
It times the three phases of ScalarFieldConnector._interp_geometry (BUILD,
QUERY, WEIGHTS) across a grid of source/target sizes and kernels, plus a
uniform-random vs Firedrake-mesh-structured QUERY cross-check, and prints the
geometry cost as a percentage of the per-age cost so the P11 decision can be
made by the numbers.

Run:
    python tests/unit/data/profile_interp_geometry.py

The phase math here is the post-P10 _interp_geometry split, factored into
named functions so each phase can be timed in isolation. The arithmetic is
identical to gadopt/gplates/connectors.py:_interp_geometry; a sanity check at
the bottom asserts the standalone weights match the real method on a small
case so the profile stays faithful.
"""

import statistics
import time

import numpy as np
from scipy.spatial import cKDTree

from gadopt.gplates.connectors import (
    InterpolationConfig,
    ScalarFieldConnector,
)

EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Phase functions (mirror _interp_geometry, split so each phase is timeable)
# ---------------------------------------------------------------------------

def phase_build(source_xyz):
    """Unit-sphere normalise the source cloud and build the cKDTree."""
    r_source = np.linalg.norm(source_xyz, axis=1)
    unit_source = source_xyz / np.maximum(r_source[:, np.newaxis], EPSILON)
    tree = cKDTree(unit_source)
    return tree


def phase_query(tree, target_coords, k):
    """Unit-sphere normalise the targets and query k nearest source neighbours."""
    r_target = np.linalg.norm(target_coords, axis=1)
    unit_target = target_coords / np.maximum(r_target[:, np.newaxis], EPSILON)
    dists, idx = tree.query(unit_target, k=k)
    return dists, idx


def phase_weights(dists, cfg):
    """Compute too_far / exact_match masks and the row-normalised weights."""
    exact_match = dists[:, 0] < EPSILON
    too_far = dists[:, 0] > cfg.distance_threshold
    if cfg.kernel == "gaussian":
        weights = np.exp(-dists**2 / (2 * cfg.gaussian_sigma**2))
    else:
        weights = 1.0 / np.maximum(dists, EPSILON)
    weight_sums = weights.sum(axis=1, keepdims=True)
    weights /= np.maximum(weight_sums, EPSILON)
    return too_far, exact_match, weights


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------

def random_unit_sphere(n, seed, radius=1.0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(int(n), 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return radius * xyz


def mesh_structured_targets(target_dofs):
    """Targets from an actual Firedrake extruded icosahedral shell, so the
    QUERY cost reflects real mesh spatial structure rather than uniform noise.

    Returns the projected SpatialCoordinate DoFs (the same array a real
    GplatesScalarFunction would hand the connector). The refinement/layers are
    chosen so total CG1 DoFs land near ``target_dofs``."""
    import firedrake as fd

    # CG1 DoFs on an extruded icosahedral shell ~= (#surface verts) * (layers+1).
    # refinement_level R surface verts = 10*4**R + 2. Pick R=4 (~2562 verts),
    # layers tuned to hit the requested DoF count.
    R = 4
    surf_verts = 10 * 4 ** R + 2
    layers = max(1, round(target_dofs / surf_verts) - 1)
    mesh2d = fd.IcosahedralSphereMesh(radius=1.208, refinement_level=R, degree=1)
    mesh = fd.ExtrudedMesh(
        mesh2d, layers=layers, layer_height=1.0 / layers, extrusion_type="radial"
    )
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    coords = fd.Function(V).interpolate(fd.SpatialCoordinate(mesh))
    return coords.dat.data_ro.copy()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_phase(fn, *args, n_runs=10):
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples), out


def profile_cell(source_xyz, target_coords, kernel, k, n_runs=10):
    cfg = InterpolationConfig(kernel=kernel, k_neighbors=k)
    # BUILD is timed fresh each run (rebuilds the tree); QUERY/WEIGHTS reuse a
    # tree/dists built once outside the loop so they isolate their own cost.
    b_med, b_min, tree = time_phase(phase_build, source_xyz, n_runs=n_runs)
    q_med, q_min, (dists, idx) = time_phase(
        phase_query, tree, target_coords, k, n_runs=n_runs
    )
    w_med, w_min, _ = time_phase(phase_weights, dists, cfg, n_runs=n_runs)
    return {
        "build_med": b_med, "build_min": b_min,
        "query_med": q_med, "query_min": q_min,
        "weights_med": w_med, "weights_min": w_min,
        "total_med": b_med + q_med + w_med,
    }


# ---------------------------------------------------------------------------
# Fidelity check: the standalone phases must match the real _interp_geometry
# ---------------------------------------------------------------------------

def _fidelity_check():
    class _DummyGplates:
        oldest_age = 100.0
        delta_t = 1.0

        def ndtime2age(self, n):
            return n * 100.0

        def age2ndtime(self, a):
            return a / 100.0

    from gadopt.gplates import Source
    from gadopt.gplates import QuinticOutput
    from mpi4py import MPI

    class _DS(Source):
        provides = frozenset({"xyz", "thickness"})

        def __init__(self):
            self.comm = MPI.COMM_WORLD
            self._is_root = True
            self.gplates_connector = _DummyGplates()

        def _compute_sources(self, age):
            return {}

    ds = _DS()
    cfg = InterpolationConfig(kernel="idw", k_neighbors=50)
    conn = ScalarFieldConnector(ds, QuinticOutput(), interpolation=cfg)
    src = random_unit_sphere(2000, seed=7, radius=6.371e6)
    tgt = random_unit_sphere(3000, seed=8, radius=2.0)

    bundle = conn._interp_geometry(src, tgt)
    tree = phase_build(src)
    dists, idx = phase_query(tree, tgt, 50)
    too_far, exact_match, weights = phase_weights(dists, cfg)
    assert np.array_equal(bundle["idx"], idx), "idx mismatch"
    assert np.array_equal(bundle["too_far"], too_far), "too_far mismatch"
    assert np.array_equal(bundle["weights"], weights), "weights mismatch"
    print("Fidelity check: standalone phases match _interp_geometry (idx, "
          "too_far, weights byte-identical).\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _fidelity_check()

    source_sizes = [2000, 10000, 40000]
    target_sizes = [10000, 100000, 1000000]
    kernels = ["idw", "gaussian"]
    k = 50

    print(f"kNN interpolation-geometry profile (k={k}, median of 10 runs, "
          f"seconds)\n")
    header = (f"{'kernel':>9} {'n_src':>7} {'n_tgt':>9} "
              f"{'BUILD':>9} {'QUERY':>9} {'WEIGHTS':>9} {'TOTAL':>9}")
    print(header)
    print("-" * len(header))

    # Cache the random clouds so the grid reuses them deterministically.
    sources = {n: random_unit_sphere(n, seed=100 + n, radius=6.371e6)
               for n in source_sizes}
    targets = {n: random_unit_sphere(n, seed=200 + n, radius=2.0)
               for n in target_sizes}

    for kernel in kernels:
        for n_src in source_sizes:
            for n_tgt in target_sizes:
                # 1e6 targets x 40k sources is the heaviest cell; fewer runs.
                n_runs = 10 if n_tgt <= 100000 else 5
                r = profile_cell(sources[n_src], targets[n_tgt], kernel, k,
                                 n_runs=n_runs)
                print(f"{kernel:>9} {n_src:>7} {n_tgt:>9} "
                      f"{r['build_med']:>9.4f} {r['query_med']:>9.4f} "
                      f"{r['weights_med']:>9.4f} {r['total_med']:>9.4f}",
                      flush=True)

    # --- QUERY representativeness anchor: uniform vs mesh-structured at ~1e5 ---
    print("\nQUERY representativeness anchor (n_src=10000, n_tgt~=1e5, "
          "idw, k=50):")
    src = sources[10000]
    uni_tgt = random_unit_sphere(100000, seed=999, radius=2.0)
    tree = phase_build(src)
    q_uni_med, q_uni_min, _ = time_phase(phase_query, tree, uni_tgt, k)
    print(f"  uniform-random 1e5 targets:   QUERY median={q_uni_med:.4f}s "
          f"(n={len(uni_tgt)})")
    try:
        mesh_tgt = mesh_structured_targets(100000)
        tree_m = phase_build(src)
        q_mesh_med, q_mesh_min, _ = time_phase(phase_query, tree_m, mesh_tgt, k)
        print(f"  mesh-structured ~1e5 targets: QUERY median={q_mesh_med:.4f}s "
              f"(n={len(mesh_tgt)})")
        ratio = q_mesh_med / q_uni_med if q_uni_med else float("nan")
        print(f"  mesh/uniform QUERY ratio: {ratio:.3f}")
    except Exception as exc:  # pragma: no cover - Firedrake optional here
        print(f"  mesh-structured anchor skipped: {exc}")


if __name__ == "__main__":
    main()
