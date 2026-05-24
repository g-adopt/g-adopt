# P11 profile: kNN interpolation geometry — decision note

**Decision: CLOSE-WITH-NOTE. No numerics-path change. Baselines untouched.**

This is the profiling deliverable for P11. The harness is
`tests/unit/data/profile_interp_geometry.py` (standalone tooling, not imported
by the suite). It times the three phases of
`ScalarFieldConnector._interp_geometry` — BUILD (unit-sphere normalise the
source cloud + `cKDTree(unit_source)`), QUERY (`tree.query`), WEIGHTS
(too_far / exact_match / kernel / row-normalise) — across a source × target ×
kernel grid at k=50, plus a uniform-random vs Firedrake-mesh-structured QUERY
cross-check. A fidelity check at the top asserts the standalone phase math is
byte-identical to the real `_interp_geometry` (idx, too_far, weights), so the
profile faithfully reflects the production code.

## Full table (median of 10 runs, 5 for the 1e6 cells; seconds)

```
   kernel   n_src     n_tgt     BUILD     QUERY   WEIGHTS     TOTAL
-------------------------------------------------------------------
      idw    2000     10000    0.0001    0.0350    0.0006    0.0357
      idw    2000    100000    0.0001    0.3489    0.0059    0.3549
      idw    2000   1000000    0.0002    3.5279    0.0622    3.5903
      idw   10000     10000    0.0014    0.0380    0.0006    0.0399
      idw   10000    100000    0.0015    0.3818    0.0059    0.3893
      idw   10000   1000000    0.0014    3.8102    0.0631    3.8747
      idw   40000     10000    0.0066    0.0393    0.0006    0.0464
      idw   40000    100000    0.0073    0.3936    0.0063    0.4072
      idw   40000   1000000    0.0070    3.9548    0.0629    4.0247
 gaussian    2000     10000    0.0002    0.0351    0.0016    0.0369
 gaussian    2000    100000    0.0002    0.3498    0.0164    0.3664
 gaussian    2000   1000000    0.0002    3.5103    0.1757    3.6862
 gaussian   10000     10000    0.0014    0.0380    0.0016    0.0410
 gaussian   10000    100000    0.0015    0.3820    0.0165    0.4000
 gaussian   10000   1000000    0.0014    3.8313    0.1692    4.0019
 gaussian   40000     10000    0.0066    0.0393    0.0016    0.0475
 gaussian   40000    100000    0.0074    0.3915    0.0166    0.4154
 gaussian   40000   1000000    0.0067    3.9598    0.1781    4.1446
```

(Machine: the worktree host, otherwise idle. Absolute numbers are
machine-dependent; the cross-phase *ratios* are what the decision rests on and
those are stable.)

### Phase breakdown — what the numbers say

- **BUILD is free.** Even at 40000 source points the tree build is 0.007 s — at
  most 0.2 % of the geometry total at any target size, and ~0.0002 % of a
  realistic per-age cost (below). The source cloud is ~10k–40k points; building
  a `cKDTree` over it is nothing.
- **QUERY is the entire geometry cost.** It tracks target size almost linearly:
  ~0.035 s at 1e4, ~0.38 s at 1e5, ~3.5–4.0 s at 1e6. Source size barely moves
  it (2000 → 40000 sources changes 1e6-target QUERY only 3.53 → 3.95 s).
- **WEIGHTS is negligible** — ≤0.06 s (idw) / ≤0.18 s (gaussian) even at 1e6.

## QUERY representativeness anchor (n_src=10000, n_tgt≈1e5, idw, k=50)

```
  uniform-random 1e5 targets:   QUERY median = 0.3819 s  (n=100000)
  mesh-structured ~1e5 targets: QUERY median = 0.1477 s  (n=99918)
  mesh/uniform QUERY ratio:     0.387
```

Real Firedrake extruded-icosahedral-shell DoFs query **2.6× faster** than
uniform-random points of the same count (the spatial coherence of mesh nodes
helps the kd-tree). So the uniform-random grid above is a *conservative
over-estimate* of the QUERY cost a real run pays. The real per-build QUERY at
1e5 DoFs is ~0.15 s, at 1e6 it would be on the order of ~1.5 s.

## Geometry as a fraction of per-age cost

Per-age cost is dominated by gtrack's `step_to` (the ocean-tracker forward
walk), which P10 already shares across siblings via the per-age source cache,
plus the Stokes/energy solve in a real run. From the isolated data-backed
regression (≈294 s for 100 ages × 2 sources), gtrack `step_to` costs on the
order of **~1.5 s per source per age** (~2.9 s/age across the two sources),
and that is *just the tracker* — a production run also pays a full coupled
solve per timestep, typically seconds to minutes at these mesh sizes.

Post-P10 the number of distinct `(source, age)` geometry builds per timestep is
**2** (the demo's two sources; the six scalar functions share them). So per age:

| mesh scale | geometry / age (2 builds, mesh-structured QUERY) | gtrack step_to / age | geometry as % of (step_to only) |
|---|---|---|---|
| ~1e4 DoF (regression mesh) | 2 × ~0.035 s ≈ 0.07 s | ~2.9 s | ~2.4 % |
| ~1e5 DoF | 2 × ~0.15 s ≈ 0.30 s | ~2.9 s | ~9 % |
| ~1e6 DoF | 2 × ~1.5 s ≈ 3.0 s | ~2.9 s | ~50 % of step_to *alone* |

The ~1e6 row looks large against `step_to` alone, but that comparison omits the
coupled Stokes/energy solve, which at 1e6 DoF on a sphere dwarfs both the
tracker and the interpolation. Against the *true* per-age cost (step_to +
solve) the shared geometry is comfortably in the single-digit-percent range at
every realistic scale. At the scale the test suite and demo actually run
(~1e4–1e5 DoF) it is ~2–9 %.

## Decision against the approved thresholds

CLOSE-WITH-NOTE fires if **any** of these hold; INVERT requires **all three** of
its conditions. Walking the numbers:

1. **Shared build+query < ~10 % of per-age cost once step_to (+solve) counted.**
   TRUE at every realistic scale: ~2.4 % at the regression mesh, ~9 % at 1e5
   against step_to alone and far less once the solve is included. This alone
   triggers CLOSE-WITH-NOTE.

2. **Amortised inversion does not net-beat the current 2×/age.** The only thing
   inverting the tree (build once over the static mesh, never clear) saves is
   BUILD — and BUILD is already 0.007 s, i.e. ~0.2 % of geometry and ~0.0002 %
   of per-age. The actual cost is QUERY, which inversion does **not** remove: it
   replaces fixed-k `tree.query` with `tree.query_ball_point(unit_source, r)`
   over the static mesh-tree, which still walks the tree per source point, then
   adds a ragged scatter-accumulate-divide (sparse COO / `np.add.at`) that
   fixed-k IDW does not pay. Trading a ~0.007 s/age saving for a fixed-radius
   scatter over ~1e5–1e6 neighbours is a net loss. Inversion does not net-beat.

3. **Fixed-radius is not a clear correctness win over fixed-k.** Fixed-k IDW
   gives every target node exactly k contributing seeds regardless of local
   source density, which is the current, tested, baseline-pinned behaviour.
   Fixed-radius `query_ball_point` gives a variable neighbour count — empty sets
   where seeds are sparse (needs a fallback), large sets where dense — and
   changes the blend. That is a different, arguably *worse*-behaved
   interpolation near sparse-seed regions, not a positive correctness argument.

All three INVERT pre-conditions fail, and CLOSE-WITH-NOTE condition (1) is
satisfied outright. **P10 already captured the win that mattered** (BUILD was
never the bottleneck; the duplicate-work cost was the repeated build+query
across the six sibling outputs, and P10's shared per-(source, age) geometry
cache removed exactly that — six builds collapse to two). Inverting the tree
would chase BUILD, which is already free, while making the numerics path more
complex, irreversibly fixed-radius, and requiring regenerated baselines for no
measurable gain.

**Outcome: close P11 with this note. No code change to the numerics path, no
baseline regeneration.**
