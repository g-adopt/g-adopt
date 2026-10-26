# Interpretation of the completed Spada study

The 250 km baseline resolves the large-scale self-gravitating response and
gives a polar-motion signal that changes little under cap mesh refinement.
It does **not** establish uniform 1% accuracy for direct spatial fields or
rates. Mesh resolution is the largest measured spatial sensitivity; the
sharp disc is appreciably less resolved than the cap. The 125 km cap checks
were useful: they substantially improve direct-profile agreement with TABOO,
but expose a geoid-rate/frame issue that should be investigated separately.

## Dataset and comparison rules

There are 28 completed production runs with 196 saved profiles: 26 planned
runs and two extra 125 km cap runs, each with seven epochs from 0 to 20 kyr.
All pass the transferred-output checks, including outer solver convergence.
The remaining planned run, `disc-on/dtn2`, failed with an MPI buffer error;
it contributes no completed sensitivity result. Four smoke runs are excluded.
Surface HDF5 and full checkpoints were not transferred or audited.

All completed runs report revision `9b759dc2` and `firedrake/2026.10.0`.
The 12 completed variants that change neither mesh size nor layer count
have the same saved mesh hash as their respective baselines. All four
one-layer variants use mesh seed 10 rather than baseline seed 9: their
differences include remeshing as well as lithosphere subdivision.

For the table below, the difference is the largest scalar geoid or Euclidean
displacement-vector difference on saved profiles within 25° of the load
centre, divided by the baseline peak **at that epoch**, then maximized over
saved epochs. These are line-profile metrics, not surface-area integrals or
relative errors at every point. The tests vary one setting at a time; no
combined cheaper configuration has been tested.

## Which settings matter?

| Change from baseline | Displacement change | Geoid change | Implication |
|---|---:|---:|---|
| Mesh 250 → 500 km | 7.72–27.44% | 13.46–36.69% | Too coarse for accurate local fields; limited cost saving |
| Mesh 250 → 125 km, cap only | 2.31–2.40% | 4.44–5.51% | Baseline mesh error is still visible |
| Lithosphere two layers → one | 1.20–11.18% | 2.12–5.96% | Keep two layers; this also changes the bulk mesh |
| K/μ 1000 → 100 | 1.97–3.11% | 0.42–0.85% | Retain 1000 for the near-incompressible benchmark |
| Double timestep | 0.55–0.61% | 0.68–1.28% | Modest value sensitivity; rates are more sensitive |
| DtN degree 5 → 2, three completed cases | 0.021–0.030% | 0.065–0.109% | Small value sensitivity in the completed tests |
| Tolerances 1e-6/1e-4 → 1e-5/1e-3, cap-on only | 0.090% | 0.162% | Promising cost saving for values; not validated for every scenario |

K/μ changes material compressibility; it is an approximation/physical-model
choice rather than a solver-convergence parameter. A two-layer versus
one-layer comparison does not prove that two layers are converged.

Doubling the timestep changes displacement rates by 4.68–8.11% and geoid
rates by 8.34–12.16%. These differences include the changed backward-difference
averaging interval. Analytic rate comparisons use matching intervals, rather
than instantaneous reference rates. Loose cap-on tolerances change rates by
0.48% and 2.96%, respectively. Retain the original timestep for rate figures;
a smaller-timestep test would be needed to establish their accuracy.

## Was the baseline accurate enough?

The degree-2–10 summary diagnostics agree fairly well with TABOO. Across
epochs, cap-off U(0), N(0) and Vmax errors are 0.53%, 1.08% and 0.67%; disc-off
errors are 1.15%, 1.91% and 0.79%. These describe the large-scale response,
and the projected summaries omit degree 0/1 contamination and higher-degree
load-edge structure.

Direct nonrotating profiles give a different assessment:

| Run | Maximum displacement error | Maximum geoid error | Displacement RMS error | Geoid RMS error |
|---|---:|---:|---:|---:|
| cap-off, 250 km | 2.34% | 6.01% | 1.25% | 2.09% |
| cap-off, 125 km | 0.50% | 1.70% | 0.47% | 1.10% |
| disc-off, 250 km | 14.74% | 35.13% | 3.98% | 8.83% |

Each column takes its own maximum across epochs. Maximum errors divide by
the reference peak; RMS errors divide by its profile RMS. The TABOO curves
use degrees 2–256, whereas the solver uses the spatial load. Reference
ringing near the sharp 10° disc edge contributes to the discrepancy, but
excluding points within 1° of that edge still leaves maxima of 7.95% for
displacement and 10.63% for geoid. The issue is not confined to the immediate
discontinuity. No 125 km disc result is available to test its improvement.

For a benchmark paper, the baseline supports a qualified large-scale
self-gravity comparison. Use the 125 km cap profiles to show the material
improvement with refinement. Do not describe the baseline disc profiles as
converged local fields, or infer a universal convergence order from the three
cap meshes: the mesh-difference patterns are not consistently aligned.

## Polar motion

The raw baseline polar-motion amplitude is about 2.4–2.6% below TABOO for the
cap and 3.1–3.3% below for the disc. The model's C−A is 2.6952e35 kg m²,
versus the paper's prescribed 2.63e35 kg m². This is consistent with much of
the cap amplitude offset, but a controlled convention check is needed to
attribute the residual. No amplitude correction was applied in this analysis.

For the cap, 125 km refinement changes the polar vector by at most 0.130%
and its amplitude by only 0.0168%. The phase error falls from 0.0678° to
0.0061°, while the raw TABOO vector error remains about 2.61%. Thus further
mesh refinement alone is unlikely to remove the amplitude bias. Numerically
stable polar motion does not imply that all associated spatial fields are
equally accurate. There is no independent rotating spatial reference in the
transferred archive, and signed LOD reference agreement remains unresolved.

## Investigate the geoid-rate dipole before claiming rate convergence

At 10 kyr, cap-off 125 km differs from the baseline in geoid rate by
0.156 mm/yr, or 21.0% of the baseline peak. A constant-plus-dipole fit explains
99.9985% of the squared difference on the saved profiles. This is a pattern
fit, not an area-weighted spherical-harmonic projection. Nothing was removed
from the reported errors. At 20 kyr, the rates agree much more closely.

The Spada driver does not enable the library's centre-of-mass constraints.
The library documents a coupled rigid-translation kernel/near-kernel for
the fluid-core problem. Frame sensitivity is therefore a plausible cause,
but these outputs do not prove it. The degree-2–10 summaries would hide a
degree-one error. A controlled frame-constrained comparison is a higher
priority than spending more on general refinement before understanding this.

## Follow-up priorities and cost

1. Investigate the frame issue with otherwise matched cap-off runs, checking
   both direct geoid and displacement rates. Confirm the effect rather than
   subtracting a fitted dipole to improve benchmark scores.
2. Resolve the C−A convention explicitly for polar motion; keep raw reference
   errors visible. Resolve the separate signed LOD convention before claiming
   LOD validation.
3. Refine the disc if local disc accuracy is part of the paper's claim.
   The existing results cannot certify it. Coordinate this with any frame
   fix so new runs remain interpretable.
4. Use a smaller timestep if deformation/geoid rates are a main result.
   Complete `disc-on/dtn2` only if a full DtN sensitivity matrix is needed.

The completed runs used 214,238.28 SU; the failed DtN attempt adds 334.53 SU,
giving 214,572.81 SU excluding smoke. The two 125 km runs cost 44,599.53 SU
combined, about 2.4–2.7 times their individual baseline costs. Loose cap-on
tolerances save 20.75%; reducing DtN degree saves only about 3–5%. The 500 km
mesh saves little despite large spatial errors, because solver performance
also changes. These are observed costs, not predictions for combinations.

## Reproduction and figures

From `tests/gia_selfgrav_benchmarks`, with NumPy and Matplotlib installed:

```bash
python3 analyse_spada_study.py outputs/spada-study
```

`analysis.md` contains generated tables and limitations. `profiles.csv`,
`scalars.csv`, `spectra.csv`, `mesh_convergence.csv`, `low_mode_diagnostics.csv`,
`runs.csv` and `coverage.csv` retain the metrics. `provenance.json` records
input/helper hashes and software versions. This interpretation is a separate
review of the transferred dataset and is not overwritten by the command.

Start with `parameter_sensitivity.pdf`, `cap_mesh_profiles.pdf`,
`disc_mesh_profiles.pdf`, `cap-on_polar_motion.pdf` and
`cap_off_geoid_rates.pdf`. PNG versions are also available.
