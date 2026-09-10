#!/bin/bash
# Step 4 of NOTES/HANDOVER.md: the coupled run at nu = 0.495.
#
# WHY THIS FILE CHANGED ON 2026-08-11
# The version before this one could not run. It called `run_b1.pbs`, which did
# not exist. It also used `--configuration coarse`, the aspect-ratio-24 mesh
# that had already killed jobs 175498591 and 175498592, and it passed neither
# the near-incompressible preconditioner nor a low-anisotropy mesh, so it would
# have repeated those failures if the PBS file had existed. Its predictions were
# wrong too, and the whole point of step 1 was to find out by how much.
#
# ---------------------------------------------------------------------------
# WHAT STEP 1 SETTLED, BEFORE ANY SERVICE UNIT IS SPENT
# ---------------------------------------------------------------------------
# NOTES/selfgravity/measurements/love_numbers_nu0495.py, on the propagator that
# reproduces TABOO to 0.46% (h) / 0.57% (l) / 0.08% (k) for n = 2..20:
#
#   nu = 0.495 IS NOT INCOMPRESSIBLE. The residual is R_h = 1.0229 at n = 2,
#   peaking at 1.0274 (n = 4) and falling to 1.0162 (n = 20).
#
# The old header predicted "U_2 ratio -> ~1.0" from 2.187/2.191, which assumed a
# perfectly incompressible run. It is wrong by exactly that 2.3%. Do not read a
# 1.02 result as a failed collapse. 1.00 would be the anomaly.
#
# ---------------------------------------------------------------------------
# PRE-COMMITTED PREDICTIONS. DO NOT ADJUST THESE AFTER SEEING NUMBERS.
# ---------------------------------------------------------------------------
# U, per degree, and ONLY for n <= 12.
#     n :    2      3      4      5      6      7      8     10     12
#   U_n : 1.0229 1.0260 1.0274 1.0272 1.0263 1.0251 1.0240 1.0219 1.0202
#   against the 2.187 .. 1.935 the nu = 0.28 run gave at the same degrees.
#
#   THE n <= 12 SCOPE IS NOT OPTIONAL. At nu = 0.28 the coupled solver's own
#   per-degree agreement with this propagator is 0.02% at n = 2 but 2.70% at
#   n = 20, while the residual to be detected FALLS from 2.3% to 1.6%. Signal
#   over noise is 139 at n = 2, 5.7 at n = 12, and 0.6 at n = 20. Above n = 12
#   this run cannot distinguish 1.016 from 1.000 whatever it prints, so no
#   claim may be made there in either direction.
#
#   Quote the PER-DEGREE U_n ratio, not the spatial U(0) ratio. The spatial one
#   moves ~3% between n_max 20 and 40; the per-degree one is truncation-free and
#   is what the propagator predicts directly. For the record the spatial
#   companion is 1.0222 against 2.0367 at nu = 0.28.
#
# V, the tangential displacement -- THE SHARPER TEST, and new.
#   V is not an open question any more. NOTES/selfgravity/measurements/
#   tangential_compressibility.py shows the recorded failure (max V at 0.057 of
#   reference, peak at 71.5 deg) is compressibility and nothing else: from Love
#   numbers alone, with no mesh in the calculation, nu = 0.28 gives +0.3796 m at
#   71.10 deg against the recorded +0.4060 m at 71.50 deg. The field is really
#   -6.05 m at 8.23 deg, about 85% of reference amplitude with the OPPOSITE
#   SIGN; the driver's signed argmax (b1_elastic.py:1142, :1381) then reports
#   the small positive lobe at 71 deg and it looks like a 20x amplitude loss.
#
#     PREDICTIONS AT nu = 0.495, by the truncation the run actually uses:
#
#       nmax  config      V_ref at ref peak   V(nu=.495)          ratio
#         20  coarse      +7.1575 @ 8.75 deg  +6.8831 @ 8.78 deg  0.9617
#         40  medium      +7.4731 @ 8.59 deg  +7.1736 @ 8.62 deg  0.9599
#         80  fine        +7.5584 @ 8.78 deg  +7.2499 @ 8.78 deg  0.9592
#        128  production  +7.5660 @ 8.93 deg  +7.2565 @ 8.93 deg  0.9591
#
#     and the nu = 0.28 arm for comparison: 0.0530 @ 71.10 deg (nmax 20),
#     0.0448 @ 68.00 (40), 0.0440 @ 71.17 (80), 0.0439 @ 70.36 (128).
#     The ratio is truncation-insensitive; the peak VALUE is not, so quote the
#     ratio.
#
#   That is a change of SIGN and of PEAK LOCATION, not a 2% amplitude shift, so
#   it is far harder to fake than the U result. If V does not come back, the
#   compressibility explanation of V is wrong and the mesh hypothesis returns.
#
# SOLVER. Expect many more outer iterations than the nu = 0.28 arms: the
#   volumetric conditioning is worse by construction. A stall is the
#   conditioning wall, not a wrong answer.
#
# LOCKING, the known risk. P2 displacement-only mechanics locks as nu -> 0.5, so
#   U would come out UNDER-predicted (ratio below the table above, too stiff)
#   and the 1000 arm would lock harder than the 100 arm. The two arms
#   DISAGREEING, with U falling as K/mu rises, is the locking signature. It
#   would say the direct route needs a mixed (u, p) element -- see NOTES/HANDOVER.md section 5.4
#   A1 -- while step 1 has already established the physics regardless.
#
# ---------------------------------------------------------------------------
# WHAT IS DIFFERENT IN THE RUN ITSELF
# ---------------------------------------------------------------------------
# --mesh          the AR-7 low-anisotropy mesh, NOT any rung of the ladder. At
#                 AR 24 the enriched near-nullspace gives 643 iterations against
#                 626 for rigid modes, i.e. it does not help; at AR 7 it does.
#                 See THE MESH below -- every ladder rung is AR ~14, so this
#                 must be generated explicitly.
#                 The cost: holding `litho_layers` at 2 cuts the lithosphere
#                 resolution, so it is the wrong mesh for a high-degree
#                 near-field question. That mattered while V was open. It is no
#                 longer open, so this is a trade rather than a confound -- but
#                 say so in the write-up.
# --u-pc          NearlyIncompressibleAssembledPC, the divergence-free enrichment.
#                 Verified to actually attach inside DtNTwoBlockSchurPC, which
#                 was the open risk: NOTES/selfgravity/measurements/
#                 step3_nearnull_attach.py Q1 reads the near-nullspace back off
#                 the assembled block and finds 5 vectors where a plain
#                 firedrake.AssembledPC gets None.
# --snes-type     ksponly. B1 sets no exponent, so the residual is linear;
#                 ksponly and newtonls agree to 0.000e+00 on the 2-D annulus
#                 (same file, Q2). Newton was spending a second linear solve, a
#                 Jacobian assembly and a full GAMG setup per solve, and a GAMG
#                 setup is 16-57 s here against ~1 s per iteration.
#
# ---------------------------------------------------------------------------
# THE MESH. READ THIS BEFORE GENERATING ANYTHING.
# ---------------------------------------------------------------------------
# `RESOLUTION_LADDER` in generate_selfgrav_sphere.py scales `litho_layers` with
# the configuration (2 / 4 / 8 / 12), deliberately, so that the lithosphere
# aspect ratio is HELD across the ladder rather than improving as the mesh
# refines -- a refinement pair with a changing aspect ratio froze three error
# components and nearly manufactured a false locking positive.
#
# The consequence, which is not written down anywhere else: REFINING NO LONGER
# RELAXES ANISOTROPY. Every rung is design AR ~14:
#
#     config      h_lat   litho_layers   h_rad   design AR   nmax
#     coarse      500 km       2         35 km      14.3       20
#     medium      250 km       4       17.5 km      14.3       40
#     fine        120 km       8       8.75 km      13.7       80
#     production   78 km      12       5.83 km      13.4      128
#
# So `--configuration fine` on its own is still a >600-iteration displacement
# block. A low-anisotropy mesh must be asked for explicitly, by holding
# `litho_layers` at 2-4 while the lateral spacing refines:
#
#     medium + 2 layers  ->  design AR 7.1     (load degrees to n = 40)
#     fine   + 4 layers  ->  design AR 6.9     (load degrees to n = 80)
#
# b2_genmesh.py's own docstring still describes the PRE-FIX generator, where the
# aspect ratio fell down the ladder (24.6 / 11.2 / 9.8). HANDOVER.md 3.3 quotes
# that table. Both are stale; the numbers above come from the live constants.
#
# GENERATE THE MESH FIRST (a login-node job of a few minutes):
#     python3 b2_genmesh.py --tag ar7 --configuration medium \
#         --litho-layers 2 --min-cells 32          # design AR 7.1
# and confirm the file name it prints matches MESH below. It prints the design
# aspect ratio; check it is ~7 and not ~14 before spending a service unit.

set -eu
cd "$(dirname "$0")"

MESH=${MESH:-b2_ar7.msh}
if [ ! -f "$MESH" ]; then
  echo "ERROR: $MESH not found. Generate it with b2_genmesh.py first;" >&2
  echo "       the AR-24 production mesh has already failed twice here." >&2
  exit 1
fi

BASE="--configuration coarse --mesh $MESH --condense --outer-rtol 1e-8"
BASE="$BASE --quad-degree 40 --u-pc gadopt.NearlyIncompressibleAssembledPC"
BASE="$BASE --snes-type ksponly"

qsub -N b1inc100 -o pbs_b1inc100.out -e pbs_b1inc100.err \
  -v LABEL=incompressible-Kmu100,ARGS="$BASE --bulk-shear-ratio 100" run_b1.pbs

# The 1000 arm (nu = 0.4995) is BOTH the locking discriminator and a second,
# sharper measurement of the physics. An earlier version of this comment said its
# 0.27% residual was "well below what this solver can resolve at any degree".
# That was wrong -- measured against the solver's own per-degree agreement with
# the propagator at nu = 0.28:
#
#     n            2      3      4      5      6      8     12
#     residual  0.227% 0.258% 0.272% 0.270% 0.261% 0.238% 0.200%
#     noise     0.016% 0.015% 0.037% 0.054% 0.059% 0.115% 0.354%
#     S/N        13.8   17.2    7.3    5.0    4.4    2.1    0.6
#
# So the 1000 arm resolves the residual for n <= 5, where the 100 arm resolves it
# to n <= 12. Its PREDICTED U_n ratios are 1.0023 (n=2), 1.0027 (n=4),
# 1.0027 (n=5) -- ten times closer to 1 than the 100 arm's.
#
# It is also where locking is most likely (P2 displacement-only at nu = 0.4995)
# and where the solve is dearest: HANDOVER 2.3 measured 333 iterations / 400 s at
# this ratio against 94 / 136 s at nu = 0.495, on the enriched arm. Submit it
# second, and consider holding it until the 100 arm has returned.
qsub -N b1inc1000 -o pbs_b1inc1000.out -e pbs_b1inc1000.err \
  -v LABEL=incompressible-Kmu1000,ARGS="$BASE --bulk-shear-ratio 1000" run_b1.pbs
