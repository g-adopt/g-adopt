#!/bin/bash
# Near-incompressible confirmation. Step 0 (love_numbers.py, validated against
# TABOO to 0.5%) showed the coupled solver's U ratio 2.19->1.70 IS the per-degree
# compressibility factor h_n(nu=0.28)/h_n(incompressible), corr 0.9993. Direct
# corollary: run the coupled solver near-incompressible (large K/mu) and U should
# COLLAPSE onto the benchmark, U_n ratio -> ~1.0.
#
# Two ratios to bracket it and expose locking: 100 (nu=0.495) and 1000 (nu=0.4995).
# Everything else identical to the discriminator baseline (coarse, condensed,
# rtol 1e-8, quad 40, cmb-buoyancy=core [the committed fix], fluid core),
# tangled b1_coarse.msh.
#
# PRE-COMMITTED PREDICTIONS (do not adjust after seeing numbers):
#   NO LOCKING (the clean case): U_2 ratio -> ~1.0 (Step 0 predicts
#     U_2(inc)=U_2(0.28)/R_h(2)=2.187/2.191=0.998), whole U spectrum -> ~1,
#     and 100 vs 1000 agree. This directly confirms U = compressibility.
#   LOCKING (the known risk): P2 displacement-only mechanics locks as nu->0.5,
#     so U comes out UNDER-predicted (ratio < 1, too stiff) and 1000 locks
#     HARDER than 100 (the two DISAGREE, U falling with K/mu). That would say the
#     direct incompressible run needs a mixed/pressure element - while Step 0
#     already proves the physics regardless.
#   SOLVER: expect many more FGMRES iterations than the 9/78 of the nu=0.28 arms
#     (worse volumetric conditioning); watch it converges inside 3 h. If it
#     stalls, that is the conditioning wall, not a wrong answer.

set -u
cd "$(dirname "$0")"
BASE="--configuration coarse --condense --outer-rtol 1e-8 --quad-degree 40"

qsub -N b1inc100 -o pbs_b1inc100.out -e pbs_b1inc100.err \
  -v LABEL=incompressible-Kmu100,ARGS="$BASE --bulk-shear-ratio 100" run_b1.pbs

qsub -N b1inc1000 -o pbs_b1inc1000.out -e pbs_b1inc1000.err \
  -v LABEL=incompressible-Kmu1000,ARGS="$BASE --bulk-shear-ratio 1000" run_b1.pbs
