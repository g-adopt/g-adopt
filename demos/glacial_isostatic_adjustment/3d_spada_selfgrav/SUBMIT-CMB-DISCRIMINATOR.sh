#!/bin/bash
# The CMB buoyancy-spring discriminator. Does correcting the CMB spring from the
# density contrast (rho_core - rho_0) to rho_core alone fix the vertical- and
# tangential-displacement mismatch?
#
# WHY. The prestress volume term already supplies the mantle's -0.5 rho_0 g u_r^2
# half at the CMB (measured to 2.9e-15 relative, NOTES/measurements/cmb_prestress_check.py;
# control: b1 adds no explicit spring at Re yet its free surface is correct). So
# the old contrast spring double-counts the mantle half: net CMB restoring 13.8%
# of physical, 7.3x too soft. Fixed in gia_gravity.py, default buoyancy_density
# = "core"; "contrast" restores the old number.
#
# All three arms hold EVERYTHING fixed (coarse, condensed, rtol 1e-8, quad 40,
# tangled b1_coarse.msh md5 93f4a9ecb93789a73a3e0b52fa8a4132 - the file the
# recorded U_2 = 2.66 was measured on) and vary ONLY the CMB treatment.
# 64 ranks, ~11 min each. Independent measurements: submit all three.
#
# PRE-COMMITTED PREDICTIONS (written before any number is seen; do not adjust):
#
#   contrast  (control) : U_2 ~= 2.66, monotone to ~1.7-1.8 at n=17, N_2 ~= 0.76.
#             This REPRODUCES the recorded baseline. If it does not, the sync or
#             setup is wrong -> STOP, nothing else is interpretable.
#
#   core      (the fix) : U_2 DROPS substantially from 2.66 (a 7.3x-stiffer CMB
#             suppresses the long-wavelength excess); high-n barely moves; the
#             tangential peak moves from 71.5 deg toward ~9 deg; |V_1| falls;
#             N barely moves (it already matched). If U_2 does NOT fall, the CMB
#             spring is not the cause and the geometry/compressibility axes reopen.
#
#   rigid     (isolate) : un=0 removes the fluid-core soft translation mode
#             entirely. If the low-degree U excess AND the tangential-peak anomaly
#             both collapse here, the fluid-core CMB treatment is convicted
#             independently of the exact coefficient. Expect a CMB-confined
#             (n<=4) change; nothing above, by the (Rc/Re)^n leverage.
#
# Read contrast vs core as the coefficient flip; core vs rigid (or contrast vs
# rigid) as the fluid-core-vs-pinned isolation.

set -u
cd "$(dirname "$0")"

BASE="--configuration coarse --condense --outer-rtol 1e-8 --quad-degree 40"

qsub -N b1cmbcon -o pbs_b1cmbcon.out -e pbs_b1cmbcon.err \
  -v LABEL=cmb-contrast-baseline,ARGS="$BASE --cmb-buoyancy contrast" run_b1.pbs

qsub -N b1cmbcore -o pbs_b1cmbcore.out -e pbs_b1cmbcore.err \
  -v LABEL=cmb-core-fix,ARGS="$BASE --cmb-buoyancy core" run_b1.pbs

qsub -N b1cmbrig -o pbs_b1cmbrig.out -e pbs_b1cmbrig.err \
  -v LABEL=cmb-rigid-core,ARGS="$BASE --rigid-core" run_b1.pbs
