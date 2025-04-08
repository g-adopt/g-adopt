#!/bin/bash -i

### Need to grab 'level' from scaling.py input args to name our
### output files appropriately
level="$4"

export MY_GADOPT="$GADOPT_CHECKOUT"
source "$GADOPT_SETUP"

mpiexec $@ -n 1 2> level_${level}_warmup.err > level_${level}_warmup.out
export PETSC_OPTIONS="-log_view :profile_${level}.txt"
mpiexec $@ 2> level_${level}_full.err > level_${level}_full.out
