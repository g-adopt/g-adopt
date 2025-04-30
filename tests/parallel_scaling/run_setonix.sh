#!/bin/bash -i
#SBATCH --exclude=nid00[2024-2055],nid00[2792-2823]

### Need to grab 'level' from scaling.py input args to name our
### output files appropriately
level="$4"

export MY_GADOPT="$GADOPT_CHECKOUT"
source "$GADOPT_SETUP"

#if [[ "${level}" -gt 5 ]]; then
#    export BINDING="-m block:block:block"
#else
#    export BINDING=""
#fi

srun ${BINDING} $@ -n 1 2> level_${level}_warmup.err > level_${level}_warmup.out
export PETSC_OPTIONS="-log_view :profile_${level}.txt"
srun ${BINDING} $@ 2> level_${level}_full.err > level_${level}_full.out
