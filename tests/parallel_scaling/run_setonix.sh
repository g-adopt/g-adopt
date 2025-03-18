#!/bin/bash -i
#SBATCH --exclude=nid00[2024-2055],nid00[2792-2823]

level=$1
shift

export MY_GADOPT="$GADOPT_CHECKOUT"
source "$GADOPT_SETUP"

srun $@ -n 1 2> level_${level}_warmup.err > level_${level}_warmup.out
export PETSC_OPTIONS="-log_view :profile_${level}.txt"
srun $@ 2> level_${level}_full.err > level_${level}_full.out
