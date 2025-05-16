#!/bin/bash -i
#SBATCH --exclude=nid00[2024-2055],nid00[2792-2823]

export MY_GADOPT="$GADOPT_CHECKOUT"
source "$GADOPT_SETUP"

srun $@