#!/bin/bash -i

export MY_GADOPT="$GADOPT_CHECKOUT"
source "$GADOPT_SETUP"

export PYOP2_SPMD_STRICT=1

mpiexec $@
