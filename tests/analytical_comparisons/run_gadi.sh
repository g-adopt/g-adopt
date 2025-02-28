#!/bin/bash -i

export MY_GADOPT="$GADOPT_CHECKOUT"
source "$GADOPT_SETUP"

mpiexec $@
