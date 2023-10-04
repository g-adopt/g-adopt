#!/bin/bash -i

module load python3/3.10.4 openmpi/4.0.7

tar -C $PBS_JOBFS -xf /g/data/xd2/ahg157/firedrake-prefixes/firedrake-prefix-current.tar.gz

cat > $PBS_JOBFS/firedrake-prefix/lib/python3.10/site-packages/petsc4py/lib/petsc.cfg <<EOF
PETSC_DIR = $PBS_JOBFS/firedrake-prefix
PETSC_ARCH =
EOF

export PETSC_DIR=$PBS_JOBFS/firedrake-prefix
export OMP_NUM_THREADS=1
export PYTHONUSERBASE=$PBS_JOBFS/firedrake-prefix
export XDG_CACHE_HOME=$PBS_JOBFS/xdg

python3 -m pip install --user --no-build-isolation $GADOPT_CHECKOUT

mpiexec $@
