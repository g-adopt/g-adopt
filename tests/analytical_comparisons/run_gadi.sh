#!/bin/bash -i

module load python3/3.10.4 openmpi/4.0.7

export PETSC_DIR=$PBS_JOBFS/firedrake-prefix
export OMP_NUM_THREADS=1
export PYTHONUSERBASE=$PBS_JOBFS/firedrake-prefix
export XDG_CACHE_HOME=$PBS_JOBFS/xdg

mpiexec --map-by ppr:1:node --np $PBS_NNODES bash -c "tar -C $PBS_JOBFS -xf $GADOPT_PREFIX_IMAGE && \
cat > $PBS_JOBFS/firedrake-prefix/lib/python3.10/site-packages/petsc4py/lib/petsc.cfg <<EOF && \
cp -r $GADOPT_CHECKOUT $PBS_JOBFS && \
python3 -m pip install --user --no-build-isolation $PBS_JOBFS/g-adopt
PETSC_DIR = $PBS_JOBFS/firedrake-prefix
PETSC_ARCH =
EOF"

export LD_LIBRARY_PATH=$PBS_JOBFS/firedrake-prefix/lib:$LD_LIBRARY_PATH

mpiexec $@
