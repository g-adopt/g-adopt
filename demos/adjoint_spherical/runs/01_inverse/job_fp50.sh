#PBS -N 40Myrs
#PBS -P xd2
#PBS -q normalsr
#PBS -l walltime=24:00:00
#PBS -l mem=10000GB
#PBS -l ncpus=2080
#PBS -l jobfs=7200GB
#PBS -l storage=scratch/xd2+gdata/xd2+gdata/fp50
#PBS -l wd
#PBS -m abe
#### Load relevant modules:

module use /g/data/fp50/modules
module use /g/data/fp50/branch_modules
export MY_GADOPT=/scratch/xd2/sg8812/g-adopt/
module load firedrake-64bit/20250219
# module load firedrake-64bit/20241030

# To bypass NCI's compiler wrappers
export OMPI_CC=/apps/intel-tools/intel-compiler-llvm/2024.2.0/bin/icx
module remove-path PATH /opt/nci/bin

# To prepend local paths for g-adopt and g-drift
export PYTHONPATH=/scratch/xd2/sg8812/g-drift/:/scratch/xd2/sg8812/local_gadopt/:${PYTHONPATH}

# Add the main adjoint script path
export PYTHONPATH=/scratch/xd2/sg8812/g-adopt/demos/adjoint_spherical/:${PYTHONPATH}

# Turn off multi-threading
export OMP_NUM_THREADS=1

# Setting Python directories
export XDG_CACHE_HOME=$PBS_JOBFS/xdg
export MPLCONFIGDIR=$PBS_JOBFS/firedrake-prefix

# This is to make sure we only compile on rank 0
# export PYOP2_CACHE_DIR=/scratch/xd2/sg8812/g-adopt/demos/adjoint_spherical/pyop/
# export PYOP2_NODE_LOCAL_COMPILATION=0
export OMPI_MCA_io="ompio"

# Making sure all nodes have matplotlib
mpiexec --map-by ppr:1:node -np $PBS_NNODES  python3 -c "import matplotlib.pyplot as plt"

# Run the main simulation
mpiexec -np $PBS_NCPUS bash -c "export PYTHONPYCACHEPREFIX=$PBS_JOBFS/PYCACHE/rank_\$OMPI_COMM_WORLD_RANK; python3 -c 'from adjoint import *; conduct_inversion()'" > inversion_2.log 2> warning.log
