#PBS -N visualisation
#PBS -P xd2
#PBS -q normalsr
#PBS -l walltime=02:00:00
#PBS -l mem=500GB
#PBS -l ncpus=104
#PBS -l jobfs=400GB
#PBS -l storage=scratch/xd2+gdata/xd2+gdata/fp50
#PBS -l wd
#PBS -m abe
#### Load relevant modules:

module use /g/data/fp50/modules
module use /g/data/fp50/branch_modules
export MY_GADOPT=/scratch/xd2/sg8812/g-adopt/
module load firedrake-64bit/gadi_fix_mem_leak
# module load firedrake-64bit/20241030

# To bypass NCI's compiler wrappers
export OMPI_CC=/apps/intel-tools/intel-compiler-llvm/2024.2.0/bin/icx
module remove-path PATH /opt/nci/bin

# To prepend local paths for g-adopt and g-drift
export PYTHONPATH=/scratch/xd2/sg8812/g-drift/:/scratch/xd2/sg8812/local_gadopt/:${PYTHONPATH}

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

mpiexec -np $PBS_NCPUS python3 -c "from inverse import *; plot_gradients()" &> taylor_test_d_$(date +"%Y%m%d%H%M").log
# mpiexec -np $PBS_NCPUS python3 -c "from forward import *; run_forward()" &> generate_${PBS_JOBID}.log
# mpiexec -np $PBS_NCPUS python3 -c "from inverse import *; my_taylor_test(0, 0, 1., 0); my_taylor_test(0, 0, 0, 1.);" &> taylor_test_d_$(date +"%Y%m%d%H%M").log
# mpiexec -np $PBS_NCPUS python3 -c "from inverse import *; my_taylor_test(-1, -1, -1, +1)" &> taylor_test_d_$(date +"%Y%m%d%H%M").log
# mpiexec -np $PBS_NCPUS python3 -c "from adjoint import *; conduct_taylor_test()" &> taylor_test_$(date +"%Y%m%d%H%M").log
# mpiexec -np $PBS_NCPUS python3 -c "from adjoint import *; conduct_inversion()" &> inversion_$(date +"%Y%m%d%H%M").log
