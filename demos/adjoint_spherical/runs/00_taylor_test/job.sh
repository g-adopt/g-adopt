#PBS -N test
#PBS -P xd2
#PBS -q normalsr
#PBS -l walltime=24:00:00
#PBS -l mem=6500GB
#PBS -l ncpus=1352
#PBS -l jobfs=5200GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#### Load relevant modules:

module load python3/3.10.4 openmpi/4.0.7

export PETSC_DIR="/tmp/fd_prefix"
export PETSC_ARC=""
export OMP_NUM_THREADS=1
export LD_PRELOAD=${INTEL_MKL_ROOT}/lib/intel64/libmkl_sequential.so:${INTEL_MKL_ROOT}/lib/intel64/libmkl_core.so:${LD_PRELOAD}
export LD_LIBRARY_PATH="/half-root/usr/lib/gcc/x86_64-redhat-linux/8/":${LD_LIBRARY_PATH}
mpiexec --map-by ppr:1:node -np $PBS_NNODES tar -C $PBS_JOBFS -xf /g/data/xd2/sg8812/FD-IMAGES/fd_prefix_2024_01_30.tar.gz
mpiexec --map-by ppr:1:node -np $PBS_NNODES ln -s $PBS_JOBFS/fd_prefix /tmp/fd_prefix

export PYTHONUSERBASE=/tmp/fd_prefix
export XDG_CACHE_HOME=$PBS_JOBFS/xdg
export MPLCONFIGDIR=$PBS_JOBFS/fd_prefix

export PYTHONPATH="/scratch/xd2/sg8812/g-adopt/demos/adjoint_spherical":${PYTHONPATH}

# Making sure all nodes have matplotlib
mpiexec --map-by ppr:1:node -np $PBS_NNODES  python3 -c "import matplotlib.pyplot as plt"

mpiexec -np 1344 python3 adjoint.py &> output.dat