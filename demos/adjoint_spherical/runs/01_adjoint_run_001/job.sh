#PBS -N ADJ001
#PBS -P xd2
#PBS -q normalsr
#PBS -l walltime=24:00:00
#PBS -l mem=6500GB
#PBS -l ncpus=1352
#PBS -l jobfs=5200GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd

#### Load relevant modules:
openmpi_version=4.0.7
petsc_arch=opt

module load python3/3.10.4 # intel-compiler/2021.10.0
module load openmpi/${openmpi_version}

export PETSC_DIR=/tmp/firedrake-prefix
export OMP_NUM_THREADS=1
export PETSC_OPTIONS="-log_view"

mpiexec --map-by ppr:1:node -np $PBS_NNODES tar -C $PBS_JOBFS -xf /g/data/xd2/sg8812/FD-IMAGES/firedrake-prefix-20231121-full-openmpi407-opt.tar.gz
mpiexec --map-by ppr:1:node -np $PBS_NNODES ln -s $PBS_JOBFS/firedrake-prefix /tmp/firedrake-prefix

export LD_PRELOAD=${INTEL_MKL_ROOT}/lib/intel64/libmkl_sequential.so:${INTEL_MKL_ROOT}/lib/intel64/libmkl_core.so:${LD_PRELOAD}
export PYTHONUSERBASE=/tmp/firedrake-prefix
export XDG_CACHE_HOME=$PBS_JOBFS/xdg
export MPLCONFIGDIR=$PBS_JOBFS/firedrake-prefix

export PYTHONPATH="/scratch/xd2/sg8812/g-adopt/demos/adjoint_spherical":${PYTHONPATH}

# Making sure all nodes have matplotlib
mpiexec --map-by ppr:1:node -np $PBS_NNODES  python3 -c "import matplotlib.pyplot as plt"

mpiexec -np 1344 python3 adjoint.py &> output.dat
