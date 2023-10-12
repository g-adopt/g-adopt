#!/bin/bash
#PBS -N L8
#PBS -P xd2
#PBS -q normal
#PBS -l walltime=05:00:00
#PBS -l mem=49150GB
#PBS -l ncpus=12288
#PBS -l jobfs=40000GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#PBS -W umask=0022
openmpi_version=4.0.7
petsc_arch=opt

module load python3/3.10.4 # intel-compiler/2021.10.0
module load openmpi/${openmpi_version}

export PETSC_DIR=/tmp/firedrake-prefix
export OMP_NUM_THREADS=1

mpiexec --map-by ppr:1:node -np $PBS_NNODES tar -C $PBS_JOBFS -xf /g/data/xd2/ahg157/firedrake-prefixes/firedrake-prefix-20231012-gamg-openmpi407-opt.tar.gz
mpiexec --map-by ppr:1:node -np $PBS_NNODES ln -s $PBS_JOBFS/firedrake-prefix /tmp/firedrake-prefix

export PYTHONUSERBASE=/tmp/firedrake-prefix
export XDG_CACHE_HOME=$PBS_JOBFS/xdg

export PYTHONPATH="/scratch/xd2/rad552/FIREDRAKE_GIT/g-adopt"
export PETSC_OPTIONS="-log_view"

#### Now run
mpiexec python3 stokes_cubed_sphere_7e3_A3_TS1.py &> output_1.dat
mpiexec python3 stokes_cubed_sphere_7e3_A3.py &> output_2.dat
