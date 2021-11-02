#!/bin/bash
#PBS -N L8_WU
#PBS -P xd2
#PBS -q normal
#PBS -l walltime=05:00:00 
#PBS -l mem=48640GB
#PBS -l ncpus=12288
#PBS -l jobfs=10GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#### Load relevant modules:
module use /g/data/xd2/modulefiles
module load firedrake/firedrake-20210820
export OMP_NUM_THREADS=1
export PETSC_OPTIONS="-log_view"
#### Now run:
mpirun -mca coll ^hcoll -np $PBS_NCPUS python stokes_cubed_sphere_7e3_A3_TS1.py &> output_1.dat &&
mpirun -mca coll ^hcoll -np $PBS_NCPUS python stokes_cubed_sphere_7e3_A3.py &> output_2.dat
