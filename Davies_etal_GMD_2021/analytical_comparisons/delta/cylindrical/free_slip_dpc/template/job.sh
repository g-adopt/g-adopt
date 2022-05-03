#!/bin/bash
TEMPLATENAME
#PBS -P xd2
TEMPLATEQUEUE
#PBS -l walltime=01:00:00 
TEMPLATEMEM
TEMPLATECPUS
#PBS -l jobfs=10GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#### Load relevant modules:
module use /g/data/xd2/modulefiles
module load firedrake/firedrake-20220301
export OMP_NUM_THREADS=1
#### Now run:
mpirun -np $PBS_NCPUS python stokes_bilinear.py NTEMPLATE TEMPLATELEVELS &> output.dat
