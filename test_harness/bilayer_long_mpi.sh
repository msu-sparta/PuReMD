#!/bin/bash -login

#PBS -l nodes=1:ppn=8
#PBS -l mem=120gb
#PBS -l walltime=4:00:00
#PBS -l feature=intel16

cd "${PBS_O_WORKDIR}"
#cd /mnt/home/korteme1/PuReMD_new/test_harness

# Import modules we will need
module purge
module load GNU/4.4.5 MPICH2/1.4.1p1

############
# MPI Runs #
############

# Run, after each run we must rename the output files, or another run will overwrite it

mpirun -np 8 ../PuReMD/bin/puremd ../data/benchmarks/bilayer/bilayer_340800.geo ../data/benchmarks/bilayer/ffield-bio bilayer_340800_mpi_control








