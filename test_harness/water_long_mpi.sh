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

mpirun -np 8 ../PuReMD/bin/puremd ../data/benchmarks/water/water_327000.geo ../data/benchmarks/water/ffield.water water_327000_mpi_control








