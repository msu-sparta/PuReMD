#!/bin/bash -login

#PBS -l nodes=5:ppn=2:gpus=2
#PBS -l mem=600gb
#PBS -l walltime=8:00:00
#PBS -l feature=gpgpu:intel16

#cd ${PBS_O_WORKDIR}
cd /mnt/home/korteme1/PuReMD_new/test_harness

# Import modules we will need
module purge
module load GNU/4.4.5 MPICH2/1.4.1p1 CUDA/6.0
module load autoconf/2.69 automake/1.15

############
# MPI Runs #
############

# Compile for MPI 
cd ..
./configure --enable-openmp=no --enable-mpi=yes
make clean && make
cd test_harness

# Run, after each run we must rename the output files, or another run will overwrite it

mpirun -np 10 ../PuReMD/bin/puremd ../data/benchmarks/bilayer/bilayer_340800.geo ../data/benchmarks/bilayer/ffield-bio bilayer_340800_control

mv bilayer.340800.log bilayer.340800_mpi.log
mv bilayer.340800.out bilayer.340800_mpi.out
mv bilayer.340800.pot bilayer.340800_mpi.pot
mv bilayer.340800.trj bilayer.340800_mpi.trj









