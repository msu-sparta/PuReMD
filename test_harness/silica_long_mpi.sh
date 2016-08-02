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

mpirun -np 10 ../PuReMD/bin/puremd ../data/benchmarks/silica/silica_300000.geo ../data/benchmarks/silica/ffield-bio silica_300000_control

mv silica.300000.log silica.300000_mpi.log
mv silica.300000.out silica.300000_mpi.out
mv silica.300000.pot silica.300000_mpi.pot
mv silica.300000.trj silica.300000_mpi.trj









