#!/bin/bash -login

#PBS -l nodes=5:ppn=2:gpus=2
#PBS -l mem=600gb
#PBS -l walltime=8:00:00
#PBS -l feature=gpgpu:intel16

#cd ${PBS_O_WORKDIR}
cd /mnt/home/korteme1/PuReMD_new/test_harness
################
# MPI-GPU Runs #
################

module purge
module load GNU/4.8.2 OpenMPI/1.6.5 CUDA/6.0
module load autoconf/2.69 automake/1.15



# Compile for MPI-GPU
cd ..
./configure --enable-openmp=no --enable-mpi-gpu=yes
make clean && make
cd test_harness


mpirun -np 10 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/silica/silica_300000.geo ../data/benchmarks/silica/ffield-bio silica_300000_control

mv silica.300000.log silica.300000_mpi_gpu.log
mv silica.300000.out silica.300000_mpi_gpu.out
mv silica.300000.pot silica.300000_mpi_gpu.pot
mv silica.300000.trj silica.300000_mpi_gpu.trj





















