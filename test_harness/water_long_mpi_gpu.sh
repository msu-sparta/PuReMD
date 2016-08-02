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


mpirun -np 10 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/water/water_327000.geo ../data/benchmarks/water/ffield.water water_327000_control

mv water.327000.log water.327000_mpi_gpu.log
mv water.327000.out water.327000_mpi_gpu.out
mv water.327000.pot water.327000_mpi_gpu.pot
mv water.327000.trj water.327000_mpi_gpu.trj





















