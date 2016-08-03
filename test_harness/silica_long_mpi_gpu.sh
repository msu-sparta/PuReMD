#!/bin/bash -login

#PBS -l nodes=1:ppn=8:gpus=8
#PBS -l mem=120gb
#PBS -l walltime=4:00:00
#PBS -l feature=gpgpu:intel16

cd "${PBS_O_WORKDIR}"
#cd /mnt/home/korteme1/PuReMD_new/test_harness

################
# MPI-GPU Runs #
################

module purge
module load GNU/4.8.2 OpenMPI/1.6.5 CUDA/6.0

mpirun -np 8 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/silica/silica_300000.geo ../data/benchmarks/silica/ffield-bio silica_300000_mpi_gpu_control





















