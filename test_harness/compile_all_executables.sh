#!/bin/bash -login

# This script does nothing but compile all executables, to be used later in the test. 

# Import modules we will need
module purge
module load GNU/4.4.5 MPICH2/1.4.1p1 CUDA/6.0
module load autoconf/2.69 automake/1.15

###############
# MPI Compile #
###############

# Compile for MPI 
cd ..
./configure --enable-openmp=no --enable-mpi=yes
make clean && make
#make
cd test_harness

# Run, after each run we must rename the output files, or another run will overwrite it

###################
# MPI-GPU Compile #
###################

module purge
module load GNU/4.8.2 OpenMPI/1.6.5 CUDA/6.0
module load autoconf/2.69 automake/1.15

# Compile for MPI-GPU
cd ..
./configure --enable-openmp=no --enable-mpi-gpu=yes
make clean && make
#make
cd test_harness





















