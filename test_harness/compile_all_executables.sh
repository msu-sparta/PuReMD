#!/bin/bash -login

# This script does nothing but compile all executables, to be used later in the test. 

# Import modules we will need

###############
# Serial Code #
###############
module purge
module load GNU/6.2
# Compile for serial (sPuReMD/bin/spuremd) 
cd ..
./configure --enable-openmp=no --enable-serial=yes
make
cd test_harness

###############
# MPI Compile #
###############
module purge
module load GNU/6.2 OpenMPI/2.0.2
# Compile for MPI (PuReMD/bin/puremd) 
cd ..
./configure --enable-openmp=no --enable-mpi-old=yes
make
cd test_harness

###################
# MPI-GPU Compile #
###################
module purge
module load GNU/6.2 OpenMPI/2.0.2 CUDA/9.0
# Compile for MPI-GPU (PG-PuReMD/bin/pg-puremd) 
cd ..
./configure --enable-openmp=no --enable-mpi-gpu=yes
make
cd test_harness
