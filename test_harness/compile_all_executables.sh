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
cd test_harness

module purge
module load GNU/4.8.2 OpenMPI/1.6.5 CUDA/6.0
module load autoconf/2.69 automake/1.15

#######################
# MPI_Not_GPU Compile #
#######################

# Requires an added layer of complexity when dealing with both the mpi_gpu and mpi_not_gpu versions, because they compile to the same name.
# Need to rename the mpi_not_gpu executable before compiling for mpi_gpu

# Make clean both versions
cd ..
./configure --enable-openmp=no --enable-mpi-gpu=yes
make clean 
./configure --enable-openmp=no --enable-mpi-not-gpu=yes
mv PG-PuReMD/bin/pg-puremd-not-gpu PG-PuReMD/bin/pg-puremd
make clean 
cd test_harness

# Compile mpi_not_gpu then rename executable
cd ..
make
mv PG-PuReMD/bin/pg-puremd PG-PuReMD/bin/pg-puremd-not-gpu
cd test_harness

###################
# MPI-GPU Compile #
###################

module purge
module load GNU/4.8.2 OpenMPI/1.6.5 CUDA/6.0
module load autoconf/2.69 automake/1.15

# Compile for MPI-GPU
cd ..
./configure --enable-openmp=no --enable-mpi-gpu=yes
make
cd test_harness

echo "Attempted to compile all executables, check if any errors before this point"



















