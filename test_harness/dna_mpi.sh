#!/bin/bash -login

#PBS -l nodes=1:ppn=2:gpus=2
#PBS -l mem=120gb
#PBS -l walltime=20:00:00
#PBS -l feature=gpgpu:intel14

cd /mnt/home/korteme1/PuReMD/puremd_rc_1003/tools

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
cd tools

# Run, after each run we must rename the output files, or another run will overwrite it

mpirun -np 2 ../PuReMD/src/puremd dna.pdb ffield-bio dna_control

mv dna.log dna_mpi.log
mv dna.out dna_mpi.out
mv dna.pot dna_mpi.pot
mv dna.trj dna_mpi.trj









