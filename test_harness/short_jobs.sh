#!/bin/bash -login

#PBS -l nodes=1:ppn=2:gpus=2
#PBS -l mem=96gb
#PBS -l walltime=30:00
#PBS -l feature=gpgpu:intel16

cd "${PBS_O_WORKDIR}"
#cd /mnt/home/korteme1/PuReMD_new/test_harness

# Import modules we will need
module purge
module load GNU/4.4.5 MPICH2/1.4.1p1 CUDA/6.0

############
# MPI Runs #
############

# Run, after each run we must rename the output files, or another run will overwrite it

mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/water/water_6540.pdb ../data/benchmarks/water/ffield.water water_6540_control

mv water.6540.log water.6540_mpi.log
mv water.6540.out water.6540_mpi.out
mv water.6540.pot water.6540_mpi.pot
mv water.6540.trj water.6540_mpi.trj

mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/silica/silica_6000.pdb ../data/benchmarks/silica/ffield-bio silica_6000_control

mv silica.6000.log silica.6000_mpi.log
mv silica.6000.out silica.6000_mpi.out
mv silica.6000.pot silica.6000_mpi.pot
mv silica.6000.trj silica.6000_mpi.trj

mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/metal/zno_6912.pdb ../data/benchmarks/metal/ffield.zno zno_6912_control

mv zno.6912.log zno.6912_mpi.log
mv zno.6912.out zno.6912_mpi.out
mv zno.6912.pot zno.6912_mpi.pot
mv zno.6912.trj zno.6912_mpi.trj

################
# MPI-GPU Runs #
################

module purge
module load GNU/4.8.2 OpenMPI/1.6.5 CUDA/6.0

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/water/water_6540.pdb ../data/benchmarks/water/ffield.water water_6540_control

mv water.6540.log water.6540_mpi_gpu.log
mv water.6540.out water.6540_mpi_gpu.out
mv water.6540.pot water.6540_mpi_gpu.pot
mv water.6540.trj water.6540_mpi_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/silica/silica_6000.pdb ../data/benchmarks/silica/ffield-bio silica_6000_control

mv silica.6000.log silica.6000_mpi_gpu.log
mv silica.6000.out silica.6000_mpi_gpu.out
mv silica.6000.pot silica.6000_mpi_gpu.pot
mv silica.6000.trj silica.6000_mpi_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/metal/zno_6912.pdb ../data/benchmarks/metal/ffield.zno zno_6912_control

mv zno.6912.log zno.6912_mpi_gpu.log
mv zno.6912.out zno.6912_mpi_gpu.out
mv zno.6912.pot zno.6912_mpi_gpu.pot
mv zno.6912.trj zno.6912_mpi_gpu.trj

####################
# MPI-NOT-GPU Runs #
####################

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/water/water_6540.pdb ../data/benchmarks/water/ffield.water water_6540_control

mv water.6540.log water.6540_mpi_not_gpu.log
mv water.6540.out water.6540_mpi_not_gpu.out
mv water.6540.pot water.6540_mpi_not_gpu.pot
mv water.6540.trj water.6540_mpi_not_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/silica/silica_6000.pdb ../data/benchmarks/silica/ffield-bio silica_6000_control

mv silica.6000.log silica.6000_mpi_not_gpu.log
mv silica.6000.out silica.6000_mpi_not_gpu.out
mv silica.6000.pot silica.6000_mpi_not_gpu.pot
mv silica.6000.trj silica.6000_mpi_not_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/metal/zno_6912.pdb ../data/benchmarks/metal/ffield.zno zno_6912_control

mv zno.6912.log zno.6912_mpi_not_gpu.log
mv zno.6912.out zno.6912_mpi_not_gpu.out
mv zno.6912.pot zno.6912_mpi_not_gpu.pot
mv zno.6912.trj zno.6912_mpi_not_gpu.trj






















