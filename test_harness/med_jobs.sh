#!/bin/bash -login

#PBS -l nodes=1:ppn=2:gpus=2
#PBS -l mem=96gb
#PBS -l walltime=4:00:00
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
mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/water/water_78480.geo ../data/benchmarks/water/ffield.water water_78480_control

mv water.78480.log water.78480_mpi.log
mv water.78480.out water.78480_mpi.out
mv water.78480.pot water.78480_mpi.pot
mv water.78480.trj water.78480_mpi.trj

mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/silica/silica_72000.geo ../data/benchmarks/silica/ffield-bio silica_72000_control

mv silica.72000.log silica.72000_mpi.log
mv silica.72000.out silica.72000_mpi.out
mv silica.72000.pot silica.72000_mpi.pot
mv silica.72000.trj silica.72000_mpi.trj

mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/bilayer/bilayer_56800.pdb ../data/benchmarks/bilayer/ffield-bio bilayer_56800_control

mv bilayer.56800.log bilayer.56800_mpi.log
mv bilayer.56800.out bilayer.56800_mpi.out
mv bilayer.56800.pot bilayer.56800_mpi.pot
mv bilayer.56800.trj bilayer.56800_mpi.trj

mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/dna/dna_19733.pdb ../data/benchmarks/dna/ffield-dna dna_control

mv dna.log dna_mpi.log
mv dna.out dna_mpi.out
mv dna.pot dna_mpi.pot
mv dna.trj dna_mpi.trj

mpirun -np 2 ../PuReMD/bin/puremd ../data/benchmarks/petn/petn_48256.pdb ../data/benchmarks/petn/ffield.petn petn_control

mv petn.log petn_mpi.log
mv petn.out petn_mpi.out
mv petn.pot petn_mpi.pot
mv petn.trj petn_mpi.trj

################
# MPI-GPU Runs #
################

module purge
module load GNU/4.8.2 OpenMPI/1.6.5 CUDA/6.0

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/water/water_78480.geo ../data/benchmarks/water/ffield.water water_78480_control

mv water.78480.log water.78480_mpi_gpu.log
mv water.78480.out water.78480_mpi_gpu.out
mv water.78480.pot water.78480_mpi_gpu.pot
mv water.78480.trj water.78480_mpi_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/silica/silica_72000.geo ../data/benchmarks/silica/ffield-bio silica_72000_control

mv silica.72000.log silica.72000_mpi_gpu.log
mv silica.72000.out silica.72000_mpi_gpu.out
mv silica.72000.pot silica.72000_mpi_gpu.pot
mv silica.72000.trj silica.72000_mpi_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/bilayer/bilayer_56800.pdb ../data/benchmarks/bilayer/ffield-bio bilayer_56800_control

mv bilayer.56800.log bilayer.56800_mpi_gpu.log
mv bilayer.56800.out bilayer.56800_mpi_gpu.out
mv bilayer.56800.pot bilayer.56800_mpi_gpu.pot
mv bilayer.56800.trj bilayer.56800_mpi_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/dna/dna_19733.pdb ../data/benchmarks/dna/ffield-dna dna_control

mv dna.log dna_mpi_gpu.log
mv dna.out dna_mpi_gpu.out
mv dna.pot dna_mpi_gpu.pot
mv dna.trj dna_mpi_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd ../data/benchmarks/petn/petn_48256.pdb ../data/benchmarks/petn/ffield.petn petn_control

mv petn.log petn_mpi_gpu.log
mv petn.out petn_mpi_gpu.out
mv petn.pot petn_mpi_gpu.pot
mv petn.trj petn_mpi_gpu.trj

####################
# MPI_Not_GPU Runs #
####################

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/water/water_78480.geo ../data/benchmarks/water/ffield.water water_78480_control

mv water.78480.log water.78480_mpi_not_gpu.log
mv water.78480.out water.78480_mpi_not_gpu.out
mv water.78480.pot water.78480_mpi_not_gpu.pot
mv water.78480.trj water.78480_mpi_not_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/silica/silica_72000.geo ../data/benchmarks/silica/ffield-bio silica_72000_control

mv silica.72000.log silica.72000_mpi_not_gpu.log
mv silica.72000.out silica.72000_mpi_not_gpu.out
mv silica.72000.pot silica.72000_mpi_not_gpu.pot
mv silica.72000.trj silica.72000_mpi_not_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/bilayer/bilayer_56800.pdb ../data/benchmarks/bilayer/ffield-bio bilayer_56800_control

mv bilayer.56800.log bilayer.56800_mpi_not_gpu.log
mv bilayer.56800.out bilayer.56800_mpi_not_gpu.out
mv bilayer.56800.pot bilayer.56800_mpi_not_gpu.pot
mv bilayer.56800.trj bilayer.56800_mpi_not_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/dna/dna_19733.pdb ../data/benchmarks/dna/ffield-dna dna_control

mv dna.log dna_mpi_not_gpu.log
mv dna.out dna_mpi_not_gpu.out
mv dna.pot dna_mpi_not_gpu.pot
mv dna.trj dna_mpi_not_gpu.trj

mpirun -np 2 ../PG-PuReMD/bin/pg-puremd-not-gpu ../data/benchmarks/petn/petn_48256.pdb ../data/benchmarks/petn/ffield.petn petn_control

mv petn.log petn_mpi_not_gpu.log
mv petn.out petn_mpi_not_gpu.out
mv petn.pot petn_mpi_not_gpu.pot
mv petn.trj petn_mpi_not_gpu.trj




















