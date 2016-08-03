#!/bin/bash -login

#####################################################
# Run this after all jobs started by sim_test_master
# have ended. This simply prints the output in an 
# easily readable format. It prints the tail of
# all output .pot files
#####################################################
echo "Small Systems:"
echo "-------------------------------------------"
echo "tail water.6540_mpi.pot:"
tail water.6540_mpi.pot;
echo "tail water.6540_mpi_gpu.pot:"
tail water.6540_mpi_gpu.pot;
echo "-------------------------------------------"

echo "tail silica.6000_mpi.pot:"
tail silica.6000_mpi.pot
echo "tail silica.6000_mpi_gpu.pot:"
tail silica.6000_mpi_gpu.pot
echo "-------------------------------------------"

echo "tail zno.6912_mpi.pot:"
tail zno.6912_mpi.pot;
echo "tail zno.6912_mpi_gpu.pot:"
tail zno.6912_mpi_gpu.pot;

echo "___________________________________________"
echo "Medium Systems"

echo "-------------------------------------------"
echo "tail water.78480_mpi.pot:"
tail water.78480_mpi.pot;
echo "tail water.78480_mpi_gpu.pot:"
tail water.78480_mpi_gpu.pot;

echo "-------------------------------------------"
echo "tail silica.72000_mpi.pot:"
tail silica.72000_mpi.pot;
echo "tail silica.72000_mpi_gpu.pot:"
tail silica.72000_mpi_gpu.pot;

echo "-------------------------------------------"
echo "tail bilayer.56800_mpi.pot:"
tail bilayer.56800_mpi.pot;
echo "tail bilayer.56800_mpi_gpu.pot:"
tail bilayer.56800_mpi_gpu.pot;

echo "-------------------------------------------"
echo "tail dna_mpi.pot:"
tail dna_mpi.pot;
echo "tail dna_mpi_gpu.pot:"
tail dna_mpi_gpu.pot;

echo "-------------------------------------------"
echo "tail petn_mpi.pot:"
tail petn_mpi.pot;
echo "tail petn_mpi_gpu.pot:"
tail petn_mpi_gpu.pot;

echo "___________________________________________"
echo "Large Systems"

echo "-------------------------------------------"
echo "tail water.327000_mpi.pot:"
tail water.327000_mpi.pot;
echo "tail water.327000_mpi_gpu.pot:"
tail water.327000_mpi_gpu.pot;

echo "-------------------------------------------"
echo "tail silica.300000_mpi.pot:"
tail silica.300000_mpi.pot;
echo "tail silica.300000_mpi_gpu.pot:"
tail silica.300000_mpi_gpu.pot;

echo "-------------------------------------------"
echo "tail bilayer.340800_mpi.pot:"
tail bilayer.340800_mpi.pot;
echo "tail bilayer.340800_mpi_gpu.pot:"
tail bilayer.340800_mpi_gpu.pot;










