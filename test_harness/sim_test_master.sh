#!/bin/bash -login
###############################################
#
#  PuReMD different versions agreement test.
#  Run this master script to start test.
#
###############################################

# Will be submitting multiple jobs
# First, one for all the short jobs
qsub short_jobs.sh
# Then, one for all the medium jobs
qsub med_jobs.sh
# Finally, one for each of the long jobs

#qsub water_long_mpi.sh
#qsub water_long_mpi_gpu.sh
#qsub water_long_mpi_not_gpu.sh

#qsub silica_long_mpi.sh
#qsub silica_long_mpi_gpu.sh
#qsub silica_long_mpi_not_gpu.sh

#qsub bilayer_long_mpi.sh
#qsub bilayer_long_mpi_gpu.sh
#qsub bilayer_long_mpi_not_gpu.sh

echo "All jobs submitted"



