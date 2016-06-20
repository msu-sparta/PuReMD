#!/bin/bash

PROCESSES="2"
MAC_FILE="mfile"

#To run parallel cpu executable
mpirun -x LD_LIBRARY_PATH=/opt/cuda5/lib64 -np $PROCESSES -machinefile $MAC_FILE puremd water_6540.pdb ffield.water parallel_control

#To run parallel gpu executable
mpirun -x LD_LIBRARY_PATH=/opt/cuda5/lib64 -np $PROCESSES -machinefile $MAC_FILE pg-puremd water_6540.pdb ffield.water parallel_control
