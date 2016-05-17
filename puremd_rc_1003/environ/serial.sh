#!/bin/bash

#To run single cpu executable
spuremd water_6540.pdb ffield.water param.gpu.water

#To run single GPU executable
puremd-gpu water_6540.pdb ffield.water param.gpu.water
