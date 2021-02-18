#!/bin/bash

export LD_LIBRARY_PATH="../../../../sPuReMD/lib/.libs"

PROG="./driver"
# first 14 atoms in BGF file are for QM region, remaining are for MM region
BGF_FILE="../AVE/fort.3"
NUM_QM_ATOMS="14"
NUM_MM_ATOMS="759"
LOG_FILE="forces_charges.txt"

"${PROG}" "${BGF_FILE}" "${NUM_QM_ATOMS}" "${NUM_MM_ATOMS}" > "${LOG_FILE}"
