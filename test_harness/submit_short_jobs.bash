#!/bin/bash


function submit_job_serial
{
	SIM_NAME="spuremd-${4}"
	qsub -l "walltime=${JOB_TIME},nodes=${JOB_NODES}:ppn=${JOB_PPN},feature=${JOB_FEATURE},mem=${JOB_MEM}" \
		-o "${SIM_NAME}" -j oe << EOF
#!/bin/bash
module purge
module load GNU/6.2 MKL/11.2
cd "\${PBS_O_WORKDIR}"
../sPuReMD/bin/spuremd "${1}" "${2}" "${3}"
EOF
}


function submit_job_mpi
{
	SIM_NAME="puremd-${4}"
	qsub -l "walltime=${JOB_TIME},nodes=${JOB_NODES}:ppn=${JOB_PPN},feature=${JOB_FEATURE},mem=${JOB_MEM}" \
		-o "${SIM_NAME}" -j oe << EOF
#!/bin/bash
module purge
module load GNU/6.2 MKL/11.2 OpenMPI/2.0.2
cd "\${PBS_O_WORKDIR}"
mpirun -np "${JOB_PPN}" "../PuReMD/bin/puremd "${1}" "${2}" "${3}"
EOF
}


function submit_job_mpi_gpu
{
	SIM_NAME="pg-puremd-${4}"
	qsub -l "walltime=${JOB_TIME},nodes=${JOB_NODES}:ppn=${JOB_PPN},feature=${JOB_FEATURE},mem=${JOB_MEM}" \
		-o "${SIM_NAME}" -j oe << EOF
#!/bin/bash
module purge
module load GNU/6.2 MKL/11.2 OpenMPI/2.0.2 CUDA/9.0
cd "\${PBS_O_WORKDIR}"
mpirun -np "${JOB_PPN}" "../PG-PuReMD/bin/pg-puremd "${1}" "${2}" "${3}"
EOF
}


############
# Serial Runs #
############
JOB_TIME="03:59:00"
JOB_NODES="1"
JOB_PPN="1"
JOB_FEATURE="lac"
JOB_MEM="120gb"

sub_job_serial  "../data/benchmarks/water/water_6540.pdb" \
	"../data/benchmarks/water/ffield.water" \
	"control_water_6540_serial" \
	"water_6540_serial"

sub_job_serial "../data/benchmarks/silica/silica_6000.pdb" \
	"../data/benchmarks/silica/ffield-bio" \
	"control_silica_6000_serial" \
	"silica_6000_serial"

sub_job_serial  "../data/benchmarks/metal/zno_6912.pdb" \
        "../data/benchmarks/metal/ffield.zno" \
	"control_zno_6912_serial" \
	"zno_6912_serial"

############
# MPI Runs #
############
JOB_TIME="03:59:00"
JOB_NODES="1"
JOB_PPN="2"
JOB_FEATURE="lac"
JOB_MEM="120gb"

sub_job_mpi "../data/benchmarks/water/water_6540.pdb" \
	"../data/benchmarks/water/ffield.water" \
       	"control_water_6540_mpi" \
	"water_6540_mpi"

sub_job_mpi "../data/benchmarks/silica/silica_6000.pdb" \
	"../data/benchmarks/silica/ffield-bio" \
	"control_silica_6000_mpi" \
	"silica_6000_mpi"

sub_job_mpi "../data/benchmarks/metal/zno_6912.pdb"
	"../data/benchmarks/metal/ffield.zno" \
	"control_zno_6912_mpi" \
	"zno_6912_mpi"

################
# MPI-GPU Runs #
################
JOB_TIME="03:59:00"
JOB_NODES="2"
JOB_PPN="1"
JOB_FEATURE="lac"
JOB_MEM="120gb"

sub_job_mpi_gpu "../data/benchmarks/water/water_6540.pdb" \
	"../data/benchmarks/water/ffield.water" \
	"control_water_6540_mpi_gpu" \
	"water_6540_mpi_gpu"

sub_job_mpi_gpu "../data/benchmarks/silica/silica_6000.pdb" \
	"../data/benchmarks/silica/ffield-bio" \
	"control_silica_6000_mpi_gpu" \
	"silica_6000_mpi_gpu"

sub_job_mpi_gpu "../data/benchmarks/metal/zno_6912.pdb" \
	"../data/benchmarks/metal/ffield.zno" 
	"control_zno_6912_mpi_gpu" \
	"zno_6912_mpi_gpu"
