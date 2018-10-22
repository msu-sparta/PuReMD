#!/bin/bash


# submit job scripts of specified type
#
# inputs: 1) job_type, 2) data set
function submit_job_serial
{
	python3 ../../tools/run_sim.py submit_jobs \
		-b "../../sPuReMD/bin/spuremd" \
		-p ensemble_type "0" \
		-p nsteps "100" \
		-p tabulate_long_range "0" \
		-p charge_method "0" \
		-p cm_solver_type "0" \
		-p cm_solver_max_iters "50" \
		-p cm_solver_restart "200" \
		-p cm_solver_q_err "1e-6" \
		-p cm_domain_sparsity "1.0" \
		-p cm_solver_pre_comp_type "1" \
		-p cm_solver_pre_comp_refactor "10000" \
		-p cm_solver_pre_comp_droptol "0.0" \
		-p cm_solver_pre_comp_sweeps "3" \
		-p cm_solver_pre_comp_sai_thres "0.1" \
		-p cm_solver_pre_app_type "0" \
		-p cm_solver_pre_app_jacobi_iters "30" \
		-p threads "1" \
		"${1}" "${2}"
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


# serial jobs (spuremd)
JOB_TIME="03:59:00"
JOB_NODES="1"
JOB_PPN="1"
JOB_FEATURE="lac"
JOB_MEM="120gb"
DATA_DIR="serial"

if [ ! -d "${DATA_DIR}" ]; then
	mkdir "${DATA_DIR}"
fi
pushd "${DATA_DIR}" &> /dev/null

submit_job_serial "slurm" "zno_6912"

submit_job_serial "slurm" "silica_6000"

submit_job_serial "slurm" "water_6540"

popd &> /dev/null

## mpi jobs (puremd)
#JOB_TIME="03:59:00"
#JOB_NODES="1"
#JOB_PPN="2"
#JOB_FEATURE="lac"
#JOB_MEM="120gb"
#DATA_DIR="mpi"
#
#if [ ! -d "${DATA_DIR}" ]; then
#	mkdir "${DATA_DIR}"
#fi
#pushd "${DATA_DIR}" &> /dev/null
#
#submit_job_mpi "../data/benchmarks/metal/zno_6912.pdb"
#	"../data/benchmarks/metal/ffield.zno" \
#	"control_zno_6912_mpi" \
#	"zno_6912_mpi"
#
#submit_job_mpi "../data/benchmarks/silica/silica_6000.pdb" \
#	"../data/benchmarks/silica/ffield-bio" \
#	"control_silica_6000_mpi" \
#	"silica_6000_mpi"
#
#submit_job_mpi "../data/benchmarks/water/water_6540.pdb" \
#	"../data/benchmarks/water/ffield.water" \
#       	"control_water_6540_mpi" \
#	"water_6540_mpi"
#
#popd &> /dev/null
#
## mpi+gpu jobs (pg-puremd)
#JOB_TIME="03:59:00"
#JOB_NODES="2"
#JOB_PPN="1"
#JOB_FEATURE="lac"
#JOB_MEM="120gb"
#DATA_DIR="mpi_gpu"
#
#if [ ! -d "${DATA_DIR}" ]; then
#	mkdir "${DATA_DIR}"
#fi
#pushd "${DATA_DIR}" &> /dev/null
#
#submit_job_mpi_gpu "../data/benchmarks/metal/zno_6912.pdb" \
#	"../data/benchmarks/metal/ffield.zno" 
#	"control_zno_6912_mpi_gpu" \
#	"zno_6912_mpi_gpu"
#
#submit_job_mpi_gpu "../data/benchmarks/silica/silica_6000.pdb" \
#	"../data/benchmarks/silica/ffield-bio" \
#	"control_silica_6000_mpi_gpu" \
#	"silica_6000_mpi_gpu"
#
#submit_job_mpi_gpu "../data/benchmarks/water/water_6540.pdb" \
#	"../data/benchmarks/water/ffield.water" \
#	"control_water_6540_mpi_gpu" \
#	"water_6540_mpi_gpu"
#
#popd &> /dev/null
