#!/bin/bash


# submit job scripts of specified type
#
# inputs: 1) job_type, 2) code version to run, 3) data set
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
		"${1}" "${2}" "${3}"
}


# submit job scripts of specified type
#
# inputs: 1) job_type, 2) code version to run, 3) data set
function submit_job_mpi
{
	python3 ../../tools/run_sim.py submit_jobs \
		-b "../../PuReMD/bin/puremd" \
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
		-m "srun:1:1:1" \
		"${1}" "${2}" "${3}"
}


# submit job scripts of specified type
#
# inputs: 1) job_type, 2) code version to run, 3) data set
function submit_job_mpi_gpu
{
	python3 ../../tools/run_sim.py submit_jobs \
		-b "../../PG-PuReMD/bin/pg-puremd" \
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
		-m "srun:1:1:1" \
		"${1}" "${2}" "${3}"
}


# serial jobs (spuremd)
DATA_DIR="serial"

if [ ! -d "${DATA_DIR}" ]; then
	mkdir "${DATA_DIR}"
fi
pushd "${DATA_DIR}" &> /dev/null

submit_job_serial "slurm" "serial" "zno_6912"

submit_job_serial "slurm" "serial" "silica_6000"

submit_job_serial "slurm" "serial" "water_6540"

popd &> /dev/null

# mpi jobs (puremd)
DATA_DIR="mpi"

if [ ! -d "${DATA_DIR}" ]; then
	mkdir "${DATA_DIR}"
fi
pushd "${DATA_DIR}" &> /dev/null

submit_job_mpi "slurm" "mpi" "zno_6912"

submit_job_mpi "slurm" "mpi" "silica_6000"

submit_job_mpi "slurm" "mpi" "water_6540"

popd &> /dev/null

# mpi+gpu jobs (pg-puremd)
DATA_DIR="mpi_gpu"

if [ ! -d "${DATA_DIR}" ]; then
	mkdir "${DATA_DIR}"
fi
pushd "${DATA_DIR}" &> /dev/null

submit_job_mpi_gpu "slurm" "mpi+gpu" "zno_6912"

submit_job_mpi_gpu "slurm" "mpi+gpu" "silica_6000"

submit_job_mpi_gpu "slurm" "mpi+gpu" "water_6540"

popd &> /dev/null
