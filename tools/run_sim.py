#!/usr/bin/env python3


class TestCase():
    def __init__(self, data_set, geo_file, ffield_file, control_file, params={}, result_header_fmt='',
            result_header='', result_body_fmt='', result_file='results.txt', geo_format='1',
            min_step=None, max_step=None):
        from re import sub

        self.__data_set = data_set
        self.__geo_file = geo_file
        self.__ffield_file = ffield_file
        self.__control_file = control_file
        self.__param_names = sorted(params.keys())
        self.__params = params
        self.__result_header_fmt = result_header_fmt
        self.__result_header = result_header
        self.__result_body_fmt = result_body_fmt
        self.__result_file = result_file
        self.__control_regexes = { \
                'name': lambda l, x: sub(
                    r'(?P<key>simulation_name\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'ensemble_type': lambda l, x: sub(
                    r'(?P<key>ensemble_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'nsteps': lambda l, x: sub(
                    r'(?P<key>nsteps\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'tabulate_long_range': lambda l, x: sub(
                    r'(?P<key>tabulate_long_range\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'charge_method': lambda l, x: sub(
                    r'(?P<key>charge_method\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_q_net': lambda l, x: sub(
                    r'(?P<key>cm_q_net\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_max_iters': lambda l, x: sub(
                    r'(?P<key>cm_solver_max_iters\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_restart': lambda l, x: sub(
                    r'(?P<key>cm_solver_restart\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_q_err': lambda l, x: sub(
                    r'(?P<key>cm_solver_q_err\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_domain_sparsity': lambda l, x: sub(
                    r'(?P<key>cm_domain_sparsity\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_init_guess_extrap1': lambda l, x: sub(
                    r'(?P<key>cm_init_guess_extrap1\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_init_guess_extrap2': lambda l, x: sub(
                    r'(?P<key>cm_init_guess_extrap2\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_droptol': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_droptol\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_refactor': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_refactor\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_sweeps': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_sweeps\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_sai_thres': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_sai_thres\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_app_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_app_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_app_jacobi_iters': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_app_jacobi_iters\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'geo_format': lambda l, x: sub(
                    r'(?P<key>geo_format\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
        }
        self.__params['geo_format'] = geo_format
        self.__min_step = min_step
        self.__max_step = max_step

    def _create_control_file(self, param, new_control_file):
        # read in template control file
#        try:
#            with open(self.__control_file, 'r') as fp:
#                lines = fp.read()
#        except Exception as e:
#            print("ERROR: Coult not read template control file!")
#            raise e

        # inline template control file
        lines = """\
simulation_name         water_6540_notab_qeq           ! output files will carry this name + their specific extension
ensemble_type           0                       ! 0: NVE, 1: Berendsen NVT, 2: nose-Hoover NVT, 3: semi-isotropic NPT, 4: isotropic NPT, 5: anisotropic NPT
nsteps                  100                     ! number of simulation steps
dt                      0.25                    ! time step in fs
periodic_boundaries     1                       ! 0: no periodic boundaries, 1: periodic boundaries

reposition_atoms        0                       ! 0: just fit to periodic boundaries, 1: CoM to the center of box, 3: CoM to the origin
restrict_bonds          0                       ! enforce the bonds given in CONECT lines of pdb file for this many steps
tabulate_long_range     0                       ! denotes the granularity of long range tabulation, 0 means no tabulation
energy_update_freq      1
remove_CoM_vel          500                     ! remove the translational and rotational vel around the center of mass at every 'this many' steps

nbrhood_cutoff          5.0                     ! near neighbors cutoff for bond calculations (Angstroms)
bond_graph_cutoff       0.3                     ! bond strength cutoff for bond graphs (Angstroms)
thb_cutoff              0.001                   ! cutoff value for three body interactions (Angstroms)
hbond_cutoff            7.50                    ! cutoff distance for hydrogen bond interactions (Angstroms)

charge_method         		  0             ! charge method: 0 = QEq, 1 = EEM, 2 = ACKS2
cm_q_net              		  0.0           ! net system charge
cm_solver_type        		  0             ! iterative linear solver for charge method: 0 = GMRES(k), 1 = GMRES_H(k), 2 = CG, 3 = SDM
cm_solver_max_iters   		  20            ! max solver iterations
cm_solver_restart     		  100           ! inner iterations of before restarting (GMRES(k)/GMRES_H(k))
cm_solver_q_err       		  1e-6          ! relative residual norm threshold used in solver
cm_domain_sparsity     		  1.0           ! scalar for scaling cut-off distance, used to sparsify charge matrix (between 0.0 and 1.0)
cm_init_guess_extrap1		  3             ! order of spline extrapolation for initial guess (s)
cm_init_guess_extrap2		  2             ! order of spline extrapolation for initial guess (t)
cm_solver_pre_comp_type           1             ! method used to compute preconditioner, if applicable
cm_solver_pre_comp_refactor       1000          ! number of steps before recomputing preconditioner
cm_solver_pre_comp_droptol        0.0           ! threshold tolerance for dropping values in preconditioner computation (ICHOLT/ILUT/FG-ILUT)
cm_solver_pre_comp_sweeps         3             ! number of sweeps used to compute preconditioner (FG-ILUT)
cm_solver_pre_comp_sai_thres      0.1           ! ratio of charge matrix NNZ's used to compute preconditioner (SAI)
cm_solver_pre_app_type            1             ! method used to apply preconditioner (ICHOLT/ILUT/FG-ILUT)
cm_solver_pre_app_jacobi_iters    50            ! num. Jacobi iterations used for applying precondition (ICHOLT/ILUT/FG-ILUT)

temp_init               0.0                     ! desired initial temperature of the simulated system
temp_final              300.0                   ! desired final temperature of the simulated system
t_mass                  0.16666                 ! 0.16666 for Nose-Hoover nvt ! 100.0 for npt! in fs, thermal inertia parameter
t_mode                  0                       ! 0: T-coupling only, 1: step-wise, 2: constant slope
t_rate                  -100.0                  ! in K
t_freq                  4.0                     ! in ps

pressure                0.000101325             ! desired pressure of the simulated system in GPa, 1atm = 0.000101325 GPa
p_mass                  5000.00                 ! in fs, pressure inertia parameter
compress                0.008134                ! in ps^2 * A / amu ( 4.5X10^(-5) bar^(-1) )
press_mode              0                       ! 0: internal + external pressure, 1: ext only, 2: int only

geo_format              1                       ! 0: custom, 1: pdb, 2: bgf
write_freq              0                       ! write trajectory after so many steps
traj_compress           0                       ! 0: no compression  1: uses zlib to compress trajectory output
traj_format             0                       ! 0: our own format (below options apply to this only), 1: xyz, 2: bgf, 3: pdb
traj_title              WATER_NVE               ! (no white spaces)
atom_info               1                       ! 0: no atom info, 1: print basic atom info in the trajectory file
atom_forces             0                       ! 0: basic atom format, 1: print force on each atom in the trajectory file
atom_velocities         0                       ! 0: basic atom format, 1: print the velocity of each atom in the trajectory file
bond_info               1                       ! 0: do not print bonds, 1: print bonds in the trajectory file
angle_info              1                       ! 0: do not print angles, 1: print angles in the trajectory file
test_forces             0                       ! 0: normal run, 1: at every timestep print each force type into a different file

molec_anal              0                       ! 1: outputs newly formed molecules as the simulation progresses
freq_molec_anal         0                       ! perform molecular analysis at every 'this many' timesteps
dipole_anal             0                       ! 1: calculate a electric dipole moment of the system
freq_dipole_anal        1                       ! calculate electric dipole moment at every 'this many' steps
diffusion_coef          0                       ! 1: calculate diffusion coefficient of the system
freq_diffusion_coef     1                       ! calculate diffusion coefficient at every 'this many' steps
restrict_type           2                       ! -1: all types of atoms, 0 and up: only this type of atoms

restart_format          1                       ! 0: restarts in ASCII  1: restarts in binary
restart_freq            0                       ! 0: do not output any restart files. >0: output a restart file at every 'this many' steps
"""

        # substitute key-value pairs in the text of the control file
        # using a dict of regexes (used to match and substitute
        # specific key-value pairs)
        for k in self.__control_regexes.keys():
            try:
                lines = self.__control_regexes[k](lines, param[k])
            except KeyError:
                # if key missing from regex dict, skip for now
                pass

        # write out new control file text using supplied file name
        with open(new_control_file, 'w') as fp:
            fp.write(lines)

    def run(self, binary, process_results=False):
        from itertools import product
        from os import environ, path, remove, rmdir
        from subprocess import Popen, PIPE
        from tempfile import mkdtemp
        from time import time

        env = dict(environ)
        temp_dir = mkdtemp()

        # create Cartesian product of all supplied sets of parameter values
        for p in product(*[self.__params[k] for k in self.__param_names]):
            param_dict = dict((k, v) for (k, v) in zip(self.__param_names, p))
            param_dict['name'] = path.basename(self.__geo_file).split('.')[0] \
                + '_cm' + param_dict['charge_method'] \
                + '_s' + param_dict['nsteps'] \
		+ '_q' + param_dict['cm_solver_type'] \
 		+ '_qtol' + param_dict['cm_solver_q_err'] \
 		+ '_qds' + param_dict['cm_domain_sparsity'] \
                + '_pc' + param_dict['cm_solver_pre_comp_type'] \
                + '_pcr' + param_dict['cm_solver_pre_comp_refactor'] \
                + '_pctol' + param_dict['cm_solver_pre_comp_droptol'] \
                + '_pcs' + param_dict['cm_solver_pre_comp_sweeps'] \
                + '_pcsai' + param_dict['cm_solver_pre_comp_sai_thres'] \
                + '_pa' + param_dict['cm_solver_pre_app_type'] \
                + '_paji' + param_dict['cm_solver_pre_app_jacobi_iters'] \
		+ '_t' + param_dict['threads']

            temp_file = path.join(temp_dir, path.basename(self.__control_file))
            self._create_control_file(param_dict, temp_file)

            env['OMP_NUM_THREADS'] = param_dict['threads']

            cmd_args = binary.split()
            cmd_args.append(self.__geo_file)
            cmd_args.append(self.__ffield_file)
            cmd_args.append("")
            cmd_args[-1] = temp_file

            start = time()
            proc_handle = Popen(cmd_args, stdout=PIPE, stderr=PIPE, env=env, universal_newlines=True)
            stdout, stderr = proc_handle.communicate()
            stop = time()

            if proc_handle.returncode < 0:
                print("WARNING: process terminated with code {0}".format(proc_handle.returncode))

            print('stdout:')
            print(stdout)
            print('stderr:')
            print(stderr)

            if path.exists(temp_file):
                remove(temp_file)

        if path.exists(temp_dir):
            rmdir(temp_dir)

    def _process_result(self, fout, param, min_step, max_step):
        time = 0.
        cm = 0.
        iters = 0.
        pre_comp = 0.
        pre_app = 0.
        spmv = 0.
        cnt = 0
        cnt_valid = 0
        line_cnt = 0
        log_file = param['name'] + '.log'

        if not path.exists(log_file):
            print('***WARNING: {0} does not exist!'.format(log_file))
            return
        with open(log_file, 'r') as fp:
            for line in fp:
                line = line.split()
                try:
                    if (not min_step and not max_step) or \
                    (min_step and not max_step and cnt_valid >= min_step) or \
                    (not min_step and max_step and cnt_valid <= max_step) or \
                    (cnt_valid >= min_step and cnt_valid <= max_step):
                        cm = cm + float(line[6])
                        iters = iters + float(line[8])
                        pre_comp = pre_comp + float(line[9])
                        pre_app = pre_app + float(line[10])
                        spmv = spmv + float(line[11])
                        cnt = cnt + 1
                    cnt_valid = cnt_valid + 1
                except Exception:
                    pass
                if line[0] == 'total:':
                    try:
                        time = float(line[1])
                    except Exception:
                        pass
                line_cnt = line_cnt + 1
            if cnt > 0:
                cm = cm / cnt
                iters = iters / cnt
                pre_comp = pre_comp / cnt
                pre_app = pre_app / cnt
                spmv = spmv / cnt

        # subtract for header, footer (total time), and extra step
        # (e.g., 100 steps means steps 0 through 100, inclusive)
        if (line_cnt - 3) == int(param['nsteps']):
            fout.write(self.__result_body_fmt.format(path.basename(self.__geo_file).split('.')[0], 
                param['nsteps'], param['charge_method'], param['cm_solver_type'],
                param['cm_solver_q_err'], param['cm_domain_sparsity'],
                param['cm_solver_pre_comp_type'], param['cm_solver_pre_comp_droptol'],
                param['cm_solver_pre_comp_sweeps'], param['cm_solver_pre_comp_sai_thres'],
                param['cm_solver_pre_app_type'], param['cm_solver_pre_app_jacobi_iters'],
                pre_comp, pre_app, iters, spmv,
                cm, param['threads'], time))
        else:
            print('**WARNING: nsteps not correct in file {0} (nsteps = {1:d}, counted steps = {2:d}).'.format(
                log_file, int(param['nsteps']), max(line_cnt - 3, 0)))
        fout.flush()

    def parse_results(self):
        from itertools import product
        from os import path

        write_header = True
        if path.exists(self.__result_file):
            write_header = False

        with open(self.__result_file, 'a') as fout:
            if write_header:
                fout.write(self.__result_header_fmt.format(*self.__result_header))
                fout.flush()

            # create Cartesian product of all supplied sets of parameter values
            for p in product(*[self.__params[k] for k in self.__param_names]):
                param_dict = dict((k, v) for (k, v) in zip(self.__param_names, p))
                param_dict['name'] = path.basename(self.__geo_file).split('.')[0] \
                    + '_cm' + param_dict['charge_method'] \
                    + '_s' + param_dict['nsteps'] \
                    + '_q' + param_dict['cm_solver_type'] \
                    + '_qtol' + param_dict['cm_solver_q_err'] \
                    + '_qds' + param_dict['cm_domain_sparsity'] \
                    + '_pc' + param_dict['cm_solver_pre_comp_type'] \
                    + '_pcr' + param_dict['cm_solver_pre_comp_refactor'] \
                    + '_pctol' + param_dict['cm_solver_pre_comp_droptol'] \
                    + '_pcs' + param_dict['cm_solver_pre_comp_sweeps'] \
                    + '_pcsai' + param_dict['cm_solver_pre_comp_sai_thres'] \
                    + '_pa' + param_dict['cm_solver_pre_app_type'] \
                    + '_paji' + param_dict['cm_solver_pre_app_jacobi_iters'] \
                    + '_t' + param_dict['threads']

                self._process_result(fout, param_dict, self.__min_step, self.__max_step)

    def _build_slurm_script(self, binary, param_values):
        from os import path

        # back up two directory levels
        base_dir = path.dirname(path.dirname(path.dirname(binary)))

        job_script = """\
#!/bin/bash -login

#SBATCH --time=03:59:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --constraint=lac

module purge
module load GCC/7.3.0-2.30

cd ${{SLURM_SUBMIT_DIR}}

python3 {0}/tools/run_sim.py run_md \\
    -b {1} \\\
""".format(base_dir, binary)

        for (k, v) in zip(self.__param_names, param_values):
            job_script += "\n    -p {0} {1} \\".format(k, v)

        job_script += "\n    {0}".format(self.__data_set)

        return job_script

    def _build_pbs_script(self, binary, param_values):
        from os import path

        # back up two directory levels
        base_dir = path.dirname(path.dirname(path.dirname(binary)))

        job_script = """\
#!/bin/bash -login

#PBS -l walltime=03:59:00,nodes=1:ppn=28,mem=120gb,feature=lac

module purge
module load GCC/7.3.0-2.30

cd ${{PBS_O_WORKDIR}}

python3 {0}/tools/run_sim.py run_md \\
    -b {1} \\\
""".format(base_dir, binary)

        for (k, v) in zip(self.__param_names, param_values):
            job_script += "\n    -p {0} {1} \\".format(k, v)

        job_script += "\n    {0}".format(self.__data_set)

        return job_script

    def submit_jobs(self, binary, job_script_type):
        from itertools import product
        from subprocess import Popen, PIPE

        for p in product(*[self.__params[k] for k in self.__param_names]):
            if job_script_type == 'slurm':
                job_script = self._build_slurm_script(binary, p)

                cmd_args = ['sbatch']

            if job_script_type == 'pbs':
                job_script = self._build_pbs_script(binary, p)
                
                cmd_args = ['qsub']

            proc_handle = Popen(cmd_args, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
            stdout, stderr = proc_handle.communicate(job_script)

            if proc_handle.returncode < 0:
                print("WARNING: process terminated with code {0}".format(proc_handle.returncode))
                print('stdout:')
                print(stdout)
                print('stderr:')
                print(stderr)
            else:
                print(stdout)


if __name__ == '__main__':
    import argparse
    from os import getcwd, path
    from sys import exit

    def setup_parser():
        DATA = [ \
                'bilayer_56800', 'bilayer_340800', \
                'dna_19733', \
                'petn_48256', \
                'silica_6000', 'silica_72000', 'silica_300000', \
                'water_6540', 'water_78480', 'water_327000', \
                ]
        JOB_TYPES = ['pbs', 'slurm']

        parser = argparse.ArgumentParser(description='Molecular dynamics simulation tools used with specified data sets.')
        subparsers = parser.add_subparsers(help="Actions types.")
        run_md_parser = subparsers.add_parser("run_md")
        parse_results_parser = subparsers.add_parser("parse_results")
        submit_jobs_parser = subparsers.add_parser("submit_jobs")

        run_md_parser.add_argument('-b', '--binary', metavar='binary', default=None, nargs=1,
                help='Binary file to run.')
        run_md_parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
                help='Paramater name and value pairs for the simulation, with multiple values comma delimited.')
        run_md_parser.add_argument('data_sets', nargs='+',
                choices=DATA, help='Data sets for which to run simulations.')
        run_md_parser.set_defaults(func=run_md)

        parse_results_parser.add_argument('-f', '--out_file', metavar='out_file', default=None, nargs=1,
                help='Output file to write results.')
        parse_results_parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
                help='Paramater name and value pairs for the simulation, with multiple values comma delimited.')
        parse_results_parser.add_argument('-n', '--min_step', metavar='min_step', default=None, nargs=1,
                help='Minimum simulation step to begin aggregating results.')
        parse_results_parser.add_argument('-x', '--max_step', metavar='max_step', default=None, nargs=1,
                help='Maxiumum simulation step for aggregating results.')
        parse_results_parser.add_argument('data_sets', nargs='+',
                choices=DATA, help='Data sets for which to parse results.')
        parse_results_parser.set_defaults(func=parse_results)

        submit_jobs_parser.add_argument('-b', '--binary', metavar='binary', default=None, nargs=1,
                help='Binary file to run.')
        submit_jobs_parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
                help='Paramater name and value pairs for the simulation, with multiple values comma delimited.')
        submit_jobs_parser.add_argument('job_script_type', nargs=1,
                choices=JOB_TYPES, help='Type of job script.')
        submit_jobs_parser.add_argument('data_sets', nargs='+',
                choices=DATA, help='Data sets for which to run simulations.')
        submit_jobs_parser.set_defaults(func=submit_jobs)

        return parser

    def setup_defaults(base_dir):
        control_dir = path.join(base_dir, 'environ')
        data_dir = path.join(base_dir, 'data/benchmarks')
        control_params_dict = {
                'ensemble_type': ['0'],
                'nsteps': ['20'],
                'tabulate_long_range': ['0'],
                'charge_method': ['0'],
                'cm_q_net': ['0.0'],
                'cm_solver_type': ['0'],
                'cm_solver_max_iters': ['20'],
                'cm_solver_restart': ['100'],
                'cm_solver_q_err': ['1e-6'],
                'cm_domain_sparsity': ['1.0'],
                'cm_init_guess_extrap1': ['3'],
                'cm_init_guess_extrap2': ['2'],
                'cm_solver_pre_comp_type': ['1'],
                'cm_solver_pre_comp_refactor': ['100'],
                'cm_solver_pre_comp_droptol': ['0.0'],
                'cm_solver_pre_comp_sweeps': ['3'],
                'cm_solver_pre_comp_sai_thres': ['0.1'],
                'cm_solver_pre_app_type': ['0'],
                'cm_solver_pre_app_jacobi_iters': ['30'],
                'threads': ['1'],
                'geo_format': [],
        }
        return control_dir, data_dir, control_params_dict

    def setup_test_cases(data_sets, data_dir, control_dir, control_params, header_fmt_str=None,
            header_str=None, body_fmt_str=None, result_file='result.txt', min_step=None, max_step=None):
        test_cases = []

        if 'water_6540' in data_sets:
            test_cases.append(
                TestCase('water_6540',
                    path.join(data_dir, 'water/water_6540.pdb'),
                    path.join(data_dir, 'water/ffield_acks2.water'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'water_78480' in data_sets:
            test_cases.append('water_78480',
                TestCase(path.join(data_dir, 'water/water_78480.geo'),
                    path.join(data_dir, 'water/ffield_acks2.water'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'water_327000' in data_sets:
            test_cases.append(
                TestCase('water_327000',
                    path.join(data_dir, 'water/water_327000.geo'),
                    path.join(data_dir, 'water/ffield_acks2.water'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'bilayer_56800' in data_sets:
            test_cases.append(
                TestCase('bilayer_56800',
                    path.join(data_dir, 'bilayer/bilayer_56800.pdb'),
                    path.join(data_dir, 'bilayer/ffield-bio'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'bilayer_340800' in data_sets:
            test_cases.append(
                TestCase('bilayer_340800',
                    path.join(data_dir, 'bilayer/bilayer_340800.geo'),
                    path.join(data_dir, 'bilayer/ffield-bio'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'dna_19733' in data_sets:
            test_cases.append(
                TestCase('dna_19733',
                    path.join(data_dir, 'dna/dna_19733.pdb'),
                    path.join(data_dir, 'dna/ffield-dna'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'silica_6000' in data_sets:
            test_cases.append(
                TestCase('silica_6000',
                    path.join(data_dir, 'silica/silica_6000.pdb'),
                    path.join(data_dir, 'silica/ffield-bio'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'silica_72000' in data_sets:
            test_cases.append(
                TestCase('silica_72000',
                    path.join(data_dir, 'silica/silica_72000.geo'),
                    path.join(data_dir, 'silica/ffield-bio'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'silica_300000' in data_sets:
            test_cases.append(
                TestCase('silica_300000',
                    path.join(data_dir, 'silica/silica_300000.geo'),
                    path.join(data_dir, 'silica/ffield-bio'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'petn_48256' in data_sets:
            test_cases.append(
                TestCase('petn_48256',
                    path.join(data_dir, 'petn/petn_48256.pdb'),
                    path.join(data_dir, 'petn/ffield.petn'),
                    path.join(control_dir, 'param.gpu.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))

        return test_cases

    def run_md(args):
        base_dir = getcwd()
        control_dir, data_dir, control_params_dict = setup_defaults(base_dir)

        if args.binary:
            binary = args.binary[0]
        else:
            binary = path.join(base_dir, 'sPuReMD/bin/spuremd')

        # overwrite default control file parameter values if supplied via command line args
        if args.params:
            for param in args.params:
                if param[0] in control_params_dict:
                    control_params_dict[param[0]] = param[1].split(',')
                else:
                    print("ERROR: Invalid parameter {0}. Terminating...".format(param[0]))
                    exit(-1)

        test_cases = setup_test_cases(args.data_sets, data_dir, control_dir, control_params_dict)

        for test in test_cases:
            test.run(binary)

    def parse_results(args):
        header_fmt_str = '{:15}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:10}|{:10}|{:10}|{:10}|{:10}|{:3}|{:10}\n'
        header_str = ['Data Set', 'Steps', 'CM', 'Solvr', 'Q Tol', 'QDS', 'PreCT', 'PreCD', 'PreCS', 'PCSAI', 'PreAT', 'PreAJ', 'Pre Comp',
                'Pre App', 'Iters', 'SpMV', 'CM', 'Thd', 'Time (s)']
        body_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:3} {:10.3f}\n'

        base_dir = getcwd()
        control_dir, data_dir, control_params_dict = setup_defaults(base_dir)

        if args.out_file:
            result_file = args.out_file[0]
        else:
            result_file = 'result.txt'

        if args.min_step:
            min_step = int(args.min_step[0])
        else:
            min_step = None

        if args.max_step:
            max_step = int(args.max_step[0])
        else:
            max_step = None

        test_cases = setup_test_cases(args.data_sets, data_dir, control_dir, control_params_dict,
                header_fmt_str=header_fmt_str, header_str=header_str, body_fmt_str=body_fmt_str,
                result_file=result_file, min_step=min_step, max_step=max_step)

        for test in test_cases:
            test.parse_results()

    def submit_jobs(args):
        base_dir = getcwd()
        control_dir, data_dir, control_params_dict = setup_defaults(base_dir)

        if args.binary:
            binary = args.binary[0]
        else:
            binary = path.join(base_dir, 'sPuReMD/bin/spuremd')

        # overwrite default control file parameter values if supplied via command line args
        if args.params:
            for param in args.params:
                if param[0] in control_params_dict:
                    control_params_dict[param[0]] = param[1].split(',')
                else:
                    print("ERROR: Invalid parameter {0}. Terminating...".format(param[0]))
                    exit(-1)

        test_cases = setup_test_cases(args.data_sets, data_dir, control_dir, control_params_dict)

        for test in test_cases:
            test.submit_jobs(binary, args.job_script_type[0])

    parser = setup_parser()
    args = parser.parse_args()
    args.func(args)
