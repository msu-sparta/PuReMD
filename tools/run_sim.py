#!/usr/bin/env python3
# coding=utf-8


class TestCase():
    def __init__(self, data_set, geo_file, ffield_file, params={}, result_header_fmt='',
            result_header='', result_body_fmt='', result_file='results.txt', geo_format='1',
            min_step=None, max_step=None):
        from re import sub

        self.__data_set = data_set
        self.__geo_file = geo_file
        self.__ffield_file = ffield_file
        self.__param_names = sorted(params.keys())
        self.__params = params
        self.__result_header_fmt = result_header_fmt
        self.__result_header = result_header
        self.__result_body_fmt = result_body_fmt
        self.__result_file = result_file
        self.__control_regexes = { \
                'name': lambda l, x: sub(
                    r'(?P<key>\bsimulation_name\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'ensemble_type': lambda l, x: sub(
                    r'(?P<key>\bensemble_type\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'nsteps': lambda l, x: sub(
                    r'(?P<key>\bnsteps\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'dt': lambda l, x: sub(
                    r'(?P<key>\bdt\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'gpus_per_node': lambda l, x: sub(
                    r'(?P<key>\bgpus_per_node\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'proc_by_dim': lambda l, x: sub(
                    r'(?P<key>\bproc_by_dim\b\s+)\S+\s+\S+\s+\S+(?P<comment>.*)',
                    r'\g<key>{0} {1} {2}\g<comment>'.format(*(x.split(':'))), l), \
                'periodic_boundaries': lambda l, x: sub(
                    r'(?P<key>\bperiodic_boundaries\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'reposition_atoms': lambda l, x: sub(
                    r'(?P<key>\breposition_atoms\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'remove_CoM_vel': lambda l, x: sub(
                    r'(?P<key>\bremove_CoM_vel\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'reneighbor': lambda l, x: sub(
                    r'(?P<key>\breneighbor\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'energy_update_freq': lambda l, x: sub(
                    r'(?P<key>\benergy_update_freq\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'tabulate_long_range': lambda l, x: sub(
                    r'(?P<key>\btabulate_long_range\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'vlist_buffer': lambda l, x: sub(
                    r'(?P<key>\bvlist_buffer\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'nbrhood_cutoff': lambda l, x: sub(
                    r'(?P<key>\bnbrhood_cutoff\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'hbond_cutoff': lambda l, x: sub(
                    r'(?P<key>\bhbond_cutoff\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'thb_cutoff': lambda l, x: sub(
                    r'(?P<key>\bthb_cutoff\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'bond_graph_cutoff': lambda l, x: sub(
                    r'(?P<key>\bbond_graph_cutoff\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'charge_method': lambda l, x: sub(
                    r'(?P<key>\bcharge_method\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_q_net': lambda l, x: sub(
                    r'(?P<key>\bcm_q_net\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_type': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_type\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_max_iters': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_max_iters\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_restart': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_restart\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_q_err': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_q_err\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_domain_sparsity': lambda l, x: sub(
                    r'(?P<key>\bcm_domain_sparsity\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_init_guess_extrap1': lambda l, x: sub(
                    r'(?P<key>\bcm_init_guess_extrap1\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_init_guess_extrap2': lambda l, x: sub(
                    r'(?P<key>\bcm_init_guess_extrap2\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_type': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_pre_comp_type\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_droptol': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_pre_comp_droptol\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_refactor': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_pre_comp_refactor\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_sweeps': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_pre_comp_sweeps\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_sai_thres': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_pre_comp_sai_thres\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_app_type': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_pre_app_type\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_app_jacobi_iters': lambda l, x: sub(
                    r'(?P<key>\bcm_solver_pre_app_jacobi_iters\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'geo_format': lambda l, x: sub(
                    r'(?P<key>\bgeo_format\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'temp_init': lambda l, x: sub(
                    r'(?P<key>\btemp_init\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'temp_final': lambda l, x: sub(
                    r'(?P<key>\btemp_final\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                't_mass': lambda l, x: sub(
                    r'(?P<key>\bt_mass\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                't_mode': lambda l, x: sub(
                    r'(?P<key>\bt_mode\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                't_rate': lambda l, x: sub(
                    r'(?P<key>\bt_rate\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                't_freq': lambda l, x: sub(
                    r'(?P<key>\bt_freq\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'pressure': lambda l, x: sub(
                    r'(?P<key>\bpressure\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'p_mass': lambda l, x: sub(
                    r'(?P<key>\bp_mass\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'pt_mass': lambda l, x: sub(
                    r'(?P<key>\bpt_mass\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'compress': lambda l, x: sub(
                    r'(?P<key>\bcompress\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'press_mode': lambda l, x: sub(
                    r'(?P<key>\bpress_mode\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'write_freq': lambda l, x: sub(
                    r'(?P<key>\bwrite_freq\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'traj_compress': lambda l, x: sub(
                    r'(?P<key>\btraj_compress\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'traj_format': lambda l, x: sub(
                    r'(?P<key>\btraj_format\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'traj_title': lambda l, x: sub(
                    r'(?P<key>\btraj_title\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'restart_format': lambda l, x: sub(
                    r'(?P<key>\brestart_format\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'restart_freq': lambda l, x: sub(
                    r'(?P<key>\brestart_freq\b\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
        }
        self.__params['geo_format'] = geo_format
        self.__min_step = min_step
        self.__max_step = max_step

    def _create_control_file(self, param, new_control_file):
        # inline template control file
        lines = """\
simulation_name         water_6540_notab_qeq           ! output files will carry this name + their specific extension
ensemble_type           0                       ! 0: NVE, 1: Berendsen NVT, 2: nose-Hoover NVT, 3: semi-isotropic NPT, 4: isotropic NPT, 5: anisotropic NPT
nsteps                  100                     ! number of simulation steps
dt                      0.25                    ! time step in fs
proc_by_dim             1 1 1                   ! distribution of processors by dimensions
gpus_per_node           1                       ! GPUs per node
periodic_boundaries     1                       ! 0: no periodic boundaries, 1: periodic boundaries

reposition_atoms        0                       ! 0: just fit to periodic boundaries, 1: CoM to the center of box, 3: CoM to the origin
remove_CoM_vel          500                     ! remove the translational and rotational vel around the center of mass at every 'this many' steps
reneighbor              1                       ! frequency to recompute Verlet lists
restrict_bonds          0                       ! enforce the bonds given in CONECT lines of pdb file for this many steps
energy_update_freq      1
tabulate_long_range     0                       ! denotes the granularity of long range tabulation, 0 means no tabulation

vlist_buffer            0.0                     ! skim distance on top of non-bonded interaction cutoff (Angstroms)
nbrhood_cutoff          5.0                     ! cutoff distance for bond interaction (Angstroms)
hbond_cutoff            7.5                     ! cutoff distance for hydrogen bond interactions (Angstroms)
thb_cutoff              0.001                   ! cutoff value for three body interactions (Angstroms)
bond_graph_cutoff       0.3                     ! bond strength cutoff for bond graphs (Angstroms)

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
freq_dipole_anal        0                       ! calculate electric dipole moment at every 'this many' steps
diffusion_coef          0                       ! 1: calculate diffusion coefficient of the system
freq_diffusion_coef     0                       ! calculate diffusion coefficient at every 'this many' steps
restrict_type           2                       ! -1: all types of atoms, 0 and up: only this type of atoms

restart_format          1                       ! 0: restarts in ASCII  1: restarts in binary
restart_freq            0                       ! 0: do not output any restart files. >0: output a restart file at every 'this many' steps\
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

    def _create_md_cmd(self, binary, control_file, run_type, mpi_cmd, mpi_cmd_extra, param_dict, env):
        from operator import mul
        from functools import reduce
        from sys import exit

        if run_type == 'openmp':
            env['OMP_NUM_THREADS'] = param_dict['threads']

        if run_type == 'serial' or run_type == 'openmp':
            cmd_args = binary.split() + [
                self.__geo_file,
                self.__ffield_file,
                control_file,
            ]
        # add MPI execution command and arguments to subprocess argument list
        elif run_type == 'mpi' or run_type == 'mpi-gpu':
            if mpi_cmd[0] == 'mpirun':
                cmd_args = [
                    'mpirun',
                    '-np',
                    # total number of MPI processes
                    str(reduce(mul,
                        map(int, param_dict['proc_by_dim'].split(':')), 1)),
                ] + binary.split() + [
                    self.__geo_file,
                    self.__ffield_file,
                    control_file,
                ]
            elif mpi_cmd[0] == 'srun':
                # slurm scheduler wraps MPI commands (e.g., NERSC)
                cmd_args = [
                    'srun',
                    '--nodes',
                    # number of nodes
                    mpi_cmd[1],
                    '--ntasks',
                    # number of tasks
                    mpi_cmd[2],
                    # number of tasks per node
                    '--tasks-per-node',
                    mpi_cmd[3],
                    # number of cores per task
                    '--cpus-per-task',
                    mpi_cmd[4],
                ] + mpi_cmd_extra[0].split() + [
                ] + binary.split() + [
                    self.__geo_file,
                    self.__ffield_file,
                    control_file,
                ]
            else:
                print("[ERROR] Invalid MPI application type ({0}). Terminating...".format(mpi_cmd[0]))
                exit(-1)

        return cmd_args, env

    def _create_output_file_base(self, run_type, param_dict):
        if run_type == 'serial' or run_type == 'openmp':
            name = path.basename(self.__geo_file).split('.')[0] \
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
        elif run_type == 'mpi' or run_type == 'mpi-gpu':
            name = path.basename(self.__geo_file).split('.')[0] \
                + '_cm' + param_dict['charge_method'] \
                + '_s' + param_dict['nsteps'] \
                + '_proc' + param_dict['proc_by_dim'].replace(':', '_') \
                + '_ren' + param_dict['reneighbor'] \
                + '_skind' + param_dict['vlist_buffer'] \
                + '_q' + param_dict['cm_solver_type'] \
                + '_qtol' + param_dict['cm_solver_q_err'] \
                + '_qds' + param_dict['cm_domain_sparsity'] \
                + '_pc' + param_dict['cm_solver_pre_comp_type'] \
                + '_pcr' + param_dict['cm_solver_pre_comp_refactor'] \
                + '_pctol' + param_dict['cm_solver_pre_comp_droptol'] \
                + '_pcs' + param_dict['cm_solver_pre_comp_sweeps'] \
                + '_pcsai' + param_dict['cm_solver_pre_comp_sai_thres'] \
                + '_pa' + param_dict['cm_solver_pre_app_type'] \
                + '_paji'+ str(param_dict['cm_solver_pre_app_jacobi_iters'])
        else:
            name = 'default_sim'

        return name

    def run_md(self, binary, run_type, mpi_cmd, mpi_cmd_extra):
        from itertools import product
        from os import environ, path, remove, rmdir
        from subprocess import Popen, PIPE
        from tempfile import mkdtemp
        from time import time

        env = dict(environ)
        temp_dir = mkdtemp(dir=getcwd())

        # create Cartesian product of all supplied sets of parameter values
        for p in product(*[self.__params[k] for k in self.__param_names]):
            param_dict = dict((k, v) for (k, v) in zip(self.__param_names, p))
            param_dict['name'] = self._create_output_file_base(run_type, param_dict)
            if not param_dict['traj_title']:
                param_dict['traj_title'] = param_dict['name'] + '.trj'

            temp_file = path.join(temp_dir, 'control')
            self._create_control_file(param_dict, temp_file)

            cmd_args, env = self._create_md_cmd(binary, temp_file, run_type,
                    mpi_cmd, mpi_cmd_extra, param_dict, env)

            start = time()
            proc_handle = Popen(cmd_args, stdout=PIPE, stderr=PIPE, env=env, universal_newlines=True)
            stdout, stderr = proc_handle.communicate()
            stop = time()

            if proc_handle.returncode < 0:
                print("[WARNING] process terminated with code {0}".format(proc_handle.returncode))

            print('stdout:\n{0}'.format(stdout), end='')
            print('stderr:\n{0}'.format(stderr), end='')

            if path.exists(temp_file):
                remove(temp_file)

        if path.exists(temp_dir):
            rmdir(temp_dir)

    def _process_result(self, fout, param, min_step, max_step, freq_step, run_type):
        if run_type == 'serial' or run_type == 'openmp':
            total_time = 0.
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
                print('[WARNING] {0} does not exist!'.format(log_file))
                return
            with open(log_file, 'r') as fp:
                for line in fp:
                    line = line.split()
                    try:
                        _cm = float(line[6])
                        _iters = float(line[8])
                        _pre_comp = float(line[9])
                        _pre_app = float(line[10])
                        _spmv = float(line[11])

                        if (not min_step and not max_step) or \
                        (min_step and not max_step and cnt_valid >= min_step) or \
                        (not min_step and max_step and cnt_valid <= max_step) or \
                        (cnt_valid >= min_step and cnt_valid <= max_step):
                            cm = cm + _cm
                            iters = iters + _iters
                            pre_comp = pre_comp + _pre_comp
                            pre_app = pre_app + _pre_app
                            spmv = spmv + _spmv

                            cnt = cnt + 1

                        cnt_valid = cnt_valid + 1
                    except Exception:
                        pass
                    if line[0] == 'total:':
                        try:
                            total_time = float(line[1])
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
            if (line_cnt - 3) == (int(param['nsteps']) / freq_step):
                fout.write(self.__result_body_fmt.format(path.basename(self.__geo_file).split('.')[0], 
                    param['nsteps'], param['charge_method'], param['cm_solver_type'],
                    param['cm_solver_q_err'], param['cm_domain_sparsity'],
                    param['cm_solver_pre_comp_type'], param['cm_solver_pre_comp_droptol'],
                    param['cm_solver_pre_comp_sweeps'], param['cm_solver_pre_comp_sai_thres'],
                    param['cm_solver_pre_app_type'], param['cm_solver_pre_app_jacobi_iters'],
                    pre_comp, pre_app, iters, spmv,
                    cm, param['threads'], total_time))
            else:
                print('[WARNING] nsteps not correct in file {0} (nsteps = {1:d}, step freq = {2:d}, counted steps = {3:d}).'.format(
                    log_file, int(param['nsteps']), freq_step, max(line_cnt - 3, 0)))
            fout.flush()
        elif run_type == 'mpi':
            from operator import mul
            from functools import reduce
            
            total_time = 0.0
            step_time = 0.0
            comm = 0.0
            neighbors = 0.0
            init = 0.0
            init_dist = 0.0
            init_cm = 0.0
            init_bond = 0.0
            bonded = 0.0
            nonbonded = 0.0
            cm = 0.0
            cm_sort = 0.0
            s_iters = 0.0
            pre_comp = 0.0
            pre_app = 0.0
            s_comm = 0.0
            s_allr = 0.0
            s_spmv = 0.0
            s_vec_ops = 0.0
            cnt = 0
            cnt_valid = 0
            line_cnt = 0
            log_file = param['name'] + '.log'
            out_file = param['name'] + '.out'

            if not path.exists(log_file):
                print('[WARNING] {0} does not exist!'.format(log_file))
                return
            with open(log_file, 'r') as fp:
                for line in fp:
                    line = line.split()
                    try:
                        _step_time = float(line[1])
                        _comm = float(line[2])
                        _neighbors = float(line[3])
                        _init = float(line[4])
                        _init_dist = float(line[5])
                        _init_cm = float(line[6])
                        _init_bond = float(line[7])
                        _bonded = float(line[8])
                        _nonbonded = float(line[9])
                        _cm = float(line[10])
                        _cm_sort = float(line[11])
                        _s_iters = float(line[12])
                        _pre_comp = float(line[13])
                        _pre_app = float(line[14])
                        _s_comm = float(line[15])
                        _s_allr = float(line[16])
                        _s_spmv = float(line[17])
                        _s_vec_ops = float(line[18])

                        if (not min_step and not max_step) or \
                        (min_step and not max_step and cnt_valid >= min_step) or \
                        (not min_step and max_step and cnt_valid <= max_step) or \
                        (cnt_valid >= min_step and cnt_valid <= max_step):
                            step_time = step_time + _step_time
                            comm = comm + _comm
                            neighbors = neighbors + _neighbors
                            init = init + _init
                            init_dist = init_dist + _init_dist
                            init_cm = init_cm + _init_cm
                            init_bond = init_bond + _init_bond
                            bonded = bonded + _bonded
                            nonbonded = nonbonded + _nonbonded
                            cm = cm + _cm
                            cm_sort = cm_sort + _cm_sort
                            s_iters = s_iters + _s_iters
                            pre_comp = pre_comp + _pre_comp
                            pre_app = pre_app + _pre_app
                            s_comm = s_comm + _s_comm
                            s_allr = s_allr + _s_allr
                            s_spmv = s_spmv + _s_spmv
                            s_vec_ops = s_vec_ops + _s_vec_ops

                            cnt = cnt + 1

                        cnt_valid = cnt_valid + 1
                    except Exception:
                        pass
                    line_cnt = line_cnt + 1

            if cnt > 0:
                step_time = step_time / cnt
                comm = comm / cnt
                neighbors = neighbors / cnt
                init = init / cnt
                init_dist = init_dist / cnt
                init_cm = init_cm / cnt
                init_bond = init_bond / cnt
                bonded = bonded / cnt
                nonbonded = nonbonded / cnt
                cm = cm / cnt
                cm_sort = cm_sort / cnt
                s_iters = s_iters / cnt
                pre_comp = pre_comp / cnt
                pre_app = pre_app / cnt
                s_comm = s_comm / cnt
                s_allr = s_allr / cnt
                s_spmv = s_spmv / cnt
                s_vec_ops = s_vec_ops / cnt

            if not path.exists(out_file):
                print('[WARNING] {0} does not exist!'.format(out_file))
                return
            with open(out_file, 'r') as fp:
                for line in fp:
                    line = line.split()
                    if line[0] == 'Total' and line[1] == 'Simulation' and line[2] == 'Time:':
                        try:
                            total_time = float(line[3])
                        except Exception:
                            pass

            # subtract for header and extra step
            # (e.g., 100 steps means steps 0 through 100, inclusive)
            if (line_cnt - 2) == (int(param['nsteps']) / freq_step):
                fout.write(self.__result_body_fmt.format(path.basename(self.__geo_file).split('.')[0],
                    str(reduce(mul, map(int, param['proc_by_dim'].split(':')), 1)),
                    param['nsteps'], param['cm_solver_pre_comp_type'],
                    param['cm_solver_q_err'],
                    param['reneighbor'],
                    param['cm_solver_pre_comp_sai_thres'],
                    total_time, step_time, comm, neighbors, init, init_dist, init_cm, init_bond,
                    bonded, nonbonded, cm, cm_sort,
                    s_iters, pre_comp, pre_app, s_comm, s_allr, s_spmv, s_vec_ops))
            else:
                print('[WARNING] nsteps not correct in file {0} (nsteps = {1:d}, step freq = {2:d}, counted steps = {3:d}).'.format(
                    log_file, int(param['nsteps']), freq_step, max(line_cnt - 3, 0)))
            fout.flush()
        elif run_type == 'mpi-gpu':
            from operator import mul
            from functools import reduce
            
            total_time = 0.0
            step_time = 0.0
            comm = 0.0
            neighbors = 0.0
            init = 0.0
            bonded = 0.0
            nonbonded = 0.0
            cm = 0.0
            s_iters = 0.0
            cnt = 0
            cnt_valid = 0
            line_cnt = 0
            log_file = param['name'] + '.log'
            out_file = param['name'] + '.out'

            if not path.exists(log_file):
                print('[WARNING] {0} does not exist!'.format(log_file))
                return
            with open(log_file, 'r') as fp:
                for line in fp:
                    line = line.split()
                    try:
                        _step_time = float(line[1])
                        _comm = float(line[2])
                        _neighbors = float(line[3])
                        _init = float(line[4])
                        _bonded = float(line[5])
                        _nonbonded = float(line[6])
                        _cm = float(line[7])
                        _s_iters = float(line[8])

                        if (not min_step and not max_step) or \
                        (min_step and not max_step and cnt_valid >= min_step) or \
                        (not min_step and max_step and cnt_valid <= max_step) or \
                        (cnt_valid >= min_step and cnt_valid <= max_step):
                            step_time = step_time + _step_time
                            comm = comm + _comm
                            neighbors = neighbors + _neighbors
                            init = init + _init
                            bonded = bonded + _bonded
                            nonbonded = nonbonded + _nonbonded
                            cm = cm + _cm
                            s_iters = s_iters + _s_iters

                            cnt = cnt + 1

                        cnt_valid = cnt_valid + 1
                    except Exception:
                        pass
                    line_cnt = line_cnt + 1

            if cnt > 0:
                step_time = step_time / cnt
                comm = comm / cnt
                neighbors = neighbors / cnt
                init = init / cnt
                bonded = bonded / cnt
                nonbonded = nonbonded / cnt
                cm = cm / cnt
                s_iters = s_iters / cnt

            if not path.exists(out_file):
                print('[WARNING] {0} does not exist!'.format(out_file))
                return
            with open(out_file, 'r') as fp:
                for line in fp:
                    line = line.split()
                    if line[0] == 'Total' and line[1] == 'Simulation' and line[2] == 'Time:':
                        try:
                            total_time = float(line[3])
                        except Exception:
                            pass

            # subtract for header and extra step
            # (e.g., 100 steps means steps 0 through 100, inclusive)
            if (line_cnt - 2) == (int(param['nsteps']) / freq_step):
                fout.write(self.__result_body_fmt.format(path.basename(self.__geo_file).split('.')[0],
                    str(reduce(mul, map(int, param['proc_by_dim'].split(':')), 1)),
                    param['nsteps'], param['cm_solver_pre_comp_type'],
                    param['cm_solver_q_err'],
                    param['reneighbor'],
                    param['cm_solver_pre_comp_sai_thres'],
                    total_time, step_time, comm, neighbors, init,
                    bonded, nonbonded, cm, s_iters))
            else:
                print('[WARNING] nsteps not correct in file {0} (nsteps = {1:d}, step freq = {2:d}, counted steps = {3:d}).'.format(
                    log_file, int(param['nsteps']), freq_step, max(line_cnt - 3, 0)))
            fout.flush()

    def parse_results(self, run_type):
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
                param_dict['name'] = self._create_output_file_base(run_type, param_dict)

                self._process_result(fout, param_dict,
                        self.__min_step, self.__max_step,
                        int(param_dict['energy_update_freq']), run_type)

    def _build_slurm_script(self, binary, run_type, mpi_cmd, mpi_cmd_extra, modules, param_values):
        from os import path

        # remove executable and back up two directory levels
        base_dir = path.dirname(path.dirname(path.dirname(path.abspath(binary))))

        job_script = """\
#!/bin/bash --login

#SBATCH --time=03:59:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --constraint=lac

module load {0}
""".format(' '.join(modules))
        job_script += """\

cd ${{SLURM_SUBMIT_DIR}}

python3 {0}/tools/run_sim.py run_md {1} \\
    -b {2} \\\
""".format(base_dir, run_type, binary)

        for (k, v) in zip(self.__param_names, param_values):
            job_script += "\n    -p {0} {1} \\".format(k, v)

        if run_type == 'mpi' or run_type == 'mpi-gpu':
            job_script += "\n    -m {0} \\".format(':'.join(mpi_cmd))
            job_script += "\n    -x {0} \\".format(mpi_cmd_extra[0])

        job_script += "\n    {0}".format(self.__data_set)

        return job_script

    def _build_pbs_script(self, binary, run_type, mpi_cmd, mpi_cmd_extra, modules, param_values):
        from os import path

        # remove executable and back up two directory levels
        base_dir = path.dirname(path.dirname(path.dirname(path.abspath(binary))))

        job_script = """\
#!/bin/bash --login

#PBS -l walltime=03:59:00,nodes=1:ppn=28,mem=120gb,feature=lac

module purge
module load {0}
""".format(' '.join(modules))
        job_script += """\

cd ${{PBS_O_WORKDIR}}

python3 {0}/tools/run_sim.py run_md {1} \\
    -b {2} \\\
""".format(base_dir, run_type, binary)

        for (k, v) in zip(self.__param_names, param_values):
            job_script += "\n    -p {0} {1} \\".format(k, v)

        if run_type == 'mpi' or run_type == 'mpi-gpu':
            job_script += "\n    -m {0} \\".format(':'.join(mpi_cmd))
            job_script += "\n    -x {0} \\".format(mpi_cmd_extra[0])

        job_script += "\n    {0}".format(self.__data_set)

        return job_script

    def submit_jobs(self, binary, run_type, job_script_type, mpi_cmd, mpi_cmd_extra, modules):
        from itertools import product
        from subprocess import Popen, PIPE

        for p in product(*[self.__params[k] for k in self.__param_names]):
            if job_script_type == 'slurm':
                job_script = self._build_slurm_script(binary, run_type, mpi_cmd, mpi_cmd_extra, modules, p)

                cmd_args = ['sbatch']

            if job_script_type == 'pbs':
                job_script = self._build_pbs_script(binary, run_type, mpi_cmd, mpi_cmd_extra, modules, p)
                
                cmd_args = ['qsub']

            proc_handle = Popen(cmd_args, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
            stdout, stderr = proc_handle.communicate(job_script)

            if proc_handle.returncode < 0:
                print("[WARNING] process terminated with code {0}".format(proc_handle.returncode))
                print('stdout:\n{0}'.format(stdout), end='')
                print('stderr:\n{0}'.format(stderr), end='')
            else:
                print(stdout, end='')


if __name__ == '__main__':
    import argparse
    from os import getcwd, path
    from sys import exit

    def setup_parser():
        DATA_SETS = [ \
                'bilayer_56800', 'bilayer_340800', \
                'dna_19733', \
                'petn_48256', \
                'silica_6000', 'silica_72000', 'silica_300000', \
                'water_6540', 'water_78480', 'water_327000', \
                'zno_6912', \
                ]
        JOB_TYPES = ['pbs', 'slurm']
        RUN_TYPES = ['serial', 'openmp', 'mpi', 'mpi-gpu']
        LOG_TYPES = ['out', 'pot', 'log']

        parser = argparse.ArgumentParser(description='Molecular dynamics simulation tools used with specified data sets.')
        subparsers = parser.add_subparsers(help="Actions.")
        run_md_parser = subparsers.add_parser("run_md")
        run_md_custom_parser = subparsers.add_parser("run_md_custom")
        parse_results_parser = subparsers.add_parser("parse_results")
        submit_jobs_parser = subparsers.add_parser("submit_jobs")
        compare_logs_parser = subparsers.add_parser("compare_logs")

        run_md_parser.add_argument('-b', '--binary', metavar='binary', default=None, nargs=1,
                help='Binary file used for running the MD simulation(s).')
        run_md_parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
                help='Paramater name and value pairs for the simulation as defined in the control file.'
                + ' Multiple values for a parameter can be specified using commas, and each'
                + ' value will constitute a separate MD simulation.')
        run_md_parser.add_argument('-m', '--mpi_cmd', metavar='mpi_cmd', default=['mpirun'], nargs=1,
                help='MPI command type and arguments. Examples: \'mpirun\', \'srun:1:32:32:1\' (nodes,tasks,tasks per node,cpus per task).')
        run_md_parser.add_argument('-x', '--mpi_cmd_extra', metavar='mpi_cmd_extra', default=[''], nargs=1,
                help='MPI command extra arguments.')
        run_md_parser.add_argument('run_type', nargs=1,
                choices=RUN_TYPES, help='Run type for the MD simulation(s).')
        run_md_parser.add_argument('data_sets', nargs='+',
                choices=DATA_SETS, help='Data set(s) used for the MD simulation(s).')
        run_md_parser.set_defaults(func=run_md)

        run_md_custom_parser.add_argument('-b', '--binary', metavar='binary', default=None, nargs=1,
                help='Binary file used for running the MD simulation(s).')
        run_md_custom_parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
                help='Paramater name and value pairs for the simulation as defined in the control file.'
                + ' Multiple values for a parameter can be specified using commas, and each'
                + ' value will constitute a separate MD simulation.')
        run_md_custom_parser.add_argument('-m', '--mpi_cmd', metavar='mpi_cmd', default=['mpirun'], nargs=1,
                help='MPI command type and arguments. Examples: \'mpirun\', \'srun:1:32:32:1\' (nodes,tasks,tasks per node,cpus per task).')
        run_md_custom_parser.add_argument('-x', '--mpi_cmd_extra', metavar='mpi_cmd_extra', default=[''], nargs=1,
                help='MPI command extra arguments.')
        run_md_custom_parser.add_argument('run_type', nargs=1,
                choices=RUN_TYPES, help='Run type for the MD simulation(s).')
        run_md_custom_parser.add_argument('geo_file', nargs=1,
                help='Geometry file used for the MD simulation.')
        run_md_custom_parser.add_argument('ffield_file', nargs=1,
                help='Force field parameter file used for the MD simulation.')
        run_md_custom_parser.set_defaults(func=run_md_custom)

        parse_results_parser.add_argument('-b', '--binary', metavar='binary', default=None, nargs=1,
                help='Binary file to run.')
        parse_results_parser.add_argument('-f', '--out_file', metavar='out_file', default=None, nargs=1,
                help='Output file to write results.')
        parse_results_parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
                help='Paramater name and value pairs for the simulation, with multiple values comma delimited.')
        parse_results_parser.add_argument('-n', '--min_step', metavar='min_step', default=None, nargs=1,
                help='Minimum simulation step to begin aggregating results.')
        parse_results_parser.add_argument('-x', '--max_step', metavar='max_step', default=None, nargs=1,
                help='Maxiumum simulation step for aggregating results.')
        parse_results_parser.add_argument('run_type', nargs=1,
                choices=RUN_TYPES, help='Run type for the MD simulation(s).')
        parse_results_parser.add_argument('geo_file', nargs=1,
                help='Geometry file used for the MD simulation.')
        parse_results_parser.add_argument('ffield_file', nargs=1,
                help='Force field parameter file used for the MD simulation.')
        parse_results_parser.set_defaults(func=parse_results)

        submit_jobs_parser.add_argument('-b', '--binary', metavar='binary', default=None, nargs=1,
                help='Binary file to run.')
        submit_jobs_parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
                help='Paramater name and value pairs for the simulation, with multiple values comma delimited.')
        submit_jobs_parser.add_argument('-m', '--mpi_cmd', metavar='mpi_cmd', default=['mpirun'], nargs=1,
                help='MPI command type and arguments. Examples: \'mpirun\', \'srun:1:32:32:1\' (nodes,tasks,tasks per node,cpus per task).')
        submit_jobs_parser.add_argument('-x', '--mpi_cmd_extra', metavar='mpi_cmd_extra', default=[''], nargs=1,
                help='MPI command extra arguments.')
        submit_jobs_parser.add_argument('-l', '--modules', metavar='modules', default=['GCC/8.2.0-2.31.1'], nargs=1,
                help='Modules to load. Multiple values are separated by the \':\' character.'
                + ' Examples: \'GCC/8.2.0-2.31.1\', \'GCC/8.2.0-2.31.1:OpenMPI/3.1.3:imkl/2019.1.144\'.')
        submit_jobs_parser.add_argument('job_script_type', nargs=1,
                choices=JOB_TYPES, help='Type of job script.')
        submit_jobs_parser.add_argument('run_type', nargs=1,
                choices=RUN_TYPES, help='Run type for the MD simulation(s).')
        submit_jobs_parser.add_argument('data_sets', nargs='+',
                choices=DATA_SETS, help='Data set(s) for which to run MD simulation(s).')
        submit_jobs_parser.set_defaults(func=submit_jobs)

        compare_logs_parser.add_argument('-t', '--tol', metavar='tolerance', default=1.0e-6, nargs=1,
                help='Tolerance used for comparing the log files.')
        compare_logs_parser.add_argument('log_file_type', nargs=1,
                choices=LOG_TYPES, help='Log file type for the MD simulation(s).')
        compare_logs_parser.add_argument('ref_log_file', nargs=1,
                help='Reference log run type and file to compare against (colon-separated).')
        compare_logs_parser.add_argument('log_file', nargs='+',
                help='Log run type and file to compare (colon-separated).')
        compare_logs_parser.set_defaults(func=compare_logs)

        return parser

    def setup_defaults(base_dir):
        data_dir = path.join(base_dir, 'data/benchmarks')
        control_params_dict = {
                'ensemble_type': ['0'],
                'nsteps': ['20'],
                'dt': ['0.25'],
                'gpus_per_node': ['1'],
                'proc_by_dim': ['1:1:1'],
                'periodic_boundaries': ['1'],
                'reposition_atoms': ['0'],
                'remove_CoM_vel': ['500'],
                'reneighbor': ['1'],
                'energy_update_freq': ['1'],
                'tabulate_long_range': ['0'],
                'vlist_buffer': ['0.0'],
                'nbrhood_cutoff': ['5.0'],
                'hbond_cutoff': ['7.5'],
                'thb_cutoff': ['0.001'],
                'bond_graph_cutoff': ['0.3'],
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
                'temp_init': ['0.0'],
                'temp_final': ['300.0'],
                't_mass': ['0.166666'],
                't_mode': ['0'],
                't_rate': ['-100.0'],
                't_freq': ['4.0'],
                'pressure': ['0.000101325'],
                'p_mass': ['5000.0'],
                'pt_mass': ['5000.0'],
                'compress': ['0.008134'],
                'press_mode': ['0'],
                'write_freq': ['0'],
                'traj_format': ['0'],
                'traj_compress': ['0'],
                'traj_title': [None],
                'restart_format': ['0'],
                'restart_freq': ['0'],
        }
        return data_dir, control_params_dict

    def setup_test_cases(data_sets, data_dir, control_params, header_fmt_str=None,
            header_str=None, body_fmt_str=None, result_file='result.txt', min_step=None, max_step=None):
        test_cases = []

        if 'water_6540' in data_sets:
            test_cases.append(
                TestCase('water_6540',
                    path.join(data_dir, 'water/water_6540.pdb'),
                    path.join(data_dir, 'water/ffield_acks2.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'water_78480' in data_sets:
            test_cases.append(
                TestCase('water_78480',
                    path.join(data_dir, 'water/water_78480.geo'),
                    path.join(data_dir, 'water/ffield_acks2.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'water_327000' in data_sets:
            test_cases.append(
                TestCase('water_327000',
                    path.join(data_dir, 'water/water_327000.geo'),
                    path.join(data_dir, 'water/ffield_acks2.water'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'bilayer_56800' in data_sets:
            test_cases.append(
                TestCase('bilayer_56800',
                    path.join(data_dir, 'bilayer/bilayer_56800.pdb'),
                    path.join(data_dir, 'bilayer/ffield-bio'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'bilayer_340800' in data_sets:
            test_cases.append(
                TestCase('bilayer_340800',
                    path.join(data_dir, 'bilayer/bilayer_340800.geo'),
                    path.join(data_dir, 'bilayer/ffield-bio'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'dna_19733' in data_sets:
            test_cases.append(
                TestCase('dna_19733',
                    path.join(data_dir, 'dna/dna_19733.pdb'),
                    path.join(data_dir, 'dna/ffield-dna'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'silica_6000' in data_sets:
            test_cases.append(
                TestCase('silica_6000',
                    path.join(data_dir, 'silica/silica_6000.pdb'),
                    path.join(data_dir, 'silica/ffield-bio'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'silica_72000' in data_sets:
            test_cases.append(
                TestCase('silica_72000',
                    path.join(data_dir, 'silica/silica_72000.geo'),
                    path.join(data_dir, 'silica/ffield-bio'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'silica_300000' in data_sets:
            test_cases.append(
                TestCase('silica_300000',
                    path.join(data_dir, 'silica/silica_300000.geo'),
                    path.join(data_dir, 'silica/ffield-bio'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['0'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'petn_48256' in data_sets:
            test_cases.append(
                TestCase('petn_48256',
                    path.join(data_dir, 'petn/petn_48256.pdb'),
                    path.join(data_dir, 'petn/ffield.petn'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))
        if 'zno_6912' in data_sets:
            test_cases.append(
                TestCase('zno_6912',
                    path.join(data_dir, 'metal/zno_6912.pdb'),
                    path.join(data_dir, 'metal/ffield.zno'),
                    params=control_params, result_header_fmt=header_fmt_str,
                    result_header = header_str, result_body_fmt=body_fmt_str,
                    geo_format=['1'], result_file=result_file,
                    min_step=min_step, max_step=max_step))

        return test_cases

    def run_md(args):
        if args.binary:
            binary = args.binary[0].split()[-1]
            # remove executable and back up two directory levels
            base_dir = path.dirname(path.dirname(path.dirname(path.abspath(binary))))
        else:
            base_dir = getcwd()
            if args.run_type[0] == 'serial' or args.run_type[0] == 'openmp':
                binary = path.join(base_dir, 'sPuReMD/bin/spuremd')
            elif args.run_type[0] == 'mpi':
                binary = path.join(base_dir, 'PuReMD/bin/puremd')
            elif args.run_type[0] == 'mpi-gpu':
                binary = path.join(base_dir, 'PG-PuReMD/bin/pg-puremd')

        data_dir, control_params_dict = setup_defaults(base_dir)

        # overwrite default control file parameter values if supplied via command line args
        if args.params:
            for param in args.params:
                if param[0] in control_params_dict:
                    control_params_dict[param[0]] = param[1].split(',')
                else:
                    print("[ERROR] Invalid parameter {0}. Terminating...".format(param[0]))
                    exit(-1)

        test_cases = setup_test_cases(args.data_sets, data_dir, control_params_dict)

        for test in test_cases:
            test.run_md(binary, args.run_type[0], args.mpi_cmd[0].split(':'), args.mpi_cmd_extra)

    def run_md_custom(args):
        if args.binary:
            binary = args.binary[0]
            # remove executable and back up two directory levels
            base_dir = path.dirname(path.dirname(path.dirname(path.abspath(binary))))
        else:
            base_dir = getcwd()
            if args.run_type[0] == 'serial' or args.run_type[0] == 'openmp':
                binary = path.join(base_dir, 'sPuReMD/bin/spuremd')
            elif args.run_type[0] == 'mpi':
                binary = path.join(base_dir, 'PuReMD/bin/puremd')
            elif args.run_type[0] == 'mpi-gpu':
                binary = path.join(base_dir, 'PG-PuReMD/bin/pg-puremd')

        _, control_params_dict = setup_defaults(base_dir)

        # overwrite default control file parameter values if supplied via command line args
        geo_format = None
        if args.params:
            for param in args.params:
                if param[0] in control_params_dict:
                    control_params_dict[param[0]] = param[1].split(',')
                    if param[0] == 'geo_format':
                        geo_format = param[1].split(',')
                else:
                    print("[ERROR] Invalid parameter {0}. Terminating...".format(param[0]))
                    exit(-1)

        geo_base, geo_ext = path.splitext(args.geo_file[0])

        if not geo_format:
            # infer geometry file format by file extension
            if geo_ext.lower() == '.pdb':
                geo_format = ['1']
            elif geo_ext.lower() == '.geo':
                geo_format = ['0']
            else:
                print("[ERROR] unrecognized geometry format {0}. Terminating...".format(geo_ext))
                exit(-1)

        test_case = TestCase(geo_base, args.geo_file[0], args.ffield_file[0],
                params=control_params_dict, geo_format=geo_format)

        test_case.run_md(binary, args.run_type[0], args.mpi_cmd[0].split(':'), args.mpi_cmd_extra)

    def parse_results(args):
        if args.run_type[0] == 'serial' or args.run_type[0] == 'openmp':
            header_fmt_str = '{:15}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:10}|{:10}|{:10}|{:10}|{:10}|{:3}|{:10}\n'
            header_str = ['Data Set', 'Steps', 'CM', 'Solvr', 'Q_Tol', 'QDS', 'PreCT', 'PreCD', 'PreCS', 'PCSAI', 'PreAT', 'PreAJ', 'Pre_Comp',
                    'Pre_App', 'Iters', 'SpMV', 'CM', 'Thd', 'Time (s)']
            body_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:3} {:10.3f}\n'
        elif args.run_type[0] == 'mpi':
            header_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10}\n'
            header_str = ['Data_Set', 'Proc', 'Steps', 'PreCT', 'Q_Tol', 'Ren', 'PCSAI',
                    'total_time', 'step_time', 'comm', 'neighbors', 'init', 'init_dist', 'init_cm', 'init_bond',
                    'bonded', 'nonbonded', 'cm', 'cm_sort',
                    's_iters', 'pre_comm', 'pre_app', 's_comm', 's_allr', 's_spmv', 's_vec_ops']
            body_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}\n'
        elif args.run_type[0] == 'mpi-gpu':
            header_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10}\n'
            header_str = ['Data_Set', 'Proc', 'Steps', 'PreCT', 'Q_Tol', 'Ren', 'PCSAI',
                    'total_time', 'step_time', 'comm', 'neighbors', 'init',
                    'bonded', 'nonbonded', 'cm', 's_iters']
            body_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}\n'

        if args.binary:
            binary = args.binary[0]
            # remove executable and back up two directory levels
            base_dir = path.dirname(path.dirname(path.dirname(path.abspath(binary))))
        else:
            base_dir = getcwd()

        _, control_params_dict = setup_defaults(base_dir)

        # overwrite default control file parameter values if supplied via command line args
        geo_format = None
        if args.params:
            for param in args.params:
                if param[0] in control_params_dict:
                    control_params_dict[param[0]] = param[1].split(',')
                    if param[0] == 'geo_format':
                        geo_format = param[1].split(',')
                else:
                    print("[ERROR] Invalid parameter {0}. Terminating...".format(param[0]))
                    exit(-1)

        geo_base, geo_ext = path.splitext(args.geo_file[0])

        if not geo_format:
            # infer geometry file format by file extension
            if geo_ext.lower() == '.pdb':
                geo_format = ['1']
            elif geo_ext.lower() == '.geo':
                geo_format = ['0']
            else:
                print("[ERROR] unrecognized geometry format {0}. Terminating...".format(geo_ext))
                exit(-1)

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

        test_case = TestCase(geo_base, args.geo_file[0], args.ffield_file[0],
                params=control_params_dict, geo_format=geo_format,
                result_header_fmt=header_fmt_str, result_header=header_str, result_body_fmt=body_fmt_str,
                result_file=result_file, min_step=min_step, max_step=max_step)

        test_case.parse_results(args.run_type[0])

    def submit_jobs(args):
        if args.binary:
            binary = args.binary[0]
            # remove executable and back up two directory levels
            base_dir = path.dirname(path.dirname(path.dirname(path.abspath(binary))))
        else:
            base_dir = getcwd()
            if args.run_type[0] == 'serial' or args.run_type[0] == 'openmp':
                binary = path.join(base_dir, 'sPuReMD/bin/spuremd')
            elif args.run_type[0] == 'mpi':
                binary = path.join(base_dir, 'PuReMD/bin/puremd')
            elif args.run_type[0] == 'mpi-gpu':
                binary = path.join(base_dir, 'PG-PuReMD/bin/pg-puremd')

        data_dir, control_params_dict = setup_defaults(base_dir)

        # overwrite default control file parameter values if supplied via command line args
        if args.params:
            for param in args.params:
                if param[0] in control_params_dict:
                    control_params_dict[param[0]] = param[1].split(',')
                else:
                    print("[ERROR] Invalid parameter {0}. Terminating...".format(param[0]))
                    exit(-1)

        test_cases = setup_test_cases(args.data_sets, data_dir, control_params_dict)

        for test in test_cases:
            test.submit_jobs(binary, args.run_type[0], args.job_script_type[0],
                    args.mpi_cmd[0].split(':'), args.mpi_cmd_extra, args.modules[0].split(':'))

    def compare_logs(args):
        import numpy as np
        import pandas as pd

        def _file_len(file_name):
            i = -1
            with open(file_name) as f:
                for i, l in enumerate(f):
                    pass
            return i + 1

        def _read_file_by_type(file_name, file_type, run_type):
            if run_type == 'serial' or run_type == 'openmp':
                if file_type == 'out':
                     names=['Step', 'Total_Energy', 'Potential_Energy',
                            'Kinetic_Energy', 'Temperature', 'Target_Temperature',
                            'Volume', 'Pressure', 'Target_Pressure']
                     dtype={'Step': np.int,
                            'Total_Energy': np.float64, 'Potential_Energy': np.float64,
                            'Kinetic_Energy': np.float64, 'Temperature': np.float64,
                            'Target_Temperature': np.float64, 'Volume': np.float64,
                            'Pressure': np.float64, 'Target_Pressure': np.float64}
                     skiprows=[0]
                elif file_type == 'pot':
                     names=['Step', 'E_Bond', 'E_Atom',
                            'E_LP', 'E_Ang', 'E_Coa',
                            'E_HB', 'E_Tor', 'E_Conj',
                            'E_vdW', 'E_Coul', 'E_Pol']
                     dtype={'Step': np.int,
                            'E_Bond': np.float64, 'E_Atom': np.float64,
                            'E_LP': np.float64, 'E_Ang': np.float64,
                            'E_Coa': np.float64, 'E_HB': np.float64,
                            'E_Tor': np.float64, 'E_Conj': np.float64,
                            'E_vdW': np.float64, 'E_Coul': np.float64,
                            'E_Pol': np.float64}
                     skiprows=[0]
                elif file_type == 'log':
                     names=['Step', 'T_Total', 'T_Nbrs',
                            'T_Init', 'T_Bonded', 'T_Nonbonded',
                            'T_CM', 'T_CM_Sort', 'Solver_Iters',
                            'T_Pre_Comp', 'T_Pre_App', 'T_Solver_SpMV',
                            'T_Solver_Vec_Ops', 'T_Solver_Orthog',
                            'T_Solver_Tri_Solve']
                     dtype={'Step': np.int,
                            'T_Total': np.float64, 'T_Nbrs': np.float64,
                            'T_Init': np.float64, 'T_Bonded': np.float64,
                            'T_Nonbonded': np.float64, 'T_CM': np.float64,
                            'T_CM_Sort': np.float64, 'Solver_Iters': np.float64,
                            'T_Pre_Comp': np.float64, 'T_Pre_App': np.float64,
                            'T_Solver_SpMV': np.float64,
                            'T_Solver_Vec_Ops': np.float64, 'T_Solver_Orthog': np.float64,
                            'T_Solver_Tri_Solve': np.float64}
                     skiprows=[0, _file_len(file_name) - 1]
                else:
                    print("[ERROR] Invalid log file type {0}. Terminating...".format(file_type))
                    exit(-1)
            elif run_type == 'mpi' or run_type == 'mpi-gpu':
                if file_type == 'out':
                     names=['Step', 'Total_Energy', 'Potential_Energy',
                            'Kinetic_Energy', 'Temperature',
                            'Volume', 'Pressure']
                     dtype={'Step': np.int,
                            'Total_Energy': np.float64, 'Potential_Energy': np.float64,
                            'Kinetic_Energy': np.float64, 'Temperature': np.float64,
                            'Volume': np.float64, 'Pressure': np.float64}
                     skiprows=[0, _file_len(file_name) - 1]
                elif file_type == 'pot':
                     names=['Step', 'E_Bond', 'E_Atom',
                            'E_LP', 'E_Ang', 'E_Coa',
                            'E_HB', 'E_Tor', 'E_Conj',
                            'E_vdW', 'E_Coul', 'E_Pol']
                     dtype={'Step': np.int,
                            'E_Bond': np.float64, 'E_Atom': np.float64,
                            'E_LP': np.float64, 'E_Ang': np.float64,
                            'E_Coa': np.float64, 'E_HB': np.float64,
                            'E_Tor': np.float64, 'E_Conj': np.float64,
                            'E_vdW': np.float64, 'E_Coul': np.float64,
                            'E_Pol': np.float64}
                     skiprows=[0]
                elif file_type == 'log':
                     names=['Step', 'T_Total', 'T_Comm', 'T_Nbrs',
                            'T_Init', 'T_Init_Dist', 'T_Init_CM', 'T_Init_Bond',
                            'T_Bonded', 'T_Nonbonded',
                            'T_CM', 'T_CM_Sort', 'Solver_Iters',
                            'T_Pre_Comp', 'T_Pre_App', 
                            'T_Solver_Comm', 'T_Solver_Allr', 'T_Solver_SpMV',
                            'T_Solver_Vec_Ops', 'T_Solver_Orthog',
                            'T_Solver_Tri_Solve']
                     dtype={'Step': np.int,
                            'T_Total': np.float64, 'T_Comm': np.float64, 'T_Nbrs': np.float64,
                            'T_Init': np.float64, 'T_Init_Dist': np.float64,
                            'T_Init_CM': np.float64, 'T_Init_Bond': np.float64, 'T_Bonded': np.float64,
                            'T_Nonbonded': np.float64, 'T_CM': np.float64,
                            'T_CM_Sort': np.float64, 'Solver_Iters': np.float64,
                            'T_Pre_Comp': np.float64, 'T_Pre_App': np.float64,
                            'T_Solver_Comm': np.float64, 'T_Solver_Allr': np.float64,
                            'T_Solver_SpMV': np.float64,
                            'T_Solver_Vec_Ops': np.float64, 'T_Solver_Orthog': np.float64,
                            'T_Solver_Tri_Solve': np.float64}
                     skiprows=[0, _file_len(file_name) - 1]
                else:
                    print("[ERROR] Invalid log file type {0}. Terminating...".format(file_type))
                    exit(-1)
            else:
                print("[ERROR] Invalid run type {0}. Terminating...".format(run_type))
                exit(-1)

            try:
                df = pd.read_csv(file_name, sep='\s+', skiprows=skiprows, names=names, dtype=dtype)
            except Exception as e: 
                print("[ERROR] failed to parse log file {0}. Terminating...".format(file_name))
                print("    [INFO] {0}".format(str(e)))

            return df

        # read reference log
        run_type, ref_log_file = args.ref_log_file[0].split(':')
        df_ref = _read_file_by_type(ref_log_file, args.log_file_type[0], run_type)

        if df_ref.empty:
            print("[ERROR] detected empty log file {0}. Terminating...".format(args.ref_log_file[0]))
            exit(-1)

        # compare other logs against reference
        for log in args.log_file:
            run_type, log_file = log.split(':')
            df = _read_file_by_type(log_file, args.log_file_type[0], run_type)

            if df.empty:
                print("[ERROR] detected empty log file {0}. Terminating...".format(log))
                exit(-1)

            if not len(df_ref.index) == len(df.index):
                print("[ERROR] detected different number of records in log files ({0:d}, {1:d}). Terminating...".format(
                    len(df_ref.index), len(df.index)))
                exit(-1)

            if args.log_file_type[0] == 'out':
                for term in df_ref.columns.values.tolist()[1:4]:
                    abs_diff = np.absolute(df_ref[term].values - df[term].values)
                    max_diff = abs_diff.max()
                    argmax_diff = abs_diff.argmax()
                    print('{0}:'.format(term))
                    print('    Max. Diff.: {0:E}'.format(max_diff))
                    print('          Step: {0:d}'.format(df_ref['Step'].iat[argmax_diff]))
                print()
            elif args.log_file_type[0] == 'pot':
                for term in df_ref.columns.values.tolist()[1:]:
                    abs_diff = np.absolute(df_ref[term].values - df[term].values)
                    max_diff = abs_diff.max()
                    argmax_diff = abs_diff.argmax()
                    print('{0}:'.format(term))
                    print('    Max. Diff.: {0:E}'.format(max_diff))
                    print('          Step: {0:d}'.format(df_ref['Step'].iat[argmax_diff]))
                print()
            elif args.log_file_type[0] == 'log':
                abs_diff = np.absolute(df_ref['T_Total'].values - df['T_Total'].values)
                max_diff = abs_diff.max()
                argmax_diff = abs_diff.argmax()
                print('Max. Diff.: {0:E}'.format(max_diff))
                print('      Step: {0:d}'.format(df_ref['Step'].iat[argmax_diff]))
                print()


    parser = setup_parser()
    args = parser.parse_args()
    args.func(args)
