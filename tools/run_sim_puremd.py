#!/usr/bin/env python3

import argparse
from fileinput import input
from itertools import product
from re import sub
from subprocess import Popen, PIPE
from os import getcwd, environ, path, remove, rename, rmdir
from sys import exit
from tempfile import mkdtemp
from time import time


class TestCase():
    def __init__(self, geo_file, ffield_file, control_file, params={}, result_header_fmt='',
            result_header='', result_body_fmt='', result_file='results.txt', geo_format='1',
            min_step=None, max_step=None):
        self.__geo_file = geo_file
        self.__ffield_file = ffield_file
        self.__control_file = control_file
        self.__param_names = sorted(params.keys())
        self.__params = params
        self.__result_header_fmt = result_header_fmt
        self.__result_header = result_header
        self.__result_body_fmt = result_body_fmt
        self.__result_file = result_file
        self.__control_res = { \
                'name': lambda l, x: sub(
                    r'(?P<key>simulation_name\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'ensemble_type': lambda l, x: sub(
                    r'(?P<key>ensemble_type\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'nsteps': lambda l, x: sub(
                    r'(?P<key>nsteps\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'proc_by_dim1': lambda l, x: sub(
                    r'(?P<key>proc_by_dim\s+)\S+(?P<proc_by_dim2>\s+\S+)(?P<proc_by_dim3>\s+\S+)(?P<comment>.*)',
                    r'\g<key>{0}\g<proc_by_dim2>\g<proc_by_dim3>\g<comment>'.format(x), l), \
                'proc_by_dim2': lambda l, x: sub(
                    r'(?P<key>proc_by_dim\s+)(?P<proc_by_dim1>\S+\s+)\S+(?P<proc_by_dim3>\s+\S+)(?P<comment>.*)',
                    r'\g<key>\g<proc_by_dim1>{0}\g<proc_by_dim3>\g<comment>'.format(x), l), \
                'proc_by_dim3': lambda l, x: sub(
                    r'(?P<key>proc_by_dim\s+)(?P<proc_by_dim1>\S+\s+)(?P<proc_by_dim2>\S+\s+)\S+(?P<comment>.*)',
                    r'\g<key>\g<proc_by_dim1>\g<proc_by_dim2>{0}\g<comment>'.format(x), l), \
                'tabulate_long_range': lambda l, x: sub(
                    r'(?P<key>tabulate_long_range\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'reneighbor': lambda l, x: sub(
                    r'(?P<key>reneighbor\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'vlist_buffer': lambda l, x: sub(
                    r'(?P<key>vlist_buffer\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'charge_method': lambda l, x: sub(
                    r'(?P<key>charge_method\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_q_net': lambda l, x: sub(
                    r'(?P<key>cm_q_net\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_type\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_max_iters': lambda l, x: sub(
                    r'(?P<key>cm_solver_max_iters\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_restart': lambda l, x: sub(
                    r'(?P<key>cm_solver_restart\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_q_err': lambda l, x: sub(
                    r'(?P<key>cm_solver_q_err\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_domain_sparsity': lambda l, x: sub(
                    r'(?P<key>cm_domain_sparsity\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_init_guess_extrap1': lambda l, x: sub(
                    r'(?P<key>cm_init_guess_extrap1\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_init_guess_extrap2': lambda l, x: sub(
                    r'(?P<key>cm_init_guess_extrap2\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_pre_comp_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_type\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_pre_comp_droptol': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_droptol\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_pre_comp_refactor': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_refactor\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_pre_comp_sweeps': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_sweeps\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_pre_comp_sai_thres': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_sai_thres\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_pre_app_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_app_type\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'cm_solver_pre_app_jacobi_iters': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_app_jacobi_iters\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
                'geo_format': lambda l, x: sub(
                    r'(?P<key>geo_format\s+)\S+(?P<comment>.*)', r'\g<key>{0}\g<comment>'.format(x), l), \
        }

        self.__params['geo_format'] = geo_format
        self.__min_step = min_step
        self.__max_step = max_step

    def _setup(self, param, temp_file):
        fp = open(self.__control_file, 'r')
        lines = fp.read()
        fp.close()
        for k in self.__control_res.keys():
            try:
                lines = self.__control_res[k](lines, param[k])
            except KeyError:
                pass
        fp_temp = open(temp_file, 'w')
        fp_temp.write(lines)
        fp_temp.close()

    def run(self, binary, process_results=False):
        # command to run as subprocess
        args = [
                'mpirun',
                '-np',
                # placeholder, substituted below
                '0',
                binary,
                self.__geo_file,
                self.__ffield_file,
        ]
        env = dict(environ)

        write_header = True
        if path.exists(self.__result_file):
            write_header = False
        fout = open(self.__result_file, 'a')
        if write_header:
            fout.write(self.__result_header_fmt.format(*self.__result_header))
            fout.flush()

        temp_dir = mkdtemp()
        temp_file = path.join(temp_dir, path.basename(self.__control_file))

        for p in product(*[self.__params[k] for k in self.__param_names]):
            param_dict = dict((k, v) for (k, v) in zip(self.__param_names, p))
            param_dict['name'] = path.basename(self.__geo_file).split('.')[0] \
                + '_cm' + param_dict['charge_method'] \
                + '_s' + param_dict['nsteps'] \
                + '_proc' + param_dict['proc_by_dim1'] \
                    + '_' + param_dict['proc_by_dim2'] \
                    + '_' + param_dict['proc_by_dim3'] \
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
        
            args[2] = str(int(param_dict['proc_by_dim1'])
                * int(param_dict['proc_by_dim2'])
                * int(param_dict['proc_by_dim3']));
            
            if not process_results:
                self._setup(param_dict, temp_file)
    
                #env['OMP_NUM_THREADS'] = param_dict['threads']
                start = time()
                args.append(temp_file);
                proc_handle = Popen(args, stdout=PIPE, stderr=PIPE, env=env, universal_newlines=True)
                stdout, stderr = proc_handle.communicate()
                stop = time()
                if proc_handle.returncode < 0:
                    print("WARNING: process terminated with code {0}".format(proc_handle.returncode))
                else:
                    print('stdout:')
                    print(stdout)
                    print('stderr:')
                    print(stderr)

            else:
                self._process_result(fout, param_dict, self.__min_step, self.__max_step)
            break

        fout.close()
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
                    (min_step and not max_step and cnt >= min_step) or \
                    (not min_step and max_step and cnt <= max_step) or \
                    (cnt >= min_step and cnt <= max_step):
                        cm = cm + float(line[6])
                        iters = iters + float(line[8])
                        pre_comp = pre_comp + float(line[9])
                        pre_app = pre_app + float(line[10])
                        spmv = spmv + float(line[11])
                        cnt = cnt + 1
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


if __name__ == '__main__':
    DATA = [ \
            'bilayer_56800', 'bilayer_340800', \
            'dna_19733', \
            'petn_48256', \
            'silica_6000', 'silica_72000', 'silica_300000', \
            'water_6540', 'water_78480', 'water_327000', \
            ]

    parser = argparse.ArgumentParser(description='Run molecular dynamics simulations on specified data sets.')
    parser.add_argument('-b', '--binary', metavar='binary', default=None, nargs=1,
            help='Binary file to run.')
    parser.add_argument('-f', '--out_file', metavar='out_file', default=None, nargs=1,
            help='Output file to write results.')
    parser.add_argument('-r', '--process_results', default=False, action='store_true',
            help='Process simulation results only (do not perform simulations).')
    parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
            help='Paramater name and value pairs for the simulation, which multiple values comma delimited.')
    parser.add_argument('-n', '--min_step', metavar='min_step', default=None, nargs=1,
            help='Minimum simulation step to begin aggregating results.')
    parser.add_argument('-x', '--max_step', metavar='max_step', default=None, nargs=1,
            help='Maxiumum simulation step for aggregating results.')
    parser.add_argument('data', nargs='+',
            choices=DATA, help='Data sets for which to run simulations.')

    # parse args and take action
    args = parser.parse_args()

    base_dir = getcwd()
    control_dir = path.join(base_dir, 'environ')
    data_dir = path.join(base_dir, 'data/benchmarks')

    header_fmt_str = '{:15}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:10}|{:10}|{:10}|{:10}|{:10}|{:3}|{:10}\n'
    header_str = ['Data Set', 'Steps', 'CM', 'Solvr', 'Q Tol', 'QDS', 'PreCT', 'PreCD', 'PreCS', 'PCSAI', 'PreAT', 'PreAJ', 'Pre Comp',
            'Pre App', 'Iters', 'SpMV', 'CM', 'Thd', 'Time (s)']
    body_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:3} {:10.3f}\n'

    params = {
            'ensemble_type': ['0'],
            'nsteps': ['100'],
            'proc_by_dim1': ['1'],
            'proc_by_dim2': ['1'],
            'proc_by_dim3': ['1'],
            'tabulate_long_range': ['0'],
            'reneighbor': ['1'],
            'vlist_buffer': ['0'],
            'charge_method': ['0'],
            'cm_q_net': ['0.0'],
            'cm_solver_type': ['2'],
            'cm_solver_max_iters': ['1000'],
            'cm_solver_restart': ['100'],
            'cm_solver_q_err': ['1e-6'],
            'cm_domain_sparsity': ['1.0'],
            'cm_solver_pre_comp_type': ['1'],
            'cm_solver_pre_comp_refactor': ['100'],
            'cm_solver_pre_comp_droptol': ['0.0'],
            'cm_solver_pre_comp_sweeps': ['3'],
            'cm_solver_pre_comp_sai_thres': ['0.1'],
            'cm_solver_pre_app_type': ['1'],
            'cm_solver_pre_app_jacobi_iters': ['50'],
            'geo_format': [1],
    }

    if args.out_file:
        result_file = args.out_file[0]
    else:
        result_file = 'result.txt'

    if args.binary:
        binary = args.binary[0]
    else:
        binary  = path.join(base_dir, 'PuReMD/bin/puremd')

    # overwrite default params, if supplied via command line args
    if args.params:
        for param in args.params:
            if param[0] in params:
                # if multiple inputs (comma seperated list), split and store in list
                params[param[0]] = param[1].split(',')
            else:
                print("[ERROR] Invalid parameter {0}. Terminating...".format(param[0]))
                exit(-1)

    if args.min_step:
        min_step = int(args.min_step[0])
    else:
        min_step = None
    if args.max_step:
        max_step = int(args.max_step[0])
    else:
        max_step = None

    test_cases = []
    if 'water_6540' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_6540.pdb'),
                path.join(data_dir, 'water/ffield_acks2.water'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'water_78480' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_78480.geo'),
                path.join(data_dir, 'water/ffield_acks2.water'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'water_327000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_327000.geo'),
                path.join(data_dir, 'water/ffield_acks2.water'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'bilayer_56800' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'bilayer/bilayer_56800.pdb'),
                path.join(data_dir, 'bilayer/ffield-bio'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'bilayer_340800' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'bilayer/bilayer_340800.geo'),
                path.join(data_dir, 'bilayer/ffield-bio'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'dna_19733' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'dna/dna_19733.pdb'),
                path.join(data_dir, 'dna/ffield-dna'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'silica_6000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_6000.pdb'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'silica_72000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_72000.geo'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'silica_300000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_300000.geo'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file,
                min_step=min_step, max_step=max_step))
    if 'petn_48256' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'petn/petn_48256.pdb'),
                path.join(data_dir, 'petn/ffield.petn'),
                path.join(control_dir, 'parallel_control'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file,
                min_step=min_step, max_step=max_step))

    for test in test_cases:
        test.run(binary, process_results=args.process_results)
