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
            result_header='', result_body_fmt='', result_file='results.txt', geo_format='1'):
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
                'cm_solver_pre_comp_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_droptol': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_droptol\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_refactor': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_refactor\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_comp_sweeps': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_comp_sweeps\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_app_type': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_app_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'cm_solver_pre_app_jacobi_iters': lambda l, x: sub(
                    r'(?P<key>cm_solver_pre_app_jacobi_iters\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'geo_format': lambda l, x: sub(
                    r'(?P<key>geo_format\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
        }

        self.__params['geo_format'] = geo_format

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

    def run(self, bin_file='sPuReMD/bin/spuremd', process_results=False):
        base_dir = getcwd()
        bin_path = path.join(base_dir, bin_file)
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
		+ '_q' + param_dict['cm_solver_type'] \
 		+ '_qtol' + param_dict['cm_solver_q_err'] \
 		+ '_qds' + param_dict['cm_domain_sparsity'] \
                + '_pc' + param_dict['cm_solver_pre_comp_type'] \
                + '_pctol' + param_dict['cm_solver_pre_comp_droptol'] \
                + '_pcs' + param_dict['cm_solver_pre_comp_sweeps'] \
                + '_pa' + param_dict['cm_solver_pre_app_type'] \
                + '_paji' + param_dict['cm_solver_pre_app_jacobi_iters'] \
		+ '_t' + param_dict['threads']


            if not process_results:
                self._setup(param_dict, temp_file)
    
                env['OMP_NUM_THREADS'] = param_dict['threads']
                start = time()
                proc_handle = Popen([bin_path, self.__geo_file, self.__ffield_file, temp_file], 
                    stdout=PIPE, stderr=PIPE, env=env, universal_newlines=True)
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
                self._process_result(fout, param_dict)

        fout.close()
        if path.exists(temp_file):
            remove(temp_file)
        if path.exists(temp_dir):
            rmdir(temp_dir)

    def _process_result(self, fout, param):
        time = 0.
        cm = 0.
        iters = 0.
        pre_comp = 0.
        pre_app = 0.
        spmv = 0.
        cnt = 0
        log_file = param['name'] + '.log'

        if not path.exists(log_file):
            print('***WARNING: {0} does not exist!'.format(log_file))
            return
        with open(log_file, 'r') as fp:
            for line in fp:
                line = line.split()
                try:
                    cm = cm + float(line[6])
                    iters = iters + float(line[8])
                    pre_comp = pre_comp + float(line[9])
                    pre_app = pre_app + float(line[10])
                    spmv = spmv + float(line[11])
                    cnt = cnt + 1
                    pass
                except Exception:
                    pass
                if line[0] == 'total:':
                    try:
                        time = float(line[1])
                    except Exception:
                        pass
            cnt = cnt - 1
            if cnt > 0:
                cm = cm / cnt
                iters = iters / cnt
                pre_comp = pre_comp / cnt
                pre_app = pre_app / cnt
                spmv = spmv / cnt

        if cnt == int(param['nsteps']):
            fout.write(self.__result_body_fmt.format(path.basename(self.__geo_file).split('.')[0], 
                param['nsteps'], param['charge_method'], param['cm_solver_type'], param['cm_solver_q_err'], param['cm_domain_sparsity'],
                param['cm_solver_pre_comp_type'], param['cm_solver_pre_comp_droptol'], param['cm_solver_pre_comp_sweeps'],
                param['cm_solver_pre_app_type'], param['cm_solver_pre_app_jacobi_iters'], pre_comp, pre_app, iters, spmv,
                cm, param['threads'], time))
        else:
            print('**WARNING: nsteps not correct in file {0}...'.format(log_file))
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
    parser.add_argument('-f', '--out_file', metavar='out_file', default=None, nargs=1,
            help='Output file to write results.')
    parser.add_argument('-r', '--process_results', default=False, action='store_true',
            help='Process simulation results only (do not perform simulations).')
    parser.add_argument('-p', '--params', metavar='params', action='append', default=None, nargs=2,
            help='Paramater name and value pairs for the simulation, which multiple values comma delimited.')
    parser.add_argument('data', nargs='+',
            choices=DATA, help='Data sets for which to run simulations.')

    # parse args and take action
    args = parser.parse_args()

    base_dir = getcwd()
    control_dir = path.join(base_dir, 'environ')
    data_dir = path.join(base_dir, 'data/benchmarks')

    header_fmt_str = '{:15}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:10}|{:10}|{:10}|{:10}|{:10}|{:3}|{:10}\n'
    header_str = ['Data Set', 'Steps', 'CM', 'Solvr', 'Q Tol', 'QDS', 'PreCT', 'PreCD', 'PreCS', 'PreAT', 'PreAJ', 'Pre Comp',
            'Pre App', 'Iters', 'SpMV', 'CM', 'Thd', 'Time (s)']
    body_fmt_str = '{:15} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:5} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:3} {:10.3f}\n'

    params = {
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
            'cm_solver_pre_comp_type': ['2'],
            'cm_solver_pre_comp_refactor': ['100'],
            'cm_solver_pre_comp_droptol': ['0.0'],
            'cm_solver_pre_comp_sweeps': ['3'],
            'cm_solver_pre_app_type': ['2'],
            'cm_solver_pre_app_jacobi_iters': ['30'],
            'threads': ['1'],
            'geo_format': [],
    }

    if args.out_file:
        result_file = args.out_file[0]
    else:
        result_file = 'result.txt'

    # overwrite default params, if supplied via command line args
    if args.params:
        for param in args.params:
            params[param[0]] = param[1].split(',')

    test_cases = []
    if 'water_6540' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_6540.pdb'),
                path.join(data_dir, 'water/ffield.water'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file))
    if 'water_78480' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_78480.geo'),
                path.join(data_dir, 'water/ffield.water'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file))
    if 'water_327000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_327000.geo'),
                path.join(data_dir, 'water/ffield.water'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file))
    if 'bilayer_56800' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'bilayer/bilayer_56800.pdb'),
                path.join(data_dir, 'bilayer/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file))
    if 'bilayer_340800' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'bilayer/bilayer_340800.geo'),
                path.join(data_dir, 'bilayer/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file))
    if 'dna_19733' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'dna/dna_19733.pdb'),
                path.join(data_dir, 'dna/ffield-dna'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file))
    if 'silica_6000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_6000.pdb'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file))
    if 'silica_72000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_72000.geo'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file))
    if 'silica_300000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_300000.geo'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0'], result_file=result_file))
    if 'petn_48256' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'petn/petn_48256.pdb'),
                path.join(data_dir, 'petn/ffield.petn'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1'], result_file=result_file))

    for test in test_cases:
        test.run(bin_file='sPuReMD/bin/spuremd', process_results=args.process_results)
