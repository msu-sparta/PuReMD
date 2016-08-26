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
                'qeq_solver_type': lambda l, x: sub(
                    r'(?P<key>qeq_solver_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'qeq_solver_q_err': lambda l, x: sub(
                    r'(?P<key>qeq_solver_q_err\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'pre_comp_type': lambda l, x: sub(
                    r'(?P<key>pre_comp_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'pre_comp_refactor': lambda l, x: sub(
                    r'(?P<key>pre_comp_refactor\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'pre_comp_sweeps': lambda l, x: sub(
                    r'(?P<key>pre_comp_sweeps\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'pre_app_type': lambda l, x: sub(
                    r'(?P<key>pre_app_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
                'pre_app_jacobi_iters': lambda l, x: sub(
                    r'(?P<key>pre_app_jacobi_iters\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
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

    def run(self, bin_file='sPuReMD/bin/spuremd'):
        base_dir = getcwd()
        bin_path = path.join(base_dir, bin_file)
        env = dict(environ)

        write_header = True
        if path.exists(self.__result_file):
            write_header = False
        fout = open(self.__result_file, 'a')
        if write_header:
            fout.write(self.__result_header_fmt.format(*self.__result_header))

        temp_dir = mkdtemp()
        temp_file = path.join(temp_dir, path.basename(self.__control_file))

        for p in product(*[self.__params[k] for k in self.__param_names]):
            param_dict = dict((k, v) for (k, v) in zip(self.__param_names, p))
            param_dict['name'] = path.basename(self.__geo_file).split('.')[0] \
                + '_step' + param_dict['nsteps'] + '_tol' + param_dict['qeq_solver_q_err'] \
                + '_precomp' + param_dict['pre_comp_type'] + '_thread' + param_dict['threads']

            self._setup(param_dict, temp_file)

            env['OMP_NUM_THREADS'] = param_dict['threads']
            start = time()
            proc_handle = Popen([bin_path, self.__geo_file, self.__ffield_file, temp_file], 
                stdout=PIPE, stderr=PIPE, env=env)
            #TODO: handle outputs?
            stdout, stderr = proc_handle.communicate()
            stop = time()

            self._process_result(fout, stop - start, param_dict)

        fout.close()
        remove(temp_file)
        rmdir(temp_dir)

    def _process_result(self, fout, time, param):
        qeq = 0.
        iters = 0.
        pre_comp = 0.
        pre_app = 0.
        spmv = 0.
        cnt = 0
        with open(param['name'] + '.log', 'r') as fp:
            for line in fp:
                line = line.split()
                try:
                    qeq = qeq + float(line[6])
                    iters = iters + float(line[7])
                    pre_comp = pre_comp + float(line[8])
                    pre_app = pre_app + float(line[9])
                    spmv = spmv + float(line[10])
                    cnt = cnt + 1
                except Exception:
                    pass
            cnt = cnt - 1
            qeq = qeq / cnt
            iters = iters / cnt
            pre_comp = pre_comp / cnt
            pre_app = pre_app / cnt
            spmv = spmv / cnt

        fout.write(self.__result_body_fmt.format(path.basename(self.__geo_file).split('.')[0], 
            param['nsteps'], param['qeq_solver_q_err'], param['pre_comp_type'], pre_comp, pre_app, iters, spmv,
            qeq, param['threads'], time))


if __name__ == '__main__':
    DATA = [ \
            'bilayer_56800', 'bilayer_340800', \
            'dna_19733', \
            'petn_48256', \
            'silica_6000', 'silica_72000', 'silica_300000', \
            'water_6540', 'water_78480', 'water_327000', \
            ]

    parser = argparse.ArgumentParser(description='Run molecular dynamics simulations on specified data sets.')
    parser.add_argument('data', nargs='+',
            choices=DATA, help='Data sets for which to run simulations.')

    # parse args and take action
    args = parser.parse_args()

    base_dir = getcwd()
    control_dir = path.join(base_dir, 'environ')
    data_dir = path.join(base_dir, 'data/benchmarks')

    header_fmt_str = '{:20}|{:5}|{:5}|{:5}|{:10}|{:10}|{:10}|{:10}|{:10}|{:10}|{:10}\n'
    header_str = ['Data Set', 'Steps', 'Q Tol', 'Pre T', 'Pre Comp',
            'Pre App', 'Iters', 'SpMV', 'QEq', 'Threads', 'Time (s)']
    body_fmt_str = '{:20} {:5} {:5} {:5} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10} {:10.6f}\n'

    params = {
            'ensemble_type': ['0'],
            'nsteps': ['20'],
#            'nsteps': ['20', '100', '500', '1000'],
            'qeq_solver_type': ['0'],
            'qeq_solver_q_err': ['1e-6'],
#            'qeq_solver_q_err': ['1e-6', '1e-10'],
#            'qeq_solver_q_err': ['1e-6', '1e-8', '1e-10', '1e-14'],
            'pre_comp_type': ['2'],
#            'pre_comp_type': ['0', '1', '2'],
            'pre_comp_refactor': ['100'],
            'pre_comp_sweeps': ['3'],
            'pre_app_type': ['2'],
            'pre_app_jacobi_iters': ['50'],
            'threads': ['2'],
#            'threads': ['1', '2', '4', '12', '24'],
            'geo_format': [],
    }

    test_cases = []
    if 'water_6540' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_6540.pdb'),
                path.join(data_dir, 'water/ffield.water'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1']))
    if 'water_78480' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_78480.pdb'),
                path.join(data_dir, 'water/ffield.water'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0']))
    if 'water_327000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'water/water_327000.geo'),
                path.join(data_dir, 'water/ffield.water'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0']))
    if 'bilayer_56800' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'bilayer/bilayer_56800.pdb'),
                path.join(data_dir, 'bilayer/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1']))
    if 'dna_19733' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'dna/dna_19733.pdb'),
                path.join(data_dir, 'dna/ffield-dna'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1']))
    if 'silica_6000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_6000.pdb'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1']))
    if 'silica_72000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_72000.geo'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0']))
    if 'silica_300000' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'silica/silica_300000.geo'),
                path.join(data_dir, 'silica/ffield-bio'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['0']))
    if 'petn_48256' in args.data:
        test_cases.append(
            TestCase(path.join(data_dir, 'petn/petn_48256.pdb'),
                path.join(data_dir, 'petn/ffield.petn'),
                path.join(control_dir, 'param.gpu.water'),
                params=params, result_header_fmt=header_fmt_str,
                result_header = header_str, result_body_fmt=body_fmt_str,
                geo_format=['1']))

    for test in test_cases:
        test.run(bin_file='sPuReMD/bin/spuremd')
