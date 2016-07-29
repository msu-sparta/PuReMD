#!/usr/bin/env python

from fileinput import input
from re import sub
from subprocess import Popen, PIPE
from os import getcwd, environ, path, rename
from sys import exit
from time import time

base_dir = getcwd()
control_dir = path.join(base_dir, 'environ')
data_dir = path.join(base_dir, 'data/benchmarks')
puremd_args = [ \
#  (path.join(data_dir, 'water/water_6540.pdb'), path.join(data_dir, 'water/ffield.water'), path.join(control_dir, 'param.gpu.water')), \
#  (path.join(data_dir, 'water/water_78480.pdb'), path.join(data_dir, 'water/ffield.water'), path.join(control_dir, 'param.gpu.water')), \
#  (path.join(data_dir, 'water/water_327000.geo'), path.join(data_dir, 'water/ffield.water'), path.join(control_dir, 'param.gpu.water')), \
#  (path.join(data_dir, 'bilayer/bilayer_56800.pdb'), path.join(data_dir, 'bilayer/ffield-bio'), path.join(control_dir, 'param.gpu.water')), \
  (path.join(data_dir, 'dna/dna_19733.pdb'), path.join(data_dir, 'dna/ffield-dna'), path.join(control_dir, 'param.gpu.water')), \
  (path.join(data_dir, 'silica/silica_6000.pdb'), path.join(data_dir, 'silica/ffield-bio'), path.join(control_dir, 'param.gpu.water')), \
#  (path.join(data_dir, 'silica/silica_72000.geo'), path.join(data_dir, 'silica/ffield-bio'), path.join(control_dir, 'param.gpu.water')), \
#  (path.join(data_dir, 'silica/silica_300000.geo'), path.join(data_dir, 'silica/ffield-bio'), path.join(control_dir, 'param.gpu.water')), \
#  (path.join(data_dir, 'petn/petn_48256.pdb'), path.join(data_dir, 'petn/ffield.petn'), path.join(control_dir, 'param.gpu.water')), \
]

header_fmt_str = '{:20}|{:5}|{:5}|{:5}|{:10}|{:10}|{:10}|{:10}|{:10}|{:10}'
body_fmt_str = '{:20} {:5} {:5} {:5} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10} {:10.6f}'

steps = ['20'] #['100']
qeq_solver_tol = ['1e-6'] #['1e-6', '1e-8', '1e-10', '1e-14']
pre_comp_type = ['2'] #['0', '1', '2']
threads = ['1', '2', '4', '12', '24']
patterns = [ \
  lambda l, x: sub(r'(?P<key>simulation_name\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
  lambda l, x: sub(r'(?P<key>nsteps\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
  lambda l, x: sub(r'(?P<key>qeq_solver_q_err\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
  lambda l, x: sub(r'(?P<key>pre_comp_type\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
  lambda l, x: sub(r'(?P<key>geo_format\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l), \
]
d = dict(environ)

print header_fmt_str.format('Data Set', 'Steps', 'Q Tol', 'Pre T', 'Pre Comp',
    'Pre App', 'Iters', 'SpMV', 'Threads', 'Time (s)')

for i in xrange(len(puremd_args)):
  for s in xrange(len(steps)):
    for j in xrange(len(qeq_solver_tol)):
      for t in xrange(len(pre_comp_type)):
        for k in xrange(len(threads)):
          for line in input(puremd_args[i][2], inplace=1):
            line = line.rstrip() 
            line = patterns[0](line, path.basename(puremd_args[i][0]).split('.')[0]
                + '_' + 'tol' + qeq_solver_tol[j] + '_precomp' + pre_comp_type[t] + '_thread' + threads[k])
            line = patterns[1](line, steps[s])
            line = patterns[2](line, qeq_solver_tol[j])
            line = patterns[3](line, pre_comp_type[t])
            print line
  
          d['OMP_NUM_THREADS'] = threads[k]
          start = time()
          p = Popen([path.join(base_dir, 'sPuReMD/bin/spuremd')] + list(puremd_args[i]), 
              stdout=PIPE, stderr=PIPE, env=d)
          stdout, stderr = p.communicate()
          stop = time()
  
          iters = 0.
          pre_comp = 0.
          pre_app = 0.
          spmv = 0.
          cnt = 0
          with open(path.basename(puremd_args[i][0]).split('.')[0]
              + '_' + 'tol' + qeq_solver_tol[j] + '_precomp' + pre_comp_type[t] + '_thread'
  	      + threads[k] + '.log', 'r') as fp:
            for line in fp:
              line = line.split()
              try:
                iters = iters + float(line[7])
                pre_comp = pre_comp + float(line[8])
                pre_app = pre_app + float(line[9])
                spmv = spmv + float(line[10])
                cnt = cnt + 1
              except Exception:
                pass
          cnt = cnt - 1
          iters = iters / cnt
          pre_comp = pre_comp / cnt
          pre_app = pre_app / cnt
          spmv = spmv / cnt
  
          print body_fmt_str.format(path.basename(puremd_args[i][0]).split('.')[0], 
              steps[s], qeq_solver_tol[j], pre_comp_type[t], pre_comp, pre_app, iters, spmv,
  	      threads[k], stop - start)
