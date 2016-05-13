#!/usr/bin/env python

from fileinput import input
from re import sub
from subprocess import Popen, PIPE
from os import environ, path, rename
from time import time

base_dir = '/mnt/home/ohearnku/PuReMD-dev/puremd_rc_1003'
control_dir = path.join(base_dir, 'environ')
data_dir = path.join(base_dir, 'data')
puremd_args = [ \
  (path.join(data_dir, 'water_6540.pdb'), path.join(data_dir, 'ffield.water'), path.join(control_dir, 'param.gpu.water')), \
  (path.join(data_dir, 'water_52320.pdb'), path.join(data_dir, 'ffield.water'), path.join(control_dir, 'param.gpu.water')), \
  (path.join(data_dir, 'water_80000.geo'), path.join(data_dir, 'ffield.water'), path.join(control_dir, 'param.gpu.water')), \
  (path.join(data_dir, 'bilayer.pdb'), path.join(data_dir, 'ffield-bio'), path.join(control_dir, 'param.gpu.water')), \
  (path.join(data_dir, 'dna.pdb'), path.join(data_dir, 'ffield-dna'), path.join(control_dir, 'param.gpu.water')), \
]

header_fmt_str = '{:20}|{:5}|{:10}|{:10}'
body_fmt_str = '{:20} {:5} {:10} {:10d}'

tols = ['1e-6', '1e-8', '1e-10', '1e-14']
threads = ['1', '2', '4', '10', '20']
patterns = [ \
  lambda l, x: sub(r'(?P<key>q_err\s+)\S+(?P<comment>.*)', r'\g<key>%s\g<comment>' % x, l.rstrip()), \
]
d = dict(environ)

print header_fmt_str.format('Data Set', 'Tol', 'Threads', 'Time (ms)')

for i in xrange(len(puremd_args)):
  for j in xrange(len(tols)):
    for line in input(puremd_args[i][2], inplace=1):
      for p in patterns:
        line = p(line, tols[j])
      print line
    for k in xrange(len(threads)):
      d['OMP_NUM_THREADS'] = threads[k]
      start = time()
      p = Popen([path.join(base_dir, 'sPuReMD/spuremd')] + list(puremd_args[i]), 
          stdout=PIPE, stderr=PIPE, env=d)
      stdout, stderr = p.communicate()
      stop = time()
      print body_fmt_str.format(path.basename(puremd_args[i][0]), tols[j], threads[k],
          int((stop - start) * 1000))

      rename('water.6.notab.log', path.basename(puremd_args[i][0]).split('.')[0]
          + '_' + tols[j] + 'tol_' + threads[k] + 'thread' + '.log')
      rename('water.6.notab.out', path.basename(puremd_args[i][0]).split('.')[0]
          + '_' + tols[j] + 'tol_' + threads[k] + 'thread' + '.out')
