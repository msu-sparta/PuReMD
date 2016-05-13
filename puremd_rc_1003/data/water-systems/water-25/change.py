#!/usr/bin/env python
import glob
import os
import sys
import fileinput

os.chdir(sys.argv[1])
for files in glob.glob("param*"):
	for line in fileinput.FileInput(files,inplace=1):
			line = line.replace("water.16", "water.25");
			print line,
