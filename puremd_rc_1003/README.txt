/*----------------------------------------------------------------------
  PuReMD - Reax Force Field Simulator
      
  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Ananth Y Grama, ayg@cs.purdue.edu
 
  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of 
  the License, or (at your option) any later version.
               
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

-------------------------------------------------------------------------
-------------------------------------------------------------------------
-------------   Instructions to get started with the codebase   ---------
-------------				PuReMD Package Release 1.0.0.0			 ---------
-------------------------------------------------------------------------
-------------------------------------------------------------------------

VERSION - 1003

1. Following package for Purdue Reactive Molecular Dynamics (PuReMD) 
consists of the following implementations: 
	a) serial (single cpu) implementation  -	sPuReMD
	b) GPU (single GPU) implementation		 -	PuReMD-GPU
	c) Parallel CPU (cluster of CPUs) implementation		- PuReMD
	d) Parallel GPU (cluster of GPUs) implementation		- PG-PuReMD

2. In the current implemtations only limited ensembles are supported
in the GPU implementations (CPU implementations supports a wide array of 
ensembles), which are as follows: 
	a) Single GPU implementation supports NVE, Nose Hoover NVT and Berendsen
	NVT.
	b) Parallel GPU implementations supports only Berendsen NVT.
	
3. "manual.pdf" describes the arguments in the control file. This file
currently describes all the parameters for single cpu and single gpu
implementations. Parallel implementations take one additional argument in the
control file, which is "procs_by_dim" - this parameter describes the
nodes/processors in each of the x, y and z dimensions. For example 1 2 3,
would correspond to 1 node on x-axis, 2 nodes on y-axis and 3 nodes on z-axis. 

4. This tar ball contains the master Makefile which invokes the appropriate
makefile depending on the build arguments. The master makefile takes the 
following arguments. 
	a) "cpu" - builds the single cpu executable (sPuReMD)
		The command is: make cpu
	b) "gpu" - builds the single gpu executable (PuReMD-GPU)
		The commnad is: make gpu
	c) "parallel-cpu" - builds the parallel cpu executable (PuReMD)
		The command is: make parallel-cpu
	d) "parallel-gpu" - builds the parallel gpu executable (PG-PuReMD)
		The command is: make parallel-gpu
	e) Sample usage of the master makefile is as follows: 
		make parallel-gpu  - this builds PG-PuReMD executable

5. The Makefiles (Makefiles are present in folders for the implementations
mentioned above)  needs to adapted to the build environment, PATH variables 
need to point to the CUDA isntallation folders appropriately. 

6. This tar ball also contains a sample water system and sample forcefield/
control files. The control file for parallel version of CPU/GPU
implementations has more parameters than that of single GPU/CPU
implementation. All these files are present in the "environ" folder of the
tar ball. 

7. The "environ" folder also contains some sample shell scripts which can run
the executables. 

8. This package is released under the GPL license.

--------------------------------------------------------------------------------
