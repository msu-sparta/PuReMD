/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

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

#include "cuda_grid.h"

#include "grid.h"
#include "index_utils.h"
#include "vector.h"

#include "cuda_utils.h"
#include "cuda_reset_utils.h"


void Cuda_Bin_Atoms (reax_system *system, static_storage *workspace )
{
    Cuda_Reset_Grid ( &system->d_g);

    Bin_Atoms ( system, workspace );

    dev_workspace->realloc.gcell_atoms = workspace->realloc.gcell_atoms;
}


void Cuda_Bin_Atoms_Sync (reax_system *system)
{
    copy_host_device (system->g.top, system->d_g.top, 
            INT_SIZE * system->g.ncell[0]*system->g.ncell[1]*system->g.ncell[2], cudaMemcpyHostToDevice, RES_GRID_TOP);

    copy_host_device (system->g.atoms, system->d_g.atoms, 
            INT_SIZE * system->g.max_atoms*system->g.ncell[0]*system->g.ncell[1]*system->g.ncell[2], cudaMemcpyHostToDevice, RES_GRID_ATOMS);
}
