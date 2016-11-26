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

#include "cuda_helpers.h"

#include "box.h"


GLOBAL void k_compute_Inc_on_T3(reax_atom *atoms, unsigned int N,
    simulation_box *box, real d1, real d2, real d3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    rvec dx;
    dx[0] = d1;
    dx[1] = d2;
    dx[2] = d3;

    if (index < N )
    {
        Inc_on_T3( atoms[index].x, dx, box );
    }
}
