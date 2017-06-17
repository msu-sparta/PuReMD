/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
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

#ifndef __GRID_H_
#define __GRID_H_

#include "mytypes.h"

void Setup_Grid( reax_system* );

void Update_Grid( reax_system* );

int  Shift( int, int, int, grid* );

void Cluster_Atoms( reax_system *, static_storage *, control_params * );

void Bin_Atoms( reax_system*, static_storage* );

void Reset_Marks( grid*, ivec*, int );

#endif
