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

#include "reax_types.h"


/* indexing routines for grid cells */
#define IDX_GRID_3D(i, j, k, g) (((i) * (g)->ncell_max[1] * (g)->ncell_max[2]) + ((j) * (g)->ncell_max[2]) + (k))
#define IDX_GRID_3D_V(x, g) (((x)[0] * (g)->ncell_max[1] * (g)->ncell_max[2]) + ((x)[1] * (g)->ncell_max[2]) + (x)[2])
#define IDX_GRID_NBRS(i, j, k, l, g) (((i) * (g)->ncell_max[1] * (g)->ncell_max[2] * (g)->max_nbrs) + ((j) * (g)->ncell_max[2] * (g)->max_nbrs) + ((k) * (g)->max_nbrs) + (l))


void Setup_Grid( reax_system * const );

void Update_Grid( reax_system * const );

void Bin_Atoms( reax_system * const, static_storage * const );

void Finalize_Grid( reax_system * const );

void Reorder_Atoms( reax_system * const, static_storage * const,
        control_params const * const );


#endif
