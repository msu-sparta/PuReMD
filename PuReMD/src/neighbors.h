/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

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

#ifndef __NEIGHBORS_H_
#define __NEIGHBORS_H_

#include "reax_types.h"

/*typedef void (*find_neighbors_function)( reax_system*, control_params*,
                     reax_list*, far_neighbor_data*,
                     reax_list*, near_neighbor_data*,
                     reax_list*, int*, near_neighbor_data*,
                     int, int*, int*, int*, int,
                     int, int, real, rvec, ivec );*/

void Generate_Neighbor_Lists( reax_system*, simulation_data*, storage*,
                              reax_list** );
int Estimate_NumNeighbors( reax_system*, reax_list**, int );

#endif
