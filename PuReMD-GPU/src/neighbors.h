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

#ifndef __NEIGHBORS_H_
#define __NEIGHBORS_H_

#include "mytypes.h"

void Generate_Neighbor_Lists( reax_system*, control_params*, simulation_data*,
                              static_storage*, list**, output_controls* );
void Cuda_Generate_Neighbor_Lists (reax_system *system,
                                   static_storage *workspace, control_params *control, bool);

int Estimate_NumNeighbors( reax_system*, control_params*,
                           static_storage*, list** );

HOST_DEVICE int Are_Far_Neighbors( rvec, rvec, simulation_box*, real, far_neighbor_data* );

GLOBAL void Estimate_NumNeighbors ( reax_atom *, grid , simulation_box *, control_params *, int *);
GLOBAL void Generate_Neighbor_Lists( reax_atom *, grid , simulation_box *, control_params *, list );

GLOBAL void Estimate_NumNeighbors ( reax_atom *,
                                    grid ,
                                    simulation_box *,
                                    control_params *,
                                    int *, int *, int, int , int, int);
GLOBAL void fix_sym_indices_far_nbrs (list , int );


#endif
