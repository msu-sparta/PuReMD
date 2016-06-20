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

#ifndef __RESET_UTILS_H_
#define __RESET_UTILS_H_

#include "mytypes.h"

void Reset_Atoms( reax_system* );

void Reset_Pressures( simulation_data* );

void Reset_Simulation_Data( simulation_data* );

#ifdef TEST_FORCES
void Reset_Test_Forces( reax_system*, static_storage* );
#endif

void Reset_Workspace( reax_system*, static_storage* );

void Reset_Neighbor_Lists( reax_system*, control_params*,
                           static_storage*, list** );

void Reset( reax_system*, control_params*, simulation_data*,
            static_storage*, list** );

//void Reset_Neighbor_Lists( reax_system*, static_storage*, list** );

void Reset_Grid( grid* );

void Reset_Marks( grid*, ivec*, int );

void Cuda_Reset_Grid( grid* );

//CUDA functions
void Cuda_Reset_Workspace (reax_system *, static_storage *);
void Cuda_Reset( reax_system*, control_params*, simulation_data*,
                 static_storage*, list** );
void Cuda_Reset_Atoms (reax_system *);

#endif
