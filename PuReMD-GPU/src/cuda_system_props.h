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

#ifndef __CUDA_SYSTEM_PROP_H_
#define __CUDA_SYSTEM_PROP_H_

#include "mytypes.h"


void prep_dev_system (reax_system *system);

void Cuda_Compute_Total_Mass( reax_system*, simulation_data* );
void Cuda_Compute_Center_of_Mass( reax_system*, simulation_data*, FILE* );
void Cuda_Compute_Kinetic_Energy( reax_system*, simulation_data* );


#endif
