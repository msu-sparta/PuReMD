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

#ifndef __CUDA_FORCES_H_
#define __CUDA_FORCES_H_

#include "mytypes.h"

void Cuda_Compute_Forces( reax_system*, control_params*, simulation_data*,
    static_storage*, list**, output_controls* );

void Cuda_Estimate_Storage_Sizes (reax_system *, control_params *, int *);

GLOBAL void Estimate_Storage_Sizes  (reax_atom *, int , single_body_parameters *,
    two_body_parameters *, global_parameters ,
    control_params *, list , int , int *);

GLOBAL void Estimate_Sparse_Matrix_Entries ( reax_atom *, control_params *,
    simulation_data *, simulation_box *, list , int , int *);

void Cuda_Threebody_List( reax_system *, static_storage *, list *, int );

bool validate_device (reax_system *, simulation_data *, static_storage *, list **);

#endif
