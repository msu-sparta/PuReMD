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

#ifndef __CUDA_BOND_ORDERS_H_
#define __CUDA_BOND_ORDERS_H_

#include "mytypes.h"

GLOBAL void Cuda_Calculate_Bond_Orders_Init (  reax_atom *, global_parameters , single_body_parameters *,
        static_storage , int , int );
GLOBAL void Cuda_Calculate_Bond_Orders ( reax_atom *, global_parameters , single_body_parameters *,
        two_body_parameters *, static_storage , list , list , list , int , int );
GLOBAL void Cuda_Update_Uncorrected_BO (  static_storage , list , int );
GLOBAL void Cuda_Update_Workspace_After_Bond_Orders(  reax_atom *, global_parameters , single_body_parameters *,
        static_storage , int );
GLOBAL void Cuda_Compute_Total_Force (reax_atom *, simulation_data *, static_storage , list , int , int );
GLOBAL void Cuda_Compute_Total_Force_PostProcess (reax_atom *, simulation_data *, static_storage , list , int , int );
//HOST_DEVICE void Cuda_Add_dBond_to_Forces( int, int, reax_atom *, static_storage*, list* );
//HOST_DEVICE void Cuda_Add_dBond_to_Forces_NPT( int, int, reax_atom *, simulation_data*, static_storage*, list* );

#endif
