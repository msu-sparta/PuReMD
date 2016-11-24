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

#ifndef __BOND_ORDERS_H_
#define __BOND_ORDERS_H_


#include "mytypes.h"


typedef struct
{
    real C1dbo, C2dbo, C3dbo;
    real C1dbopi, C2dbopi, C3dbopi, C4dbopi;
    real C1dbopi2, C2dbopi2, C3dbopi2, C4dbopi2;
    real C1dDelta, C2dDelta, C3dDelta;
} dbond_coefficients;


#ifdef TEST_FORCES
void Get_dBO( reax_system*, list**, int, int, real, rvec* );
void Get_dBOpinpi2( reax_system*, list**, int, int, real, real, rvec*, rvec* );

void Add_dBO( reax_system*, list**, int, int, real, rvec* );
void Add_dBOpinpi2( reax_system*, list**, int, int, real, real, rvec*, rvec* );

void Add_dBO_to_Forces( reax_system*, list**, int, int, real );
void Add_dBOpinpi2_to_Forces( reax_system*, list**, int, int, real, real );

void Add_dDelta( reax_system*, list**, int, real, rvec* );
void Add_dDelta_to_Forces( reax_system *, list**, int, real );
#endif

void Add_dBond_to_Forces( int, int, reax_system*, simulation_data*,
                          static_storage*, list** );
void Add_dBond_to_Forces_NPT( int, int, reax_system*, simulation_data*,
                              static_storage*, list** );
void Calculate_Bond_Orders( reax_system*, control_params*, simulation_data*,
                            static_storage*, list**, output_controls* );

//CUDA Functions
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
