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

#ifndef __CUDA_TWO_BODY_INTERACTIONS_H_
#define __CUDA_TWO_BODY_INTERACTIONS_H_

#include "mytypes.h"


GLOBAL void Cuda_Bond_Energy ( reax_atom *, global_parameters , single_body_parameters *, two_body_parameters *,
        simulation_data *, static_storage , list , int , int, real *);

GLOBAL void Cuda_vdW_Coulomb_Energy( reax_atom *, two_body_parameters *,
        global_parameters , control_params *, simulation_data *, list , real *, real *, rvec *,
        int , int );

GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy ( reax_atom *, control_params *, simulation_data *,
        list , real *, real *, rvec *,
        LR_lookup_table *, int , int , int ) ;

GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy_1 ( reax_atom *, control_params *, simulation_data *,
        list , real *, real *, rvec *,
        LR_lookup_table *, int , int , int ) ;

GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy_2 ( reax_atom *, control_params *, simulation_data *,
        list , real *, real *, rvec *,
        LR_lookup_table *, int , int , int ) ;

DEVICE void LR_vdW_Coulomb( global_parameters, two_body_parameters *,
        control_params *, int , int , real , LR_data * , int);


#endif

