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

#ifndef __TWO_BODY_INTERACTIONS_H_
#define __TWO_BODY_INTERACTIONS_H_

#include "mytypes.h"


void Bond_Energy( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );

void vdW_Coulomb_Energy( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );

void LR_vdW_Coulomb( reax_system*, control_params*, int, int, real, LR_data* );

void Tabulated_vdW_Coulomb_Energy( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );


#endif
