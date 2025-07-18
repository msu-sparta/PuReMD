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

#ifndef __NONBONDED_H_
#define __NONBONDED_H_

#include "reax_types.h"


void Compute_Polarization_Energy( reax_system *, control_params *, simulation_data *, static_storage * );

void vdW_Coulomb_Energy_Type1( reax_system*, control_params*, simulation_data*,
        static_storage*, reax_list**, output_controls* );

void vdW_Coulomb_Energy_Type2( reax_system*, control_params*, simulation_data*,
        static_storage*, reax_list**, output_controls* );

void vdW_Coulomb_Energy_Type3( reax_system*, control_params*, simulation_data*,
        static_storage*, reax_list**, output_controls* );

void Coulomb_Energy_ACKS2( reax_system*, control_params*, simulation_data*,
        static_storage*, reax_list**, output_controls* );

void Tabulated_vdW_Coulomb_Energy( reax_system*, control_params*, simulation_data*,
        static_storage*, reax_list**, output_controls* );

void LR_vdW_Coulomb( reax_system*, control_params*, static_storage *,
        uint32_t, uint32_t, real, LR_data* );


#endif
