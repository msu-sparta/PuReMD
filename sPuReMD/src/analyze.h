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

#ifndef __ANALYZE_H_
#define __ANALYZE_H_

#include "reax_types.h"


void Analysis( reax_system*, control_params*, simulation_data*,
               static_storage*, reax_list**, output_controls* );

//void Copy_Bond_List( reax_system*, control_params*, reax_list** );

//void Analyze_Molecules( reax_system*, control_params*, simulation_data*,
//                        static_storage*, reax_list**, FILE* );

//void Analyze_Bonding( reax_system*, control_params*, simulation_data*,
//                      static_storage*, reax_list**, FILE* );

//void Analyze_Silica( reax_system*, control_params*, simulation_data*,
//                     static_storage*, reax_list**, FILE* );

//void Calculate_Dipole_Moment( reax_system*, control_params*, simulation_data*,
//                              static_storage *, reax_list*, FILE* );

//void Copy_Positions( reax_system*, static_storage* );

//void Calculate_Drift( reax_system*, control_params*,
//                      simulation_data*, static_storage*, FILE* );

//void Calculate_Density_3DMesh( reax_system*, simulation_data*, FILE* );

//void Calculate_Density_Slice( reax_system*, simulation_data*, FILE* );


#endif
