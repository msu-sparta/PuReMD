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

#ifndef __THREE_BODY_INTERACTIONS_H_
#define __THREE_BODY_INTERACTIONS_H_

#include "mytypes.h"

void Three_Body_Interactions( reax_system*, control_params*, simulation_data*,
                              static_storage*, list**, output_controls* );

void Hydrogen_Bonds( reax_system*, control_params*, simulation_data*,
                     static_storage*, list**, output_controls* );

void Calculate_Theta( rvec, real, rvec, real, real*, real* );

void Calculate_dCos_Theta( rvec, real, rvec, real, rvec*, rvec*, rvec* );

#endif
