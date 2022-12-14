/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

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

#ifndef __SYSTEM_PROP_H_
#define __SYSTEM_PROP_H_

#include "reax_types.h"

void Temperature_Control( control_params*, simulation_data* );

void Compute_Kinetic_Energy( reax_system*, simulation_data*, MPI_Comm );

void Compute_System_Energy( reax_system*, simulation_data*, MPI_Comm );

void Compute_Total_Mass( reax_system*, simulation_data*, MPI_Comm );

void Compute_Center_of_Mass( reax_system*, simulation_data*,
                             mpi_datatypes*, MPI_Comm );

void Compute_Pressure( reax_system*, control_params*,
                       simulation_data*, mpi_datatypes* );
//void Compute_Pressure( reax_system*, simulation_data* );
#endif
