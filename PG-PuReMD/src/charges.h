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

#ifndef __CHARGES_H_
#define __CHARGES_H_

#include "reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

int is_refactoring_step ( control_params const * const, simulation_data * const );

void Compute_Charges( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, const output_controls * const,
        mpi_datatypes * const );

#ifdef __cplusplus
}
#endif


#endif
