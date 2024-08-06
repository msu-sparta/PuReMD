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

#ifndef __LOOKUP_H_
#define __LOOKUP_H_

#include "reax_types.h"


enum LR_lookup_table_entry_type
{
    LR_E_VDW = 0,
    LR_CE_VDW = 1,
    LR_E_CLMB = 2,
    LR_CE_CLMB = 3,
    LR_CM = 4,
};


#ifdef __cplusplus
extern "C" {
#endif

real LR_Lookup_Entry( LR_lookup_table * const, real, int );

void Make_LR_Lookup_Table( reax_system * const, control_params * const, storage * const,
        mpi_datatypes * const );

void Finalize_LR_Lookup_Table( reax_system * const, control_params * const,
       storage * const, mpi_datatypes * const );

#ifdef __cplusplus
}
#endif


#endif
