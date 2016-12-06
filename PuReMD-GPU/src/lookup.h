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

#ifndef __LOOKUP_H_
#define __LOOKUP_H_

#include "mytypes.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Make_Lookup_Table( real, real, int, lookup_function, lookup_table* );
int  Lookup_Index_Of( real, lookup_table* );
real Lookup( real, lookup_table* );

void Make_LR_Lookup_Table( reax_system*, control_params* );

#ifdef __cplusplus
}
#endif


#endif
