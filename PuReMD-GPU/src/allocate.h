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

#ifndef __ALLOCATE_H_
#define __ALLOCATE_H_

#include "mytypes.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Reallocate( reax_system*, static_storage*, list**, int );

int Allocate_Matrix( sparse_matrix*, int, int );
void Deallocate_Matrix( sparse_matrix *);

int Allocate_HBond_List( int, int, int*, int*, list* );

int Allocate_Bond_List( int, int*, list* );

#ifdef __cplusplus
}
#endif


#endif
