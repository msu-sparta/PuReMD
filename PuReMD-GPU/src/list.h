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

#ifndef __LIST_H_
#define __LIST_H_

#include "mytypes.h"


char Make_List( int, int, int, list* );
void Delete_List( list* );


static inline HOST_DEVICE int Num_Entries(int i, list* l)
{
    return l->end_index[i] - l->index[i];
}


static inline HOST_DEVICE int Start_Index(int i, list *l )
{
    return l->index[i];
}


static inline HOST_DEVICE int End_Index( int i, list *l )
{
    return l->end_index[i];
}


static inline HOST_DEVICE void Set_Start_Index(int i, int val, list *l)
{
    l->index[i] = val;
}


static inline HOST_DEVICE void Set_End_Index(int i, int val, list *l)
{
    l->end_index[i] = val;
}


#endif
