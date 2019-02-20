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

#ifndef __LIST_H_
#define __LIST_H_

#include "reax_types.h"


int Make_List( int, int, int, int, reax_list*, MPI_Comm );

void Delete_List( reax_list*, MPI_Comm );


static inline int Num_Entries( int i, reax_list *l )
{
    return l->end_index[i] - l->index[i];
}


static inline int Start_Index( int i, reax_list *l )
{
    return l->index[i];
}


static inline int End_Index( int i, reax_list *l )
{
    return l->end_index[i];
}


static inline void Set_Start_Index( int i, int val, reax_list *l )
{
    l->index[i] = val;
}


static inline void Set_End_Index( int i, int val, reax_list *l )
{
    l->end_index[i] = val;
}


#endif
