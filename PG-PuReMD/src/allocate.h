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

#ifndef __ALLOCATE_H_
#define __ALLOCATE_H_

#include "reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Init_Matrix_Row_Indices( sparse_matrix *, int * );

void PreAllocate_Space( reax_system*, control_params*, storage* );

void ReAllocate_System( reax_system*, int, int );

void Allocate_Workspace( reax_system*, control_params*, storage*,
        int, int );

void Allocate_Grid( reax_system*, MPI_Comm );

void Deallocate_Grid( grid* );

void Allocate_MPI_Buffers( mpi_datatypes*, int, neighbor_proc* );

void Allocate_Matrix( sparse_matrix*, int, int );

void Deallocate_Matrix( sparse_matrix * );

int Allocate_HBond_List( int, int, int*, int*, reax_list* );

int Allocate_Bond_List( int, int*, reax_list* );

void Deallocate_MPI_Buffers( mpi_datatypes * );

void ReAllocate( reax_system*, control_params*, simulation_data*, storage*,
        reax_list**, mpi_datatypes* );

#ifdef __cplusplus
}
#endif


#endif
