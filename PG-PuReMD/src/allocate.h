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

void Init_Matrix_Row_Indices( sparse_matrix * const, int * const );

void PreAllocate_Space( reax_system * const, control_params * const, storage * const );

void ReAllocate_System( reax_system * const, int, int );

void Allocate_Workspace( reax_system * const, control_params * const, storage * const,
        int, int );

void Allocate_Grid( reax_system * const, MPI_Comm );

void Deallocate_Grid( grid * const );

void Allocate_MPI_Buffers( mpi_datatypes * const, int, neighbor_proc * const );

void Allocate_Matrix( sparse_matrix * const, int, int, int );

void Reallocate_Matrix( sparse_matrix * const, int, int, int );

//void Allocate_Matrix_SAI( sparse_matrix ** const, int, int, int );

void Deallocate_Matrix( sparse_matrix * const );

int Allocate_HBond_List( int, int, int * const, int * const, reax_list * const );

int Allocate_Bond_List( int, int * const, reax_list * const );

void Deallocate_MPI_Buffers( mpi_datatypes * const );

void ReAllocate( reax_system * const, control_params * const,
        simulation_data * const, storage * const,
        reax_list** const, mpi_datatypes * const );

#ifdef __cplusplus
}
#endif


#endif
