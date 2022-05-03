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

#ifndef __CUDA_BASIC_COMM_H_
#define __CUDA_BASIC_COMM_H_

#include "../reax_types.h"


enum pointer_type
{
    INT_PTR_TYPE = 0,
    REAL_PTR_TYPE = 1,
    RVEC_PTR_TYPE = 2,
    RVEC2_PTR_TYPE = 3,
};


void Cuda_Dist( reax_system const * const, storage * const, mpi_datatypes * const,
        void const * const, int, MPI_Datatype, cudaStream_t );

void Cuda_Coll( reax_system const * const, mpi_datatypes * const,
        void * const , int, MPI_Datatype, cudaStream_t );


#endif
