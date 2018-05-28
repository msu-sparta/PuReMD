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

#ifndef __BASIC_COMM_H_
#define __BASIC_COMM_H_

#include "reax_types.h"


enum pointer_type
{
    INT_PTR_TYPE = 0,
    REAL_PTR_TYPE = 1,
    RVEC_PTR_TYPE = 2,
    RVEC2_PTR_TYPE = 3,
};


#ifdef __cplusplus
extern "C" {
#endif

void Dist( const reax_system * const, const mpi_datatypes * const,
        void*, int, MPI_Datatype );

void Coll( const reax_system * const, const mpi_datatypes * const,
        void*, int, MPI_Datatype );

real Parallel_Norm( const real * const, const int, MPI_Comm );

real Parallel_Dot( const real * const, const real * const, const int, MPI_Comm );

real Parallel_Vector_Acc( const real * const, const int, MPI_Comm );

#if defined(TEST_FORCES)
void Coll_ids_at_Master( reax_system*, storage*, mpi_datatypes* );

void Coll_rvecs_at_Master( reax_system*, storage*, mpi_datatypes*, rvec* );
#endif

#ifdef __cplusplus
}
#endif


#endif
