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

#ifndef __COMM_TOOLS_H_
#define __COMM_TOOLS_H_

#include "reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void Check_MPI_Error( int, const char * const );

void Setup_Comm( reax_system * const, control_params * const, mpi_datatypes * const );

void Update_Comm( reax_system * const );

void Estimate_Boundary_Atoms( reax_system * const, int, int, int, mpi_out_data * const );

void Unpack_Estimate_Message( reax_system * const, int, void * const, int,
        neighbor_proc * const, int );

int SendRecv( reax_system * const, mpi_datatypes * const, MPI_Datatype, int * const,
        message_sorter, unpacker, int );

void Comm_Atoms( reax_system * const, control_params * const, simulation_data * const, storage * const,
        mpi_datatypes * const, int );

#ifdef __cplusplus
}
#endif


#endif
