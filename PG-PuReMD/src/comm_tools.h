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

void Check_MPI_Error( int, const char * );

void Setup_Comm( reax_system*, control_params*, mpi_datatypes* );

void Update_Comm( reax_system* );

void Sort_Boundary_Atoms( reax_system*, int, int, int, mpi_out_data* );

void Estimate_Boundary_Atoms( reax_system*, int, int, int, mpi_out_data* );

void Unpack_Exchange_Message( reax_system*, int, void*, int,
        neighbor_proc*, int );

void Unpack_Estimate_Message( reax_system*, int, void*, int,
        neighbor_proc*, int );

int SendRecv( reax_system*, mpi_datatypes*_data, MPI_Datatype, int*,
        message_sorter, unpacker, int );

void Comm_Atoms( reax_system*, control_params*, simulation_data*, storage*,
        mpi_datatypes*, int );

#ifdef __cplusplus
}
#endif


#endif
