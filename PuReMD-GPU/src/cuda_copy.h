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

#ifndef __CUDA_COPY_H_
#define __CUDA_COPY_H_

#include "cuda_utils.h"

#include "mytypes.h"
#include "list.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Sync_Host_Device_Grid( grid *, grid *, enum cudaMemcpyKind );
void Sync_Host_Device_Sys( reax_system *, enum cudaMemcpyKind );
void Sync_Host_Device_Params( control_params *, control_params *, enum cudaMemcpyKind );
void Sync_Host_Device_Data( simulation_data *, simulation_data *, enum cudaMemcpyKind );
void Sync_Host_Device_Mat( sparse_matrix *, sparse_matrix *, enum cudaMemcpyKind );
void Sync_Host_Device_Control( output_controls *, enum cudaMemcpyKind );

void Prep_Device_For_Output( reax_system *, simulation_data * );
void Sync_Host_Device_List( list *host, list *device, int type );

#ifdef __cplusplus
}
#endif


#endif
