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
#include "cuda.h"
#include "mytypes.h"
#include "list.h"

void Sync_Host_Device (grid *, grid *, enum cudaMemcpyKind);
void Sync_Host_Device (reax_system *, enum cudaMemcpyKind);
void Sync_Host_Device (control_params *, control_params *, enum cudaMemcpyKind);
void Sync_Host_Device (simulation_data *, simulation_data *, enum cudaMemcpyKind);
void Sync_Host_Device (sparse_matrix *, sparse_matrix *, enum cudaMemcpyKind);
void Sync_Host_Device (output_controls *, enum cudaMemcpyKind);

void Prep_Device_For_Output (reax_system *, simulation_data *);
void Sync_Host_Device (list *host, list *device, int type);

#endif
