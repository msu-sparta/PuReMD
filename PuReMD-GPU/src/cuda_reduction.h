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

#ifndef __REDUCTION_H__
#define __REDUCTION_H__

#include "mytypes.h"

#define INITIAL 0
#define FINAL       1

GLOBAL void Cuda_reduction (const real *input, real *per_block_results, const size_t n);
GLOBAL void Cuda_Norm (const real *input, real *per_block_results, const size_t n, int pass);
GLOBAL void Cuda_Dot (const real *a, const real *b, real *per_block_results, const size_t n);
GLOBAL void Cuda_reduction (const int *input, int *per_block_results, const size_t n);
GLOBAL void Cuda_reduction_rvec (rvec *, rvec *, size_t n);

GLOBAL void Cuda_Vector_Sum( real* , real , real* , real , real* , int ) ;
GLOBAL void Cuda_Vector_Scale( real* , real , real* , int ) ;
GLOBAL void Cuda_Vector_Add( real* , real , real* , int );

#endif
