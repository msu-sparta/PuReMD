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

#ifndef __CUDA_LINEAR_SOLVERS_H_
#define __CUDA_LINEAR_SOLVERS_H_

#include "reax_types.h"

#ifdef __cplusplus
extern "C" {
#endif
void get_from_device (real *host, real *device, unsigned int bytes, char *);
void put_on_device (real *host, real *device, unsigned int bytes, char *);
void Cuda_Vector_Sum (real *res, real a, real *x, real b, real *y, int count);
void Cuda_CG_Preconditioner (real *res, real *a, real *b, int count);
void Cuda_CG_Diagnol_Preconditioner (storage *workspace, rvec2 *b, int n);
void Cuda_DualCG_Preconditioer (storage *workspace, rvec2 *, rvec2 alpha, int n, rvec2 result);
void Cuda_Norm (rvec2 *arr, int n, rvec2 result);
void Cuda_Dot (rvec2 *a, rvec2 *b, rvec2 result, int n);
void Cuda_Vector_Sum_Rvec2 (rvec2 *x, rvec2 *, rvec2 , rvec2 *c, int n);
void Cuda_RvecCopy_From (real *dst, rvec2 *src, int index, int n);
void Cuda_RvecCopy_To (rvec2 *dst, real *src, int index, int n);
void Cuda_Dual_Matvec (sparse_matrix *, rvec2 *, rvec2 *, int , int); 
void Cuda_Matvec (sparse_matrix *, real *, real *, int , int); 


#ifdef __cplusplus
}
#endif

#endif
