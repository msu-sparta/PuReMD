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

#ifndef __GMRES_H_
#define __GMRES_H_

#define SIGN(x) (x < 0.0 ? -1 : 1);

#include "mytypes.h"

int GMRES( static_storage*, sparse_matrix*, 
	   real*, real, real*, FILE* ,reax_system* );

int Cuda_GMRES( static_storage *, real *b, real tol, real *x );
int Cublas_GMRES( reax_system *, static_storage *, real *b, real tol, real *x );

int GMRES_HouseHolder( static_storage*, sparse_matrix*, 
		       real*, real, real*, FILE* ,reax_system*  );

int PGMRES( static_storage*, sparse_matrix*, real*, real, 
	    sparse_matrix*, sparse_matrix*, real*, FILE*, reax_system* );

int PCG( static_storage*, sparse_matrix*, real*, real, 
	sparse_matrix*, sparse_matrix*, real*, FILE*, reax_system* );

int CG( static_storage*, sparse_matrix*, 
	 real*, real, real*, FILE*, reax_system* );

int uyduruk_GMRES( static_storage*, sparse_matrix*, 
		   real*, real, real*, int, FILE*, reax_system* );

#endif
