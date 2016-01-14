/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator
      
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

#ifndef __GMRES_H_
#define __GMRES_H_

#define SIGN(x) (x < 0.0 ? -1 : 1);

#include "mytypes.h"

int GMRES( static_storage*, sparse_matrix*, 
	   real*, real, real*, FILE* );

int GMRES_HouseHolder( static_storage*, sparse_matrix*, 
		       real*, real, real*, FILE* );

int PGMRES( static_storage*, sparse_matrix*, real*, real, 
	    sparse_matrix*, sparse_matrix*, real*, FILE* );

int PCG( static_storage*, sparse_matrix*, real*, real, 
	sparse_matrix*, sparse_matrix*, real*, FILE* );

int CG( static_storage*, sparse_matrix*, 
	 real*, real, real*, FILE* );

int uyduruk_GMRES( static_storage*, sparse_matrix*, 
		   real*, real, real*, int, FILE* );

#endif
