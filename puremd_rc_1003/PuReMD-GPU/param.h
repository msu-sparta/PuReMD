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

#ifndef __PARAM_H_
#define __PARAM_H_

#include "mytypes.h"

#define MAX_LINE 1024
#define MAX_TOKENS 20
#define MAX_TOKEN_LEN 1024

int  Get_Atom_Type( reax_interaction*, char* );

int  Tokenize( char*, char*** );

char Read_Force_Field( FILE*, reax_interaction* );

char Read_Control_File( FILE*, reax_system*, control_params*, 
			output_controls* );

#endif
