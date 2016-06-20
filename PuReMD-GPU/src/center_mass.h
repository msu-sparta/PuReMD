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

#ifndef __CENTER_MASS_H__
#define __CENTER_MASS_H__

#include "mytypes.h"

GLOBAL void center_of_mass_blocks (single_body_parameters *, reax_atom *,
                            rvec *res_xcm, 
                            rvec *res_vcm, 
                            rvec *res_amcm, 
                            size_t n);

GLOBAL void center_of_mass (rvec *xcm, 
                            rvec *vcm, 
                            rvec *amcm, 
                            rvec *res_xcm,
                            rvec *res_vcm,
                            rvec *res_amcm,
                            size_t n);

GLOBAL void compute_center_mass (single_body_parameters *sbp, 
                                reax_atom *atoms,
                                real *results, 
								real xcm0, real xcm1, real xcm2,
                                size_t n);

GLOBAL void compute_center_mass (real *input, real *output, size_t n);

#endif
