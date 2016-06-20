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

#ifndef __THREE_BODY_INTERACTIONS_H_
#define __THREE_BODY_INTERACTIONS_H_

#include "mytypes.h"

void Three_Body_Interactions( reax_system*, control_params*, simulation_data*,
			      static_storage*, list**, output_controls* );

void Hydrogen_Bonds( reax_system*, control_params*, simulation_data*,
		     static_storage*, list**, output_controls* );


//CUDA Functions.
HOST_DEVICE void Calculate_Theta( rvec, real, rvec, real, real*, real* );
		      
HOST_DEVICE void Calculate_dCos_Theta( rvec, real, rvec, real, rvec*, rvec*, rvec* );

GLOBAL void Three_Body_Interactions( reax_atom *, single_body_parameters *, three_body_header *,
                                     global_parameters , control_params *, simulation_data *,
                                     static_storage , 
                                     list , list , int , int , real *, real *, real *, rvec *);

GLOBAL void Three_Body_Interactions_results (  reax_atom *, 
																	control_params *,
                                                   static_storage , 
																	list , int );

GLOBAL void Three_Body_Estimate ( reax_atom *atoms, 
                                    control_params *control,
												list p_bonds, int N, 
												int *count);

GLOBAL void Hydrogen_Bonds (  reax_atom *,
                              single_body_parameters *, hbond_parameters *,
										control_params *, simulation_data *, static_storage , 
										list , list , int , int, real *, rvec *, rvec *);
GLOBAL void Hydrogen_Bonds_HB (  reax_atom *,
                              single_body_parameters *, hbond_parameters *,
										control_params *, simulation_data *, static_storage , 
										list , list , int , int, real *, rvec *, rvec *);

GLOBAL void Hydrogen_Bonds_Postprocess (  reax_atom *,
														single_body_parameters *, 
														static_storage , list, 
														list , list , int, real * );
GLOBAL void Hydrogen_Bonds_Far_Nbrs (  reax_atom *,
														single_body_parameters *, 
														static_storage , list, 
														list , list , int );
GLOBAL void Hydrogen_Bonds_HNbrs (  reax_atom *,
														single_body_parameters *, 
														static_storage , list, 
														list , list , int );
#endif
