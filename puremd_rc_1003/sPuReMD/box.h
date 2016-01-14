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

#ifndef __BOX_H__
#define __BOX_H__

#include "mytypes.h"

/* Initializes box from CRYST1 line of PDB */
void Init_Box_From_CRYST(real, real, real, real, real, real, 
			 simulation_box*/*, int*/);

/* Initializes box from box rtensor */
void Update_Box(rtensor, simulation_box* /*, int*/);
void Update_Box_Isotropic(simulation_box*, real /*, int*/);
void Update_Box_SemiIsotropic( simulation_box*, rvec /*, int*/ );

/* Computes all the transformations, 
   metric and other quantities from box rtensor */
void Make_Consistent(simulation_box*/*, int*/ );

/* Applies transformation to and from 
   Cartesian to Triclinic coordinates based on flag */
/* Use -1 flag for Cartesian -> Triclinic and +1 for otherway */
void Transform( rvec, simulation_box*, char, rvec );
void Transform_to_UnitBox( rvec, simulation_box*, char, rvec );

int Are_Far_Neighbors( rvec, rvec, simulation_box*, real, far_neighbor_data* );
void Get_NonPeriodic_Far_Neighbors( rvec, rvec, simulation_box*, 
				    control_params*, far_neighbor_data*, int* );
void Get_Periodic_Far_Neighbors_Big_Box( rvec, rvec, simulation_box*, 
					 control_params*, far_neighbor_data*, 
					 int* );
void Get_Periodic_Far_Neighbors_Small_Box( rvec, rvec, simulation_box*, 
					   control_params*, far_neighbor_data*, 
					   int* );
void Distance_on_T3_Gen( rvec, rvec, simulation_box*, rvec );
void Inc_on_T3_Gen( rvec, rvec, simulation_box* );

/*int Get_Nbr_Box( simulation_box*, int, int, int );
rvec Get_Nbr_Box_Press( simulation_box*, int, int, int );
void Inc_Nbr_Box_Press( simulation_box*, int, int, int, rvec );*/

/* These functions assume that the coordinates are in triclinic system */
/* this function returns cartesian norm but triclinic distance vector */
real Sq_Distance_on_T3( rvec, rvec, simulation_box*, rvec );
void Inc_on_T3( rvec, rvec, simulation_box* );

real Metric_Product( rvec, rvec, simulation_box* );

void Print_Box_Information( simulation_box*, FILE* );

#endif
