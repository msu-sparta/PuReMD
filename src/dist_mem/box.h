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

#ifndef __BOX_H_
#define __BOX_H_

#include "reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void Make_Consistent( simulation_box * const );

/* initializes simulation boxes */
void Setup_Big_Box( real, real, real, real, real, real, simulation_box * const );

void Init_Box( rtensor, simulation_box * const );

void Setup_My_Box( reax_system * const, control_params * const );

void Setup_My_Ext_Box( reax_system * const, control_params * const );

void Setup_Environment( reax_system * const, control_params * const,
        mpi_datatypes * const );

/* scales simulation box for NPT ensembles */
void Scale_Box( reax_system * const, control_params * const,
        simulation_data * const, mpi_datatypes * const );

/* applies transformation to/from Cartesian/ Triclinic coordinates */
/* use -1 flag for Cartesian -> Triclinic and +1 for otherway */
//void Transform( rvec, simulation_box * const, char, rvec );

//void Distance_on_T3_Gen( rvec, rvec, simulation_box * const, rvec );

//void Inc_on_T3_Gen( rvec, rvec, simulation_box * const );

//int Get_Nbr_Box( simulation_box * const, int, int, int );

//rvec Get_Nbr_Box_Press( simulation_box * const, int, int, int );

//void Inc_Nbr_Box_Press( simulation_box * const, int, int, int, rvec );

/* these functions assume that the coordinates are in triclinic system
   this function returns cartesian norm but triclinic distance vector */
//real Sq_Distance_on_T3( rvec, rvec, simulation_box * const, rvec );

//void Inc_on_T3( rvec, rvec, simulation_box * const );

//real Metric_Product( rvec, rvec, simulation_box * const );

#ifdef __cplusplus
}
#endif


#endif
