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

#ifndef __BOX_H__
#define __BOX_H__


#include "mytypes.h"


void Setup_Box( real, real, real, real, real, real, simulation_box* );

/* Initializes box from box rtensor */
void Update_Box(rtensor, simulation_box* /*, int*/);
void Update_Box_Isotropic(simulation_box*, real /*, int*/);
void Update_Box_SemiIsotropic( simulation_box*, rvec /*, int*/ );

/* Computes all the transformations,
   metric and other quantities from box rtensor */
void Make_Consistent( simulation_box* );

int Are_Far_Neighbors( rvec, rvec, simulation_box*, real, far_neighbor_data* );
void Get_NonPeriodic_Far_Neighbors( rvec, rvec, simulation_box*,
        control_params*, far_neighbor_data*, int* );
void Get_Periodic_Far_Neighbors_Big_Box( rvec, rvec, simulation_box*,
        control_params*, far_neighbor_data*, int* );
void Get_Periodic_Far_Neighbors_Small_Box( rvec, rvec, simulation_box*,
        control_params*, far_neighbor_data*, int* );
void Distance_on_T3_Gen( rvec, rvec, simulation_box*, rvec );
void Inc_on_T3_Gen( rvec, rvec, simulation_box* );

/*int Get_Nbr_Box( simulation_box*, int, int, int );
rvec Get_Nbr_Box_Press( simulation_box*, int, int, int );
void Inc_Nbr_Box_Press( simulation_box*, int, int, int, rvec );*/

/* These functions assume that the coordinates are in triclinic system */
/* this function returns cartesian norm but triclinic distance vector */
static inline HOST_DEVICE real Sq_Distance_on_T3( rvec x1, rvec x2, simulation_box* box, rvec r)
{

    real norm = 0.0;
    real d, tmp;
    int i;

    for (i = 0; i < 3; i++)
    {
        d = x2[i] - x1[i];
        tmp = SQR(d);

        if ( tmp >= SQR( box->box_norms[i] / 2.0 ) )
        {
            if (x2[i] > x1[i])
                d -= box->box_norms[i];
            else
                d += box->box_norms[i];

            r[i] = d;
            norm += SQR(d);
        }
        else
        {
            r[i] = d;
            norm += tmp;
        }
    }

    return norm;

}


static inline HOST_DEVICE void Inc_on_T3( rvec x, rvec dx, simulation_box *box )
{
    int i;
    real tmp;

    for (i = 0; i < 3; i++)
    {
        tmp = x[i] + dx[i];
        if ( tmp <= -box->box_norms[i] || tmp >= box->box_norms[i] )
            tmp = fmod( tmp, box->box_norms[i] );

        if ( tmp < 0 ) tmp += box->box_norms[i];
        x[i] = tmp;
    }
}

real Metric_Product( rvec, rvec, simulation_box* );

void Print_Box( simulation_box*, FILE* );


#endif
