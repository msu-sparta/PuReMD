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

#ifndef __NEIGHBORS_H_
#define __NEIGHBORS_H_

#include "mytypes.h"


void Generate_Neighbor_Lists( reax_system*, control_params*, simulation_data*,
   static_storage*, list**, output_controls* );

int Estimate_NumNeighbors( reax_system*, control_params*,
   static_storage*, list** );


static inline HOST_DEVICE int index_grid_debug( int x, int y, int z, int blocksize )
{
    return x * 8 * 8 * blocksize +  
        y * 8 * blocksize +  
        z * blocksize ;
}


static inline HOST_DEVICE real DistSqr_to_CP( rvec cp, rvec x )
{
    int  i;
    real d_sqr = 0;

    for( i = 0; i < 3; ++i )
        if( cp[i] > NEG_INF )
            d_sqr += SQR( cp[i] - x[i] );

    return d_sqr;
}


#endif
