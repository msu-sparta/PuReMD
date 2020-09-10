
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

#include <stdio.h>
#include <stdlib.h>

#include "spuremd.h"

#define INVALID_INPUT (-1)


static void usage( char * argv[] )
{
    fprintf( stderr, "usage:-------- ./%s geometry_file force_field_file control_file\n", argv[0] );
}


int main( int argc, char* argv[] )
{
    void *handle;
    int ret;

    if ( argc != 4 )
    {
        usage( argv );
        exit( INVALID_INPUT );
    }

    handle = setup( argv[1], argv[2], argv[3] );
    ret = SPUREMD_FAILURE;
	fprintf(stderr,"before simulate");
    if ( handle != NULL )
    {
        ret = simulate( handle );
    }

    if ( ret == SPUREMD_SUCCESS )
    {
        ret = cleanup( handle );
    }

    return (ret == SPUREMD_SUCCESS) ? 0 : (-1);
}
