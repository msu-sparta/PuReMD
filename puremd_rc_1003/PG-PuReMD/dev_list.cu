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

#include "reax_types.h"
#include "cuda_utils.h"

#if defined(PURE_REAX)
#include "list.h"
#include "tool_box.h"
#elif defined(LAMMPS_REAX)
#include "reax_list.h"
#include "reax_tool_box.h"
#endif

extern "C" {


/************* allocate list space ******************/
int Dev_Make_List(int n, int num_intrs, int type, reax_list *l)
{
  l->allocated = 1;
  
  l->n = n;
  l->num_intrs = num_intrs;

  		cuda_malloc ((void **) &l->index, n * sizeof (int), 1, "list:index");
  		cuda_malloc ((void **) &l->end_index, n * sizeof (int), 1, "list:end_index");

  		l->type = type;
		#if defined(DEBUG_FOCUS)
  		fprintf( stderr, "list: n=%d num_intrs=%d type=%d\n", n, num_intrs, type );
		#endif

  		switch(l->type) {

  		case TYP_FAR_NEIGHBOR:
		  cuda_malloc ((void **) &l->select.far_nbr_list, 
		  	l->num_intrs * sizeof (far_neighbor_data), 1, "list:far_nbrs");
  		  break;

  		case TYP_THREE_BODY:
			cuda_malloc ((void **) &l->select.three_body_list,
				l->num_intrs * sizeof (three_body_interaction_data), 1, 
  		        "list:three_bodies" );
  		  break;

  		case TYP_HBOND:
  		  cuda_malloc ((void **) &l->select.hbond_list, 
  		    l->num_intrs * sizeof(hbond_data), 1, "list:hbonds" );
  		  break;			

  		case TYP_BOND:
  		  cuda_malloc ((void **) &l->select.bond_list,
  		    l->num_intrs * sizeof(bond_data), 1, "list:bonds" );
  		  break;

  		default:
  		  fprintf( stderr, "ERROR: no %d list type defined!\n", l->type );
  		  MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
		}

  return SUCCESS;
}


void Dev_Delete_List( reax_list *l)
{
  if( l->allocated == 0 )
    return;
  l->allocated = 0;

		cuda_free ( l->index, "index");
		cuda_free ( l->end_index, "end_index" );

		switch (l->type) {
  		case TYP_HBOND:
  		  cuda_free( l->select.hbond_list, "list:hbonds" );
  		  break;
  		case TYP_FAR_NEIGHBOR:
  		  cuda_free( l->select.far_nbr_list, "list:far_nbrs" );
  		  break;
  		case TYP_BOND:
  		  cuda_free( l->select.bond_list, "list:bonds" );
  		  break;
  		case TYP_THREE_BODY:
  		  cuda_free( l->select.three_body_list, "list:three_bodies" );
  		  break;
		default:
			fprintf (stderr, "ERROR no %d list type defined !\n", l->type);
  		  	MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
		}
}

}
