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
void Dev_Make_List( int n, int num_intrs, int type, reax_list *l )
{
    l->allocated = TRUE;

    l->n = n;
    l->num_intrs = num_intrs;

    cuda_malloc( (void **) &l->index, n * sizeof(int), TRUE, "dev_list:index" );
    cuda_malloc( (void **) &l->end_index, n * sizeof(int), TRUE, "dev_list:end_index" );

    l->type = type;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "dev_list: n=%d num_intrs=%d type=%d\n", n, num_intrs, type );
#endif

    switch( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            cuda_malloc( (void **) &l->select.far_nbr_list, 
                    l->num_intrs * sizeof(far_neighbor_data), TRUE, "dev_list:far_nbrs" );
            break;

        case TYP_THREE_BODY:
            cuda_malloc( (void **) &l->select.three_body_list,
                    l->num_intrs * sizeof(three_body_interaction_data), TRUE,
                    "dev_list:three_bodies" );
            break;

        case TYP_HBOND:
            cuda_malloc( (void **) &l->select.hbond_list, 
                    l->num_intrs * sizeof(hbond_data), TRUE, "dev_list:hbonds" );
            break;            

        case TYP_BOND:
            cuda_malloc( (void **) &l->select.bond_list,
                    l->num_intrs * sizeof(bond_data), TRUE, "dev_list:bonds" );
            break;

        default:
            fprintf( stderr, "ERROR: no %d dev_list type defined!\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }
}


void Dev_Delete_List( reax_list *l )
{
    if( l->allocated == FALSE )
    {
        return;
    }
    l->allocated = FALSE;

    cuda_free( l->index, "dev_index" );
    cuda_free( l->end_index, "dev_end_index" );

    switch (l->type)
    {
        case TYP_HBOND:
            cuda_free( l->select.hbond_list, "dev_list:hbonds" );
            break;
        case TYP_FAR_NEIGHBOR:
            cuda_free( l->select.far_nbr_list, "dev_list:far_nbrs" );
            break;
        case TYP_BOND:
            cuda_free( l->select.bond_list, "dev_list:bonds" );
            break;
        case TYP_THREE_BODY:
            cuda_free( l->select.three_body_list, "dev_list:three_bodies" );
            break;
        default:
            fprintf (stderr, "ERROR no %d dev_list type defined !\n", l->type);
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }
}

}
