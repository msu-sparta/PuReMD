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

#include "cuda_utils.h"

#if defined(PURE_REAX)
  #include "../list.h"
  #include "../tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "../reax_list.h"
  #include "../reax_tool_box.h"
#endif


extern "C" {


/* Allocate space for interaction list
 *
 * n: num. of elements to be allocated for list
 * max_intrs: max. num. of interactions for which to allocate space
 * type: list interaction type
 * l: pointer to list to be allocated
 * */
void Dev_Make_List( int n, int max_intrs, int type, reax_list *l )
{
    if ( l->allocated == TRUE )
    {
        fprintf( stderr, "[WARNING] attempted to allocate list which was already allocated."
                " Returning without allocation...\n" );
        return;
    }

    l->allocated = TRUE;
    l->n = n;
    l->max_intrs = max_intrs;
    l->type = type;

    cuda_malloc( (void **) &l->index, n * sizeof(int), TRUE, "Dev_Make_List::index" );
    cuda_malloc( (void **) &l->end_index, n * sizeof(int), TRUE, "Dev_Make_List::end_index" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "dev_list: n=%d max_intrs=%d type=%d\n", n, max_intrs, type );
#endif

    switch ( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            cuda_malloc( (void **) &l->far_nbr_list, 
                    l->max_intrs * sizeof(far_neighbor_data), TRUE, "Dev_Make_List::far_nbrs" );
            break;

        case TYP_THREE_BODY:
            cuda_malloc( (void **) &l->three_body_list,
                    l->max_intrs * sizeof(three_body_interaction_data), TRUE,
                    "Dev_Make_List::three_bodies" );
            break;

        case TYP_HBOND:
            cuda_malloc( (void **) &l->hbond_list, 
                    l->max_intrs * sizeof(hbond_data), TRUE, "Dev_Make_List::hbonds" );
            break;            

        case TYP_BOND:
            cuda_malloc( (void **) &l->bond_list,
                    l->max_intrs * sizeof(bond_data), TRUE, "Dev_Make_List::bonds" );
            break;

        default:
            fprintf( stderr, "[ERROR] unknown devive list type (%d)\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }
}


void Dev_Delete_List( reax_list *l )
{
    if ( l->allocated == FALSE )
    {
        fprintf( stderr, "[WARNING] attempted to free list which was not allocated."
                " Returning without deallocation...\n" );
        return;
    }

    l->allocated = FALSE;

    cuda_free( l->index, "Dev_Delete_List::index" );
    cuda_free( l->end_index, "Dev_Delete_List::end_index" );

    switch ( l->type )
    {
        case TYP_HBOND:
            cuda_free( l->hbond_list, "Dev_Delete_List::hbonds" );
            break;
        case TYP_FAR_NEIGHBOR:
            cuda_free( l->far_nbr_list, "Dev_Delete_List::far_nbrs" );
            break;
        case TYP_BOND:
            cuda_free( l->bond_list, "Dev_Delete_List::bonds" );
            break;
        case TYP_THREE_BODY:
            cuda_free( l->three_body_list, "Dev_Delete_List::three_bodies" );
            break;
        default:
            fprintf( stderr, "[ERROR] unknown devive list type (%d)\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }
}


}
