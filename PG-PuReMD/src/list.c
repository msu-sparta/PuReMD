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

#if defined(PURE_REAX)
  #include "list.h"
  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_list.h"
  #include "reax_tool_box.h"
#endif


void Print_List( reax_list* list )
{
    int i;

    for( i = 0; i < list->n; i++ )
    {
        printf( "%d %d\n", list->index[i], list->end_index[i] );
        if ( i > 1 && list->end_index[i - 1] >= list->index[i] )
        {
            printf( "===> malformed list %d", i );
        }
    }
}


/* Allocate space for interaction list
 *
 * n: num. of elements to be allocated for list
 * max_intrs: max. num. of interactions for which to allocate space
 * type: list interaction type
 * l: pointer to list to be allocated
 * */
void Make_List( int n, int max_intrs, int type, reax_list *l )
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

    l->index = smalloc( sizeof(int) * n, "Make_List::index" );
    l->end_index = smalloc( sizeof(int) * n, "Make_List::end_index" );

    switch ( l->type )
    {
    case TYP_VOID:
        l->v = smalloc( sizeof(void*) * l->max_intrs, "Make_List::v" );
        break;

    case TYP_BOND:
        l->bond_list = smalloc( sizeof(bond_data) * l->max_intrs, "Make_List::bonds" );
        break;

    case TYP_THREE_BODY:
        l->three_body_list = smalloc( sizeof(three_body_interaction_data) * l->max_intrs,
                "Make_List::three_bodies" );
        break;

    case TYP_HBOND:
        l->hbond_list = smalloc( sizeof(hbond_data) * l->max_intrs, "Make_List::hbonds" );
        break;

    case TYP_FAR_NEIGHBOR:
        l->far_nbr_list = smalloc( sizeof(far_neighbor_data) * l->max_intrs,
                "Make_List::far_nbrs" );
        break;

    case TYP_DBO:
        l->dbo_list = smalloc( sizeof(dbond_data) * l->max_intrs, "Make_List::dbonds" );
        break;

    case TYP_DDELTA:
        l->dDelta_list = smalloc( sizeof(dDelta_data) * l->max_intrs, "Make_List::dDeltas" );
        break;

    default:
        fprintf( stderr, "[ERROR] unknown list type (%d)\n", l->type );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;
    }
}


void Delete_List( reax_list *l )
{
    if ( l->allocated == FALSE )
    {
        fprintf( stderr, "[WARNING] attempted to free list which was not allocated."
                " Returning without deallocation...\n" );
        return;
    }

    l->allocated = FALSE;
    l->n = 0;
    l->max_intrs = 0;

    sfree( l->index, "Delete_List::index" );
    sfree( l->end_index, "Delete_List::end_index" );

    switch ( l->type )
    {
    case TYP_VOID:
        sfree( l->v, "Delete_List::v" );
        break;

    case TYP_BOND:
        sfree( l->bond_list, "Delete_List::bonds" );
        break;

    case TYP_THREE_BODY:
        sfree( l->three_body_list, "Delete_List::three_bodies" );
        break;

    case TYP_HBOND:
        sfree( l->hbond_list, "Delete_List::hbonds" );
        break;

    case TYP_FAR_NEIGHBOR:
        sfree( l->far_nbr_list, "Delete_List::far_nbrs" );
        break;

    case TYP_DBO:
        sfree( l->dbo_list, "Delete_List::dbos" );
        break;

    case TYP_DDELTA:
        sfree( l->dDelta_list, "Delete_List::dDeltas" );
        break;

    default:
        fprintf( stderr, "[ERROR] unknown list type (%d)\n", l->type );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;
    }
}


/* Initialize list indices
 *
 * list: pointer to list
 * max_intrs: max. num. of interactions for each list element
 * */
void Init_List_Indices( reax_list *list, int *max_intrs )
{
    int i;

    /* exclusive prefix sum of max_intrs replaces start indices,
     * set end indices to the same as start indices for safety */
    Set_Start_Index( 0, 0, list );
    Set_End_Index( 0, 0, list );
    for ( i = 1; i < list->n; ++i )
    {
        Set_Start_Index( i, Start_Index( i - 1, list ) + max_intrs[i - 1], list );
        Set_End_Index( i, Start_Index( i, list ), list );
    }
}
