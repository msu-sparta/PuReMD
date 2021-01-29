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

#include "list.h"

#include "tool_box.h"


void Make_List( int n, int n_max, int total_intrs, int type, reax_list* l )
{
    assert( n > 0 );
    assert( n_max > 0 );
    assert( n_max >= n );
    assert( total_intrs >= 0 );
    assert( l != NULL );

    if ( l->allocated == TRUE )
    {
        fprintf( stderr, "[WARNING] attempted to allocate list which was already allocated."
                " Returning without allocation...\n" );
        return;
    }

    l->allocated = TRUE;
    l->n = n;
    l->n_max = n_max;
    l->total_intrs = total_intrs;

    l->index = smalloc( n_max * sizeof(int), "Make_List::l->index" );
    l->end_index = smalloc( n_max * sizeof(int), "Make_List::l->end_index" );

    switch ( type )
    {
    case TYP_VOID:
        if ( l->total_intrs > 0 )
        {
            l->v = smalloc( l->total_intrs * sizeof(void),
                    "Make_List::l->v" );
        }
        else
        {
            l->v = NULL;
        }
        break;

    case TYP_THREE_BODY:
        if ( l->total_intrs > 0 )
        {
            l->three_body_list = smalloc( l->total_intrs * sizeof(three_body_interaction_data),
                    "Make_List::l->three_body_list" );
        }
        else
        {
            l->three_body_list = NULL;
        }
        break;

    case TYP_BOND:
        if ( l->total_intrs > 0 )
        {
            l->bond_list = smalloc( l->total_intrs * sizeof(bond_data),
                    "Make_List::l->bond_list" );
        }
        else
        {
            l->bond_list = NULL;
        }
        break;

    case TYP_DBO:
        if ( l->total_intrs > 0 )
        {
            l->dbo_list = smalloc( l->total_intrs * sizeof(dbond_data),
                    "Make_List::l->dbo_list" );
        }
        else
        {
            l->dbo_list = NULL;
        }
        break;

    case TYP_DDELTA:
        if ( l->total_intrs > 0 )
        {
            l->dDelta_list = smalloc( l->total_intrs * sizeof(dDelta_data),
                    "Make_List::l->dDelta_list" );
        }
        else
        {
            l->dDelta_list = NULL;
        }
        break;

    case TYP_FAR_NEIGHBOR:
        if ( l->total_intrs > 0 )
        {
            l->far_nbr_list = smalloc( l->total_intrs * sizeof(far_neighbor_data),
                    "Make_List::l->far_nbr_list" );
        }
        else
        {
            l->far_nbr_list = NULL;
        }
        break;

    case TYP_NEAR_NEIGHBOR:
        if ( l->total_intrs > 0 )
        {
            l->near_nbr_list = smalloc( l->total_intrs * sizeof(near_neighbor_data),
                    "Make_List::l->near_nbr_list" );
        }
        else
        {
            l->near_nbr_list = NULL;
        }
        break;

    case TYP_HBOND:
        if ( l->total_intrs > 0 )
        {
            l->hbond_list = smalloc( l->total_intrs * sizeof(hbond_data),
                    "Make_List::l->hbond_list" );
        }
        else
        {
            l->hbond_list = NULL;
        }
        break;

    default:
        fprintf( stderr, "[ERROR] unknown list type. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }
}


void Delete_List( int type, reax_list* l )
{
    assert( l != NULL );

    if ( l->allocated == FALSE )
    {
        fprintf( stderr, "[WARNING] attempted to free list which was not allocated."
                " Returning without deallocation...\n" );
        return;
    }

    l->allocated = FALSE;
    l->n = 0;
    l->n_max = 0;
    l->total_intrs = 0;

    sfree( l->index, "Delete_List::l->index" );
    sfree( l->end_index, "Delete_List::l->end_index" );

    switch ( type )
    {
    case TYP_VOID:
        if ( l->v != NULL )
        {
            sfree( l->v, "Delete_List::l->v" );
        }
        break;

    case TYP_THREE_BODY:
        if ( l->three_body_list != NULL )
        {
            sfree( l->three_body_list, "Delete_List::l->three_body_list" );
        }
        break;

    case TYP_BOND:
        if ( l->bond_list != NULL )
        {
            sfree( l->bond_list, "Delete_List::l->bond_list" );
        }
        break;

    case TYP_DBO:
        if ( l->dbo_list != NULL )
        {
            sfree( l->dbo_list, "Delete_List::l->dbo_list" );
        }
        break;

    case TYP_DDELTA:
        if ( l->dDelta_list != NULL )
        {
            sfree( l->dDelta_list, "Delete_List::l->dDelta_list" );
        }
        break;

    case TYP_FAR_NEIGHBOR:
        if ( l->far_nbr_list != NULL )
        {
            sfree( l->far_nbr_list, "Delete_List::l->far_nbr_list" );
        }
        break;

    case TYP_NEAR_NEIGHBOR:
        if ( l->near_nbr_list != NULL )
        {
            sfree( l->near_nbr_list, "Delete_List::l->near_nbr_list" );
        }
        break;

    case TYP_HBOND:
        if ( l->hbond_list != NULL )
        {
            sfree( l->hbond_list, "Delete_List::l->hbond_list" );
        }
        break;

    default:
        fprintf( stderr, "[ERROR] unknown list type. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }
}
