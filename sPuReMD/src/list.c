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
    assert( total_intrs > 0 );
    assert( l != NULL );

    l->n = n;
    l->n_max = n_max;
    l->total_intrs = total_intrs;

    l->index = smalloc( n_max * sizeof(int), "Make_List::l->index" );
    l->end_index = smalloc( n_max * sizeof(int), "Make_List::l->end_index" );

    switch ( type )
    {
    case TYP_VOID:
        l->v = smalloc( l->total_intrs * sizeof(void),
                "Make_List::l->v" );
        break;

    case TYP_THREE_BODY:
        l->three_body_list = smalloc( l->total_intrs * sizeof(three_body_interaction_data),
                "Make_List::l->three_body_list" );
        break;

    case TYP_BOND:
        l->bond_list = smalloc( l->total_intrs * sizeof(bond_data),
                "Make_List::l->bond_list" );
        break;

    case TYP_DBO:
        l->dbo_list = smalloc( l->total_intrs * sizeof(dbond_data),
                "Make_List::l->dbo_list" );
        break;

    case TYP_DDELTA:
        l->dDelta_list = smalloc( l->total_intrs * sizeof(dDelta_data),
                "Make_List::l->dDelta_list" );
        break;

    case TYP_FAR_NEIGHBOR:
        l->far_nbr_list = smalloc( l->total_intrs * sizeof(far_neighbor_data),
                "Make_List::l->far_nbr_list" );
        break;

    case TYP_NEAR_NEIGHBOR:
        l->near_nbr_list = smalloc( l->total_intrs * sizeof(near_neighbor_data),
                "Make_List::l->near_nbr_list" );
        break;

    case TYP_HBOND:
        l->hbond_list = smalloc( l->total_intrs * sizeof(hbond_data),
                "Make_List::l->hbond_list" );
        break;

    default:
        l->v = smalloc( l->total_intrs * sizeof(void),
                "Make_List::l->v" );
        break;
    }
}


void Delete_List( int type, reax_list* l )
{
    sfree( l->index, "Delete_List::l->index" );
    sfree( l->end_index, "Delete_List::l->end_index" );

    switch ( type )
    {
    case TYP_VOID:
        sfree( l->v, "Delete_List::l->v" );
        break;

    case TYP_THREE_BODY:
        sfree( l->three_body_list, "Delete_List::l->three_body_list" );
        break;

    case TYP_BOND:
        sfree( l->bond_list, "Delete_List::l->bond_list" );
        break;

    case TYP_DBO:
        sfree( l->dbo_list, "Delete_List::l->dbo_list" );
        break;

    case TYP_DDELTA:
        sfree( l->dDelta_list, "Delete_List::l->dDelta_list" );
        break;

    case TYP_FAR_NEIGHBOR:
        sfree( l->far_nbr_list, "Delete_List::l->far_nbr_list" );
        break;

    case TYP_NEAR_NEIGHBOR:
        sfree( l->near_nbr_list, "Delete_List::l->near_nbr_list" );
        break;

    case TYP_HBOND:
        sfree( l->hbond_list, "Delete_List::l->hbond_list" );
        break;

    default:
        fprintf( stderr, "[ERROR] unknown list type. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

}
