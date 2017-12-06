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


void Make_List( int n, int total_intrs, int type, reax_list* l )
{
    l->n = n;
    l->total_intrs = total_intrs;

    l->index = (int*) smalloc( n * sizeof(int), "Make_List::l->index" );
    l->end_index = (int*) smalloc( n * sizeof(int), "Make_List::l->end_index" );

    switch ( type )
    {
    case TYP_VOID:
        l->select.v = (void *) smalloc( l->total_intrs * sizeof(void),
                "Make_List::l->select.v" );
        break;

    case TYP_THREE_BODY:
        l->select.three_body_list = (three_body_interaction_data*)
            smalloc( l->total_intrs * sizeof(three_body_interaction_data),
                    "Make_List::l->select.three_body_list" );
        break;

    case TYP_BOND:
        l->select.bond_list = (bond_data*)
            smalloc( l->total_intrs * sizeof(bond_data),
                    "Make_List::l->select.bond_list" );
        break;

    case TYP_DBO:
        l->select.dbo_list = (dbond_data*)
            smalloc( l->total_intrs * sizeof(dbond_data),
                    "Make_List::l->select.dbo_list" );
        break;

    case TYP_DDELTA:
        l->select.dDelta_list = (dDelta_data*)
            smalloc( l->total_intrs * sizeof(dDelta_data),
                    "Make_List::l->select.dDelta_list" );
        break;

    case TYP_FAR_NEIGHBOR:
        l->select.far_nbr_list = (far_neighbor_data*)
            smalloc( l->total_intrs * sizeof(far_neighbor_data),
                    "Make_List::l->select.far_nbr_list" );
        break;

    case TYP_NEAR_NEIGHBOR:
        l->select.near_nbr_list = (near_neighbor_data*)
            smalloc( l->total_intrs * sizeof(near_neighbor_data),
                    "Make_List::l->select.near_nbr_list" );
        break;

    case TYP_HBOND:
        l->select.hbond_list = (hbond_data*)
            smalloc( l->total_intrs * sizeof(hbond_data),
                    "Make_List::l->select.hbond_list" );
        break;

    default:
        l->select.v = (void *) smalloc( l->total_intrs * sizeof(void),
                "Make_List::l->select.v" );
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
        if ( l->select.v != NULL )
        {
            sfree( l->select.v, "Delete_List::l->select.v" );
        }
        break;
    case TYP_THREE_BODY:
        if ( l->select.three_body_list != NULL )
        {
            sfree( l->select.three_body_list, "Delete_List::l->select.three_body_list" );
        }
        break;
    case TYP_BOND:
        if ( l->select.bond_list != NULL )
        {
            sfree( l->select.bond_list, "Delete_List::l->select.bond_list" );
        }
        break;
    case TYP_DBO:
        if ( l->select.dbo_list != NULL )
        {
            sfree( l->select.dbo_list, "Delete_List::l->select.dbo_list" );
        }
        break;
    case TYP_DDELTA:
        if ( l->select.dDelta_list != NULL )
        {
            sfree( l->select.dDelta_list, "Delete_List::l->select.dDelta_list" );
        }
        break;
    case TYP_FAR_NEIGHBOR:
        if ( l->select.far_nbr_list != NULL )
        {
            sfree( l->select.far_nbr_list, "Delete_List::l->select.far_nbr_list" );
        }
        break;
    case TYP_NEAR_NEIGHBOR:
        if ( l->select.near_nbr_list != NULL )
        {
            sfree( l->select.near_nbr_list, "Delete_List::l->select.near_nbr_list" );
        }
        break;
    case TYP_HBOND:
        if ( l->select.hbond_list != NULL )
        {
            sfree( l->select.hbond_list, "Delete_List::l->select.hbond_list" );
        }
        break;

    default:
        fprintf( stderr, "[ERROR] unknown list type. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

}
