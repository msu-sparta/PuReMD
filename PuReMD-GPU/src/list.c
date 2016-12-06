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

#include "list.h"


char Make_List(int n, int num_intrs, int type, list* l)
{
    char success=1;

    l->n = n;
    l->num_intrs = num_intrs;

    l->index = (int*) malloc( n * sizeof(int) );
    l->end_index = (int*) malloc( n * sizeof(int) );

    if (l->index == NULL) success = 0;
    if (l->end_index == NULL) success = 0;

    l->type = type;

    switch(type)
    {
        case TYP_VOID:
            l->select.v = (void *) malloc(l->num_intrs*sizeof(void));
            if (l->select.v == NULL) success = 0;
            break;

        case TYP_THREE_BODY:
            l->select.three_body_list = (three_body_interaction_data*) 
                malloc(l->num_intrs*sizeof(three_body_interaction_data));
            if (l->select.three_body_list == NULL) success = 0;
            break;

        case TYP_BOND:
            l->select.bond_list = (bond_data*) 
                malloc(l->num_intrs * sizeof(bond_data));
            if (l->select.bond_list == NULL) success = 0;
            break;

        case TYP_DBO:
            l->select.dbo_list = (dbond_data*) 
                malloc(l->num_intrs * sizeof(dbond_data));
            if (l->select.dbo_list == NULL) success = 0;
            break;

        case TYP_DDELTA:
            l->select.dDelta_list = (dDelta_data*) 
                malloc(l->num_intrs*sizeof(dDelta_data));
            if (l->select.dDelta_list == NULL) success = 0;
            break;

        case TYP_FAR_NEIGHBOR:
            l->select.far_nbr_list = (far_neighbor_data*) 
                malloc(l->num_intrs*sizeof(far_neighbor_data));
            if (l->select.far_nbr_list == NULL) success = 0;
            break;

        case TYP_NEAR_NEIGHBOR:
            l->select.near_nbr_list = (near_neighbor_data*) 
                malloc(l->num_intrs*sizeof(near_neighbor_data));
            if (l->select.near_nbr_list == NULL) success = 0;
            break;

        case TYP_HBOND:
            l->select.hbond_list = (hbond_data*)
                malloc( l->num_intrs * sizeof(hbond_data) );
            if (l->select.hbond_list == NULL) success = 0;
            break;            

        default:
            l->select.v = (void *) malloc(l->num_intrs*sizeof(void));
            if (l->select.v == NULL) success = 0;
            l->type = TYP_VOID;
            break;      
    }

    return success;
}


void Delete_List(list* l)
{
    if( l->index != NULL )
        free(l->index);
    if( l->end_index != NULL )
        free(l->end_index);

    switch(l->type)
    {
        case TYP_VOID:
            if( l->select.v != NULL )
                free(l->select.v);
            break;
        case TYP_THREE_BODY:
            if( l->select.three_body_list != NULL )
                free(l->select.three_body_list);
            break;
        case TYP_BOND:
            if( l->select.bond_list != NULL )
                free(l->select.bond_list);
            break;
        case TYP_DBO:
            if( l->select.dbo_list != NULL )
                free(l->select.dbo_list);
            break;
        case TYP_DDELTA:
            if( l->select.dDelta_list != NULL )
                free(l->select.dDelta_list);
            break;
        case TYP_FAR_NEIGHBOR:
            if( l->select.far_nbr_list != NULL )
                free(l->select.far_nbr_list);
            break;
        case TYP_NEAR_NEIGHBOR:
            if( l->select.near_nbr_list != NULL )
                free(l->select.near_nbr_list);
            break;
        case TYP_HBOND:
            if( l->select.hbond_list != NULL )
                free(l->select.hbond_list);
            break;

        default:
            // Report fatal error
            break;
    }
}

