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


/************* allocate list space ******************/
int Make_List( int n, int num_intrs, int type, int format,
        reax_list *l, MPI_Comm comm )
{
    l->allocated = 1;

    l->n = n;
    l->num_intrs = num_intrs;

    l->index = smalloc( n * sizeof(int), "Make_List:index", comm );
    l->end_index = smalloc( n * sizeof(int), "Make_List:end_index", comm );

    l->type = type;
    l->format = format;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "list: n=%d num_intrs=%d type=%d\n", l->n, l->num_intrs, l->type );
#endif

    switch ( l->type )
    {
    case TYP_VOID:
        l->v = smalloc( l->num_intrs * sizeof(void*),
                "Make_List:v", comm );
        break;

    case TYP_THREE_BODY:
        l->three_body_list = smalloc( l->num_intrs * sizeof(three_body_interaction_data),
                "Make_List:three_bodies", comm );
        break;

    case TYP_BOND:
        l->bond_list = smalloc( l->num_intrs * sizeof(bond_data),
                "Make_List:bonds", comm );
        break;

    case TYP_DBO:
        l->dbo_list = smalloc( l->num_intrs * sizeof(dbond_data),
                "Make_List:dbonds", comm );
        break;

    case TYP_DDELTA:
        l->dDelta_list = smalloc( l->num_intrs * sizeof(dDelta_data),
                "Make_List:dDeltas", comm );
        break;

    case TYP_FAR_NEIGHBOR:
        l->far_nbr_list.nbr = smalloc( l->num_intrs * sizeof(int),
                "Make_List:far_nbr_list.nbr", comm );
        l->far_nbr_list.rel_box = smalloc( l->num_intrs * sizeof(ivec),
                "Make_List:far_nbr_list.rel_box", comm );
        l->far_nbr_list.d = smalloc( l->num_intrs * sizeof(real),
                "Make_List:far_nbr_list.d", comm );
        l->far_nbr_list.dvec = smalloc( l->num_intrs * sizeof(rvec),
                "Make_List:far_nbr_list.dvec", comm );
        break;

    case TYP_HBOND:
        l->hbond_list = smalloc( l->num_intrs * sizeof(hbond_data),
                "Make_List:hbonds", comm );
        break;

    default:
        fprintf( stderr, "[ERROR]: no %d list type defined!\n", l->type );
        MPI_Abort( comm, INVALID_INPUT );
    }

    return SUCCESS;
}


void Delete_List( reax_list *l, MPI_Comm comm )
{
    if ( l->allocated == 0 )
        return;
    l->allocated = 0;

    sfree( l->index, "Delete_List:index" );
    sfree( l->end_index, "Delete_List:end_index" );

    switch (l->type)
    {
    case TYP_VOID:
        sfree( l->v, "Delete_List:v" );
        break;
    case TYP_HBOND:
        sfree( l->hbond_list, "Delete_List:hbonds" );
        break;
    case TYP_FAR_NEIGHBOR:
        sfree( l->far_nbr_list.nbr, "Delete_List:far_nbr_list.nbr" );
        sfree( l->far_nbr_list.rel_box, "Delete_List:far_nbr_list.rel_box" );
        sfree( l->far_nbr_list.d, "Delete_List:far_nbr_list.d" );
        sfree( l->far_nbr_list.dvec, "Delete_List:far_nbr_list.dvec" );
        break;
    case TYP_BOND:
        sfree( l->bond_list, "Delete_List:bonds" );
        break;
    case TYP_DBO:
        sfree( l->dbo_list, "Delete_List:dbos" );
        break;
    case TYP_DDELTA:
        sfree( l->dDelta_list, "Delete_List:dDeltas" );
        break;
    case TYP_THREE_BODY:
        sfree( l->three_body_list, "Delete_List:three_bodies" );
        break;

    default:
        fprintf( stderr, "ERROR: no %d list type defined!\n", l->type );
        MPI_Abort( comm, INVALID_INPUT );
    }
}
