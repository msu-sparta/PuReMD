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


/* Allocate space for interaction list
 *
 * n: num. of elements to be allocated for list
 * max_intrs: max. num. of interactions for which to allocate space
 * type: list interaction type
 * l: pointer to list to be allocated
 * */
extern "C" void Cuda_Make_List( int n, int max_intrs, int type, reax_list * const l )
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
//    l->format = format;

    cuda_malloc( (void **) &l->index, sizeof(int) * n,
            TRUE, "Cuda_Make_List::index" );
    cuda_malloc( (void **) &l->end_index, sizeof(int) * n,
            TRUE, "Cuda_Make_List::end_index" );

    switch ( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            cuda_malloc( (void **) &l->far_nbr_list.nbr, 
                    sizeof(int) * l->max_intrs, TRUE,
                    "Cuda_Make_List::far_nbr_list.nbr" );
            cuda_malloc( (void **) &l->far_nbr_list.rel_box, 
                    sizeof(ivec) * l->max_intrs, TRUE,
                    "Cuda_Make_List::far_nbr_list.rel_box" );
            cuda_malloc( (void **) &l->far_nbr_list.d, 
                    sizeof(real) * l->max_intrs, TRUE,
                    "Cuda_Make_List::far_nbr_list.d" );
            cuda_malloc( (void **) &l->far_nbr_list.dvec, 
                    sizeof(rvec) * l->max_intrs, TRUE,
                    "Cuda_Make_List::far_nbr_list.dvec" );
            break;

        case TYP_BOND:
            cuda_malloc( (void **) &l->bond_list,
                    sizeof(bond_data) * l->max_intrs, TRUE,
                    "Cuda_Make_List::bonds" );
            break;

        case TYP_HBOND:
            cuda_malloc( (void **) &l->hbond_list, 
                    sizeof(hbond_data) * l->max_intrs, TRUE,
                    "Cuda_Make_List::hbonds" );
            break;            

        case TYP_THREE_BODY:
            cuda_malloc( (void **) &l->three_body_list,
                    sizeof(three_body_interaction_data) * l->max_intrs, TRUE,
                    "Cuda_Make_List::three_bodies" );
            break;

        default:
            fprintf( stderr, "[ERROR] unknown devive list type (%d)\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }
}


extern "C" void Cuda_Delete_List( reax_list *l )
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

    cuda_free( l->index, "Cuda_Delete_List::index" );
    cuda_free( l->end_index, "Cuda_Delete_List::end_index" );

    switch ( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            cuda_free( l->far_nbr_list.nbr, "Cuda_Delete_List::far_nbr_list.nbr" );
            cuda_free( l->far_nbr_list.rel_box, "Cuda_Delete_List::far_nbr_list.rel_box" );
            cuda_free( l->far_nbr_list.d, "Cuda_Delete_List::far_nbr_list.d" );
            cuda_free( l->far_nbr_list.dvec, "Cuda_Delete_List::far_nbr_list.dvec" );
            break;

        case TYP_BOND:
            cuda_free( l->bond_list, "Cuda_Delete_List::bonds" );
            break;

        case TYP_HBOND:
            cuda_free( l->hbond_list, "Cuda_Delete_List::hbonds" );
            break;

        case TYP_THREE_BODY:
            cuda_free( l->three_body_list, "Cuda_Delete_List::three_bodies" );
            break;

        default:
            fprintf( stderr, "[ERROR] unknown devive list type (%d)\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }
}
