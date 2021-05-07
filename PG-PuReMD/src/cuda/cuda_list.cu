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

    sCudaMalloc( (void **) &l->index, sizeof(int) * n, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &l->end_index, sizeof(int) * n, __FILE__, __LINE__ );

    switch ( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            sCudaMalloc( (void **) &l->far_nbr_list.nbr, 
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->far_nbr_list.rel_box, 
                    sizeof(ivec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->far_nbr_list.d, 
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->far_nbr_list.dvec, 
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            break;

        case TYP_BOND:
            sCudaMalloc( (void **) &l->bond_list,
                    sizeof(bond_data) * l->max_intrs, __FILE__, __LINE__ );
            break;

        case TYP_HBOND:
            sCudaMalloc( (void **) &l->hbond_list, 
                    sizeof(hbond_data) * l->max_intrs, __FILE__, __LINE__ );
            break;            

        case TYP_THREE_BODY:
            sCudaMalloc( (void **) &l->three_body_list,
                    sizeof(three_body_interaction_data) * l->max_intrs,
                    __FILE__, __LINE__ );
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

    sCudaFree( l->index, __FILE__, __LINE__ );
    sCudaFree( l->end_index, __FILE__, __LINE__ );

    switch ( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            sCudaFree( l->far_nbr_list.nbr, __FILE__, __LINE__ );
            sCudaFree( l->far_nbr_list.rel_box, __FILE__, __LINE__ );
            sCudaFree( l->far_nbr_list.d, __FILE__, __LINE__ );
            sCudaFree( l->far_nbr_list.dvec, __FILE__, __LINE__ );
            break;

        case TYP_BOND:
            sCudaFree( l->bond_list, __FILE__, __LINE__ );
            break;

        case TYP_HBOND:
            sCudaFree( l->hbond_list, __FILE__, __LINE__ );
            break;

        case TYP_THREE_BODY:
            sCudaFree( l->three_body_list, __FILE__, __LINE__ );
            break;

        default:
            fprintf( stderr, "[ERROR] unknown devive list type (%d)\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }
}
