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

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "../../common/include/config.h"
#endif

#if defined(PURE_REAX)
  #include "list.h"

  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_list.h"

  #include "reax_tool_box.h"
#endif


void Print_List_Indices( reax_list * const l )
{
    int i;

    assert( l != NULL );

    for ( i = 0; i < l->n; i++ )
    {
        printf( "%d %d\n", l->index[i], l->end_index[i] );
        if ( i > 1 && l->end_index[i - 1] >= l->index[i] )
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
 * format: list format type
 * l: pointer to list to be allocated
 * */
void Make_List( int n, int max_intrs, int type, int format, reax_list * const l )
{
    assert( n > 0 );
    assert( max_intrs > 0 );
    assert( l != NULL );

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
    l->format = format;

    l->index = smalloc( sizeof(int) * n, __FILE__, __LINE__ );
    l->end_index = smalloc( sizeof(int) * n, __FILE__, __LINE__ );

    switch ( l->type )
    {
    case TYP_VOID:
        l->v = smalloc( sizeof(void*) * l->max_intrs, __FILE__, __LINE__ );
        break;

    case TYP_FAR_NEIGHBOR:
        l->far_nbr_list.nbr = smalloc( sizeof(int) * l->max_intrs,
                __FILE__, __LINE__ );
        l->far_nbr_list.rel_box = smalloc( sizeof(ivec) * l->max_intrs,
                __FILE__, __LINE__ );
        l->far_nbr_list.d = smalloc( sizeof(real) * l->max_intrs,
                __FILE__, __LINE__ );
        l->far_nbr_list.dvec = smalloc( sizeof(rvec) * l->max_intrs,
                __FILE__, __LINE__ );
        break;

    case TYP_BOND:
        l->bond_list = smalloc( sizeof(bond_data) * l->max_intrs, __FILE__, __LINE__ );
        break;

    case TYP_HBOND:
        l->hbond_list.nbr = smalloc( sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
        l->hbond_list.scl = smalloc( sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
        l->hbond_list.ptr = smalloc( sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
#if (defined(HAVE_CUDA) || defined(HAVE_HIP)) && !defined(GPU_KERNEL_ATOMIC)
//        l->hbond_list.sym_index = smalloc( sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
//        l->hbond_list.hb_f = smalloc( sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
#endif
        break;

    case TYP_THREE_BODY:
        l->three_body_list = smalloc( sizeof(three_body_interaction_data) * l->max_intrs,
                __FILE__, __LINE__ );
        break;

#if defined(TEST_FORCES)
    case TYP_DBO:
        l->dbo_list = smalloc( sizeof(dbond_data) * l->max_intrs, __FILE__, __LINE__ );
        break;

    case TYP_DDELTA:
        l->dDelta_list = smalloc( sizeof(dDelta_data) * l->max_intrs, __FILE__, __LINE__ );
        break;
#endif

    default:
        fprintf( stderr, "[ERROR] unknown list type (%d)\n", l->type );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;
    }
}


void Delete_List( reax_list * const l )
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
    l->max_intrs = 0;

    sfree( l->index, __FILE__, __LINE__ );
    sfree( l->end_index, __FILE__, __LINE__ );

    switch ( l->type )
    {
    case TYP_VOID:
        sfree( l->v, __FILE__, __LINE__ );
        break;

    case TYP_FAR_NEIGHBOR:
        sfree( l->far_nbr_list.nbr, __FILE__, __LINE__ );
        sfree( l->far_nbr_list.rel_box, __FILE__, __LINE__ );
        sfree( l->far_nbr_list.d, __FILE__, __LINE__ );
        sfree( l->far_nbr_list.dvec, __FILE__, __LINE__ );
        break;

    case TYP_BOND:
        sfree( l->bond_list, __FILE__, __LINE__ );
        break;

    case TYP_HBOND:
        sfree( l->hbond_list.nbr, __FILE__, __LINE__ );
        sfree( l->hbond_list.scl, __FILE__, __LINE__ );
        sfree( l->hbond_list.ptr, __FILE__, __LINE__ );
#if (defined(HAVE_CUDA) || defined(HAVE_HIP)) && !defined(GPU_KERNEL_ATOMIC)
//        sfree( l->hbond_list.sym_index, __FILE__, __LINE__ );
//        sfree( l->hbond_list.hb_f, __FILE__, __LINE__ );
#endif
        break;

    case TYP_THREE_BODY:
        sfree( l->three_body_list, __FILE__, __LINE__ );
        break;

#if defined(TEST_FORCES)
    case TYP_DBO:
        sfree( l->dbo_list, __FILE__, __LINE__ );
        break;

    case TYP_DDELTA:
        sfree( l->dDelta_list, __FILE__, __LINE__ );
        break;
#endif

    default:
        fprintf( stderr, "[ERROR] unknown list type (%d)\n", l->type );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;
    }
}


/* Initialize list indices
 *
 * l: pointer to list
 * max_intrs: max. num. of interactions for each list element
 * */
void Init_List_Indices( reax_list * const l, int * const max_intrs )
{
    int i;

    assert( l != NULL );
    assert( l->n > 0 );
    assert( max_intrs > 0 );

    /* exclusive prefix sum of max_intrs replaces start indices,
     * set end indices to the same as start indices for safety */
    Set_Start_Index( 0, 0, l );
    Set_End_Index( 0, 0, l );
    for ( i = 1; i < l->n; ++i )
    {
        Set_Start_Index( i, Start_Index( i - 1, l ) + max_intrs[i - 1], l );
        Set_End_Index( i, Start_Index( i, l ), l );
    }
}
