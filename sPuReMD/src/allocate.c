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

#include "allocate.h"

#include "list.h"
#include "tool_box.h"


/* allocate space for atoms */
void PreAllocate_Space( reax_system * const system,
        control_params const  * const control,
        static_storage * const workspace, int n )
{
    int i;

    if ( system->prealloc_allocated == FALSE )
    {
        system->prealloc_allocated = TRUE;

        system->atoms = scalloc( n, sizeof(reax_atom),
                "PreAllocate_Space::system->atoms" );
        workspace->orig_id = scalloc( n, sizeof(int),
                "PreAllocate_Space::workspace->orid_id" );

        /* bond restriction info */
        if ( control->restrict_bonds )
        {
            workspace->restricted = scalloc( n, sizeof(int),
                    "PreAllocate_Space::workspace->restricted_atoms" );

            workspace->restricted_list = scalloc( n, sizeof(int*),
                    "PreAllocate_Space::workspace->restricted_list" );

            for ( i = 0; i < n; ++i )
            {
                workspace->restricted_list[i] = scalloc( MAX_RESTRICT, sizeof(int),
                        "PreAllocate_Space::workspace->restricted_list[i]" );
            }
        }

        if ( control->geo_format == BGF
                || control->geo_format == ASCII_RESTART
                || control->geo_format == BINARY_RESTART )
        {
            workspace->map_serials = scalloc( MAX_ATOM_ID, sizeof(int),
                    "Read_BGF::workspace->map_serials" );
        }
    }
    else
    {
        sfree( system->atoms, "PreAllocate_Space::system->atoms" );
        sfree( workspace->orig_id, "PreAllocate_Space::workspace->orid_id" );

        /* bond restriction info */
        if ( control->restrict_bonds )
        {
            sfree( workspace->restricted,
                    "PreAllocate_Space::workspace->restricted_atoms" );

            for ( i = 0; i < n; ++i )
            {
                sfree( workspace->restricted_list[i],
                        "PreAllocate_Space::workspace->restricted_list[i]" );
            }

            sfree( workspace->restricted_list,
                    "PreAllocate_Space::workspace->restricted_list" );
        }

        system->atoms = scalloc( n, sizeof(reax_atom),
                "PreAllocate_Space::system->atoms" );
        workspace->orig_id = scalloc( n, sizeof(int),
                "PreAllocate_Space::workspace->orid_id" );

        /* bond restriction info */
        if ( control->restrict_bonds )
        {
            workspace->restricted = scalloc( n, sizeof(int),
                    "PreAllocate_Space::workspace->restricted_atoms" );

            workspace->restricted_list = scalloc( n, sizeof(int*),
                    "PreAllocate_Space::workspace->restricted_list" );

            for ( i = 0; i < n; ++i )
            {
                workspace->restricted_list[i] = scalloc( MAX_RESTRICT, sizeof(int),
                        "PreAllocate_Space::workspace->restricted_list[i]" );
            }
        }
    }
}


static void Reallocate_Neighbor_List( reax_list *far_nbr_list, int n, int n_max, int num_intrs )
{
    if ( far_nbr_list->allocated == TRUE )
    {
        Delete_List( TYP_FAR_NEIGHBOR, far_nbr_list );
    }

    Make_List( n, n_max, num_intrs, TYP_FAR_NEIGHBOR, far_nbr_list );
}


/* Dynamic allocation of memory for matrix in CSR format
 *
 * pH (output): pointer to sparse matrix for which to allocate
 * n: num. rows of the matrix
 * n_max: max. num. rows of the matrix
 * m: number of nonzeros to allocate space for in matrix
 * */
void Allocate_Matrix( sparse_matrix * const H, int n, int n_max, int m )
{
    H->allocated = TRUE;

    H->n = n;
    H->n_max = n_max;
    H->m = m;

    H->start = smalloc( sizeof(unsigned int) * (n_max + 1), "Allocate_Matrix::H->start" );
    H->j = smalloc( sizeof(unsigned int) * m, "Allocate_Matrix::H->j" );
    H->val = smalloc( sizeof(real) * m, "Allocate_Matrix::H->val" );
}


/* Deallocate memory for matrix in CSR format
 *
 * H (output): pointer to sparse matrix for which to allocate
 * */
void Deallocate_Matrix( sparse_matrix * const H )
{
    H->allocated = FALSE;

    sfree( H->start, "Deallocate_Matrix::H->start" );
    sfree( H->j, "Deallocate_Matrix::H->j" );
    sfree( H->val, "Deallocate_Matrix::H->val" );
}


static void Reallocate_Matrix( sparse_matrix *H, int n, int n_max, int m )
{
    Deallocate_Matrix( H );

    Allocate_Matrix( H, n, n_max, m );
}


void Initialize_HBond_List( int n, int const * const h_index,
        int * const hb_top, reax_list * const hbond_list )
{
    int i;

    /* find starting indexes for each H and the total number of hbonds */
    for ( i = 1; i < n; ++i )
    {
        hb_top[i] += hb_top[i - 1];
    }

    for ( i = 0; i < n; ++i )
    {
        if ( h_index[i] == 0 )
        {
            Set_Start_Index( 0, 0, hbond_list );
            Set_End_Index( 0, 0, hbond_list );
        }
        else if ( h_index[i] > 0 )
        {
            Set_Start_Index( h_index[i], hb_top[i - 1], hbond_list );
            Set_End_Index( h_index[i], hb_top[i - 1], hbond_list );
        }
    }
}


static void Reallocate_Initialize_HBond_List( int n, int num_h, int num_h_max,
        int *h_index, reax_list *hbond_list )
{
    int i, num_hbonds, *hb_top;

    hb_top = scalloc( n, sizeof(int),
            "Reallocate_Initialize_HBond_List::hb_top" );
    num_hbonds = 0;

    for ( i = 0; i < n; ++i )
    {
        if ( h_index[i] >= 0 )
        {
            hb_top[i] = MAX( Num_Entries( h_index[i], hbond_list ) * SAFE_HBONDS,
                    MIN_HBONDS );
            num_hbonds += hb_top[i];
        }
    }

    if ( hbond_list->allocated == TRUE )
    {
        Delete_List( TYP_HBOND, hbond_list );
    }
    Make_List( num_h, num_h_max, num_hbonds, TYP_HBOND, hbond_list );

    Initialize_HBond_List( n, h_index, hb_top, hbond_list );

    sfree( hb_top, "Reallocate_Initialize_HBond_List::hb_top" );
}


void Initialize_Bond_List( int * const bond_top,
        reax_list * const bond_list )
{
    int i;

    /* find starting indexes for each atom and the total number of bonds */
    for ( i = 1; i < bond_list->n; ++i )
    {
        bond_top[i] += bond_top[i - 1];
    }

    Set_Start_Index( 0, 0, bond_list );
    Set_End_Index( 0, 0, bond_list );
    for ( i = 1; i < bond_list->n; ++i )
    {
        Set_Start_Index( i, bond_top[i - 1], bond_list );
        Set_End_Index( i, bond_top[i - 1], bond_list );
    }
}


static void Reallocate_Initialize_Bond_List( int n, int n_max,
        reax_list *bond_list, int *num_bonds, int *est_3body )
{
    int i;
    int *bond_top;

    bond_top = (int *) scalloc( n, sizeof(int),
            "Reallocate_Initialize_Bond_List::hb_top" );
    *num_bonds = 0;
    *est_3body = 0;

    for ( i = 0; i < n; ++i )
    {
        *est_3body += SQR( Num_Entries( i, bond_list ) );
        bond_top[i] = MAX( Num_Entries( i, bond_list ) * 2, MIN_BONDS );
        *num_bonds += bond_top[i];
    }

    if ( bond_list->allocated == TRUE )
    {
        Delete_List( TYP_BOND, bond_list );
    }
    Make_List( n, n_max, (int) CEIL( *num_bonds * SAFE_ZONE ),
            TYP_BOND, bond_list );

    Initialize_Bond_List( bond_top, bond_list );

    sfree( bond_top, "Reallocate_Initialize_Bond_List::bond_top" );
}


void Reallocate( reax_system * const system, control_params const * const control,
        static_storage * const workspace, reax_list ** const lists,
        int nbr_flag )
{
    int i, j, k;
    int num_bonds, est_3body;
    reallocate_data *realloc;
    grid *g;

    realloc = &workspace->realloc;
    g = &system->g;

    if ( realloc->num_far > 0 && nbr_flag == TRUE )
    {
        Reallocate_Neighbor_List( lists[FAR_NBRS],
                system->N, system->N_max, realloc->num_far * SAFE_ZONE );
        realloc->num_far = -1;
    }

    if ( realloc->Htop > 0 )
    {
        Reallocate_Matrix( &workspace->H, system->N_cm, system->N_cm_max,
                realloc->Htop * SAFE_ZONE );
        realloc->Htop = -1;

        Deallocate_Matrix( &workspace->L );
        Deallocate_Matrix( &workspace->U );
    }

    if ( control->hbond_cut > 0.0 && realloc->hbonds > 0 )
    {
        Reallocate_Initialize_HBond_List( system->N, workspace->num_H,
                workspace->num_H_max, workspace->hbond_index, lists[HBONDS] );
        realloc->hbonds = -1;
    }

    num_bonds = est_3body = -1;
    if ( realloc->bonds > 0 )
    {
        Reallocate_Initialize_Bond_List( system->N, system->N_max, lists[BONDS], &num_bonds, &est_3body );
        realloc->bonds = -1;
        realloc->num_3body = MAX( realloc->num_3body, est_3body );
    }

    if ( realloc->num_3body > 0 )
    {
        if ( lists[THREE_BODIES]->allocated == TRUE )
        {
            Delete_List( TYP_THREE_BODY, lists[THREE_BODIES] );
        }

        if ( num_bonds == -1 )
        {
            num_bonds = lists[BONDS]->total_intrs;
        }
        realloc->num_3body *= SAFE_ZONE;

        Make_List( num_bonds, num_bonds, realloc->num_3body,
                TYP_THREE_BODY, lists[THREE_BODIES] );
        realloc->num_3body = -1;

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "[INFO] reallocating 3 bodies\n" );
        fprintf( stderr, "[INFO] reallocated - num_bonds: %d\n", num_bonds );
        fprintf( stderr, "[INFO] reallocated - num_3body: %d\n", realloc->num_3body );
        fprintf( stderr, "[INFO] reallocated 3body memory: %ldMB\n",
                 realloc->num_3body * sizeof(three_body_interaction_data) /
                 (1024 * 1024) );
#endif
    }

    if ( realloc->gcell_atoms > -1 )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "[INFO] reallocating gcell: g->max_atoms: %d\n", g->max_atoms );
#endif

        for ( i = 0; i < g->ncell_max[0]; i++ )
        {
            for ( j = 0; j < g->ncell_max[1]; j++ )
            {
                for ( k = 0; k < g->ncell_max[2]; k++ )
                {
                    sfree( g->atoms[i][j][k], "Reallocate::g->atoms[i][j][k]" );
                    g->atoms[i][j][k] = scalloc( workspace->realloc.gcell_atoms, sizeof(int),
                                "Reallocate::g->atoms[i][j][k]" );
                }
            }
        }

        realloc->gcell_atoms = -1;
    }
}
