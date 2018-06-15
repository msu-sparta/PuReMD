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
void PreAllocate_Space( reax_system *system, control_params *control,
        static_storage *workspace )
{
    int i;

    system->atoms = (reax_atom*) scalloc( system->N,
            sizeof(reax_atom), "PreAllocate_Space::system->atoms" );
    workspace->orig_id = (int*) scalloc( system->N,
            sizeof(int), "PreAllocate_Space::workspace->orid_id" );

    /* space for keeping restriction info, if any */
    if ( control->restrict_bonds )
    {
        workspace->restricted = (int*) scalloc( system->N,
                sizeof(int), "PreAllocate_Space::workspace->restricted_atoms" );

        workspace->restricted_list = (int**) scalloc( system->N,
                sizeof(int*), "PreAllocate_Space::workspace->restricted_list" );

        for ( i = 0; i < system->N; ++i )
        {
            workspace->restricted_list[i] = (int*) scalloc( MAX_RESTRICT,
                    sizeof(int), "PreAllocate_Space::workspace->restricted_list[i]" );
        }
    }
}


static void Reallocate_Neighbor_List( reax_list *far_nbrs, int n, int num_intrs )
{
    Delete_List( TYP_FAR_NEIGHBOR, far_nbrs );

    Make_List( n, num_intrs, TYP_FAR_NEIGHBOR, far_nbrs );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "num_far = %d, far_nbrs = %d -> reallocating!\n",
             num_intrs, far_nbrs->total_intrs );
    fprintf( stderr, "memory allocated: far_nbrs = %ldMB\n",
             num_intrs * sizeof(far_neighbor_data) / (1024 * 1024) );
#endif
}


/* Dynamic allocation of memory for matrix in CSR format
 *
 * pH (output): pointer to sparse matrix for which to allocate
 * n: dimension of the matrix
 * m: number of nonzeros to allocate space for in matrix
 * */
void Allocate_Matrix( sparse_matrix **pH, int n, int m )
{
    sparse_matrix *H;

    *pH = (sparse_matrix*) smalloc( sizeof(sparse_matrix),
            "Allocate_Matrix::pH" );

    H = *pH;
    H->n = n;
    H->m = m;

    H->start = (unsigned int*) smalloc( sizeof(unsigned int) * (n + 1),
            "Allocate_Matrix::H->start" );
    H->j = (unsigned int*) smalloc( sizeof(unsigned int) * m,
            "Allocate_Matrix::H->j" );
    H->val = (real*) smalloc( sizeof(real) * m,
            "Allocate_Matrix::H->val" );
}


/* Deallocate memory for matrix in CSR format
 *
 * H (output): pointer to sparse matrix for which to allocate
 * */
void Deallocate_Matrix( sparse_matrix *H )
{
    sfree( H->start, "Deallocate_Matrix::H->start" );
    sfree( H->j, "Deallocate_Matrix::H->j" );
    sfree( H->val, "Deallocate_Matrix::H->val" );
    sfree( H, "Deallocate_Matrix::H" );
}


static void Reallocate_Matrix( sparse_matrix **H, int n, int m )
{
    Deallocate_Matrix( *H );

    Allocate_Matrix( H, n, m );
}


void Allocate_HBond_List( int n, int num_h, int *h_index, int *hb_top,
        reax_list *hbonds )
{
    int i, num_hbonds;

    num_hbonds = 0;
    /* find starting indexes for each H and the total number of hbonds */
    for ( i = 1; i < n; ++i )
    {
        hb_top[i] += hb_top[i - 1];
    }
    num_hbonds = hb_top[n - 1];

    Make_List( num_h, num_hbonds, TYP_HBOND, hbonds );

    for ( i = 0; i < n; ++i )
    {
        if ( h_index[i] == 0 )
        {
            Set_Start_Index( 0, 0, hbonds );
            Set_End_Index( 0, 0, hbonds );
        }
        else if ( h_index[i] > 0 )
        {
            Set_Start_Index( h_index[i], hb_top[i - 1], hbonds );
            Set_End_Index( h_index[i], hb_top[i - 1], hbonds );
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "allocating hbonds - num_hbonds: %d\n", num_hbonds );
    fprintf( stderr, "memory allocated: hbonds = %ldMB\n",
             num_hbonds * sizeof(hbond_data) / (1024 * 1024) );
#endif
}


static void Reallocate_HBonds_List( int n, int num_h, int *h_index, reax_list *hbonds )
{
    int i;
    int *hb_top;

    hb_top = scalloc( n, sizeof(int), "Reallocate_HBonds_List::hb_top" );
    for ( i = 0; i < n; ++i )
    {
        if ( h_index[i] >= 0 )
        {
            hb_top[i] = MAX(Num_Entries(h_index[i], hbonds) * SAFE_HBONDS, MIN_HBONDS);
        }
    }

    Delete_List( TYP_HBOND, hbonds );

    Allocate_HBond_List( n, num_h, h_index, hb_top, hbonds );

    sfree( hb_top, "Reallocate_HBonds_List::hb_top" );
}


void Allocate_Bond_List( int n, int *bond_top, reax_list *bonds )
{
    int i, num_bonds;

    num_bonds = 0;
    /* find starting indexes for each atom and the total number of bonds */
    for ( i = 1; i < n; ++i )
    {
        bond_top[i] += bond_top[i - 1];
    }
    num_bonds = bond_top[n - 1];

    Make_List( n, num_bonds, TYP_BOND, bonds );

    Set_Start_Index( 0, 0, bonds );
    Set_End_Index( 0, 0, bonds );
    for ( i = 1; i < n; ++i )
    {
        Set_Start_Index( i, bond_top[i - 1], bonds );
        Set_End_Index( i, bond_top[i - 1], bonds );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "allocating bonds - num_bonds: %d\n", num_bonds );
    fprintf( stderr, "memory allocated: bonds = %ldMB\n",
             num_bonds * sizeof(bond_data) / (1024 * 1024) );
#endif
}


static void Reallocate_Bonds_List( int n, reax_list *bonds, int *num_bonds, int *est_3body )
{
    int i;
    int *bond_top;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "reallocating bonds\n" );
#endif

    bond_top = (int *) scalloc( n, sizeof(int), "Reallocate_Bonds_List::hb_top" );
    *est_3body = 0;

    for ( i = 0; i < n; ++i )
    {
        *est_3body += SQR( Num_Entries( i, bonds ) );
        bond_top[i] = MAX( Num_Entries( i, bonds ) * 2, MIN_BONDS );
    }

    Delete_List( TYP_BOND, bonds );

    Allocate_Bond_List( n, bond_top, bonds );
    *num_bonds = bond_top[n - 1];

    sfree( bond_top, "Reallocate_Bonds_List::bond_top" );
}


void Reallocate( reax_system *system, control_params *control, static_storage *workspace, reax_list **lists,
        int nbr_flag )
{
    int i, j, k;
    int num_bonds, est_3body;
    reallocate_data *realloc;
    grid *g;

    realloc = &workspace->realloc;
    g = &system->g;

    if ( realloc->num_far > 0 && nbr_flag )
    {
        Reallocate_Neighbor_List( lists[FAR_NBRS],
                system->N, realloc->num_far * SAFE_ZONE );
        realloc->num_far = -1;
    }

    if ( realloc->Htop > 0 )
    {
        Reallocate_Matrix( &workspace->H, system->N_cm,
                realloc->Htop * SAFE_ZONE );
        realloc->Htop = -1;

        Deallocate_Matrix( workspace->L );
        Deallocate_Matrix( workspace->U );
        workspace->L = NULL;
        workspace->U = NULL;
    }

    if ( control->hb_cut > 0.0 && realloc->hbonds > 0 )
    {
        Reallocate_HBonds_List( system->N, workspace->num_H, workspace->hbond_index,
                lists[HBONDS] );
        realloc->hbonds = -1;
    }

    num_bonds = est_3body = -1;
    if ( realloc->bonds > 0 )
    {
        Reallocate_Bonds_List( system->N, lists[BONDS], &num_bonds, &est_3body );
        realloc->bonds = -1;
        realloc->num_3body = MAX( realloc->num_3body, est_3body );
    }

    if ( realloc->num_3body > 0 )
    {
        Delete_List( TYP_THREE_BODY, lists[THREE_BODIES] );

        if ( num_bonds == -1 )
        {
            num_bonds = lists[BONDS]->total_intrs;
        }
        realloc->num_3body *= SAFE_ZONE;

        Make_List( num_bonds, realloc->num_3body,
                TYP_THREE_BODY, lists[THREE_BODIES] );
        realloc->num_3body = -1;
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "reallocating 3 bodies\n" );
        fprintf( stderr, "reallocated - num_bonds: %d\n", num_bonds );
        fprintf( stderr, "reallocated - num_3body: %d\n", realloc->num_3body );
        fprintf( stderr, "reallocated 3body memory: %ldMB\n",
                 realloc->num_3body * sizeof(three_body_interaction_data) /
                 (1024 * 1024) );
#endif
    }

    if ( realloc->gcell_atoms > -1 )
    {
#if defined(DEBUG_FOCUS)
        fprintf(stderr, "reallocating gcell: g->max_atoms: %d\n", g->max_atoms);
#endif

        for ( i = 0; i < g->ncell[0]; i++ )
        {
            for ( j = 0; j < g->ncell[1]; j++ )
            {
                for ( k = 0; k < g->ncell[2]; k++ )
                {
                    // reallocate g->atoms
                    sfree( g->atoms[i][j][k], "Reallocate::g->atoms[i][j][k]" );
                    g->atoms[i][j][k] = scalloc( workspace->realloc.gcell_atoms, sizeof(int),
                                "Reallocate::g->atoms[i][j][k]" );
                }
            }
        }

        realloc->gcell_atoms = -1;
    }
}
