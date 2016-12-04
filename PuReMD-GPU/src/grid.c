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

#include "grid.h"

#include "reset_utils.h"
#include "vector.h"
#include "index_utils.h"


int Estimate_GCell_Population( reax_system* system )
{
    int i, j, k, l;
    int max_atoms;
    grid *g;

    g = &( system->g );
    Reset_Grid( g );

    for( l = 0; l < system->N; l++ ) {
        i = (int)(system->atoms[l].x[0] * g->inv_len[0]);
        j = (int)(system->atoms[l].x[1] * g->inv_len[1]);
        k = (int)(system->atoms[l].x[2] * g->inv_len[2]);
        g->top[index_grid_3d (i, j, k, g)]++;
        // fprintf( stderr, "\tatom%-6d (%8.3f%8.3f%8.3f) --> (%3d%3d%3d)\n", 
        // l, system->atoms[l].x[0], system->atoms[l].x[1], system->atoms[l].x[2],
        // i, j, k );
    }

    max_atoms = 0;
    for( i = 0; i < g->ncell[0]; i++ )
        for( j = 0; j < g->ncell[1]; j++ )
            for( k = 0; k < g->ncell[2]; k++ )
                if( max_atoms < g->top[index_grid_3d (i, j, k, g)] )
                    max_atoms = g->top[index_grid_3d (i, j, k, g)];  

    return MAX(max_atoms*SAFE_ZONE, MIN_GCELL_POPL); 
}


void Allocate_Space_for_Grid( reax_system *system )
{
    int i, j, k, l;
    grid *g = &(system->g);

    int total = g->ncell[0] * g->ncell[1] * g->ncell[2];

    g = &(system->g);
    g->max_nbrs = (2*g->spread[0]+1) * (2*g->spread[1]+1) * (2*g->spread[2]+1)+3; 

    /* allocate space for the new grid */
    g->top = (int*) calloc( total, sizeof( int ));
    g->mark = (int*) calloc( total, sizeof( int ));
    g->start = (int*) calloc( total, sizeof( int ));
    g->end = (int*) calloc( total, sizeof( int ));
    g->nbrs = (ivec*) calloc( total * g->max_nbrs, sizeof( ivec ));
    g->nbrs_cp = (rvec*) calloc( total * g->max_nbrs, sizeof( rvec ));

    for( i = 0; i < g->ncell[0]; i++ ) {
        for( j = 0; j < g->ncell[1]; j++ ) {
            for( k = 0; k < g->ncell[2]; k++ ) {
                for( l = 0; l < g->max_nbrs; ++l ){ 
                    g->nbrs[ index_grid_nbrs (i, j, k, l, g) ][0] = -1;
                    g->nbrs[ index_grid_nbrs (i, j, k, l, g) ][1] = -1;
                    g->nbrs[ index_grid_nbrs (i, j, k, l, g) ][2] = -1;

                    g->nbrs_cp[ index_grid_nbrs (i, j, k, l, g) ][0] = -1;
                    g->nbrs_cp[ index_grid_nbrs (i, j, k, l, g) ][1] = -1;
                    g->nbrs_cp[ index_grid_nbrs (i, j, k, l, g) ][2] = -1;
                }
            }
        }
    }

    g->max_atoms = Estimate_GCell_Population( system );

    g->atoms = (int*) calloc( total * g->max_atoms, sizeof( int ));
}


void Deallocate_Grid_Space( grid *g )
{
    free( g->atoms );
    free( g->top );
    free( g->mark );
    free( g->nbrs );
    free( g->nbrs_cp );
}


int Shift(int p, int dp, int dim, grid *g )
{
    int dim_len = 0;
    int newp = p + dp;

    switch( dim ) {
        case 0: dim_len = g->ncell[0];
            break;
        case 1: dim_len = g->ncell[1];
            break;
        case 2: dim_len = g->ncell[2];
    }

    while( newp < 0 )        newp = newp + dim_len;
    while( newp >= dim_len ) newp = newp - dim_len;
    return newp;
}


/* finds the closest point between two grid cells denoted by c1 and c2.
   periodic boundary conditions are taken into consideration as well. */
void Find_Closest_Point( grid *g, int c1x, int c1y, int c1z, 
        int c2x, int c2y, int c2z, rvec closest_point )
{
    int  i, d;
    ivec c1 = { c1x, c1y, c1z };
    ivec c2 = { c2x, c2y, c2z };

    for( i = 0; i < 3; i++ ) {
        if( g->ncell[i] < 5 ) {
            closest_point[i] = NEG_INF - 1.;
            continue;
        }

        d = c2[i] - c1[i];
        if( abs(d) <= g->ncell[i] / 2 ) {
            if( d > 0 )
                closest_point[i] = c2[i] * g->len[i];
            else if ( d == 0 )
                closest_point[i] = NEG_INF - 1.;
            else
                closest_point[i] = ( c2[i] + 1 ) * g->len[i];
        }
        else {
            if( d > 0 )
                closest_point[i] = ( c2[i] - g->ncell[i] + 1 ) * g->len[i];
            else    
                closest_point[i] = ( c2[i] + g->ncell[i] ) * g->len[i];
        }
    }
}


void Find_Neighbor_GridCells( grid *g )
{
    int i, j, k;
    int di, dj, dk;
    int x, y, z;
    int stack_top;
    ivec *nbrs_stack;
    rvec *cp_stack;

    /* pick up a cell in the grid */
    for( i = 0; i < g->ncell[0]; i++ )
        for( j = 0; j < g->ncell[1]; j++ )
            for( k = 0; k < g->ncell[2]; k++ ) {
                nbrs_stack = &( g->nbrs[ index_grid_nbrs (i, j, k, 0, g) ] );
                cp_stack = &( g->nbrs_cp[ index_grid_nbrs (i, j, k, 0, g) ] );
                stack_top = 0;
                //fprintf( stderr, "grid1: %d %d %d\n", i, j, k );

                /* choose an unmarked neighbor cell*/
                for( di = -g->spread[0]; di <= g->spread[0]; di++ ) {
                    x = Shift( i, di, 0, g );

                    for( dj = -g->spread[1]; dj <= g->spread[1]; dj++ ) {
                        y = Shift( j, dj, 1, g );

                        for( dk = -g->spread[2]; dk <= g->spread[2]; dk++ ) {
                            z = Shift( k, dk, 2, g );
                            //fprintf( stderr, "\tgrid2: %d %d %d\n", x, y, z );

                            if( !g->mark[ index_grid_3d (x, y, z, g) ] ) {
                                /*(di < 0 || // 9 combinations
                                  (di == 0 && dj < 0) || // 3 combinations
                                  (di == 0 && dj == 0 && dk < 0) ) )*/ 
                                /* put the neighbor cell into the stack and mark it */
                                nbrs_stack[stack_top][0] = x;
                                nbrs_stack[stack_top][1] = y;
                                nbrs_stack[stack_top][2] = z;
                                g->mark[ index_grid_3d(x,y,z,g) ] = 1;

                                Find_Closest_Point( g, i, j, k, x, y, z, cp_stack[stack_top] );
                                //fprintf( stderr, "\tcp: %lf %lf %lf\n", 
                                // cp_stack[stack_top][0], cp_stack[stack_top][1], 
                                // cp_stack[stack_top][2]);
                                stack_top++;
                            }
                        }
                    }
                }

                /*nbrs_stack[stack_top][0] = i;
                  nbrs_stack[stack_top][1] = j;
                  nbrs_stack[stack_top][2] = k;
                  Find_Closest_Point( g, i, j, k, i, j, k, cp_stack[stack_top] );
                  nbrs_stack[stack_top+1][0] = -1;
                  nbrs_stack[stack_top+1][1] = -1;
                  nbrs_stack[stack_top+1][2] = -1;
                  Reset_Marks( g, nbrs_stack, stack_top+1 );*/
                nbrs_stack[stack_top][0] = -1;
                nbrs_stack[stack_top][1] = -1;
                nbrs_stack[stack_top][2] = -1;
                Reset_Marks( g, nbrs_stack, stack_top );
            }
}



void Setup_Grid( reax_system* system )
{
    int  d;
    ivec ncell;
    grid *g = &( system->g );
    simulation_box *my_box = &( system->box );

    /* determine number of grid cells in each direction */
    ivec_rScale( ncell, 1. / g->cell_size, my_box->box_norms );

    for( d = 0; d < 3; ++d )
        if( ncell[d] <= 0 )
            ncell[d] = 1;

    /* find the number of grid cells */
    g->total = ncell[0] * ncell[1] * ncell[2];
    ivec_Copy( g->ncell, ncell );

    /* compute cell lengths */
    rvec_iDivide( g->len, my_box->box_norms, g->ncell );
    rvec_Invert( g->inv_len, g->len );

    Allocate_Space_for_Grid( system );
    Find_Neighbor_GridCells( g );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "setting up the grid: " );
    fprintf( stderr, "ncell[%d %d %d] ", g->ncell[0], g->ncell[1], g->ncell[2] );
    fprintf( stderr, "len[%5.2f %5.2f %5.2f] ", g->len[0], g->len[1], g->len[2] );
    fprintf( stderr, "g->max_atoms = %d\n", g->max_atoms );
#endif
}


void Update_Grid( reax_system* system )
{
    int  d, i, j, k, x, y, z, itr;
    ivec ncell;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g = &( system->g );
    simulation_box *my_box = &( system->box );

    /* determine number of grid cells in each direction */
    ivec_rScale( ncell, 1. / g->cell_size, my_box->box_norms );

    for( d = 0; d < 3; ++d )
        if( ncell[d] == 0 )
            ncell[d] = 1;

    if( ivec_isEqual( ncell, g->ncell ) ) {/* ncell are unchanged */
        /* update cell lengths */
        rvec_iDivide( g->len, my_box->box_norms, g->ncell );
        rvec_Invert( g->inv_len, g->len );

        /* update closest point distances between gcells */
        for( i = 0; i < g->ncell[0]; i++ )
            for( j = 0; j < g->ncell[1]; j++ )
                for( k = 0; k < g->ncell[2]; k++ ) {
                    nbrs = &( g->nbrs[ index_grid_nbrs (i, j, k, 0, g) ] );
                    nbrs_cp = &( g->nbrs_cp[ index_grid_nbrs (i, j, k, 0, g) ] );
                    //fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

                    itr = 0;
                    while( nbrs[itr][0] >= 0 ){
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];

                        Find_Closest_Point( g, i, j, k, x, y, z, nbrs_cp[itr] );
                        ++itr;
                    }
                }
    }
    else{  /* at least one of ncell has changed */
        Deallocate_Grid_Space( g );    
        /* update number of grid cells */
        g->total = ncell[0] * ncell[1] * ncell[2];
        ivec_Copy( g->ncell, ncell );
        /* update cell lengths */
        rvec_iDivide( g->len, my_box->box_norms, g->ncell );
        rvec_Invert( g->inv_len, g->len );

        Allocate_Space_for_Grid( system );
        Find_Neighbor_GridCells( g );
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "updated grid: " );
        fprintf( stderr, "ncell[%d %d %d] ", 
                g->ncell[0], g->ncell[1], g->ncell[2] );
        fprintf( stderr, "len[%5.2f %5.2f %5.2f] ", 
                g->len[0], g->len[1], g->len[2] );
        fprintf( stderr, "g->max_atoms = %d\n", g->max_atoms );
#endif
    }
}


void Bin_Atoms( reax_system* system, static_storage *workspace )
{
    int i, j, k, l;
    int max_atoms;
    grid *g = &( system->g );

    Reset_Grid( g );

    for( l = 0; l < system->N; l++ ) {
        i = (int)(system->atoms[l].x[0] * g->inv_len[0]);
        j = (int)(system->atoms[l].x[1] * g->inv_len[1]);
        k = (int)(system->atoms[l].x[2] * g->inv_len[2]);

#ifdef __BNVT_FIX__
        if (i >= g->ncell[0]) i = g->ncell[0]-1;
        if (j >= g->ncell[1]) j = g->ncell[1]-1;
        if (k >= g->ncell[2]) k = g->ncell[2]-1;
#endif

        g->atoms[ index_grid_atoms (i,j,k,g->top[ index_grid_3d (i,j,k,g) ], g) ] = l;
        g->top[index_grid_3d (i,j,k,g) ]++;

        //fprintf( stderr, "\tatom%-6d (%8.3f%8.3f%8.3f) --> (%3d%3d%3d)\n", 
        //l, system->atoms[l].x[0], system->atoms[l].x[1], system->atoms[l].x[2],
        //i, j, k );
    }

    max_atoms = 0;
    for( i = 0; i < g->ncell[0]; i++ )
        for( j = 0; j < g->ncell[1]; j++ )
            for( k = 0; k < g->ncell[2]; k++ )
                if( max_atoms < g->top[ index_grid_3d (i, j, k, g) ] )
                    max_atoms = g->top[ index_grid_3d (i, j, k, g) ];  

    /* check if current gcell->max_atoms is safe */
    if( max_atoms >= g->max_atoms * SAFE_ZONE ) 
        workspace->realloc.gcell_atoms = MAX(max_atoms*SAFE_ZONE,MIN_GCELL_POPL); 
}


inline void reax_atom_Copy( reax_atom *dest, reax_atom *src )
{
    dest->type = src->type;
    rvec_Copy( dest->x, src->x );
    rvec_Copy( dest->v, src->v );
    strcpy( dest->name, src->name );
}


void Copy_Storage( reax_system *system, static_storage *workspace, 
        int top, int old_id, int old_type, 
        int *num_H, real *v, real *s, real *t, 
        int *orig_id, rvec *f_old )
{
    int i;

    for( i = 0; i < RESTART+1; ++i )
        v[ index_wkspace_sys (i,top, system->N) ] = workspace->v[ index_wkspace_sys (i,old_id, system->N) ];

    for( i = 0; i < 3; ++i ) {
        s[ index_wkspace_sys (i,top, system->N) ] = workspace->s[ index_wkspace_sys (i,old_id, system->N) ];
        t[ index_wkspace_sys (i,top, system->N) ] = workspace->t[ index_wkspace_sys (i,old_id, system->N) ];
    }

    orig_id[top]  = workspace->orig_id[old_id];

    workspace->Hdia_inv[top] = 1. / system->reaxprm.sbp[ old_type ].eta;
    workspace->b_s[top] = -system->reaxprm.sbp[ old_type ].chi;
    workspace->b_t[top] = -1.0;          

    if( system->reaxprm.sbp[ old_type ].p_hbond == 1 ) // H atom
        workspace->hbond_index[top] = (*num_H)++;
    else workspace->hbond_index[top] = -1;

    rvec_Copy( f_old[top], workspace->f_old[old_id] );
}


void Free_Storage( static_storage *workspace )
{
    free( workspace->v );
    free( workspace->s );
    free( workspace->t );
    free( workspace->orig_id );  
}


void Assign_New_Storage( static_storage *workspace, 
        real *v, real *s, real *t, 
        int *orig_id, rvec *f_old )
{
    workspace->v = v;

    workspace->s = s;
    workspace->t = t;

    workspace->orig_id = orig_id;

    workspace->f_old = f_old;
}


void Cluster_Atoms( reax_system *system, static_storage *workspace )
{
    int         i, j, k, l, top, old_id, num_H = 0;
    reax_atom  *old_atom;
    grid       *g = &( system->g );
    reax_atom  *new_atoms = (reax_atom*) calloc( system->N, sizeof(reax_atom) );
    int        *orig_id = (int  *) calloc( system->N, sizeof( int ) );
    real       *v;
    real       *s, *t;
    rvec       *f_old = (rvec*) calloc( system->N, sizeof(rvec) );

    s = (real*) calloc( 3, sizeof( real ) * system->N );
    t = (real*) calloc( 3, sizeof( real ) * system->N );
    v = (real*) calloc( RESTART+1, sizeof( real ) * system->N );

    top = 0;

    for( i = 0; i < g->ncell[0]; i++ )
        for( j = 0; j < g->ncell[1]; j++ )
            for( k = 0; k < g->ncell[2]; k++ ) {
                g->start[ index_grid_3d (i, j, k, g) ] = top;

                for( l = 0; l < g->top[ index_grid_3d (i, j, k, g) ]; ++l ) {
                    old_id   = g->atoms[ index_grid_atoms (i, j, k, l, g) ];
                    old_atom = &( system->atoms[old_id] );
                    // fprintf( stderr, "%d <-- %d\n", top, old_id );

                    reax_atom_Copy( &(new_atoms[top]), old_atom );
                    Copy_Storage( system, workspace, top, old_id, old_atom->type, 
                            &num_H, v, s, t, orig_id, f_old );
                    ++top;
                }

                g->end[ index_grid_3d (i, j, k, g) ] = top;
            }


    free( system->atoms );
    Free_Storage( workspace );

    system->atoms = new_atoms;
    Assign_New_Storage( workspace, v, s, t, orig_id, f_old );
}
