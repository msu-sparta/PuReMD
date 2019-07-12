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

#include "grid.h"

#include "reset_tools.h"
#include "tool_box.h"
#include "vector.h"


static int Estimate_GCell_Population( reax_system* system )
{
    int i, j, k, l;
    int max_atoms;
    grid *g;

    g = &system->g;

    Reset_Grid( g );

    for ( l = 0; l < system->N; l++ )
    {
        i = (int)(system->atoms[l].x[0] * g->inv_len[0]);
        j = (int)(system->atoms[l].x[1] * g->inv_len[1]);
        k = (int)(system->atoms[l].x[2] * g->inv_len[2]);
        g->top[i][j][k]++;

//        fprintf( stderr, "\tatom%-6d (%8.3f%8.3f%8.3f) --> (%3d%3d%3d)\n",
//                l, system->atoms[l].x[0], system->atoms[l].x[1], system->atoms[l].x[2],
//                i, j, k );
    }

    max_atoms = 0;
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                if ( max_atoms < g->top[i][j][k] )
                {
                    max_atoms = g->top[i][j][k];
                }
            }
        }
    }

    return MAX( max_atoms * SAFE_ZONE, MIN_GCELL_POPL );
}


static void Allocate_Space_for_Grid( reax_system *system )
{
    int i, j, k, l;
    grid *g;

    g = &system->g;
    g->max_nbrs = (2 * g->spread[0] + 1)
        * (2 * g->spread[1] + 1) * (2 * g->spread[2] + 1) + 3;

    /* allocate space for the new grid */
    g->atoms = (int****) scalloc( g->ncell[0], sizeof( int*** ),
            "Allocate_Space_for_Grid::g->atoms" );
    g->top = (int***) scalloc( g->ncell[0], sizeof( int** ),
            "Allocate_Space_for_Grid::g->top" );
    g->mark = (int***) scalloc( g->ncell[0], sizeof( int** ),
            "Allocate_Space_for_Grid::g->mark" );
    g->start = (int***) scalloc( g->ncell[0], sizeof( int** ),
            "Allocate_Space_for_Grid::g->start" );
    g->end = (int***) scalloc( g->ncell[0], sizeof( int** ),
            "Allocate_Space_for_Grid::g->end" );
    g->nbrs = (ivec****) scalloc( g->ncell[0], sizeof( ivec*** ),
            "Allocate_Space_for_Grid::g->nbrs" );
    g->nbrs_cp = (rvec****) scalloc( g->ncell[0], sizeof( rvec*** ),
            "Allocate_Space_for_Grid::g->nbrs_cp" );

    for ( i = 0; i < g->ncell[0]; i++ )
    {
        g->atoms[i] = (int***) scalloc( g->ncell[1], sizeof( int** ),
                "Allocate_Space_for_Grid::g->atoms[i]" );
        g->top [i] = (int**) scalloc( g->ncell[1], sizeof( int* ),
                "Allocate_Space_for_Grid::g->top[i]" );
        g->mark[i] = (int**) scalloc( g->ncell[1], sizeof( int* ),
                "Allocate_Space_for_Grid::g->mark[i]" );
        g->start[i] = (int**) scalloc( g->ncell[1], sizeof( int* ),
                "Allocate_Space_for_Grid::g->start[i]" );
        g->end[i] = (int**) scalloc( g->ncell[1], sizeof( int* ),
                "Allocate_Space_for_Grid::g->end[i]" );
        g->nbrs[i] = (ivec***) scalloc( g->ncell[1], sizeof( ivec** ),
                "Allocate_Space_for_Grid::g->nbrs[i]" );
        g->nbrs_cp[i] = (rvec***) scalloc( g->ncell[1], sizeof( rvec** ),
                "Allocate_Space_for_Grid::g->nbrs_cp[i]" );

        for ( j = 0; j < g->ncell[1]; j++ )
        {
            g->atoms[i][j] = (int**) scalloc( g->ncell[2], sizeof( int* ),
                    "Allocate_Space_for_Grid::g->atoms[i][j]" );
            g->top[i][j] = (int*) scalloc( g->ncell[2], sizeof( int ),
                    "Allocate_Space_for_Grid::g->top[i][j]" );
            g->mark[i][j] = (int*) scalloc( g->ncell[2], sizeof( int ),
                    "Allocate_Space_for_Grid::g->mark[i][j]" );
            g->start[i][j] = (int*) scalloc( g->ncell[2], sizeof( int ),
                    "Allocate_Space_for_Grid::g->start[i][j]" );
            g->end[i][j] = (int*) scalloc( g->ncell[2], sizeof( int ),
                    "Allocate_Space_for_Grid::g->end[i][j]" );
            g->nbrs[i][j] = (ivec**) scalloc( g->ncell[2], sizeof( ivec* ),
                    "Allocate_Space_for_Grid::g->nbrs[i][j]" );
            g->nbrs_cp[i][j] = (rvec**) scalloc( g->ncell[2], sizeof( rvec* ),
                    "Allocate_Space_for_Grid::g->nbrs_cp[i][j]" );

            for ( k = 0; k < g->ncell[2]; k++ )
            {
                g->top[i][j][k] = 0;
                g->mark[i][j][k] = 0;
                g->start[i][j][k] = 0;
                g->end[i][j][k] = 0;
                g->nbrs[i][j][k] = (ivec*) smalloc( g->max_nbrs * sizeof( ivec ),
                       "Allocate_Space_for_Grid::g->nbrs[i][j]][k]" );
                g->nbrs_cp[i][j][k] = (rvec*) smalloc( g->max_nbrs * sizeof( rvec ),
                       "Allocate_Space_for_Grid::g->nbrs_cp[i][j]][k]" );

                for ( l = 0; l < g->max_nbrs; ++l )
                {
                    g->nbrs[i][j][k][l][0] = -1;
                    g->nbrs[i][j][k][l][1] = -1;
                    g->nbrs[i][j][k][l][2] = -1;

                    g->nbrs_cp[i][j][k][l][0] = -1.0;
                    g->nbrs_cp[i][j][k][l][1] = -1.0;
                    g->nbrs_cp[i][j][k][l][2] = -1.0;
                }
            }
        }
    }

    g->max_atoms = Estimate_GCell_Population( system );

    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                g->atoms[i][j][k] = (int*) smalloc( g->max_atoms * sizeof( int ),
                       "Allocate_Space_for_Grid::g->atoms[i][j]][k]" );

                for ( l = 0; l < g->max_atoms; ++l )
                {
                    g->atoms[i][j][k][l] = -1;
                }
            }
        }
    }

}


static void Deallocate_Grid_Space( grid *g )
{
    int i, j, k;

    /* deallocate the old grid */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                sfree( g->atoms[i][j][k], "Deallocate_Grid_Space::g->atoms[i][j][k]" );
                sfree( g->nbrs[i][j][k], "Deallocate_Grid_Space::g->nbrs[i][j][k]" );
                sfree( g->nbrs_cp[i][j][k], "Deallocate_Grid_Space::g->nbrs_cp[i][j][k]" );
            }

            sfree( g->atoms[i][j], "Deallocate_Grid_Space::g->atoms[i][j]" );
            sfree( g->top[i][j], "Deallocate_Grid_Space::g->top[i][j]" );
            sfree( g->mark[i][j], "Deallocate_Grid_Space::g->mark[i][j]" );
            sfree( g->start[i][j], "Deallocate_Grid_Space::g->start[i][j]" );
            sfree( g->end[i][j], "Deallocate_Grid_Space::g->end[i][j]" );
            sfree( g->nbrs[i][j], "Deallocate_Grid_Space::g->nbrs[i][j]" );
            sfree( g->nbrs_cp[i][j], "Deallocate_Grid_Space::g->nbrs_cp[i][j]" );
        }

        sfree( g->atoms[i], "Deallocate_Grid_Space::g->atoms[i]" );
        sfree( g->top[i], "Deallocate_Grid_Space::g->top[i]" );
        sfree( g->mark[i], "Deallocate_Grid_Space::g->mark[i]" );
        sfree( g->start[i], "Deallocate_Grid_Space::g->start[i]" );
        sfree( g->end[i], "Deallocate_Grid_Space::g->end[i]" );
        sfree( g->nbrs[i], "Deallocate_Grid_Space::g->nbrs[i]" );
        sfree( g->nbrs_cp[i], "Deallocate_Grid_Space::g->nbrs_cp[i]" );
    }

    sfree( g->atoms, "Deallocate_Grid_Space::g->atoms" );
    sfree( g->top, "Deallocate_Grid_Space::g->top" );
    sfree( g->mark, "Deallocate_Grid_Space::g->mark" );
    sfree( g->start, "Deallocate_Grid_Space::g->start" );
    sfree( g->end, "Deallocate_Grid_Space::g->end" );
    sfree( g->nbrs, "Deallocate_Grid_Space::g->nbrs" );
    sfree( g->nbrs_cp, "Deallocate_Grid_Space::g->nbrs_cp" );
}


static inline int Shift( int p, int dp, int dim, grid *g )
{
    int dim_len, newp;

    dim_len = 0;
    newp = p + dp;

    switch ( dim )
    {
    case 0:
        dim_len = g->ncell[0];
        break;
    case 1:
        dim_len = g->ncell[1];
        break;
    case 2:
        dim_len = g->ncell[2];
    }

    while ( newp < 0 )
    {
        newp = newp + dim_len;
    }
    while ( newp >= dim_len )
    {
        newp = newp - dim_len;
    }

    return newp;
}


/* finds the closest point between two grid cells denoted by c1 and c2.
 * periodic boundary conditions are taken into consideration as well. */
static void Find_Closest_Point( grid *g, int c1x, int c1y, int c1z,
        int c2x, int c2y, int c2z, rvec closest_point )
{
    int i, d;
    ivec c1 = { c1x, c1y, c1z };
    ivec c2 = { c2x, c2y, c2z };

    for ( i = 0; i < 3; i++ )
    {
        if ( g->ncell[i] < 5 )
        {
            closest_point[i] = NEG_INF - 1.0;
            continue;
        }

        d = c2[i] - c1[i];
        if ( abs(d) <= g->ncell[i] / 2 )
        {
            if ( d > 0 )
            {
                closest_point[i] = c2[i] * g->len[i];
            }
            else if ( d == 0 )
            {
                closest_point[i] = NEG_INF - 1.;
            }
            else
            {
                closest_point[i] = ( c2[i] + 1 ) * g->len[i];
            }
        }
        else
        {
            if ( d > 0 )
            {
                closest_point[i] = ( c2[i] - g->ncell[i] + 1 ) * g->len[i];
            }
            else
            {
                closest_point[i] = ( c2[i] + g->ncell[i] ) * g->len[i];
            }
        }
    }
}


static void Find_Neighbor_Grid_Cells( grid *g )
{
    int i, j, k;
    int di, dj, dk;
    int x, y, z;
    int stack_top;
    ivec *nbrs_stack;
    rvec *cp_stack;

    /* for each cell in the grid */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs_stack = g->nbrs[i][j][k];
                cp_stack = g->nbrs_cp[i][j][k];
                stack_top = 0;
                //fprintf( stderr, "grid1: %d %d %d\n", i, j, k );

                /* choose an unmarked neighbor cell */
                for ( di = -g->spread[0]; di <= g->spread[0]; di++ )
                {
                    x = Shift( i, di, 0, g );

                    for ( dj = -g->spread[1]; dj <= g->spread[1]; dj++ )
                    {
                        y = Shift( j, dj, 1, g );

                        for ( dk = -g->spread[2]; dk <= g->spread[2]; dk++ )
                        {
                            z = Shift( k, dk, 2, g );
                            //fprintf( stderr, "\tgrid2: %d %d %d\n", x, y, z );

                            if ( !g->mark[x][y][z] )
                            {
                                /*(di < 0 || // 9 combinations
                                 (di == 0 && dj < 0) || // 3 combinations
                                 (di == 0 && dj == 0 && dk < 0) ) )*/
                                /* put the neighbor cell into the stack and mark it */
                                nbrs_stack[stack_top][0] = x;
                                nbrs_stack[stack_top][1] = y;
                                nbrs_stack[stack_top][2] = z;
                                g->mark[x][y][z] = 1;

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
    }
}


void Setup_Grid( reax_system* system )
{
    int d;
    ivec ncell;
    grid *g;
    simulation_box *my_box;

    g = &system->g;
    my_box = &system->box;

    /* determine number of grid cells in each direction */
    ivec_rScale( ncell, 1.0 / g->cell_size, my_box->box_norms );

    for ( d = 0; d < 3; ++d )
    {
        if ( ncell[d] <= 0 )
        {
            ncell[d] = 1;
        }
    }

    /* find the number of grid cells */
    g->total = ncell[0] * ncell[1] * ncell[2];
    ivec_Copy( g->ncell, ncell );

    /* compute cell lengths */
    rvec_iDivide( g->len, my_box->box_norms, g->ncell );
    rvec_Invert( g->inv_len, g->len );

    Allocate_Space_for_Grid( system );
    Find_Neighbor_Grid_Cells( g );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "setting up the grid: " );
    fprintf( stderr, "ncell[%d %d %d] ", g->ncell[0], g->ncell[1], g->ncell[2] );
    fprintf( stderr, "len[%5.2f %5.2f %5.2f] ", g->len[0], g->len[1], g->len[2] );
    fprintf( stderr, "g->max_atoms = %d\n", g->max_atoms );
#endif
}


void Update_Grid( reax_system* system )
{
    int d, i, j, k, x, y, z, itr;
    ivec ncell;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g;
    simulation_box *my_box;

    g = &system->g;
    my_box = &system->box;

    /* determine number of grid cells in each direction */
    ivec_rScale( ncell, 1.0 / g->cell_size, my_box->box_norms );

    for ( d = 0; d < 3; ++d )
    {
        if ( ncell[d] == 0 )
        {
            ncell[d] = 1;
        }
    }

    /* ncell are unchanged */
    if ( ivec_isEqual( ncell, g->ncell ) )
    {
        /* update cell lengths */
        rvec_iDivide( g->len, my_box->box_norms, g->ncell );
        rvec_Invert( g->inv_len, g->len );

        /* update closest point distances between gcells */
        for ( i = 0; i < g->ncell[0]; i++ )
        {
            for ( j = 0; j < g->ncell[1]; j++ )
            {
                for ( k = 0; k < g->ncell[2]; k++ )
                {
                    nbrs = g->nbrs[i][j][k];
                    nbrs_cp = g->nbrs_cp[i][j][k];
                    //fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

                    itr = 0;
                    while ( nbrs[itr][0] >= 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];

                        Find_Closest_Point( g, i, j, k, x, y, z, nbrs_cp[itr] );
                        ++itr;
                    }
                }
            }
        }
    }
    /* at least one of ncell has changed */
    else
    {
        Deallocate_Grid_Space( g );

        /* update number of grid cells */
        g->total = ncell[0] * ncell[1] * ncell[2];
        ivec_Copy( g->ncell, ncell );

        /* update cell lengths */
        rvec_iDivide( g->len, my_box->box_norms, g->ncell );
        rvec_Invert( g->inv_len, g->len );

        Allocate_Space_for_Grid( system );
        Find_Neighbor_Grid_Cells( g );

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
    grid *g;

    g = &system->g;

    Reset_Grid( g );

    for ( l = 0; l < system->N; l++ )
    {
        i = (int)(system->atoms[l].x[0] * g->inv_len[0]);
        j = (int)(system->atoms[l].x[1] * g->inv_len[1]);
        k = (int)(system->atoms[l].x[2] * g->inv_len[2]);
        g->atoms[i][j][k][g->top[i][j][k]] = l;
        g->top[i][j][k]++;

        // fprintf( stderr, "\tatom%-6d (%8.3f%8.3f%8.3f) --> (%3d%3d%3d)\n",
        // l, system->atoms[l].x[0], system->atoms[l].x[1], system->atoms[l].x[2],
        // i, j, k );
    }

    max_atoms = 0;
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                if ( max_atoms < g->top[i][j][k] )
                {
                    max_atoms = g->top[i][j][k];
                }
            }
        }
    }

    /* check if current gcell->max_atoms is safe */
    if ( max_atoms >= g->max_atoms * SAFE_ZONE )
    {
        workspace->realloc.gcell_atoms = MAX( max_atoms * SAFE_ZONE,
                MIN_GCELL_POPL );
    }
}


void Finalize_Grid( reax_system* system )
{
    Deallocate_Grid_Space( &system->g );
}


static inline void reax_atom_Copy( reax_atom *dest, reax_atom *src )
{
    dest->type = src->type;
    strncpy( dest->name, src->name, sizeof(dest->name) - 1 );
    dest->name[sizeof(dest->name) - 1] = '\0';
    rvec_Copy( dest->x, src->x );
    rvec_Copy( dest->v, src->v );
    rvec_Copy( dest->f, src->f );
    dest->q = src->q;
}


static void Copy_Storage( reax_system *system, static_storage *workspace,
        control_params *control, int top, int old_id, int old_type, int *num_H,
        real **v, real **s, real **t, int *orig_id, rvec *f_old )
{
    int i;

    for ( i = 0; i < control->cm_solver_restart + 1; ++i )
    {
        v[i][top] = workspace->v[i][old_id];
    }

    for ( i = 0; i < 5; ++i )
    {
        s[i][top] = workspace->s[i][old_id];
        t[i][top] = workspace->t[i][old_id];
    }

    orig_id[top] = workspace->orig_id[old_id];

    workspace->b_s[top] = -system->reax_param.sbp[ old_type ].chi;
    workspace->b_t[top] = -1.0;

    if ( system->reax_param.sbp[ old_type ].p_hbond == 1 ) // H atom
    {
        workspace->hbond_index[top] = (*num_H)++;
    }
    else
    {
        workspace->hbond_index[top] = -1;
    }

    rvec_Copy( f_old[top], workspace->f_old[old_id] );
}


static void Free_Storage( static_storage *workspace, control_params * control )
{
    int i;

    for ( i = 0; i < control->cm_solver_restart + 1; ++i )
    {
        sfree( workspace->v[i], "Free_Storage::workspace->v[i]" );
    }
    sfree( workspace->v, "Free_Storage::workspace->v" );

    for ( i = 0; i < 3; ++i )
    {
        sfree( workspace->s[i], "Free_Storage::workspace->s[i]" );
        sfree( workspace->t[i], "Free_Storage::workspace->t[i]" );
    }
    sfree( workspace->s, "Free_Storage::workspace->s" );
    sfree( workspace->t, "Free_Storage::workspace->t" );

    sfree( workspace->orig_id, "Free_Storage::workspace->orig_id" );
}


static void Assign_New_Storage( static_storage *workspace,
        real **v, real **s, real **t, int *orig_id, rvec *f_old )
{
    workspace->v = v;
    workspace->s = s;
    workspace->t = t;
    workspace->orig_id = orig_id;
    workspace->f_old = f_old;
}


void Cluster_Atoms( reax_system *system, static_storage *workspace,
        control_params *control )
{
    int i, j, k, l, top, old_id, num_H;
    reax_atom *old_atom, *new_atoms;
    grid *g;
    int *orig_id;
    real **v;
    real **s, **t;
    rvec *f_old;

    num_H = 0;
    top = 0;
    g = &system->g;

    new_atoms = scalloc( system->N, sizeof(reax_atom), "Cluster_Atoms::new_atoms" );
    orig_id = scalloc( system->N, sizeof( int ), "Cluster_Atoms::orig_id" );
    f_old = scalloc( system->N, sizeof(rvec), "Cluster_Atoms::f_old" );

    s = scalloc( 5, sizeof( real* ), "Cluster_Atoms::s" );
    t = scalloc( 5, sizeof( real* ), "Cluster_Atoms::t" );
    for ( i = 0; i < 5; ++i )
    {
        s[i] = scalloc( system->N_cm, sizeof( real ), "Cluster_Atoms::s[i]" );
        t[i] = scalloc( system->N_cm, sizeof( real ), "Cluster_Atoms::t[i]" );
    }

    v = scalloc( control->cm_solver_restart + 1, sizeof( real* ),
            "Cluster_Atoms::v" );
    for ( i = 0; i < control->cm_solver_restart + 1; ++i )
    {
        v[i] = scalloc( system->N_cm, sizeof( real ), "Cluster_Atoms::v[i]" );
    }

    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                g->start[i][j][k] = top;

                for ( l = 0; l < g->top[i][j][k]; ++l )
                {
                    old_id = g->atoms[i][j][k][l];
                    old_atom = &system->atoms[old_id];
                    // fprintf( stderr, "%d <-- %d\n", top, old_id );

                    reax_atom_Copy( &new_atoms[top], old_atom );
                    Copy_Storage( system, workspace, control, top, old_id, old_atom->type,
                            &num_H, v, s, t, orig_id, f_old );

                    ++top;
                }

                g->end[i][j][k] = top;
            }
        }
    }

    sfree( system->atoms, "Cluster_Atoms::system->atoms" );
    Free_Storage( workspace, control );

    system->atoms = new_atoms;
    Assign_New_Storage( workspace, v, s, t, orig_id, f_old );
}
