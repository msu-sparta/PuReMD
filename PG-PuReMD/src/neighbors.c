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

#include "neighbors.h"

#include "index_utils.h"
#include "io_tools.h"
#include "list.h"
#include "tool_box.h"
#include "vector.h"


#if defined(TEST_ENERGY) || defined(TEST_FORCES)
int compare_far_nbrs( const void *p1, const void *p2 )
{
    return ((far_neighbor_data *)p1)->nbr - ((far_neighbor_data *)p2)->nbr;
}
#endif


int Generate_Neighbor_Lists( reax_system *system, simulation_data *data,
        storage *workspace, reax_list **lists )
{
    int i, j, k, l, m, itr, num_far;
    real d, cutoff;
    rvec dvec;
    grid *g;
    reax_list *far_nbr_list;
    reax_atom *atom1, *atom2;
#if defined(LOG_PERFORMANCE)
    double time;
    
    time = Get_Time( );
#endif

    far_nbr_list = lists[FAR_NBRS];
    g = &system->my_grid;

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncells[0]; i++ )
    {
        for ( j = 0; j < g->ncells[1]; j++ )
        {
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                cutoff = SQR( g->cutoff[ index_grid_3d(i, j, k, g) ] );

                /* pick up an atom from the current cell */
                for ( l = g->str[ index_grid_3d(i, j, k, g) ];
                        l < g->end[ index_grid_3d(i, j, k, g) ]; ++l )
                {
                    atom1 = &system->my_atoms[l];
#if defined(NEUTRAL_TERRITORY)
                    if( g->type[ index_grid_3d(i, j, k, g) ] >= NT_NBRS
                            && g->type[ index_grid_3d(i, j, k, g) ] < NT_NBRS + 6 )
                    {
                        atom1->nt_dir = g->type[ index_grid_3d(i, j, k, g) ] - NT_NBRS;
                    }
                    else
                    {
                        atom1->nt_dir = -1;
                    }
#endif
                    num_far = Start_Index( l, far_nbr_list );
                    itr = 0;

                    /* search through neighbor grid cell candidates of current cell */
                    while ( g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ][0] >= 0 )
                    {
                        if ( ((far_nbr_list->format == HALF_LIST
                                        && g->str[ index_grid_3d(i, j, k, g) ] <=
                                        g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ])
                                    || far_nbr_list->format == FULL_LIST)
                                && DistSqr_to_Special_Point( g->nbrs_cp[ index_grid_nbrs(i, j, k, itr, g) ], atom1->x ) <= cutoff )
                        {
                            /* pick up another atom from the neighbor grid cell */
                            for ( m = g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ];
                                    m < g->end[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ]; ++m )
                            {
                                /* HALF_LIST: prevent recounting same pairs within a gcell and
                                 *  make half-list
                                 * FULL_LIST: prevent recounting same pairs within a gcell */
                                if ( (far_nbr_list->format == HALF_LIST && l < m)
                                        || (far_nbr_list->format == FULL_LIST && l != m) )
                                {
                                    atom2 = &system->my_atoms[m];
                                    dvec[0] = atom2->x[0] - atom1->x[0];
                                    dvec[1] = atom2->x[1] - atom1->x[1];
                                    dvec[2] = atom2->x[2] - atom1->x[2];
                                    d = rvec_Norm_Sqr( dvec );

                                    if ( d <= cutoff )
                                    {
                                        far_nbr_list->far_nbr_list.nbr[num_far] = m;
                                        far_nbr_list->far_nbr_list.d[num_far] = SQRT( d );
                                        rvec_Copy( far_nbr_list->far_nbr_list.dvec[num_far], dvec );
                                        ivec_ScaledSum( far_nbr_list->far_nbr_list.rel_box[num_far],
                                                1, g->rel_box[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ],
                                                -1, g->rel_box[ index_grid_3d(i, j, k, g) ] );
                                        ++num_far;
                                    }
                                }
                            }
                        }

                        ++itr;
                    }

                    Set_End_Index( l, num_far, far_nbr_list );
                }
            }
        }
    }

    for ( i = 0; i < system->total_cap; i++ )
    {
        /* reallocation check */
        if ( Num_Entries( i, far_nbr_list ) > system->max_far_nbrs[i] )
        {
            workspace->realloc.far_nbrs = TRUE;
        }
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.nbrs );
#endif

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
    for ( i = 0; i < system->N; ++i )
    {
        qsort( &far_nbr_list->far_nbr_list[ Start_Index(i, far_nbr_list) ],
                Num_Entries(i, far_nbr_list), sizeof(far_neighbor_data),
                compare_far_nbrs );
    }
#endif

    return (workspace->realloc.far_nbrs == FALSE) ? SUCCESS : FAILURE;
}


void Estimate_Num_Neighbors( reax_system *system, simulation_data *data,
        int far_nbr_list_format )
{
    int i, j, k, l, m, itr;
    real d, cutoff;
    rvec dvec;
    grid *g;
    reax_atom *atom1, *atom2;
#if defined(LOG_PERFORMANCE)
    double time;
    
    time = Get_Time( );
#endif

    g = &system->my_grid;

    for ( i = 0; i < system->total_cap; ++i )
    {
        system->far_nbrs[i] = 0;
    }

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncells[0]; i++ )
    {
        for ( j = 0; j < g->ncells[1]; j++ )
        {
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                cutoff = SQR( g->cutoff[ index_grid_3d(i, j, k, g) ] );

                /* pick up an atom from the current cell */
                for ( l = g->str[ index_grid_3d(i, j, k, g) ];
                        l < g->end[ index_grid_3d( i, j, k, g) ]; ++l )
                {
                    atom1 = &system->my_atoms[l];
#if defined(NEUTRAL_TERRITORY)
                    if( g->type[ index_grid_3d(i, j, k, g) ] >= NT_NBRS
                            && g->type[ index_grid_3d(i, j, k, g) ] < NT_NBRS + 6 )
                    {
                        atom1->nt_dir = g->type[ index_grid_3d(i, j, k, g) ] - NT_NBRS;
                    }
                    else
                    {
                        atom1->nt_dir = -1;
                    }
#endif
                    itr = 0;

                    while ( g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ][0] >= 0 )
                    {
                        /* only search half of grid cells according to stencil used (upper-right) */
                        if ( (far_nbr_list_format == HALF_LIST
                                    && g->str[ index_grid_3d(i, j, k, g) ] <=
                                        g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ])
                                && DistSqr_to_Special_Point( g->nbrs_cp[ index_grid_nbrs(i, j, k, itr, g) ], atom1->x ) <= cutoff )
                        {
                            /* pick up another atom from the neighbor cell */
                            for ( m = g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ];
                                    m < g->end[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ]; ++m )
                            {
                                /* HALF_LIST: prevent recounting same pairs within a gcell and
                                 *  make half-list
                                 * FULL_LIST: prevent recounting same pairs within a gcell */
                                if ( (far_nbr_list_format == HALF_LIST && l < m)
                                        || (far_nbr_list_format == FULL_LIST && l != m) )
                                {
                                    atom2 = &system->my_atoms[m];
                                    dvec[0] = atom2->x[0] - atom1->x[0];
                                    dvec[1] = atom2->x[1] - atom1->x[1];
                                    dvec[2] = atom2->x[2] - atom1->x[2];
                                    d = rvec_Norm_Sqr( dvec );

                                    if ( d <= cutoff )
                                    {
                                        ++system->far_nbrs[l];
                                    }
                                }
                            }
                        }

                        ++itr;
                    }
                }
            }
        }
    }

    /* reduction for total num. of interactions */
    system->total_far_nbrs = 0;
    for ( i = 0; i < system->total_cap; ++i )
    {
        system->max_far_nbrs[i] = MAX( (int) CEIL( system->far_nbrs[i] * SAFE_ZONE ), MIN_NBRS );
        system->total_far_nbrs += system->max_far_nbrs[i];
    }
    system->total_far_nbrs = (int) CEIL( system->total_far_nbrs * SAFE_ZONE );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.nbrs );
#endif
}
