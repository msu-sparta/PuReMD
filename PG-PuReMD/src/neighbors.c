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


void Draw_Near_Neighbor_Box( reax_system *system, control_params *control,
        storage *workspace )
{
    int i;
    reax_atom *atom;
    simulation_box *my_box;
    boundary_cutoff *bc;

    my_box = &system->my_box;
    bc = &system->bndry_cuts;

    /* all native atoms are within near neighbor skin */
    for ( i = 0; i < system->n; ++i )
    {
        workspace->within_bond_box[i] = 1;
    }

    /* loop over imported atoms */
    for ( i = system->n; i < system->N; ++i )
    {
        atom = &system->my_atoms[i];

        if ( my_box->min[0] - bc->ghost_bond <= atom->x[0] &&
                atom->x[0] <= my_box->max[0] + bc->ghost_bond &&
                my_box->min[1] - bc->ghost_bond <= atom->x[1] &&
                atom->x[1] <= my_box->max[1] + bc->ghost_bond &&
                my_box->min[2] - bc->ghost_bond <= atom->x[2] &&
                atom->x[2] <= my_box->max[2] + bc->ghost_bond )
        {
            workspace->within_bond_box[i] = 1;
        }
        else
        {
            workspace->within_bond_box[i] = 0;
        }
    }
}


int Generate_Neighbor_Lists( reax_system *system, simulation_data *data,
        storage *workspace, reax_list **lists )
{
    int i, j, k, l, m, itr, num_far;
    real d, cutoff;
    rvec dvec;
    grid *g;
    reax_list *far_nbr_list;
    far_neighbor_data *nbr_data;
    reax_atom *atom1, *atom2;

#if defined(LOG_PERFORMANCE)
    real t_start = 0.0;
    real t_elapsed = 0.0;

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
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
                    num_far = Start_Index( l, far_nbr_list );
                    itr = 0;

                    /* search through neighbor grid cell candidates of current cell */
                    while ( g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ][0] >= 0 )
                    {
                        if ( g->str[ index_grid_3d(i, j, k, g) ] <=
                                g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ]
                            && DistSqr_to_Special_Point( g->nbrs_cp[ index_grid_nbrs(i, j, k, itr, g) ], atom1->x ) <= cutoff )
                        {
                            /* pick up another atom from the neighbor grid cell */
                            for ( m = g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ];
                                    m < g->end[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ]; ++m )
                            {
                                /* prevent recounting same pairs within a gcell */
                                if ( l < m )
                                {
                                    atom2 = &system->my_atoms[m];
                                    dvec[0] = atom2->x[0] - atom1->x[0];
                                    dvec[1] = atom2->x[1] - atom1->x[1];
                                    dvec[2] = atom2->x[2] - atom1->x[2];
                                    d = rvec_Norm_Sqr( dvec );

                                    if ( d <= cutoff )
                                    {
                                        nbr_data = &far_nbr_list->far_nbr_list[num_far];
                                        nbr_data->nbr = m;
                                        nbr_data->d = SQRT( d );
                                        rvec_Copy( nbr_data->dvec, dvec );
                                        ivec_ScaledSum( nbr_data->rel_box, 1,
                                                g->rel_box[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ],
                                                -1, g->rel_box[ index_grid_3d(i, j, k, g) ] );
                                        ++num_far;
                                    }
                                }
                            }
                        }

                        ++itr;
                    }

                    Set_End_Index( l, num_far, far_nbr_list );

                    /* reallocation check */
                    if ( Num_Entries( l, far_nbr_list ) > system->max_far_nbrs[l] )
                    {
                        workspace->realloc.far_nbrs = TRUE;
                    }
                }
            }
        }
    }

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_elapsed = Get_Timing_Info( t_start );
        data->timing.nbrs += t_elapsed;
    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: nbrs done - total_num_far=%d\n",
             system->my_rank, data->step, total_num_far );
    MPI_Barrier( MPI_COMM_WORLD );
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


void Estimate_Num_Neighbors( reax_system *system )
{
    int i, j, k, l, m, itr;
    real d, cutoff;
    rvec dvec;
    grid *g;
    reax_atom *atom1, *atom2;

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
                    itr = 0;

                    while ( g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ][0] >= 0 )
                    {
                        /* only search half of grid cells according to stencil used (upper-right) */
                        if ( g->str[ index_grid_3d(i, j, k, g) ] <=
                                g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ]
                            && DistSqr_to_Special_Point( g->nbrs_cp[ index_grid_nbrs(i, j, k, itr, g) ], atom1->x ) <= cutoff )
                        {
                            /* pick up another atom from the neighbor cell */
                            for ( m = g->str[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ];
                                    m < g->end[ index_grid_3d_v(g->nbrs_x[ index_grid_nbrs(i, j, k, itr, g) ], g) ]; ++m )
                            {
                                /* half-list for case when l and m point to the same grid cell */
                                if ( l < m )
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

    /* reduction for total */
    system->total_far_nbrs = 0;
    for ( i = 0; i < system->total_cap; ++i )
    {
        system->max_far_nbrs[i] = MAX( (int)(system->far_nbrs[i] * SAFE_ZONE), MIN_NBRS );
        system->total_far_nbrs += system->max_far_nbrs[i];
    }
    system->total_far_nbrs = MAX( system->total_far_nbrs, MIN_CAP * MIN_NBRS );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: estimate nbrs done - system->total_far_nbrs=%d\n",
             system->my_rank, system->total_far_nbrs );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}
