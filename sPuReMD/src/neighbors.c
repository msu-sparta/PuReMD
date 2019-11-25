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

#include "neighbors.h"

#include "box.h"
#include "grid.h"
#if defined(DEBUG_FOCUS)
  #include "io_tools.h"
#endif
#include "list.h"
#include "reset_tools.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


/* If the code is not compiled to handle small periodic boxes (i.e. a
 * simulation box with any dimension less than twice the Verlet list cutoff
 * distance, vlist_cut), it will use the optimized Generate_Neighbor_Lists
 * function.  Otherwise it will execute the neighbor routine with small
 * periodic box support.
 * 
 * Define the preprocessor definition SMALL_BOX_SUPPORT to enable (in
 * reax_types.h). */
typedef int (*count_far_neighbors_function)( rvec, rvec, int, int,
        simulation_box*, real );

typedef int (*find_far_neighbors_function)( rvec, rvec, int, int,
        simulation_box*, real, far_neighbor_data* );


static void Choose_Neighbor_Counter( reax_system *system, control_params *control,
        count_far_neighbors_function *Count_Far_Neighbors )
{
    if ( control->periodic_boundaries == TRUE )
    {
#if defined(SMALL_BOX_SUPPORT)
        if ( system->box.box_norms[0] >= 2.0 * control->vlist_cut
                && system->box.box_norms[1] >= 2.0 * control->vlist_cut
                && system->box.box_norms[2] >= 2.0 * control->vlist_cut )
        {
            *Count_Far_Neighbors = &Count_Periodic_Far_Neighbors_Big_Box;
        }
        else
        {
            *Count_Far_Neighbors = &Count_Periodic_Far_Neighbors_Small_Box;
        }
#else
        *Count_Far_Neighbors = &Count_Periodic_Far_Neighbors_Big_Box;
#endif
    }
    else
    {
        *Count_Far_Neighbors = &Count_Non_Periodic_Far_Neighbors;
    }
}


static void Choose_Neighbor_Finder( reax_system *system, control_params *control,
        find_far_neighbors_function *Find_Far_Neighbors )
{
    if ( control->periodic_boundaries == TRUE )
    {
#if defined(SMALL_BOX_SUPPORT)
        if ( system->box.box_norms[0] >= 2.0 * control->vlist_cut
                && system->box.box_norms[1] >= 2.0 * control->vlist_cut
                && system->box.box_norms[2] >= 2.0 * control->vlist_cut )
        {
            *Find_Far_Neighbors = &Find_Periodic_Far_Neighbors_Big_Box;
        }
        else
        {
            *Find_Far_Neighbors = &Find_Periodic_Far_Neighbors_Small_Box;
        }
#else
        *Find_Far_Neighbors = &Find_Periodic_Far_Neighbors_Big_Box;
#endif
    }
    else
    {
        *Find_Far_Neighbors = &Find_Non_Periodic_Far_Neighbors;
    }
}


#if defined(DEBUG_FOCUS)
static int compare_far_nbrs(const void *v1, const void *v2)
{
    return ((*(far_neighbor_data *)v1).nbr - (*(far_neighbor_data *)v2).nbr);
}
#endif


static inline real DistSqr_to_CP( rvec cp, rvec x )
{
    int i;
    real d_sqr;

    d_sqr = 0.0;

    for ( i = 0; i < 3; ++i )
    {
        if ( cp[i] > NEG_INF )
        {
            d_sqr += SQR( cp[i] - x[i] );
        }
    }

    return d_sqr;
}


int Estimate_Num_Neighbors( reax_system *system, control_params *control,
        static_storage *workspace, reax_list **lists )
{
    int i, j, k, l, m, itr;
    int x, y, z;
    int atom1, atom2, max;
    int num_far, count;
    int *nbr_atoms;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g;
    count_far_neighbors_function Count_Far_Neighbors;

    g = &system->g;
    num_far = 0;

    Choose_Neighbor_Counter( system, control, &Count_Far_Neighbors );

    Bin_Atoms( system, workspace );

    /* for each cell in the grid along the 3
     * Cartesian directions: (i, j, k) => (x, y, z) */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs = g->nbrs[i][j][k];
                nbrs_cp = g->nbrs_cp[i][j][k];

                /* for each atom in the current cell */
                for ( l = 0; l < g->top[i][j][k]; ++l )
                {
                    atom1 = g->atoms[i][j][k][l];
                    itr = 0;

                    /* for each of the neighboring grid cells within
                     * the Verlet list cutoff distance */
                    while ( nbrs[itr][0] >= 0 )
                    {
                        /* if the Verlet list cutoff covers the closest point
                         * in the neighboring grid cell, then search through the cell's atoms */
                        if ( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x )
                                <= SQR(control->vlist_cut) )
                        {
                            x = nbrs[itr][0];
                            y = nbrs[itr][1];
                            z = nbrs[itr][2];
                            nbr_atoms = g->atoms[x][y][z];
                            max = g->top[x][y][z];

                            /* pick up another atom from the neighbor cell;
                             * we have to compare atom1 with its own periodic images as well
                             * in the case of periodic boundary conditions,
                             * hence the equality in the if stmt below */
                            for ( m = 0; m < max; ++m )
                            {
                                atom2 = nbr_atoms[m];

                                if ( atom1 >= atom2 )
                                {
                                    count = Count_Far_Neighbors( system->atoms[atom1].x,
                                                system->atoms[atom2].x, atom1, atom2, 
                                                &system->box, control->vlist_cut );

                                    num_far += count;
                                }
                            }
                        }

                        ++itr;
                    }
                }
            }
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] Estimate_Num_Neighbors: num_far = %d\n",
            (int) CEIL( num_far * SAFE_ZONE ) );
#endif

    return (int) CEIL( num_far * SAFE_ZONE );
}


void Generate_Neighbor_Lists( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, j, k, l, m, itr;
    int x, y, z;
    int atom1, atom2, max;
    int num_far, count;
    int *nbr_atoms;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g;
    reax_list *far_nbrs;
    far_neighbor_data *nbr_data;
    find_far_neighbors_function Find_Far_Neighbors;
    real t_start, t_elapsed;

    t_start = Get_Time( );
    g = &system->g;
    num_far = 0;
    far_nbrs = lists[FAR_NBRS];

    Choose_Neighbor_Finder( system, control, &Find_Far_Neighbors );

    Bin_Atoms( system, workspace );

#if defined(REORDER_ATOMS)
    //Cluster_Atoms( system, workspace, control );
#endif

    /* for each cell in the grid along the 3
     * Cartesian directions: (i, j, k) => (x, y, z) */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs = g->nbrs[i][j][k];
                nbrs_cp = g->nbrs_cp[i][j][k];

                /* for each atom in the current cell */
                for ( l = 0; l < g->top[i][j][k]; ++l )
                {
                    atom1 = g->atoms[i][j][k][l];
                    Set_Start_Index( atom1, num_far, far_nbrs );
                    itr = 0;

                    /* for each of the neighboring grid cells within
                     * the Verlet list cutoff distance */
                    while ( nbrs[itr][0] >= 0 )
                    {
                        /* if the Verlet list cutoff covers the closest point
                         * in the neighboring grid cell, then search through the cell's atoms */
                        if ( DistSqr_to_CP( nbrs_cp[itr], system->atoms[atom1].x )
                                <= SQR(control->vlist_cut) )
                        {
                            x = nbrs[itr][0];
                            y = nbrs[itr][1];
                            z = nbrs[itr][2];
                            nbr_atoms = g->atoms[x][y][z];
                            max = g->top[x][y][z];

                            /* pick up another atom from the neighbor cell;
                             * we have to compare atom1 with its own periodic images as well
                             * in the case of periodic boundary conditions,
                             * hence the equality in the if stmt below */
                            for ( m = 0; m < max; ++m )
                            {
                                atom2 = nbr_atoms[m];

                                if ( atom1 >= atom2 )
                                {
                                    nbr_data = &far_nbrs->far_nbr_list[num_far];

                                    count = Find_Far_Neighbors( system->atoms[atom1].x,
                                            system->atoms[atom2].x, atom1, atom2,
                                            &system->box, control->vlist_cut, nbr_data );

                                    num_far += count;
                                }
                            }
                        }

                        ++itr;
                    }

                    Set_End_Index( atom1, num_far, far_nbrs );

#if defined(DEBUG_FOCUS)
                    fprintf( stderr, "[INFO] Generate_Neighbor_Lists: i = %d, start = %d, end = %d, itr = %d\n",
                            atom1, Start_Index(atom1,far_nbrs), End_Index(atom1,far_nbrs), itr );
#endif
                }
            }
        }
    }

    //TODO: conditionally perform these assignments if periodic boundary conditions are enabled
    for ( i = 0; i < system->N; i++ )
    {
        ivec_MakeZero( system->atoms[i].rel_map );
    }

    if ( num_far > far_nbrs->total_intrs * DANGER_ZONE )
    {
        workspace->realloc.num_far = num_far;

        if ( num_far > far_nbrs->total_intrs )
        {
            fprintf( stderr, "[ERROR] Generate_Neighbor_Lists: step%d-ran out of space on far_nbrs: top=%d, max=%d",
                     data->step, num_far, far_nbrs->total_intrs );
            exit( INSUFFICIENT_MEMORY );
        }
    }

#if defined(DEBUG_FOCUS)
    for ( i = 0; i < system->N; ++i )
    {
        qsort( &far_nbrs->far_nbr_list[ Start_Index(i, far_nbrs) ],
                Num_Entries(i, far_nbrs), sizeof(far_neighbor_data),
                compare_far_nbrs );
    }
#endif

#if defined(DEBUG_FOCUS)
    Print_Far_Neighbors( system, control, data, workspace, lists );
#endif

    t_elapsed = Get_Timing_Info( t_start );
    data->timing.nbrs += t_elapsed;
}
