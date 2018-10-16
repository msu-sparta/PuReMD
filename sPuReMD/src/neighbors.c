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
#include "list.h"
#include "reset_utils.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


static inline real DistSqr_to_CP( rvec cp, rvec x )
{
    int i;
    real d_sqr = 0.0;

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
    int num_far;
    int *nbr_atoms;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g;
    far_neighbor_data nbr_data;

    g = &system->g;
    num_far = 0;

    Bin_Atoms( system, workspace );

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs = g->nbrs[i][j][k];
                nbrs_cp = g->nbrs_cp[i][j][k];

                /* pick up an atom from the current cell */
                for ( l = 0; l < g->top[i][j][k]; ++l )
                {
                    atom1 = g->atoms[i][j][k][l];

                    itr = 0;
                    while ( nbrs[itr][0] >= 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];

                        if ( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x )
                                <= SQR(control->vlist_cut) )
                        {
                            nbr_atoms = g->atoms[x][y][z];
                            max = g->top[x][y][z];

                            /* pick up another atom from the neighbor cell -
                             * we have to compare atom1 with its own periodic images as well,
                             * that's why there is also equality in the if stmt below */
                            for ( m = 0; m < max; ++m )
                            {
                                atom2 = nbr_atoms[m];

                                //if( nbrs[itr+1][0] >= 0 || atom1 > atom2 ) {
                                if ( atom1 > atom2 )
                                {
                                    /* assume periodic boundary conditions since it is
                                     * safe to over-estimate for the non-periodic case */
                                    if ( Count_Periodic_Far_Neighbors_Big_Box(system->atoms[atom1].x,
                                                system->atoms[atom2].x,
                                                &system->box, control->vlist_cut, &nbr_data) )
                                    {
                                        ++num_far;
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

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "estimate nbrs done, num_far: %d\n", num_far );
#endif

    return num_far * SAFE_ZONE;
}


/* If the code is not compiled to handle small periodic boxes (i.e. a
 * simulation box with any dimension less than twice the Verlet list cutoff
 * distance, vlist_cut), it will use the optimized Generate_Neighbor_Lists
 * function.  Otherwise it will execute the neighbor routine with small
 * periodic box support.
 * 
 * Define the preprocessor definition SMALL_BOX_SUPPORT to enable (in
 * reax_types.h). */
typedef int (*find_far_neighbors_function)( rvec, rvec, int, int,
        simulation_box*, real, far_neighbor_data* );


void Choose_Neighbor_Finder( reax_system *system, control_params *control,
        find_far_neighbors_function *Find_Far_Neighbors )
{
    if ( control->periodic_boundaries )
    {
#ifdef SMALL_BOX_SUPPORT
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


#ifdef DEBUG
int compare_far_nbrs(const void *v1, const void *v2)
{
    return ((*(far_neighbor_data *)v1).nbr - (*(far_neighbor_data *)v2).nbr);
}
#endif


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
    far_nbrs = lists[FAR_NBRS];

    Bin_Atoms( system, workspace );

#ifdef REORDER_ATOMS
    //Cluster_Atoms( system, workspace, control );
#endif

    Choose_Neighbor_Finder( system, control, &Find_Far_Neighbors );

    num_far = 0;

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs = g->nbrs[i][j][k];
                nbrs_cp = g->nbrs_cp[i][j][k];

                /* pick up an atom from the current cell */
                for ( l = 0; l < g->top[i][j][k]; ++l )
                {
                    atom1 = g->atoms[i][j][k][l];
                    Set_Start_Index( atom1, num_far, far_nbrs );
                    itr = 0;

                    while ( nbrs[itr][0] >= 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];

                        if ( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x )
                                <= SQR(control->vlist_cut) )
                        {
                            nbr_atoms = g->atoms[x][y][z];
                            max = g->top[x][y][z];

                            /* pick up another atom from the neighbor cell */
                            for ( m = 0; m < max; ++m )
                            {
                                atom2 = nbr_atoms[m];

                                if ( atom1 > atom2 )
                                {
                                    nbr_data = &far_nbrs->far_nbr_list[num_far];

                                    count = Find_Far_Neighbors( system->atoms[atom1].x, system->atoms[atom2].x,
                                            atom1, atom2, &system->box, control->vlist_cut, nbr_data );

                                    num_far += count;
                                }
                            }
                        }

                        ++itr;
                    }

                    Set_End_Index( atom1, num_far, far_nbrs );

//                    fprintf( stderr, "i:%d, start: %d, end: %d - itr: %d\n",
//                            atom1, Start_Index(atom1,far_nbrs), End_Index(atom1,far_nbrs), itr );
                }
            }
        }
    }

    if ( num_far > far_nbrs->total_intrs * DANGER_ZONE )
    {
        workspace->realloc.num_far = num_far;
        if ( num_far > far_nbrs->total_intrs )
        {
            fprintf( stderr, "step%d-ran out of space on far_nbrs: top=%d, max=%d",
                     data->step, num_far, far_nbrs->total_intrs );
            exit( INSUFFICIENT_MEMORY );
        }
    }

#if defined(DEBUG)
    for ( i = 0; i < system->N; ++i )
    {
        qsort( &far_nbrs->far_nbr_list[ Start_Index(i, far_nbrs) ],
                Num_Entries(i, far_nbrs), sizeof(far_neighbor_data),
                compare_far_nbrs );
    }
#endif

#if defined(TEST_ENERGY)
    //Print_Far_Neighbors( system, control, workspace, lists );
#endif

    t_elapsed = Get_Timing_Info( t_start );
    data->timing.nbrs += t_elapsed;
}


#if defined(LEGACY)  
/* check if atom2 is on atom1's near neighbor list */
static inline int is_Near_Neighbor( reax_list *near_nbrs, int atom1, int atom2 )
{
    int i;

    for ( i = Start_Index(atom1, near_nbrs); i < End_Index(atom1, near_nbrs); ++i )
    {
        if ( near_nbrs->near_nbr_list[i].nbr == atom2 )
        {
            return FALSE;
        }
    }

    return TRUE;
}


int compare_near_nbrs(const void *v1, const void *v2)
{
    return ((*(near_neighbor_data *)v1).nbr - (*(near_neighbor_data *)v2).nbr);
}


static inline void Set_Near_Neighbor( near_neighbor_data *dest, int nbr, real d, real C,
        rvec dvec, ivec rel_box/*, rvec ext_factor*/ )
{
    dest->nbr = nbr;
    dest->d = d;
    rvec_Scale( dest->dvec, C, dvec );
    ivec_Scale( dest->rel_box, C, rel_box );
    // rvec_Scale( dest->ext_factor, C, ext_factor );
}


void Generate_Neighbor_Lists( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, j, k;
    int x, y, z;
    int *nbr_atoms;
    int atom1, atom2, max;
    int num_far;
    int c, count;
    int grid_top;
    grid *g;
    reax_list *far_nbrs;
    find_far_neighbors_function Find_Far_Neighbors;
    far_neighbor_data *nbr_data;
    far_neighbor_data new_nbrs[125];
    real t_start, t_elapsed;

    t_start = Get_Time( );
    g = &system->g;
    far_nbrs = lists[FAR_NBRS];

    if ( control->ensemble == iNPT || control->ensemble == sNPT
            || control->ensemble == aNPT )
    {
        Update_Grid( system );
    }

    Bin_Atoms( system, workspace );

#ifdef REORDER_ATOMS
    //Cluster_Atoms( system, workspace, control );
#endif

    Choose_Neighbor_Finder( system, control, &Find_Far_Neighbors );

    num_far = 0;
    num_near = 0;
    c = 0;

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs = g->nbrs[i][j][k];
                nbrs_cp = g->nbrs_cp[i][j][k];

                /* pick up an atom from the current cell */
                for ( l = 0; l < g->top[i][j][k]; ++l )
                {
                    atom1 = g->atoms[i][j][k][l];
                    Set_End_Index( atom1, num_far, far_nbrs );

                    itr = 0;
                    while ( nbrs[itr][0] > 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];

                        // if( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <=
                        //     SQR(control->r_cut))
                        nbr_atoms = g->atoms[x][y][z];
                        max = g->top[x][y][z];

                        /* pick up another atom from the neighbor cell -
                         * we have to compare atom1 with its own periodic images as well,
                         * that's why there is also equality in the if stmt below */
                        for ( m = 0; m < max; ++m )
                        {
                            atom2 = nbr_atoms[m];

                            if ( atom1 >= atom2 )
                            {
                                nbr_data = &far_nbrs->far_nbr_list[num_far];

                                //top_near1 = End_Index( atom1, near_nbrs );
                                //Set_Start_Index( atom1, num_far, far_nbrs );
                                count = Find_Far_Neighbors( system->atoms[atom1].x, system->atoms[atom2].x,
                                        atom1, atom2, &system->box, control->vlist_cut, nbr_data );

                                num_far += count;
                            }
                        }
                    }

                    Set_End_Index( atom1, top_far1, far_nbrs );
                }
            }
        }
    }

   /* Below is some piece of legacy code. It tries to force bonded 
    * interactions between atoms which are given to be bonded in the 
    * BGF file through the CONECT lines. The aim is to ensure that 
    * molecules stay intact during an energy minimization procudure.
    * Energy minimization is not actually not implemented as of now 
    * (July 7, 2014), but we mimic it by running at very low temperature
    * and small dt (timestep length) NVT simulation.
    * However, I (HMA) am not sure if it can possibly achieve the desired
    * effect.  Therefore, this functionality is currently disabled. */
    /* apply restrictions on near neighbors only */
    if ( (data->step - data->prev_steps) < control->restrict_bonds )
    {
        for ( atom1 = 0; atom1 < system->N; ++atom1 )
        {
            if ( workspace->restricted[ atom1 ] )
            {
                top_near1 = End_Index( atom1, near_nbrs );

                for ( j = 0; j < workspace->restricted[ atom1 ]; ++j )
                {
                    atom2 = workspace->restricted_list[atom1][j];

                    if ( is_Near_Neighbor(near_nbrs, atom1, atom2) == FALSE )
                    {
                        fprintf( stderr, "%3d-%3d: added bond by applying restrictions!\n",
                                atom1, atom2 );

                        top_near2 = End_Index( atom2, near_nbrs );

                        /* we just would like to get the nearest image, so a call to
                         * Get_Periodic_Far_Neighbors_Big_Box is good enough. */
                        Get_Periodic_Far_Neighbors_Big_Box( system->atoms[ atom1 ].x,
                                system->atoms[ atom2 ].x, &system->box, control,
                                new_nbrs, &count );

                        Set_Near_Neighbor( &near_nbrs->near_nbr_list[ top_near1 ], atom2,
                                new_nbrs[c].d, 1.0, new_nbrs[c].dvec, new_nbrs[c].rel_box );
                        ++top_near1;

                        Set_Near_Neighbor( &near_nbrs->near_nbr_list[ top_near2 ], atom1,
                                new_nbrs[c].d, -1.0, new_nbrs[c].dvec, new_nbrs[c].rel_box );
                        Set_End_Index( atom2, top_near2 + 1, near_nbrs );
                    }
                }

                Set_End_Index( atom1, top_near1, near_nbrs );
            }
        }
    }

    /* verify nbrlists, count total_intrs, sort nearnbrs */
    near_nbrs->total_intrs = 0;
    far_nbrs->total_intrs = 0;
    for ( i = 0; i < system->N - 1; ++i )
    {
        if ( End_Index(i, near_nbrs) > Start_Index(i + 1, near_nbrs) )
        {
            fprintf( stderr,
                     "step%3d: nearnbr list of atom%d is overwritten by atom%d\n",
                     data->step, i + 1, i );
            exit( RUNTIME_ERROR );
        }

        near_nbrs->total_intrs += Num_Entries( i, near_nbrs );

        if ( End_Index(i, far_nbrs) > Start_Index(i + 1, far_nbrs) )
        {
            fprintf( stderr,
                     "step%3d: farnbr list of atom%d is overwritten by atom%d\n",
                     data->step, i + 1, i );
            exit( RUNTIME_ERROR );
        }

        far_nbrs->total_intrs += Num_Entries( i, far_nbrs );
    }

    for ( i = 0; i < system->N; ++i )
    {
        qsort( &near_nbrs->near_nbr_list[ Start_Index(i, near_nbrs) ],
               Num_Entries(i, near_nbrs), sizeof(near_neighbor_data),
               compare_near_nbrs );
    }

#ifdef TEST_ENERGY
//    for ( i = 0; i < system->N; ++i )
//    {
//       qsort( &far_nbrs->far_nbr_list[ Start_Index(i, far_nbrs) ],
//               Num_Entries(i, far_nbrs), sizeof(far_neighbor_data), compare_far_nbrs );
//    }

    fprintf( stderr, "Near neighbors/atom: %d (compare to 150)\n",
             num_near / system->N );
#endif

    t_elapsed = Get_Timing_Info( t_start );
    data->timing.nbrs += t_elapsed;
}
#endif
