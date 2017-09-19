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
#include "index_utils.h"
#include "list.h"
#include "reset_utils.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


void Generate_Neighbor_Lists( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        list **lists, output_controls *out_control )
{
    int  i, j, k, l, m, itr;
    int  x, y, z;
    int  atom1, atom2, max;
    int  num_far;
    int  *nbr_atoms;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g;
    list *far_nbrs;
    far_neighbor_data *nbr_data;
    real t_start, t_elapsed;

    t_start = Get_Time( );
    // fprintf( stderr, "\n\tentered nbrs - " );
    g = &( system->g );
    far_nbrs = (*lists) + FAR_NBRS;
    Bin_Atoms( system, workspace );
    // fprintf( stderr, "atoms sorted - " );
    num_far = 0;

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs = &g->nbrs[ index_grid_nbrs(i,j,k,0,g) ];
                nbrs_cp = &g->nbrs_cp[ index_grid_nbrs(i,j,k,0,g) ];
                //fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

                /* pick up an atom from the current cell */
                for(l = 0; l < g->top[ index_grid_3d(i,j,k,g) ]; ++l )
                {
                    atom1 = g->atoms[ index_grid_atoms(i,j,k,l,g) ];
                    Set_Start_Index( atom1, num_far, far_nbrs );
                    //fprintf( stderr, "\tatom %d\n", atom1 );

                    itr = 0;
                    while ( nbrs[itr][0] >= 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];
                        //fprintf( stderr, "\t\tgridcell %d %d %d\n", x, y, z );

                        if ( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <=
                                SQR(control->vlist_cut) )
                        {
                            nbr_atoms = &g->atoms[ index_grid_atoms(x,y,z,0,g) ];
                            max = g->top[ index_grid_3d(x,y,z,g) ];
                            //fprintf( stderr, "\t\tmax: %d\n", max );

                            /* pick up another atom from the neighbor cell */
                            for ( m = 0; m < max; ++m )
                            {
                                atom2 = nbr_atoms[m];
                                if ( atom1 > atom2 )
                                {
                                    nbr_data = &(far_nbrs->select.far_nbr_list[num_far]);
                                    //fprintf (stderr, " %f %f %f \n", nbr_data->dvec[0], nbr_data->dvec[1], nbr_data->dvec[2]);
                                    if (Are_Far_Neighbors(system->atoms[atom1].x,
                                                          system->atoms[atom2].x,
                                                          &(system->box), control->vlist_cut,
                                                          nbr_data))
                                    {
                                        nbr_data->nbr = atom2;
                                        ++num_far;
                                    }
                                }
                            }
                        }

                        ++itr;
                    }

                    Set_End_Index( atom1, num_far, far_nbrs );
                    //fprintf(stderr, "i:%d, start: %d, end: %d - itr: %d\n",
                    //  atom1,Start_Index(atom1,far_nbrs),End_Index(atom1,far_nbrs),
                    //  itr);
                }
            }
        }
    }

    if ( num_far > far_nbrs->num_intrs * DANGER_ZONE )
    {
        workspace->realloc.num_far = num_far;
        if ( num_far > far_nbrs->num_intrs )
        {
            fprintf( stderr, "step%d-ran out of space on far_nbrs: top=%d, max=%d",
                     data->step, num_far, far_nbrs->num_intrs );
            exit( INSUFFICIENT_MEMORY );
        }
    }

    t_elapsed = Get_Timing_Info( t_start );
    data->timing.nbrs += t_elapsed;

#if defined(DEBUG)
    for ( i = 0; i < system->N; ++i )
    {
        qsort( &(far_nbrs->select.far_nbr_list[ Start_Index(i, far_nbrs) ]),
               Num_Entries(i, far_nbrs), sizeof(far_neighbor_data),
               compare_far_nbrs );
    }
#endif
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nbrs - ");
    fprintf( stderr, "nbrs done, num_far: %d\n", num_far );
#endif
#if defined(TEST_ENERGY)
    //Print_Far_Neighbors( system, control, workspace, lists );
#endif
}


int Estimate_NumNeighbors( reax_system *system, control_params *control,
        static_storage *workspace, list **lists )
{
    int  i, j, k, l, m, itr;
    int  x, y, z;
    int  atom1, atom2, max;
    int  num_far;
    int  *nbr_atoms;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g;
    far_neighbor_data nbr_data;
#ifdef HAVE_CUDA
    int start = 0, finish = 0;
#endif

    // fprintf( stderr, "\n\tentered nbrs - " );
    g = &( system->g );
    Bin_Atoms( system, workspace );
    // fprintf( stderr, "atoms sorted - " );
    num_far = 0;
#ifdef HAVE_CUDA
    g->max_cuda_nbrs = 0;
#endif

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncell[0]; i++ )
    {
        for ( j = 0; j < g->ncell[1]; j++ )
        {
            for ( k = 0; k < g->ncell[2]; k++ )
            {
                nbrs = &g->nbrs[ index_grid_nbrs(i,j,k,0,g) ];
                nbrs_cp = &g->nbrs_cp[ index_grid_nbrs(i,j,k,0,g) ];
                //fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

                /* pick up an atom from the current cell */
                for(l = 0; l < g->top[ index_grid_3d(i,j,k,g) ]; ++l )
                {
                    atom1 = g->atoms[ index_grid_atoms(i,j,k,l,g) ];

                    itr = 0;
                    while ( nbrs[itr][0] >= 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];
                        //fprintf( stderr, "\t\tgridcell %d %d %d\n", x, y, z );

                        if ( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <=
                                SQR(control->vlist_cut) )
                        {
                            nbr_atoms = &g->atoms[ index_grid_atoms(x,y,z,0,g) ];
                            max = g->top[ index_grid_3d(x,y,z,g) ];
                            //fprintf( stderr, "\t\tmax: %d\n", max );

                            /* pick up another atom from the neighbor cell -
                            we have to compare atom1 with its own periodic images as well,
                             that's why there is also equality in the if stmt below */
                            for ( m = 0; m < max; ++m )
                            {
                                atom2 = nbr_atoms[m];
                                //if( nbrs[itr+1][0] >= 0 || atom1 > atom2 ) {
                                if ( atom1 > atom2 )
                                {
                                    if (Are_Far_Neighbors(system->atoms[atom1].x,
                                                          system->atoms[atom2].x,
                                                          &(system->box), control->vlist_cut,
                                                          &nbr_data))
                                        ++num_far;
                                }
                            }
                        }

                        ++itr;
                    }

#ifdef HAVE_CUDA
                    finish = num_far;
                    if (g->max_cuda_nbrs <= (finish - start))
                    {
                        g->max_cuda_nbrs    = finish - start;
                    }
#endif
                }
            }
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "estimate nbrs done, num_far: %d\n", num_far );
#endif
    return num_far * SAFE_ZONE;
}


#if defined DONE
void Choose_Neighbor_Finder( reax_system *system, control_params *control,
        get_far_neighbors_function *Get_Far_Neighbors )
{
    if ( control->periodic_boundaries )
    {
        if ( system->box.box_norms[0] > 2.0 * control->vlist_cut &&
                system->box.box_norms[1] > 2.0 * control->vlist_cut &&
                system->box.box_norms[2] > 2.0 * control->vlist_cut )
        {
            (*Get_Far_Neighbors) = Get_Periodic_Far_Neighbors_Big_Box;
        }
        else
        {
            (*Get_Far_Neighbors) = Get_Periodic_Far_Neighbors_Small_Box;
        }
    }
    else
    {
        (*Get_Far_Neighbors) = Get_NonPeriodic_Far_Neighbors;
    }
}


int compare_near_nbrs(const void *v1, const void *v2)
{
    return ((*(near_neighbor_data *)v1).nbr - (*(near_neighbor_data *)v2).nbr);
}


int compare_far_nbrs(const void *v1, const void *v2)
{
    return ((*(far_neighbor_data *)v1).nbr - (*(far_neighbor_data *)v2).nbr);
}


inline void Set_Far_Neighbor( far_neighbor_data *dest, int nbr, real d, real C,
        rvec dvec, ivec rel_box/*, rvec ext_factor*/ )
{
    dest->nbr = nbr;
    dest->d = d;
    rvec_Scale( dest->dvec, C, dvec );
    ivec_Copy( dest->rel_box, rel_box );
    // rvec_Scale( dest->ext_factor, C, ext_factor );
}


inline void Set_Near_Neighbor(near_neighbor_data *dest, int nbr, real d, real C,
        rvec dvec, ivec rel_box/*, rvec ext_factor*/)
{
    dest->nbr = nbr;
    dest->d = d;
    rvec_Scale( dest->dvec, C, dvec );
    ivec_Scale( dest->rel_box, C, rel_box );
    // rvec_Scale( dest->ext_factor, C, ext_factor );
}


/* In case bond restrictions are applied, this method checks if
   atom1 and atom2 are allowed to bond with each other */
inline int can_Bond( static_storage *workspace, int atom1, int atom2 )
{
    int i;

    // fprintf( stderr, "can bond %6d %6d?\n", atom1, atom2 );

    if ( !workspace->restricted[ atom1 ] && !workspace->restricted[ atom2 ] )
    {
        return FALSE;
    }

    for ( i = 0; i < workspace->restricted[ atom1 ]; ++i )
    {
        if ( workspace->restricted_list[ atom1 ][i] == atom2 )
        {
            return FALSE;
        }
    }

    for ( i = 0; i < workspace->restricted[ atom2 ]; ++i )
    {
        if ( workspace->restricted_list[ atom2 ][i] == atom1 )
        {
            return FALSE;
        }
    }

    return TRUE;
}


/* check if atom2 is on atom1's near neighbor list */
inline int is_Near_Neighbor( list *near_nbrs, int atom1, int atom2 )
{
    int i;

    for ( i = Start_Index(atom1, near_nbrs); i < End_Index(atom1, near_nbrs); ++i )
    {
        if ( near_nbrs->select.near_nbr_list[i].nbr == atom2 )
        {
            // fprintf( stderr, "near neighbors %6d %6d\n", atom1, atom2 );
            return FALSE;
        }
    }

    return TRUE;
}


void Generate_Neighbor_Lists( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        list **lists, output_controls *out_control )
{
    int  i, j, k;
    int  x, y, z;
    int  *nbr_atoms;
    int  atom1, atom2, max;
    int   num_far;
    int   c, count;
    int   grid_top;
    grid *g = &( system->g );
    list *far_nbrs = (*lists) + FAR_NBRS;
    //int   hb_type1, hb_type2;
    //list *hbonds = (*lists) + HBOND;
    //int   top_hbond1, top_hbond2;
    get_far_neighbors_function Get_Far_Neighbors;
    far_neighbor_data new_nbrs[125];

    // fprintf( stderr, "\n\tentered nbrs - " );
    if ( control->ensemble == iNPT || control->ensemble == sNPT ||
            control->ensemble == NPT )
    {
        Update_Grid( system );
    }
    // fprintf( stderr, "grid updated - " );

    Bin_Atoms( system, out_control );
    // fprintf( stderr, "atoms sorted - " );

#ifdef REORDER_ATOMS
    Cluster_Atoms( system, workspace );
    // fprintf( stderr, "atoms clustered - " );
#endif

    Choose_Neighbor_Finder( system, control, &Get_Far_Neighbors );
    // fprintf( stderr, "function chosen - " );

    Reset_Neighbor_Lists( system, workspace, lists );
    // fprintf( stderr, "lists cleared - " );

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
                //#ifdef REORDER_ATOMS
                //  for(atom1 = g->start[i][j][k]; atom1 < g->end[i][j][k]; atom1++)
                //#else
                for (l = 0; l < g->top[i][j][k]; ++l )
                {
                    atom1 = g->atoms[i][j][k][l];
                    Set_End_Index( atom1, num_far, far_nbrs );
                    // fprintf( stderr, "atom %d:\n", atom1 );

                    itr = 0;
                    while ( nbrs[itr][0] > 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];

                        // if( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <=
                        //     SQR(control->r_cut))
                        nbr_atoms = g->atoms[x][y][z];
                        max_atoms = g->top[x][y][z];

                        /* pick up another atom from the neighbor cell -
                           we have to compare atom1 with its own periodic images as well,
                           that's why there is also equality in the if stmt below */
                        //#ifdef REORDER_ATOMS
                        //for(atom2=g->start[x][y][z]; atom2<g->end[x][y][z]; atom2++)
                        //#else
                        for ( m = 0, atom2 = nbr_atoms[m]; m < max; ++m, atom2 = nbr_atoms[m] )
                        {
                            if ( atom1 >= atom2 )
                            {
                                //fprintf( stderr, "\tatom2 %d", atom2 );
                                //top_near1 = End_Index( atom1, near_nbrs );
                                //Set_Start_Index( atom1, num_far, far_nbrs );
                                //hb_type1=system->reaxprm.sbp[system->atoms[atom1].type].p_hbond;
                                Get_Far_Neighbors( system->atoms[atom1].x,
                                                   system->atoms[atom2].x,
                                                   &(system->box), control, new_nbrs, &count );
                                fprintf( stderr, "\t%d count:%d\n", atom2, count );

                                for ( c = 0; c < count; ++c )
                                {
                                    if (atom1 != atom2 || (atom1 == atom2 && new_nbrs[c].d >= 0.1))
                                    {
                                        Set_Far_Neighbor(&(far_nbrs->select.far_nbr_list[num_far]),
                                                         atom2, new_nbrs[c].d, 1.0,
                                                         new_nbrs[c].dvec, new_nbrs[c].rel_box );
                                        ++num_far;

                                        /*fprintf(stderr,"FARNBR:%6d%6d%8.3f[%8.3f%8.3f%8.3f]\n",
                                          atom1, atom2, new_nbrs[c].d,
                                          new_nbrs[c].dvec[0], new_nbrs[c].dvec[1],
                                          new_nbrs[c].dvec[2] ); */


                                        /* hydrogen bond lists */
                                        /*if( control->hb_cut > 0.1 &&
                                          new_nbrs[c].d <= control->hb_cut ) {
                                          // fprintf( stderr, "%d %d\n", atom1, atom2 );
                                          hb_type2=system->reaxprm.sbp[system->atoms[atom2].type].p_hbond;
                                          if( hb_type1 == 1 && hb_type2 == 2 ) {
                                          top_hbond1=End_Index(workspace->hbond_index[atom1],hbonds);
                                          Set_Near_Neighbor(&(hbonds->select.hbond_list[top_hbond1]),
                                          atom2, new_nbrs[c].d, 1.0, new_nbrs[c].dvec,
                                          new_nbrs[c].rel_box );
                                          Set_End_Index( workspace->hbond_index[atom1],
                                          top_hbond1 + 1, hbonds );
                                          }
                                          else if( hb_type1 == 2 && hb_type2 == 1 ) {
                                          top_hbond2 = End_Index( workspace->hbond_index[atom2], hbonds );
                                          Set_Near_Neighbor(&(hbonds->select.hbond_list[top_hbond2]),
                                          atom1, new_nbrs[c].d, -1.0, new_nbrs[c].dvec,
                                          new_nbrs[c].rel_box );
                                          Set_End_Index( workspace->hbond_index[atom2],
                                          top_hbond2 + 1, hbonds );
                                          }*/
                                    }
                                }
                            }
                        }
                    }

                    Set_End_Index( atom1, top_far1, far_nbrs );
                }
            }
        }
    }

    fprintf( stderr, "nbrs done-" );


    /* apply restrictions on near neighbors only */
    if ( (data->step - data->prev_steps) < control->restrict_bonds )
    {
        for ( atom1 = 0; atom1 < system->N; ++atom1 )
        {
            if ( workspace->restricted[ atom1 ] )
            {
                // fprintf( stderr, "atom1: %d\n", atom1 );

                top_near1 = End_Index( atom1, near_nbrs );

                for ( j = 0; j < workspace->restricted[ atom1 ]; ++j )
                {
                    if (is_Near_Neighbor(near_nbrs, atom1,
                          atom2 = workspace->restricted_list[atom1][j]) == FALSE)
                    {
                        fprintf( stderr, "%3d-%3d: added bond by applying restrictions!\n",
                                 atom1, atom2 );

                        top_near2 = End_Index( atom2, near_nbrs );

                        /* we just would like to get the nearest image, so a call to
                           Get_Periodic_Far_Neighbors_Big_Box is good enough. */
                        Get_Periodic_Far_Neighbors_Big_Box( system->atoms[ atom1 ].x,
                                                            system->atoms[ atom2 ].x,
                                                            &(system->box), control,
                                                            new_nbrs, &count );

                        Set_Near_Neighbor( &(near_nbrs->select.near_nbr_list[ top_near1 ]),
                                           atom2, new_nbrs[c].d, 1.0,
                                           new_nbrs[c].dvec, new_nbrs[c].rel_box );
                        ++top_near1;

                        Set_Near_Neighbor( &(near_nbrs->select.near_nbr_list[ top_near2 ]),
                                           atom1, new_nbrs[c].d, -1.0,
                                           new_nbrs[c].dvec, new_nbrs[c].rel_box );
                        Set_End_Index( atom2, top_near2 + 1, near_nbrs );
                    }
                }

                Set_End_Index( atom1, top_near1, near_nbrs );
            }
        }
    }
    // fprintf( stderr, "restrictions applied-" );


    /* verify nbrlists, count num_intrs, sort nearnbrs */
    near_nbrs->num_intrs = 0;
    far_nbrs->num_intrs = 0;
    for ( i = 0; i < system->N - 1; ++i )
    {
        if ( End_Index(i, near_nbrs) > Start_Index(i + 1, near_nbrs) )
        {
            fprintf( stderr,
                     "step%3d: nearnbr list of atom%d is overwritten by atom%d\n",
                     data->step, i + 1, i );
            exit( RUNTIME_ERROR );
        }

        near_nbrs->num_intrs += Num_Entries(i, near_nbrs);

        if ( End_Index(i, far_nbrs) > Start_Index(i + 1, far_nbrs) )
        {
            fprintf( stderr,
                     "step%3d: farnbr list of atom%d is overwritten by atom%d\n",
                     data->step, i + 1, i );
            exit( RUNTIME_ERROR );
        }

        far_nbrs->num_intrs += Num_Entries(i, far_nbrs);
    }

    for ( i = 0; i < system->N; ++i )
    {
        qsort( &(near_nbrs->select.near_nbr_list[ Start_Index(i, near_nbrs) ]),
               Num_Entries(i, near_nbrs), sizeof(near_neighbor_data),
               compare_near_nbrs );
    }
    // fprintf( stderr, "near nbrs sorted\n" );


#ifdef TEST_ENERGY
    /* for( i = 0; i < system->N; ++i ) {
       qsort( &(far_nbrs->select.far_nbr_list[ Start_Index(i, far_nbrs) ]),
       Num_Entries(i, far_nbrs), sizeof(far_neighbor_data),
       compare_far_nbrs );
       } */

    fprintf( stderr, "Near neighbors/atom: %d (compare to 150)\n",
             num_near / system->N );
    fprintf( stderr, "Far neighbors per atom: %d (compare to %d)\n",
             num_far / system->N, control->max_far_nbrs );
#endif

    //fprintf( stderr, "step%d: num of nearnbrs = %6d   num of farnbrs: %6d\n",
    //       data->step, num_near, num_far );

    //fprintf( stderr, "\talloc nearnbrs = %6d   alloc farnbrs: %6d\n",
    //   system->N * near_nbrs->intrs_per_unit,
    //   system->N * far_nbrs->intrs_per_unit );
}


void Generate_Neighbor_Lists( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        list **lists, output_controls *out_control )
{
    int  i, j, k, l, m, itr;
    int  x, y, z;
    int  atom1, atom2, max;
    int  num_far, c, count;
    int  *nbr_atoms;
    ivec *nbrs;
    rvec *nbrs_cp;
    grid *g;
    list *far_nbrs;
    get_far_neighbors_function Get_Far_Neighbors;
    far_neighbor_data new_nbrs[125];

    g = &( system->g );
    far_nbrs = (*lists) + FAR_NBRS;

    // fprintf( stderr, "\n\tentered nbrs - " );
    if ( control->ensemble == iNPT ||
            control->ensemble == sNPT ||
            control->ensemble == NPT )
    {
        Update_Grid( system );
    }
    // fprintf( stderr, "grid updated - " );

    Bin_Atoms( system, out_control );
    // fprintf( stderr, "atoms sorted - " );
    Choose_Neighbor_Finder( system, control, &Get_Far_Neighbors );
    // fprintf( stderr, "function chosen - " );
    Reset_Neighbor_Lists( system, workspace, lists );
    // fprintf( stderr, "lists cleared - " );

    num_far = 0;
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
                fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

                /* pick up an atom from the current cell */
                for (l = 0; l < g->top[i][j][k]; ++l )
                {
                    atom1 = g->atoms[i][j][k][l];
                    Set_Start_Index( atom1, num_far, far_nbrs );
                    fprintf( stderr, "\tatom %d\n", atom1 );

                    itr = 0;
                    while ( nbrs[itr][0] > 0 )
                    {
                        x = nbrs[itr][0];
                        y = nbrs[itr][1];
                        z = nbrs[itr][2];
                        fprintf( stderr, "\t\tgridcell %d %d %d\n", x, y, z );

                        // if( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <=
                        //     SQR(control->r_cut))
                        nbr_atoms = g->atoms[x][y][z];
                        max = g->top[x][y][z];
                        fprintf( stderr, "\t\tmax: %d\n", max );


                        /* pick up another atom from the neighbor cell -
                           we have to compare atom1 with its own periodic images as well,
                           that's why there is also equality in the if stmt below */
                        for ( m = 0, atom2 = nbr_atoms[m]; m < max; ++m, atom2 = nbr_atoms[m] )
                        {
                            if ( atom1 >= atom2 )
                            {
                                Get_Far_Neighbors( system->atoms[atom1].x,
                                                   system->atoms[atom2].x,
                                                   &(system->box), control, new_nbrs, &count );
                                fprintf( stderr, "\t\t\t%d count:%d\n", atom2, count );

                                for ( c = 0; c < count; ++c )
                                    if (atom1 != atom2 || (atom1 == atom2 && new_nbrs[c].d >= 0.1))
                                    {
                                        Set_Far_Neighbor(&(far_nbrs->select.far_nbr_list[num_far]),
                                                         atom2, new_nbrs[c].d, 1.0,
                                                         new_nbrs[c].dvec, new_nbrs[c].rel_box );
                                        ++num_far;

                                        /*fprintf(stderr,"FARNBR:%6d%6d%8.3f[%8.3f%8.3f%8.3f]\n",
                                          atom1, atom2, new_nbrs[c].d,
                                          new_nbrs[c].dvec[0], new_nbrs[c].dvec[1],
                                          new_nbrs[c].dvec[2] ); */
                                    }
                            }
                        }

                        ++itr;
                    }

                    Set_End_Index( atom1, num_far, far_nbrs );
                }
            }
        }
    }

    far_nbrs->num_intrs = num_far;
    fprintf( stderr, "nbrs done, num_far: %d\n", num_far );

#if defined(DEBUG)
    for ( i = 0; i < system->N; ++i )
    {
        qsort( &(far_nbrs->select.far_nbr_list[ Start_Index(i, far_nbrs) ]),
               Num_Entries(i, far_nbrs), sizeof(far_neighbor_data),
               compare_far_nbrs );
    }

    fprintf( stderr, "step%d: num of farnbrs=%6d\n", data->step, num_far );
    fprintf( stderr, "\tallocated farnbrs: %6d\n",
             system->N * far_nbrs->intrs_per_unit );
#endif
}



#endif
