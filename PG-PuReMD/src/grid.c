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

#include "grid.h"
#include "allocate.h"
#include "io_tools.h"
#include "reset_tools.h"
#include "tool_box.h"
#include "vector.h"

#include "index_utils.h"


/* determines the exchange boundaries with nbrs in terms of gcells */
void Mark_GCells( reax_system* system, grid *g, ivec procs, MPI_Comm comm )
{
    int x, y, z, d;
    ivec r, nbr_coord, prdc;
    ivec send_span, recv_span;
    ivec str_send, end_send;
    ivec str_recv, end_recv;

    /* clear all gcell type info */
    for ( x = 0; x < g->ncells[0]; x++ )
    {
        for ( y = 0; y < g->ncells[1]; y++ )
        {
            for ( z = 0; z < g->ncells[2]; z++ )
            {
                //SUDHIR
                //g->cells[x][y][z].type = 0;
                g->cells[ index_grid_3d (x, y, z, g) ].type = 0;
            }
        }
    }

    /* mark native cells */
    for ( x = g->native_str[0]; x < g->native_end[0]; x++ )
    {
        for ( y = g->native_str[1]; y < g->native_end[1]; y++ )
        {
            for ( z = g->native_str[2]; z < g->native_end[2]; z++ )
            {
                //SUDHIR
                //g->cells[x][y][z].type = NATIVE;
                //ivec_MakeZero( g->cells[x][y][z].rel_box );
                g->cells[ index_grid_3d (x, y, z, g) ].type = NATIVE;
                //ivec_MakeZero( g->cells[ index_grid_3d(x, y, z, g) ].rel_box );
                ivec_MakeZero( g->rel_box[ index_grid_3d(x, y, z, g) ]);
            }
        }
    }

    /* loop over neighbors */
    for ( r[0] = -1; r[0] <= 1; ++r[0])
    {
        for ( r[1] = -1; r[1] <= 1; ++r[1] )
        {
            for ( r[2] = -1; r[2] <= 1; ++r[2] )
            {
                /* determine the width of exchange with nbr_pr */
                ivec_Copy( send_span, g->ghost_span );
                ivec_Copy( recv_span, g->ghost_span );
                /* calculate actual neighbor coordinates */
                ivec_Sum( nbr_coord, system->my_coords, r );

                for ( d = 0; d < 3; ++d )
                {
                    /* determine the periodicity of this neighbor */
                    if ( nbr_coord[d] < 0 )
                    {
                        prdc[d] = -1;
                    }
                    else if ( nbr_coord[d] >= procs[d] )
                    {
                        prdc[d] = +1;
                    }
                    else
                    {
                        prdc[d] = 0;
                    }

                    /* determine gcells to be sent & recv'd */
                    if ( r[d] == -1 )
                    {
                        str_send[d] = g->native_str[d];
                        end_send[d] = g->native_str[d] + send_span[d];
                        str_recv[d] = g->native_str[d] - recv_span[d];
                        end_recv[d] = g->native_str[d];
                    }
                    else if ( r[d] == 0 )
                    {
                        str_send[d] = g->native_str[d];
                        end_send[d] = g->native_end[d];
                        str_recv[d] = g->native_str[d];
                        end_recv[d] = g->native_end[d];
                    }
                    else   // r[d] == +1
                    {
                        str_send[d] = g->native_end[d] - send_span[d];
                        end_send[d] = g->native_end[d];
                        str_recv[d] = g->native_end[d];
                        end_recv[d] = g->native_end[d] + recv_span[d];
                    }
                }

                for ( x = str_recv[0]; x < end_recv[0]; ++x )
                {
                    for ( y = str_recv[1]; y < end_recv[1]; ++y )
                    {
                        for ( z = str_recv[2]; z < end_recv[2]; ++z )
                        {
                            //SUDHIR
                            //ivec_Copy( g->cells[x][y][z].rel_box, prdc );
                            //ivec_Copy( g->cells[ index_grid_3d(x, y, z, g) ].rel_box, prdc );
                            ivec_Copy( g->rel_box[ index_grid_3d(x, y, z, g) ], prdc );
                        }
                    }
                }
            }
        }
    }
}


/* finds the closest point between two grid cells denoted by c1 and c2.
   periodic boundary conditions are taken into consideration as well. */
void Find_Closest_Point( grid *g, ivec c1, ivec c2, rvec closest_point )
{
    int  i, d;

    for ( i = 0; i < 3; i++ )
    {
        d = c2[i] - c1[i];

        if ( d > 0 )
        {
            //SUDHIR
            //closest_point[i] = g->cells[c2[0]][c2[1]][c2[2]].min[i];
            closest_point[i] = g->cells[ index_grid_3d(c2[0], c2[1], c2[2], g) ].min[i];
        }
        else if ( d == 0 )
        {
            closest_point[i] = NEG_INF - 1.;
        }
        else
        {
            //SUDHIR
            //closest_point[i] = g->cells[c2[0]][c2[1]][c2[2]].max[i];
            closest_point[i] = g->cells[ index_grid_3d(c2[0], c2[1], c2[2], g) ].max[i];
        }
    }
}


/* mark gcells based on the kind of nbrs we will be looking for in them */
void Find_Neighbor_GridCells( grid *g, control_params *control )
{
    int  d, top;
    ivec ci, cj, cmin, cmax, span;
    grid_cell *gc;

    //TODO
    //TODO
    //TODO
    //TODO
    //fprintf (stderr, " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    //fprintf (stderr, " CHANGED TO WORK NEIGHBOR LISTS \n");
    //fprintf (stderr, " DEBUG THIS ISSUE \n");
    //fprintf (stderr, " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    //fprintf (stderr, " vlist_cut: %f \n", control->vlist_cut);
    //fprintf (stderr, " bond_cut: %f \n", control->bond_cut);
    //TODO
    //TODO
    //TODO
    //TODO

    /* pick up a cell in the grid */
    for ( ci[0] = 0; ci[0] < g->ncells[0]; ci[0]++ )
    {
        for ( ci[1] = 0; ci[1] < g->ncells[1]; ci[1]++ )
        {
            for ( ci[2] = 0; ci[2] < g->ncells[2]; ci[2]++ )
            {
                //SUDHIR
                //gc = &(g->cells[ci[0]][ci[1]][ci[2]]);
                gc = &(g->cells[ index_grid_3d(ci[0], ci[1], ci[2], g) ]);
                top = 0;
                //fprintf( stderr, "grid1: %d %d %d:\n", ci[0], ci[1], ci[2] );

                //TODO
                /////////////////////////////////////////////////////////////////
                //if( gc->type == NATIVE )
                // gc->cutoff = control->vlist_cut;
                //else gc->cutoff = control->bond_cut;
                //gc->cutoff = control->vlist_cut;
                if (gc->type == NATIVE)
                {
                    g->cutoff [index_grid_3d (ci[0], ci[1], ci[2], g)] = control->vlist_cut;
                }
                else
                {
                    g->cutoff [index_grid_3d (ci[0], ci[1], ci[2], g)] = control->bond_cut;
                }

                /////////////////////////////////////////////////////////////////
                //TODO


                for ( d = 0; d < 3; ++d )
                {
                    //span[d] = (int)ceil( gc->cutoff / g->cell_len[d] );
                    span[d] = (int)ceil( control->vlist_cut / g->cell_len[d] );
                    cmin[d] = MAX(ci[d] - span[d], 0 );
                    cmax[d] = MIN(ci[d] + span[d] + 1, g->ncells[d] );
                }

                /* loop over neighboring gcells */
                for ( cj[0] = cmin[0]; cj[0] < cmax[0]; ++cj[0] )
                {
                    for ( cj[1] = cmin[1]; cj[1] < cmax[1]; ++cj[1] )
                    {
                        for ( cj[2] = cmin[2]; cj[2] < cmax[2]; ++cj[2] )
                        {
                            //fprintf( stderr, "\tgrid2: %d %d %d (%d - %d) - ", cj[0], cj[1], cj[2], top, g->max_nbrs );
                            //SUDHIR
                            //gc->nbrs[top] = &(g->cells[cj[0]][cj[1]][cj[2]]);
                            //gc->nbrs[top] = &(g->cells[ index_grid_3d(cj[0],cj[1],cj[2],g) ]);
                            ivec_Copy( g->nbrs_x[index_grid_nbrs (ci[0], ci[1], ci[2], top, g)], cj );
                            //fprintf (stderr, " index: %d - %d \n", index_grid_nbrs (ci[0], ci[1], ci[2], top, g), g->total * g->max_nbrs);
                            Find_Closest_Point( /*ext_box,*/ g, ci, cj, g->nbrs_cp[index_grid_nbrs (ci[0], ci[1], ci[2], top, g)] );
                            //fprintf( stderr, "cp: %f %f %f\n",
                            //       gc->nbrs_cp[top][0], gc->nbrs_cp[top][1],
                            //       gc->nbrs_cp[top][2] );
                            ++top;
                        }
                    }
                }
                //gc->nbrs[top] = NULL;
                //fprintf( stderr, "top=%d\n", top );
            }
        }
    }
}


void Reorder_GridCells( grid *g )
{
    int i, j, k, x, y, z, top;
    ivec dblock, nblocks;

    dblock[0] = 1; //3; //4; //(int)(ceil( sqrt(g->ncells[0]) ));
    dblock[1] = 1; //3; //4; //(int)(ceil( sqrt(g->ncells[1]) ));
    dblock[2] = 1; //3; //4; //(int)(ceil( sqrt(g->ncells[2]) ));
    nblocks[0] = (int)(ceil( (real)g->ncells[0] / dblock[0] ));
    nblocks[1] = (int)(ceil( (real)g->ncells[1] / dblock[1] ));
    nblocks[2] = (int)(ceil( (real)g->ncells[2] / dblock[2] ));

    top = 0;
    for ( i = 0; i < nblocks[0]; ++i )
    {
        for ( j = 0; j < nblocks[1]; ++j )
        {
            for ( k = 0; k < nblocks[2]; ++k )
            {
                for ( x = i * dblock[0]; x < MIN((i + 1)*dblock[0], g->ncells[0]); ++x )
                {
                    for ( y = j * dblock[1]; y < MIN((j + 1)*dblock[1], g->ncells[1]); ++y )
                    {
                        for ( z = k * dblock[2]; z < MIN((k + 1)*dblock[2], g->ncells[2]); ++z )
                        {
                            g->order[top][0] = x;
                            g->order[top][1] = y;
                            g->order[top][2] = z;
                            ++top;
                        }
                    }
                }
            }
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "reorder_gcells: total_gcells=%d top=%d\n", g->total, top );
    fprintf( stderr, "dblock: %d %d %d\n", dblock[0], dblock[1], dblock[2] );
    fprintf( stderr, "nblocks: %d %d %d\n", nblocks[0], nblocks[1], nblocks[2] );
    fprintf( stderr, "reordered gcells:\n" );
    for ( i = 0; i < top; ++i )
    {
        fprintf( stderr, "order%d: %d %d %d\n",
                 i, g->order[i][0], g->order[i][1], g->order[i][2] );
    }
#endif
}


void Setup_New_Grid( reax_system* system, control_params* control,
                     MPI_Comm comm )
{
    int              d, i, j, k;
    grid            *g;
    simulation_box  *my_box, *my_ext_box;
    boundary_cutoff *bc;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: setup new grid\n", system->my_rank );
#endif

    g = &( system->my_grid );
    my_box = &( system->my_box );
    my_ext_box = &( system->my_ext_box );
    bc = &(system->bndry_cuts);

    /* compute number of grid cells and props in each direction */
    for ( d = 0; d < 3; ++d )
    {
        /* estimate the number of native cells */
        g->native_cells[d] = (int)(my_box->box_norms[d] / (control->vlist_cut / 2));
        if ( g->native_cells[d] == 0 )
        {
            g->native_cells[d] = 1;
        }
    }

    /* cell lengths */
    rvec_iDivide( g->cell_len, my_box->box_norms, g->native_cells );
    rvec_Invert( g->inv_len, g->cell_len );

    for ( d = 0; d < 3; ++d )
    {
        /* # of surrounding grid cells to look into for nonbonded & bonded nbrs */
        g->vlist_span[d] = (int)ceil( control->vlist_cut / g->cell_len[d] );
        g->nonb_span[d] = (int)ceil( control->nonb_cut / g->cell_len[d] );
        g->bond_span[d] = (int)ceil( control->bond_cut / g->cell_len[d] );
        /* span of the ghost region in terms of gcells */
        g->ghost_span[d] = (int)ceil(system->bndry_cuts.ghost_cutoff /
                                     g->cell_len[d]);
        g->ghost_nonb_span[d] = (int)ceil(system->bndry_cuts.ghost_nonb /
                                          g->cell_len[d]);
        g->ghost_hbond_span[d] = (int)ceil( system->bndry_cuts.ghost_hbond /
                                            g->cell_len[d] );
        g->ghost_bond_span[d] = (int)ceil( system->bndry_cuts.ghost_bond /
                                           g->cell_len[d] );
    }

    /* total number of grid cells */
    ivec_ScaledSum( g->ncells, 1, g->native_cells, 2, g->ghost_span );
    g->total = g->ncells[0] * g->ncells[1] * g->ncells[2];
    //fprintf (stderr, " dimensions (%d, %d, %d) \n", g->ncells[0], g->ncells[1], g->ncells[2]);

    /* native cell start & ends */
    ivec_Copy( g->native_str, g->ghost_span );
    ivec_Sum( g->native_end, g->native_str, g->native_cells );

    /* upper bound on the number of gcells to be exchanged with a single nbr */
    system->gcell_cap =
        MAX3( g->native_cells[0] * g->native_cells[1] * g->ghost_span[2],
              g->native_cells[0] * g->native_cells[2] * g->ghost_span[1],
              g->native_cells[1] * g->native_cells[2] * g->ghost_span[0] ) + 1;

    /* allocate grid space */
    Allocate_Grid( system, comm );

    /* compute min and max coords for each grid cell */
    for ( i = 0; i < g->ncells[0]; i++ )
    {
        for ( j = 0; j < g->ncells[1]; j++ )
        {
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                /*
                g->cells[i][j][k].min[0] = my_ext_box->min[0] + i * g->cell_len[0];
                g->cells[i][j][k].min[1] = my_ext_box->min[1] + j * g->cell_len[1];
                g->cells[i][j][k].min[2] = my_ext_box->min[2] + k * g->cell_len[2];

                g->cells[i][j][k].max[0] = my_ext_box->min[0] + (i+1)*g->cell_len[0];
                g->cells[i][j][k].max[1] = my_ext_box->min[1] + (j+1)*g->cell_len[1];
                g->cells[i][j][k].max[2] = my_ext_box->min[2] + (k+1)*g->cell_len[2];
                */
                g->cells[ index_grid_3d(i, j, k, g) ].min[0] = my_ext_box->min[0] + i * g->cell_len[0];
                g->cells[ index_grid_3d(i, j, k, g) ].min[1] = my_ext_box->min[1] + j * g->cell_len[1];
                g->cells[ index_grid_3d(i, j, k, g) ].min[2] = my_ext_box->min[2] + k * g->cell_len[2];

                g->cells[ index_grid_3d(i, j, k, g) ].max[0] = my_ext_box->min[0] + (i + 1) * g->cell_len[0];
                g->cells[ index_grid_3d(i, j, k, g) ].max[1] = my_ext_box->min[1] + (j + 1) * g->cell_len[1];
                g->cells[ index_grid_3d(i, j, k, g) ].max[2] = my_ext_box->min[2] + (k + 1) * g->cell_len[2];
            }
        }
    }

    /* determine the exchange boundaries with nbrs in terms of gcells */
    Mark_GCells( system, g, control->procs_by_dim, comm );

    /* determine what kind of nbrs we will be looking for in boundary gcells */
    Find_Neighbor_GridCells( g, control );

    Reorder_GridCells( g );
}



void Update_Grid( reax_system* system, control_params* control, MPI_Comm comm )
{
    int  d, i, j, k, itr;
    ivec ci, native_cells, nonb_span, bond_span;
    ivec ghost_span, ghost_nonb_span, ghost_bond_span, ghost_hbond_span;;
    rvec cell_len, inv_len;
    grid *g;
    grid_cell *gc;
    simulation_box *my_box;
    simulation_box *my_ext_box;
    boundary_cutoff *bc;

    g = &( system->my_grid );
    my_box = &( system->my_box );
    my_ext_box = &( system->my_ext_box );
    bc = &(system->bndry_cuts);

    /* compute number of grid cells and props in each direction */
    for ( d = 0; d < 3; ++d )
    {
        /* estimate the number of native cells */
        native_cells[d] = (int)(my_box->box_norms[d] / (control->vlist_cut / 2));
        if ( native_cells[d] == 0 )
        {
            native_cells[d] = 1;
        }
    }

    /* cell lengths */
    rvec_iDivide( cell_len, my_box->box_norms, native_cells );
    rvec_Invert( inv_len, cell_len );

    for ( d = 0; d < 3; ++d )
    {
        /* # of surrounding grid cells to look into for nonbonded & bonded nbrs */
        nonb_span[d] = (int)ceil( control->nonb_cut / cell_len[d] );
        bond_span[d] = (int)ceil( control->bond_cut / cell_len[d] );
        /* span of the ghost region in terms of gcells */
        ghost_span[d] = (int)ceil(system->bndry_cuts.ghost_cutoff / cell_len[d]);
        ghost_nonb_span[d] = (int)ceil(system->bndry_cuts.ghost_nonb / cell_len[d]);
        ghost_hbond_span[d] = (int)ceil( system->bndry_cuts.ghost_hbond /
                                         cell_len[d] );
        ghost_bond_span[d] = (int)ceil( system->bndry_cuts.ghost_bond /
                                        cell_len[d] );
    }


    if ( ivec_isEqual( native_cells, g->native_cells ) &&
            ivec_isEqual( ghost_span, g->ghost_span ) )   // gcells are unchanged
    {
        /* update cell lengths */
        rvec_Copy( g->cell_len, cell_len );
        rvec_Copy( g->inv_len, inv_len );

        /* compute min and max coords for each grid cell */
        for ( i = 0; i < g->ncells[0]; i++ )
        {
            for ( j = 0; j < g->ncells[1]; j++ )
            {
                for ( k = 0; k < g->ncells[2]; k++ )
                {
                    /*
                      g->cells[i][j][k].min[0] = my_ext_box->min[0] + i * g->cell_len[0];
                      g->cells[i][j][k].min[1] = my_ext_box->min[1] + j * g->cell_len[1];
                      g->cells[i][j][k].min[2] = my_ext_box->min[2] + k * g->cell_len[2];

                      g->cells[i][j][k].max[0] = my_ext_box->min[0]+(i+1)*g->cell_len[0];
                      g->cells[i][j][k].max[1] = my_ext_box->min[1]+(j+1)*g->cell_len[1];
                      g->cells[i][j][k].max[2] = my_ext_box->min[2]+(k+1)*g->cell_len[2];
                      */
                    g->cells[ index_grid_3d(i, j, k, g) ].min[0] = my_ext_box->min[0] + i * g->cell_len[0];
                    g->cells[ index_grid_3d(i, j, k, g) ].min[1] = my_ext_box->min[1] + j * g->cell_len[1];
                    g->cells[ index_grid_3d(i, j, k, g) ].min[2] = my_ext_box->min[2] + k * g->cell_len[2];

                    g->cells[ index_grid_3d(i, j, k, g) ].max[0] = my_ext_box->min[0] + (i + 1) * g->cell_len[0];
                    g->cells[ index_grid_3d(i, j, k, g) ].max[1] = my_ext_box->min[1] + (j + 1) * g->cell_len[1];
                    g->cells[ index_grid_3d(i, j, k, g) ].max[2] = my_ext_box->min[2] + (k + 1) * g->cell_len[2];
                }
            }
        }

        /* pick up a cell in the grid */
        for ( ci[0] = 0; ci[0] < g->ncells[0]; ci[0]++ )
        {
            for ( ci[1] = 0; ci[1] < g->ncells[1]; ci[1]++ )
            {
                for ( ci[2] = 0; ci[2] < g->ncells[2]; ci[2]++ )
                {
                    //SUDHIR
                    //gc = &(g->cells[ci[0]][ci[1]][ci[2]]);
                    gc = &(g->cells[ index_grid_3d (ci[0], ci[1], ci[2], g) ]);

                    itr = 0;
                    //while( g->nbrs[itr] != NULL ) {
                    while ( g->nbrs_x[itr][0] >= 0 )
                    {
                        //Find_Closest_Point( g, ci, gc->nbrs_x[itr], gc->nbrs_cp[itr] );
                        Find_Closest_Point( g, ci, g->nbrs_x[index_grid_nbrs (ci[0], ci[1], ci[2], itr, g)],
                                            g->nbrs_cp[index_grid_nbrs (ci[0], ci[1], ci[2], itr, g)] );
                        ++itr;
                    }
                }
            }
        }
    }
    else   // the grid has changed!
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: whole grid is being updated\n", system->my_rank );
#endif

        Deallocate_Grid( g );
        Setup_New_Grid( system, control, comm );
    }
}


/* bin my atoms into grid cells */
void Bin_My_Atoms( reax_system *system, reallocate_data *realloc )
{
    int  i, j, k, l, d, max_atoms;
    ivec c;
    simulation_box *big_box, *my_box, *my_ext_box;
    grid  *g;
    grid_cell *gc;
    reax_atom *atoms;

    big_box    = &(system->big_box);
    my_ext_box = &(system->my_ext_box);
    my_box     = &(system->my_box);
    g          = &(system->my_grid);
    atoms      = system->my_atoms;

#if defined(DEBUG)
    fprintf( stderr, "p%d bin_my_atoms: entered\n", system->my_rank );
#endif

    Reset_Grid( g );

    for ( l = 0; l < system->n; l++ )
    {
        // outgoing atoms are marked with orig_id = -1
        if ( atoms[l].orig_id >= 0 )
        {
            for ( d = 0; d < 3; ++d )
            {
                //if( atoms[l].x[d] < big_box->min[d] )
                //  atoms[l].x[d] += big_box->box_norms[d];
                //else if( atoms[l].x[d] >= big_box->max[d] )
                //  atoms[l].x[d] -= big_box->box_norms[d];
                if ( atoms[l].x[d] < my_box->min[d] || atoms[l].x[d] > my_box->max[d] )
                {
                    fprintf( stderr, "p%d: local atom%d [%f %f %f] is out of my box!\n",
                             system->my_rank, l,
                             atoms[l].x[0], atoms[l].x[1], atoms[l].x[2] );
                    fprintf( stderr, "p%d: my_box=[%f-%f, %f-%f, %f-%f]\n",
                             system->my_rank, my_box->min[0], my_box->max[0],
                             my_box->min[1], my_box->max[1],
                             my_box->min[2], my_box->max[2] );
                    MPI_Abort( MPI_COMM_WORLD, -1 );
                }

                c[d] = (int)((atoms[l].x[d] - my_ext_box->min[d]) * g->inv_len[d]);
                if ( c[d] >= g->native_end[d] )
                {
                    c[d] = g->native_end[d] - 1;
                }
                else if ( c[d] < g->native_str[d] )
                {
                    c[d] = g->native_str[d];
                }
            }

#if defined(DEBUG)
            fprintf( stderr, "p%d bin_my_atoms: l:%d - atom%d @ %.5f %.5f %.5f"\
                     "--> cell: %d %d %d\n",
                     system->my_rank, l, atoms[l].orig_id,
                     atoms[l].x[0], atoms[l].x[1], atoms[l].x[2],
                     c[0], c[1], c[2] );
#endif

            //SUDHIR
            //gc = &( g->cells[c[0]][c[1]][c[2]] );
            gc = &( g->cells[ index_grid_3d(c[0], c[1], c[2], g) ] );
            gc->atoms[ gc->top++ ] = l;
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d bin_my_atoms: sorted atoms\n", system->my_rank );
#endif

    max_atoms = 0;
    for ( i = g->native_str[0]; i < g->native_end[0]; i++ )
    {
        for ( j = g->native_str[1]; j < g->native_end[1]; j++ )
        {
            for ( k = g->native_str[2]; k < g->native_end[2]; k++ )
            {
                //SUDHIR
                //gc = &(g->cells[i][j][k]);
                gc = &(g->cells[ index_grid_3d(i, j, k, g) ]);
                if ( max_atoms < gc->top )
                {
                    max_atoms = gc->top;
                }

#if defined(DEBUG)
                fprintf( stderr, "p%d gc[%d,%d,%d]->top=%d\n",
                         system->my_rank, i, j, k, gc->top );
#endif
            }
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d max_atoms=%d, g->max_atoms=%d\n",
             system->my_rank, max_atoms, g->max_atoms );
#endif
    /* check if current gcell->max_atoms is safe */
    //fprintf (stderr, "*********** grid max_atoms: %d \n", g->max_atoms );
    if ( max_atoms >= g->max_atoms * DANGER_ZONE )
    {
        realloc->gcell_atoms = MAX( max_atoms * SAFE_ZONE, MIN_GCELL_POPL );
#ifdef HAVE_CUDA
        dev_workspace->realloc.gcell_atoms = MAX( max_atoms * SAFE_ZONE, MIN_GCELL_POPL );
#endif
    }
    else
    {
        realloc->gcell_atoms = -1;
#ifdef HAVE_CUDA
        dev_workspace->realloc.gcell_atoms = -1;
#endif
    }
}


/* reorder atoms falling into the same gcell together in the atom list */
void Reorder_My_Atoms( reax_system *system, storage *workspace )
{
    int        i, l, x, y, z;
    int        top, old_id;
    grid      *g;
    grid_cell *gc;
    reax_atom *old_atom, *new_atoms;

    /* allocate storage space for est_N */
    new_atoms = (reax_atom*) malloc( system->total_cap * sizeof(reax_atom) );
    top = 0;
    g = &( system->my_grid );

    //for( i = 0; i < g->ncells[0]; i++ )
    //for( j = 0; j < g->ncells[1]; j++ )
    //  for( k = 0; k < g->ncells[2]; k++ ) {
    for ( i = 0; i < g->total; ++i )
    {
        x = g->order[i][0];
        y = g->order[i][1];
        z = g->order[i][2];
        //SUDHIR
        //gc = &( g->cells[x][y][z] );
        gc = &( g->cells[ index_grid_3d (x, y, z, g) ] );
        g->str[index_grid_3d(x, y, z, g)] = top;

        for ( l = 0; l < gc->top; ++l )
        {
            old_id   = gc->atoms[l];
            old_atom = &( system->my_atoms[old_id] );
            //fprintf( stderr, "%d <-- %d\n", top, old_id );
            //reax_atom_Copy( &(new_atoms[top]), old_atom );
            memcpy( new_atoms + top, old_atom, sizeof(reax_atom) );
            new_atoms[top].imprt_id = -1;
            ++top;
        }
        g->end[index_grid_3d(x, y, z, g)] = top;
    }

    /* deallocate old storage */
    free( system->my_atoms );
    /* start using clustered storages */
    system->my_atoms = new_atoms;
    system->n = top;
    system->N = system->n;
}


void Get_Boundary_GCell( grid *g, rvec base, rvec x, grid_cell **gc,
                         rvec *cur_min, rvec *cur_max, ivec gcell_cood )
{
    int d;
    ivec c;
    rvec loosen = {1e-6, 1e-6, 1e-6};

    for ( d = 0; d < 3; ++d )
    {
        c[d] = (int)((x[d] - base[d]) * g->inv_len[d]);
        if ( c[d] < 0 )
        {
            c[d] = 0;
        }
        //else if( c[d] == g->native_str[d] ) --c[d];
        //else if( c[d] == g->native_end[d] - 1 ) ++c[d];
        else if ( c[d] >= g->ncells[d] )
        {
            c[d] = g->ncells[d] - 1;
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "get_bndry_gc: base=[%f %f %f] x=[%f %f %f] c=[%d %d %d]\n",
             base[0], base[1], base[2], x[0], x[1], x[2], c[0], c[1], c[2] );
#endif

    ivec_Copy (gcell_cood, c);

    //SUDHIR
    //*gc = &( g->cells[c[0]][c[1]][c[2]] );
    *gc = &( g->cells[ index_grid_3d(c[0], c[1], c[2], g) ] );
    rvec_ScaledSum( *cur_min, 1, (*gc)->min, -1, loosen );
    rvec_Sum( *cur_max, (*gc)->max, loosen );

#if defined(DEBUG)
    fprintf( stderr, "get_bndry_gc: gcmin=[%f %f %f] gcmax=[%f %f %f]\n",
             (*gc)->min[0], (*gc)->min[1], (*gc)->min[2],
             (*gc)->max[0], (*gc)->max[1], (*gc)->max[2] );
    fprintf( stderr, "get_bndry_gc: curmin=[%f %f %f] curmax=[%f %f %f]\n",
             (*cur_min)[0], (*cur_min)[1], (*cur_min)[2],
             (*cur_max)[0], (*cur_max)[1], (*cur_max)[2] );
#endif
}


int is_Within_GCell( rvec x, rvec cur_min, rvec cur_max )
{
    int d;

    for ( d = 0; d < 3; ++d )
    {
        if ( x[d] < cur_min[d] || x[d] > cur_max[d] )
        {
            return FALSE;
        }
    }

    return TRUE;
}


/* bin my atoms into grid cells */
void Bin_Boundary_Atoms( reax_system *system )
{
    int  i, start, end;
    rvec base, cur_min, cur_max;
    grid *g;
    grid_cell *gc;
    reax_atom *atoms;
    simulation_box *ext_box;
    ivec gcell_cood;

#if defined(DEBUG)
    fprintf( stderr, "p%d bin_boundary_atoms: entered with start: %d, end: %d\n", system->my_rank, system->n, system->N );
#endif

    g = &(system->my_grid);
    atoms = system->my_atoms;
    start = system->n;
    end = system->N;
    if ( start == end )
    {
        return;
    }

    ext_box = &(system->my_ext_box);
    memcpy( base, ext_box->min, sizeof(rvec) );

    Get_Boundary_GCell( g, base, atoms[start].x, &gc, &cur_min, &cur_max, gcell_cood );
    g->str[index_grid_3d (gcell_cood[0], gcell_cood[1], gcell_cood[2], g)] = start;
    gc->top = 1;
    /* error check */
    if ( is_Within_GCell( atoms[start].x, ext_box->min, ext_box->max ) == FALSE )
    {
        fprintf( stderr, "p%d: (start):ghost atom%d [%f %f %f] is out of my box!\n",
                 system->my_rank, start,
                 atoms[start].x[0], atoms[start].x[1], atoms[start].x[2] );
        //MPI_Abort( MPI_COMM_WORLD, -1 );
    }

    for ( i = start + 1; i < end; i++ )
    {
        /* error check */
        //if(atoms[i].x[0]<ext_box->min[0] || atoms[i].x[0]>ext_box->max[0] ||
        // atoms[i].x[1]<ext_box->min[1] || atoms[i].x[1]>ext_box->max[1] ||
        // atoms[i].x[2]<ext_box->min[2] || atoms[i].x[2]>ext_box->max[2] ){
        if ( is_Within_GCell( atoms[i].x, ext_box->min, ext_box->max ) == FALSE )
        {
            fprintf( stderr, "p%d: (middle )ghost atom%d [%f %f %f] is out of my box!\n",
                     system->my_rank, i,
                     atoms[i].x[0], atoms[i].x[1], atoms[i].x[2] );
            //MPI_Abort( MPI_COMM_WORLD, -1 );
        }

        if ( is_Within_GCell( atoms[i].x, cur_min, cur_max ) == TRUE )
        {
            ++gc->top;
        }
        else
        {
            g->end[index_grid_3d (gcell_cood[0], gcell_cood[1], gcell_cood[2], g)] = i;
            Get_Boundary_GCell( g, base, atoms[i].x, &gc, &cur_min, &cur_max, gcell_cood );
            /* sanity check! */
            if ( gc->top != 0 )
            {
                fprintf( stderr, "p%d bin_boundary_atoms: atom%d map was unexpected! ",
                         system->my_rank, i );
                fprintf( stderr, "[%f %f %f] --> [%f %f %f] to [%f %f %f]\n",
                         atoms[i].x[0], atoms[i].x[1], atoms[i].x[2],
                         gc->min[0], gc->min[1], gc->min[2],
                         gc->max[0], gc->max[1], gc->max[2] );
                //MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            }
            g->str[index_grid_3d (gcell_cood[0], gcell_cood[1], gcell_cood[2], g)] = i;
            gc->top = 1;
        }
    }

    /* mark last gcell's end position */
    g->end[index_grid_3d (gcell_cood[0], gcell_cood[1], gcell_cood[2], g)] = i;

#if defined(DEBUG)
    fprintf( stderr, "p%d bin_boundary_atoms: done\n", system->my_rank );
#endif

    //MPI_Abort( MPI_COMM_WORLD, -1 );
}
