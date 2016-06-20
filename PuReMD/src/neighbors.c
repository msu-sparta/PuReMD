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

#include "neighbors.h"
#include "io_tools.h"
#include "list.h"
#include "tool_box.h"
#include "vector.h"


int compare_far_nbrs( const void *p1, const void *p2 )
{
    return ((far_neighbor_data *)p1)->nbr - ((far_neighbor_data *)p2)->nbr;
}


void Draw_Near_Neighbor_Box( reax_system *system, control_params *control,
                             storage *workspace )
{
    int  i;
    reax_atom       *atom;
    simulation_box  *my_box;
    boundary_cutoff *bc;

    my_box = &( system->my_box );
    bc = &( system->bndry_cuts );

    /* all native atoms are within near neighbor skin */
    for ( i = 0; i < system->n; ++i )
        workspace->within_bond_box[i] = 1;

    /* loop over imported atoms */
    for ( i = system->n; i < system->N; ++i )
    {
        atom = &(system->my_atoms[i]);

        if ( my_box->min[0] - bc->ghost_bond <= atom->x[0] &&
                atom->x[0] <= my_box->max[0] + bc->ghost_bond &&
                my_box->min[1] - bc->ghost_bond <= atom->x[1] &&
                atom->x[1] <= my_box->max[1] + bc->ghost_bond &&
                my_box->min[2] - bc->ghost_bond <= atom->x[2] &&
                atom->x[2] <= my_box->max[2] + bc->ghost_bond )
            workspace->within_bond_box[i] = 1;
        else
            workspace->within_bond_box[i] = 0;
    }
}


void Generate_Neighbor_Lists( reax_system *system, simulation_data *data,
                              storage *workspace, reax_list **lists )
{
    int  i, j, k, l, m, itr, num_far;
    real d, cutoff;
    rvec dvec;
    grid *g;
    grid_cell *gci, *gcj;
    reax_list *far_nbrs;
    far_neighbor_data *nbr_data;
    reax_atom *atom1, *atom2;

#if defined(LOG_PERFORMANCE)
    real t_start = 0, t_elapsed = 0;

    if ( system->my_rank == MASTER_NODE )
        t_start = Get_Time( );
#endif

    // fprintf( stderr, "\n\tentered nbrs - " );
    g = &( system->my_grid );
    far_nbrs = (*lists) + FAR_NBRS;
    num_far = 0;

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncells[0]; i++ )
        for ( j = 0; j < g->ncells[1]; j++ )
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                gci = &(g->cells[i][j][k]);
                cutoff = SQR(gci->cutoff);
                //fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

                /* pick up an atom from the current cell */
                for (l = gci->str; l < gci->end; ++l )
                {
                    atom1 = &(system->my_atoms[l]);
                    Set_Start_Index( l, num_far, far_nbrs );
                    //fprintf( stderr, "\tatom %d\n", atom1 );

                    itr = 0;
                    while ( (gcj = gci->nbrs[itr]) != NULL )
                    {
                        if ( gci->str <= gcj->str &&
                                (DistSqr_to_Special_Point(gci->nbrs_cp[itr], atom1->x) <= cutoff) )
                            /* pick up another atom from the neighbor cell */
                            for ( m = gcj->str; m < gcj->end; ++m )
                                if ( l < m )  // prevent recounting same pairs within a gcell
                                {
                                    atom2 = &(system->my_atoms[m]);
                                    dvec[0] = atom2->x[0] - atom1->x[0];
                                    dvec[1] = atom2->x[1] - atom1->x[1];
                                    dvec[2] = atom2->x[2] - atom1->x[2];
                                    d = rvec_Norm_Sqr( dvec );
                                    if ( d <= cutoff )
                                    {
                                        nbr_data = &(far_nbrs->far_nbr_list[num_far]);
                                        nbr_data->nbr = m;
                                        nbr_data->d = sqrt(d);
                                        rvec_Copy( nbr_data->dvec, dvec );
                                        //ivec_Copy( nbr_data->rel_box, gcj->rel_box );
                                        ivec_ScaledSum( nbr_data->rel_box,
                                                        1, gcj->rel_box, -1, gci->rel_box );
                                        ++num_far;
                                    }
                                }
                        ++itr;
                    }
                    Set_End_Index( l, num_far, far_nbrs );
                    //fprintf(stderr, "i:%d, start: %d, end: %d - itr: %d\n",
                    //  atom1,Start_Index(atom1,far_nbrs),End_Index(atom1,far_nbrs),
                    //  itr);
                }
            }

    workspace->realloc.num_far = num_far;

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_elapsed = Get_Timing_Info( t_start );
        data->timing.nbrs += t_elapsed;
    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: nbrs done - num_far=%d\n",
             system->my_rank, data->step, num_far );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
    for ( i = 0; i < system->N; ++i )
        qsort( &(far_nbrs->far_nbr_list[ Start_Index(i, far_nbrs) ]),
               Num_Entries(i, far_nbrs), sizeof(far_neighbor_data),
               compare_far_nbrs );
#endif
}


int Estimate_NumNeighbors( reax_system *system, reax_list **lists )
{
    int  i, j, k, l, m, itr, num_far; //, tmp, tested;
    real d, cutoff;
    rvec dvec;
    grid *g;
    grid_cell *gci, *gcj;
    reax_atom *atom1, *atom2;

    // fprintf( stderr, "\n\tentered nbrs - " );
    g = &( system->my_grid );
    num_far = 0;

    /* first pick up a cell in the grid */
    for ( i = 0; i < g->ncells[0]; i++ )
        for ( j = 0; j < g->ncells[1]; j++ )
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                gci = &(g->cells[i][j][k]);
                cutoff = SQR(gci->cutoff);
                //fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

                /* pick up an atom from the current cell */
                for ( l = gci->str; l < gci->end; ++l )
                {
                    atom1 = &(system->my_atoms[l]);
                    //fprintf( stderr, "\tatom %d: ", l );
                    //tmp = num_far; tested = 0;
                    itr = 0;
                    while ( (gcj = gci->nbrs[itr]) != NULL )
                    {
                        if (gci->str <= gcj->str &&
                                (DistSqr_to_Special_Point(gci->nbrs_cp[itr], atom1->x) <= cutoff))
                            //fprintf( stderr, "\t\tgcell2: %d\n", itr );
                            /* pick up another atom from the neighbor cell */
                            for ( m = gcj->str; m < gcj->end; ++m )
                                if ( l < m )
                                {
                                    //fprintf( stderr, "\t\t\tatom2=%d\n", m );
                                    atom2 = &(system->my_atoms[m]);
                                    dvec[0] = atom2->x[0] - atom1->x[0];
                                    dvec[1] = atom2->x[1] - atom1->x[1];
                                    dvec[2] = atom2->x[2] - atom1->x[2];
                                    d = rvec_Norm_Sqr( dvec );
                                    if ( d <= cutoff )
                                        ++num_far;
                                }

                        ++itr;
                    }
                    //fprintf( stderr, "itr: %d, tested: %d, num_nbrs: %d\n",
                    //   itr, tested, num_far-tmp );
                }
            }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: estimate nbrs done - num_far=%d\n",
             system->my_rank, num_far );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
    return MAX( num_far * SAFE_ZONE, MIN_CAP * MIN_NBRS );
}
