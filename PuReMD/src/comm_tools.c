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

#include "comm_tools.h"
#include "grid.h"
#include "reset_tools.h"
#include "tool_box.h"
#include "vector.h"


void Setup_NT_Comm( reax_system* system, control_params* control,
                 mpi_datatypes *mpi_data )
{
    int i, d;
    real bndry_cut;
    neighbor_proc *nbr_pr;
    simulation_box *my_box;
    ivec nbr_coords;
    ivec r[12] = {
        {0, 0, -1}, // -z
        {0, 0, +1}, // +z
        {0, -1, 0}, // -y
        {-1, -1, 0}, // -x-y
        {-1, 0, 0}, // -x
        {-1, +1, 0},  // -x+y

        {0, 0, +1}, // +z
        {0, 0, -1}, // -z
        {0, +1, 0}, // +y
        {+1, +1, 0}, // +x+y
        {+1, 0, 0}, // -x
        {+1, -1, 0}  // -x+y
    };
    my_box = &(system->my_box);
    bndry_cut = system->bndry_cuts.ghost_cutoff;

    /* identify my neighbors */
    system->num_nt_nbrs = MAX_NT_NBRS;
    for ( i = 0; i < system->num_nt_nbrs; ++i )
    {
        ivec_Sum( nbr_coords, system->my_coords, r[i] ); /* actual nbr coords */
        nbr_pr = &(system->my_nt_nbrs[i]);
        MPI_Cart_rank( mpi_data->comm_mesh3D, nbr_coords, &(nbr_pr->rank) );
        
        /* set the rank of the neighbor processor in the receiving direction */
        ivec_Sum( nbr_coords, system->my_coords, r[i + 6] ); /* actual nbr coords */
        MPI_Cart_rank( mpi_data->comm_mesh3D, nbr_coords, &(nbr_pr->receive_rank) );

        for ( d = 0; d < 3; ++d )
        {
            /* determine the boundary area with this nbr */
            if ( r[i][d] < 0 )
            {
                nbr_pr->bndry_min[d] = my_box->min[d];
                nbr_pr->bndry_max[d] = my_box->min[d] + bndry_cut;
            }
            else if ( r[i][d] > 0 )
            {
                nbr_pr->bndry_min[d] = my_box->max[d] - bndry_cut;
                nbr_pr->bndry_max[d] = my_box->max[d];
            }
            else
            {
                nbr_pr->bndry_min[d] = my_box->min[d];
                nbr_pr->bndry_max[d] = my_box->max[d];
            }

            /* determine if it is a periodic neighbor */
            if ( nbr_coords[d] < 0 )
                nbr_pr->prdc[d] = -1;
            else if ( nbr_coords[d] >= control->procs_by_dim[d] )
                nbr_pr->prdc[d] = +1;
            else
                nbr_pr->prdc[d] = 0;
        }

    }
}


void Sort_Neutral_Territory( reax_system *system, int dir, mpi_out_data *out_bufs )
{
    int i, d, p, out_cnt;
    reax_atom *atoms;
    simulation_box *my_box;
    boundary_atom *out_buf;
    neighbor_proc *nbr_pr;

    atoms = system->my_atoms;
    my_box = &( system->my_box );

    /* place each atom into the appropriate outgoing list */
    nbr_pr = &( system->my_nt_nbrs[dir] );
    for ( i = 0; i < system->n; ++i )
    {
        if ( nbr_pr->bndry_min[0] <= atoms[i].x[0] &&
            atoms[i].x[0] < nbr_pr->bndry_max[0] &&
            nbr_pr->bndry_min[1] <= atoms[i].x[1] &&
            atoms[i].x[1] < nbr_pr->bndry_max[1] &&
            nbr_pr->bndry_min[2] <= atoms[i].x[2] &&
            atoms[i].x[2] < nbr_pr->bndry_max[2] )
        {
            out_bufs[dir].index[out_cnt] = i;
            out_bufs[dir].cnt++;
        }
    }
}


void Init_Neutral_Territory( reax_system* system, mpi_datatypes *mpi_data )
{
    int d, start, end, cnt;
    mpi_out_data *out_bufs;
    void *in;
    MPI_Comm comm;
    MPI_Request req;
    MPI_Status stat;
    neighbor_proc *nbr;

    Reset_Out_Buffers( mpi_data->out_nt_buffers, system->num_nt_nbrs );
    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_nt_buffers;
    cnt = 0;
    end = system->n;

    for ( d = 0; d < 6; ++d )
    {
        Sort_Neutral_Territory( system, d, out_bufs );
        start = end;
        
        MPI_Irecv( &cnt, 1, MPI_INT, nbr->receive_rank, d, comm, &req );
        MPI_Send( &(out_bufs[d].cnt), 1, MPI_INT, nbr->rank, d, comm );
        MPI_Wait( &req, &stat );
        
        if( mpi_data->in_nt_buffer[d] == NULL )
        {
            mpi_data->in_nt_buffer[d] = (void *) smalloc( SAFER_ZONE * out_bufs[d].cnt * sizeof(real), "in", comm );
        }

        nbr = &(system->my_nt_nbrs[d]);
        nbr->atoms_str = end;
        nbr->atoms_cnt = cnt;
        end += cnt;
    }
}


void Setup_Comm( reax_system* system, control_params* control,
                 mpi_datatypes *mpi_data )
{
    int i, d;
    real bndry_cut;
    neighbor_proc *nbr_pr;
    simulation_box *my_box;
    ivec nbr_coords;
    ivec r[6] = {{ -1, 0, 0}, { +1, 0, 0}, // -x, +x
        {0, -1, 0}, {0, +1, 0}, // -y, +y
        {0, 0, -1}, {0, 0, +1}
    };// -z, +z
    my_box = &(system->my_box);
    bndry_cut = system->bndry_cuts.ghost_cutoff;

    /* identify my neighbors */
    system->num_nbrs = MAX_NBRS;
    for ( i = 0; i < system->num_nbrs; ++i )
    {
        ivec_Sum( nbr_coords, system->my_coords, r[i] ); /* actual nbr coords */
        nbr_pr = &(system->my_nbrs[i]);
        ivec_Copy( nbr_pr->rltv, r[i] );
        MPI_Cart_rank( mpi_data->comm_mesh3D, nbr_coords, &(nbr_pr->rank) );

        for ( d = 0; d < 3; ++d )
        {
            /* determine the boundary area with this nbr */
            if ( r[i][d] < 0 )
            {
                nbr_pr->bndry_min[d] = my_box->min[d];
                nbr_pr->bndry_max[d] = my_box->min[d] + bndry_cut;
            }
            else if ( r[i][d] > 0 )
            {
                nbr_pr->bndry_min[d] = my_box->max[d] - bndry_cut;
                nbr_pr->bndry_max[d] = my_box->max[d];
            }
            else
            {
                nbr_pr->bndry_min[d] = nbr_pr->bndry_max[d] = NEG_INF;
            }

            /* determine if it is a periodic neighbor */
            if ( nbr_coords[d] < 0 )
                nbr_pr->prdc[d] = -1;
            else if ( nbr_coords[d] >= control->procs_by_dim[d] )
                nbr_pr->prdc[d] = +1;
            else
                nbr_pr->prdc[d] = 0;
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d-nbr%d: r[%2d %2d %2d] -> c[%2d %2d %2d] -> rank=%d\n",
                 system->my_rank, i, r[i][0], r[i][1], r[i][2],
                 nbr_coords[0], nbr_coords[1], nbr_coords[2], nbr_pr->rank );
#endif
    }
}


void Update_Comm( reax_system* system )
{
    int i, d;
    real bndry_cut;
    neighbor_proc *nbr_pr;
    simulation_box *my_box;
    ivec r[6] = {{ -1, 0, 0}, { +1, 0, 0}, // -x, +x
        {0, -1, 0}, {0, +1, 0}, // -y, +y
        {0, 0, -1}, {0, 0, +1}
    };// -z, +z
    my_box = &(system->my_box);
    bndry_cut = system->bndry_cuts.ghost_cutoff;

    /* identify my neighbors */
    for ( i = 0; i < system->num_nbrs; ++i )
    {
        nbr_pr = &(system->my_nbrs[i]);

        for ( d = 0; d < 3; ++d )
            /* determine the boundary area with this nbr */
            if ( r[i][d] < 0 )
            {
                nbr_pr->bndry_min[d] = my_box->min[d];
                nbr_pr->bndry_max[d] = my_box->min[d] + bndry_cut;
            }
            else if ( r[i][d] > 0 )
            {
                nbr_pr->bndry_min[d] = my_box->max[d] - bndry_cut;
                nbr_pr->bndry_max[d] = my_box->max[d];
            }
            else
            {
                nbr_pr->bndry_min[d] = nbr_pr->bndry_max[d] = NEG_INF;
            }
    }
}


/********************* ATOM TRANSFER ***********************/

/***************** PACK & UNPACK ATOMS *********************/
void Pack_MPI_Atom( mpi_atom *matm, reax_atom *ratm, int i )
{
    matm->orig_id  = ratm->orig_id;
    matm->imprt_id = i;
    matm->type     = ratm->type;
    matm->num_bonds = ratm->num_bonds;
    matm->num_hbonds = ratm->num_hbonds;
    strcpy( matm->name, ratm->name );
    rvec_Copy( matm->x, ratm->x );
    rvec_Copy( matm->v, ratm->v );
    rvec_Copy( matm->f_old, ratm->f_old );
    memcpy( matm->s, ratm->s, sizeof(rvec4) ); //rvec_Copy( matm->s, ratm->s );
    memcpy( matm->t, ratm->t, sizeof(rvec4) ); //rvec_Copy( matm->t, ratm->t );
}


void Unpack_MPI_Atom( reax_atom *ratm, mpi_atom *matm )
{
    ratm->orig_id  = matm->orig_id;
    ratm->imprt_id = matm->imprt_id;
    ratm->type     = matm->type;
    ratm->num_bonds = matm->num_bonds;
    ratm->num_hbonds = matm->num_hbonds;
    strcpy( ratm->name, matm->name );
    rvec_Copy( ratm->x, matm->x );
    rvec_Copy( ratm->v, matm->v );
    rvec_Copy( ratm->f_old, matm->f_old );
    memcpy( ratm->s, matm->s, sizeof(rvec4) ); //rvec_Copy( ratm->s, matm->s );
    memcpy( ratm->t, matm->t, sizeof(rvec4) ); //rvec_Copy( ratm->t, matm->t );
}


/*********************** SORTER **************************/
void Sort_Transfer_Atoms( reax_system *system, int start, int end,
                          int dim, mpi_out_data *out_bufs )
{
    int i, d, out_cnt;
    reax_atom *atoms;
    simulation_box *my_box;
    mpi_atom *out_buf;

#if defined(DEBUG)
    fprintf( stderr, "p%d sort_transfers: start=%d end=%d dim=%d starting...\n",
             system->my_rank, start, end, dim );
#endif
    atoms = system->my_atoms;
    my_box = &( system->my_box );

    /* place each atom into the appropriate outgoing list */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
            if ( atoms[i].x[d] < my_box->min[d] )
            {
                out_cnt = out_bufs[2 * d].cnt++;
                out_buf = out_bufs[2 * d].out_atoms;
                Pack_MPI_Atom( out_buf + out_cnt, atoms + i, i );
                atoms[i].orig_id = -1;
                break;
            }
            else if ( atoms[i].x[d] >= my_box->max[d] )
            {
                out_cnt = out_bufs[2 * d + 1].cnt++;
                out_buf = out_bufs[2 * d + 1].out_atoms;
                Pack_MPI_Atom( out_buf + out_cnt, atoms + i, i );
                atoms[i].orig_id = -1;
                break;
            }
    }

#if defined(DEBUG_FOCUS)
    for ( d = 2 * dim; d < 2 * dim + 2; ++d )
        if ( out_bufs[d].cnt )
        {
            fprintf( stderr, "p%d to p%d(nbr%d) # of transfers = %d\n",
                     system->my_rank, system->my_nbrs[d].rank, d, out_bufs[d].cnt );
            out_buf = out_bufs[d].out_atoms;
            for ( i = 0; i < out_bufs[d].cnt; ++i )
                fprintf( stderr, "p%d to p%d: transfer atom%d [%.3f %.3f %.3f]\n",
                         system->my_rank, system->my_nbrs[d].rank, out_buf[i].imprt_id,
                         out_buf[i].x[0],  out_buf[i].x[1],  out_buf[i].x[2] );
        }
    //fprintf( stderr, "p%d sort_transfers: start=%d end=%d dim=%d done!\n",
    //   system->my_rank, start, end, dim );
#endif
}


/*********************** UNPACKER **************************/
void Unpack_Transfer_Message( reax_system *system, int end, void *dummy,
                              int cnt, neighbor_proc *nbr, int dim )
{
    int i;
    real dx;
    reax_atom *dest;
    mpi_atom* src = (mpi_atom*) dummy;

    dest = system->my_atoms + end;
    for ( i = 0; i < cnt; ++i )
        Unpack_MPI_Atom( dest + i, src + i );

    /* adjust coordinates of recved atoms if nbr is a periodic one */
    if ( nbr->prdc[dim] )
    {
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];
        for ( i = 0; i < cnt; ++i )
            dest[i].x[dim] += dx;
    }
}


/************** EXCHANGE BOUNDARY ATOMS *****************/

/************ PACK & UNPACK BOUNDARY ATOMS **************/
void Pack_Boundary_Atom( boundary_atom *matm, reax_atom *ratm, int i )
{
    matm->orig_id  = ratm->orig_id;
    matm->imprt_id = i;
    matm->type = ratm->type;
    matm->num_bonds = ratm->num_bonds;
    matm->num_hbonds = ratm->num_hbonds;
    rvec_Copy( matm->x, ratm->x );
}


void Unpack_Boundary_Atom( reax_atom *ratm, boundary_atom *matm )
{
    ratm->orig_id = matm->orig_id;
    ratm->imprt_id = matm->imprt_id;
    ratm->type = matm->type;
    ratm->num_bonds = matm->num_bonds;
    ratm->num_hbonds = matm->num_hbonds;
    //ratm->renumber = offset + matm->imprt_id;
    rvec_Copy( ratm->x, matm->x );
}


/*********************** SORTER **************************/
void Sort_Boundary_Atoms( reax_system *system, int start, int end,
                          int dim, mpi_out_data *out_bufs )
{
    int i, d, p, out_cnt;
    reax_atom *atoms;
    simulation_box *my_box;
    boundary_atom *out_buf;
    neighbor_proc *nbr_pr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d sort_exchange: start=%d end=%d dim=%d starting...\n",
             system->my_rank, start, end, dim );
#endif

    atoms = system->my_atoms;
    my_box = &( system->my_box );

    /* place each atom into the appropriate outgoing list */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
        {
            for ( p = 2 * d; p < 2 * d + 2; ++p )
            {
                nbr_pr = &( system->my_nbrs[p] );
                if ( nbr_pr->bndry_min[d] <= atoms[i].x[d] &&
                        atoms[i].x[d] < nbr_pr->bndry_max[d] )
                {
                    out_cnt = out_bufs[p].cnt++;
                    out_bufs[p].index[out_cnt] = i;
                    out_buf = out_bufs[p].out_atoms;
                    Pack_Boundary_Atom( out_buf + out_cnt, atoms + i, i );
                }
            }
        }
    }

#if defined(DEBUG_FOCUS)
    for ( i = 2 * dim; i < 2 * dim + 2; ++i )
    {
        fprintf( stderr, "p%d to p%d(nbr%d) # of exchanges to send = %d\n",
                 system->my_rank, system->my_nbrs[i].rank, i,
                 out_bufs[i].cnt );
    }
    fprintf( stderr, "p%d sort_exchange: start=%d end=%d dim=%d done!\n",
             system->my_rank, start, end, dim );
#endif
}


void Estimate_Boundary_Atoms( reax_system *system, int start, int end,
                              int d, mpi_out_data *out_bufs )
{
    int i, p, out_cnt;
    reax_atom *atoms;
    simulation_box *my_box;
    boundary_atom *out_buf;
    neighbor_proc *nbr1, *nbr2, *nbr_pr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d estimate_exchange: start=%d end=%d dim=%d starting.\n",
             system->my_rank, start, end, d );
#endif
    atoms = system->my_atoms;
    my_box = &( system->my_box );
    nbr1 = &(system->my_nbrs[2 * d]);
    nbr2 = &(system->my_nbrs[2 * d + 1]);
    nbr1->est_send = 0;
    nbr2->est_send = 0;

    /* count the number of atoms in each processor's outgoing list */
    for ( i = 0; i < end; ++i )
    {
        if ( nbr1->bndry_min[d] <= atoms[i].x[d] && atoms[i].x[d] < nbr1->bndry_max[d] )
            nbr1->est_send++;
        if ( nbr2->bndry_min[d] <= atoms[i].x[d] && atoms[i].x[d] < nbr2->bndry_max[d] )
            nbr2->est_send++;
    }

    /* estimate the space based on the count above */
    nbr1->est_send = MAX( MIN_SEND, nbr1->est_send * SAFER_ZONE );
    nbr2->est_send = MAX( MIN_SEND, nbr2->est_send * SAFER_ZONE );
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d estimate_exchange: end=%d dim=%d est1=%d est2=%d\n",
             system->my_rank, end, d, nbr1->est_send, nbr2->est_send );
#endif

    /* allocate the estimated space */
    for ( p = 2 * d; p < 2 * d + 2; ++p )
    {
        nbr_pr = &( system->my_nbrs[p] );
        out_bufs[p].index = (int*) calloc( nbr_pr->est_send, sizeof(int) );
        out_bufs[p].out_atoms = (void*)
                                calloc( nbr_pr->est_send, sizeof(boundary_atom) );
    }

    /* sort the atoms to their outgoing buffers */
    for ( i = 0; i < end; ++i )
    {
        for ( p = 2 * d; p < 2 * d + 2; ++p )
        {
            nbr_pr = &( system->my_nbrs[p] );
            if ( nbr_pr->bndry_min[d] <= atoms[i].x[d] &&
                    atoms[i].x[d] < nbr_pr->bndry_max[d] )
            {
                out_cnt = out_bufs[p].cnt++;
                out_bufs[p].index[out_cnt] = i;
                out_buf = out_bufs[p].out_atoms;
                Pack_Boundary_Atom( out_buf + out_cnt, atoms + i, i );
            }
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d estimate_exchange: end=%d dim=%d done!\n",
             system->my_rank, end, d );
#endif
}


void Estimate_Init_Storage( int me, neighbor_proc *nbr1, neighbor_proc *nbr2,
                            int d, int *max, int *nrecv,
                            void **in1, void **in2, MPI_Comm comm )
{
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    int new_max;

    /* first exchange the estimates, then allocate buffers */
    MPI_Irecv( &nbr1->est_recv, 1, MPI_INT, nbr1->rank, 2 * d + 1, comm, &req1 );
    MPI_Irecv( &nbr2->est_recv, 1, MPI_INT, nbr2->rank, 2 * d, comm, &req2 );
    MPI_Send( &nbr1->est_send, 1, MPI_INT, nbr1->rank, 2 * d, comm );
    MPI_Send( &nbr2->est_send, 1, MPI_INT, nbr2->rank, 2 * d + 1, comm );
    MPI_Wait( &req1, &stat1 );
    MPI_Wait( &req2, &stat2 );
    nrecv[2 * d] = nbr1->est_recv;
    nrecv[2 * d + 1] = nbr2->est_recv;
    new_max = MAX( nbr1->est_recv, nbr2->est_recv );
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d-p%d(nbr%d) est_send=%d est_recv=%d\n",
             me, nbr1->rank, 2 * d, nbr1->est_send, nbr1->est_recv );
    fprintf( stderr, "p%d-p%d(nbr%d) est_send=%d est_recv=%d\n",
             me, nbr2->rank, 2 * d + 1, nbr2->est_send, nbr2->est_recv );
    fprintf( stderr, "max=%d  new_max=%d\n", *max, new_max );
#endif

    if ( new_max > *max )
    {
        *max = new_max;
        if (*in1) sfree( *in1, "in1" );
        if (*in2) sfree( *in2, "in2" );
        *in1 = (void *) smalloc( new_max * sizeof(boundary_atom), "in1", comm );
        *in2 = (void *) smalloc( new_max * sizeof(boundary_atom), "in2", comm );
    }
}


/*********************** UNPACKER **************************/
void Unpack_Exchange_Message( reax_system *system, int end, void *dummy,
                              int cnt, neighbor_proc *nbr, int dim )
{
    int i;
    real dx;
    reax_atom *dest;
    boundary_atom* src = (boundary_atom*) dummy;

    dest = system->my_atoms + end;
    for ( i = 0; i < cnt; ++i )
        Unpack_Boundary_Atom( dest + i, src + i );

    /* record the atoms recv'd from this nbr */
    nbr->atoms_str = end;
    nbr->atoms_cnt = cnt;
    /* update est_recv */
    nbr->est_recv = MAX( cnt * SAFER_ZONE, MIN_SEND );

    /* update max_recv to make sure that we reallocate at the right time */
    if ( cnt > system->max_recved )
        system->max_recved = cnt;

    /* adjust coordinates of recved atoms if nbr is a periodic one */
    if ( nbr->prdc[dim] )
    {
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];
        for ( i = 0; i < cnt; ++i )
            dest[i].x[dim] += dx;
    }
}


void Unpack_Estimate_Message( reax_system *system, int end, void *dummy,
                              int cnt, neighbor_proc *nbr, int dim )
{
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d-p%d unpack_estimate: end=%d cnt=%d - unpacking\n",
             system->my_rank, nbr->rank, end, cnt );
#endif

    system->my_atoms = (reax_atom*)
                       realloc( system->my_atoms, (end + cnt) * sizeof(reax_atom) );

    Unpack_Exchange_Message( system, end, dummy, cnt, nbr, dim );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d-p%d unpack_estimate: end=%d cnt=%d - done\n",
             system->my_rank, nbr->rank, end, cnt );
#endif
}


/**************** UPDATE ATOM POSITIONS *******************/

/**************** PACK POSITION UPDATES *******************/
void Sort_Position_Updates( reax_system *system, int start, int end,
                            int dim, mpi_out_data *out_bufs )
{
    int i, p;
    reax_atom *atoms;
    rvec *out;

    atoms = system->my_atoms;

    for ( p = 2 * dim; p < 2 * dim + 2; ++p )
    {
        out = (rvec*) out_bufs[p].out_atoms;
        for ( i = 0; i < out_bufs[p].cnt; ++i )
            memcpy( out[i], atoms[ out_bufs[p].index[i] ].x, sizeof(rvec) );
    }
}

/*************** UNPACK POSITION UPDATES ******************/
void Unpack_Position_Updates( reax_system *system, int end, void *dummy,
                              int cnt, neighbor_proc *nbr, int dim )
{
    int i, start;
    reax_atom *atoms;
    real dx;
    rvec* src = (rvec*) dummy;

    atoms = system->my_atoms;
    start = nbr->atoms_str;

    for ( i = 0; i < cnt; ++i )
        memcpy( atoms[start + i].x, src[i], sizeof(rvec) );

    /* adjust coordinates of recved atoms if nbr is a periodic one */
    if ( nbr->prdc[dim] )
    {
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];
        for ( i = 0; i < cnt; ++i )
            atoms[start + i].x[dim] += dx;
    }
}


int SendRecv( reax_system* system, mpi_datatypes *mpi_data, MPI_Datatype type,
              int* nrecv, message_sorter sort_func, unpacker unpack, int clr )
{
    int d, cnt, start, end, max, est_flag;
    mpi_out_data *out_bufs;
    void *in1, *in2;
    MPI_Comm comm;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    neighbor_proc *nbr1, *nbr2;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d sendrecv: entered\n", system->my_rank );
#endif

    if ( clr )
    {
        Reset_Out_Buffers( mpi_data->out_buffers, system->num_nbrs );
    }
    comm = mpi_data->comm_mesh3D;
    in1 = mpi_data->in1_buffer;
    in2 = mpi_data->in2_buffer;
    out_bufs = mpi_data->out_buffers;
    start = 0;
    end = system->n;
    cnt = 0;
    max = 0;
    est_flag = (mpi_data->in1_buffer == NULL) || (mpi_data->in2_buffer == NULL);

    for ( d = 0; d < 3; ++d )
    {
        sort_func( system, start, end, d, out_bufs );
        start = end;
        nbr1 = &(system->my_nbrs[2 * d]);
        nbr2 = &(system->my_nbrs[2 * d + 1]);

        /* for estimates in1_buffer & in2_buffer will be NULL */
        if ( est_flag )
        {
            Estimate_Init_Storage( system->my_rank, nbr1, nbr2, d,
                    &max, nrecv, &in1, &in2, comm );
        }

        /* initiate recvs */
        MPI_Irecv( in1, nrecv[2 * d], type, nbr1->rank, 2 * d + 1, comm, &req1 );
        MPI_Irecv( in2, nrecv[2 * d + 1], type, nbr2->rank, 2 * d, comm, &req2 );

        /* send both messages in dimension d */
        MPI_Send( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt, type,
                nbr1->rank, 2 * d, comm );
        MPI_Send( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt, type,
                nbr2->rank, 2 * d + 1, comm );

        /* recv and unpack atoms from nbr1 in dimension d */
        MPI_Wait( &req1, &stat1 );
        MPI_Get_count( &stat1, type, &cnt );
        unpack( system, end, in1, cnt, nbr1, d );
        end += cnt;

        /* recv and unpack atoms from nbr2 in dimension d */
        MPI_Wait( &req2, &stat2 );
        MPI_Get_count( &stat2, type, &cnt );
        unpack( system, end, in2, cnt, nbr2, d );
        end += cnt;
    }

    if ( est_flag )
    {
        system->est_recv = max;
        system->est_trans = (max * sizeof(boundary_atom)) / sizeof(mpi_atom);
        mpi_data->in1_buffer = in1;
        mpi_data->in2_buffer = in2;
    }

    return end;
}


void Comm_Atoms( reax_system *system, control_params *control,
                 simulation_data *data, storage *workspace, reax_list **lists,
                 mpi_datatypes *mpi_data, int renbr )
{
    int i;
    int nrecv[MAX_NBRS];
#if defined(LOG_PERFORMANCE)
    real t_start = 0, t_elapsed = 0;

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
#endif

    if ( renbr )
    {
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nrecv[i] = system->est_trans;
        }
        system->n = SendRecv( system, mpi_data, mpi_data->mpi_atom_type, nrecv,
                Sort_Transfer_Atoms, Unpack_Transfer_Message, 1 );
        Bin_My_Atoms( system, &(workspace->realloc) );
        Reorder_My_Atoms( system, workspace );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d updated local atoms, n=%d\n",
                 system->my_rank, system->n );
        MPI_Barrier( mpi_data->world );
#endif

        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nrecv[i] = system->my_nbrs[i].est_recv;
        }
        system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
                Sort_Boundary_Atoms, Unpack_Exchange_Message, 1 );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: exchanged boundary atoms, N=%d\n",
                system->my_rank, system->N );

        for ( i = 0; i < MAX_NBRS; ++i )
        {
            fprintf( stderr, "p%d: nbr%d(p%d) str=%d cnt=%d end=%d\n",
                     system->my_rank, i, system->my_nbrs[i].rank,
                     system->my_nbrs[i].atoms_str,  system->my_nbrs[i].atoms_cnt,
                     system->my_nbrs[i].atoms_str + system->my_nbrs[i].atoms_cnt );
        }

        MPI_Barrier( mpi_data->world );
#endif

        Bin_Boundary_Atoms( system );

#if defined(NEUTRAL_TERRITORY)
        Init_Neutral_Territory( system, mpi_data );
#endif
    }
    else
    {
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nrecv[i] = system->my_nbrs[i].atoms_cnt;
        }

        SendRecv( system, mpi_data, mpi_data->mpi_rvec, nrecv,
                Sort_Position_Updates, Unpack_Position_Updates, 0 );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: updated positions\n", system->my_rank );
        MPI_Barrier( mpi_data->world );
#endif
    }

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_elapsed = Get_Timing_Info( t_start );
        data->timing.comm += t_elapsed;
    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf(stderr, "p%d @ renbr=%d: comm_atoms done\n", system->my_rank, renbr);
    fprintf( stderr, "p%d: system->n = %d, system->N = %d\n",
             system->my_rank, system->n, system->N );
    //Print_My_Ext_Atoms( system );
    //Print_All_GCells( system );
    //MPI_Barrier( MPI_COMM_WORLD );
#endif
}
