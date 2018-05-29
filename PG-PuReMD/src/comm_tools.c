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

#include "comm_tools.h"

#include "grid.h"
#include "reset_tools.h"
#include "tool_box.h"
#include "vector.h"


void Check_MPI_Error( int code, const char * const msg )
{
    char err_msg[MPI_MAX_ERROR_STRING];
    int len;

    if ( code != MPI_SUCCESS )
    {
        MPI_Error_string( code, err_msg, &len );

        fprintf( stderr, "[ERROR] MPI error code %d, from %s\n",
                code, msg );
        fprintf( stderr, "    [INFO] MPI error message: %s\n", err_msg );
        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


void Setup_Comm( reax_system * const system, control_params * const control,
        mpi_datatypes * const mpi_data )
{
    int i, d;
    const real bndry_cut = system->bndry_cuts.ghost_cutoff;
    neighbor_proc *nbr_pr;
    simulation_box * const my_box = &system->my_box;
    ivec nbr_coords;
    ivec r[6] = {
        { -1, 0, 0}, { 1, 0, 0}, // -x, +x
        {0, -1, 0}, {0, 1, 0}, // -y, +y
        {0, 0, -1}, {0, 0, 1}, // -z, +z
    };

    /* identify my neighbors */
    system->num_nbrs = MAX_NBRS;
    for ( i = 0; i < system->num_nbrs; ++i )
    {
        ivec_Sum( nbr_coords, system->my_coords, r[i] ); /* actual nbr coords */
        nbr_pr = &system->my_nbrs[i];
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
            {
                nbr_pr->prdc[d] = -1;
            }
            else if ( nbr_coords[d] >= control->procs_by_dim[d] )
            {
                nbr_pr->prdc[d] = +1;
            }
            else
            {
                nbr_pr->prdc[d] = 0;
            }
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d-nbr%d: r[%2d %2d %2d] -> c[%2d %2d %2d] -> rank=%d\n",
                 system->my_rank, i, r[i][0], r[i][1], r[i][2],
                 nbr_coords[0], nbr_coords[1], nbr_coords[2], nbr_pr->rank );
#endif
    }
}


void Update_Comm( reax_system * const system )
{
    int i, d;
    const real bndry_cut = system->bndry_cuts.ghost_cutoff;
    neighbor_proc *nbr_pr;
    simulation_box * const my_box = &system->my_box;
    ivec r[6] = {
        { -1, 0, 0}, { 1, 0, 0}, // -x, +x
        {0, -1, 0}, {0, 1, 0}, // -y, +y
        {0, 0, -1}, {0, 0, 1}, // -z, +z
    };

    /* identify my neighbors */
    for ( i = 0; i < system->num_nbrs; ++i )
    {
        nbr_pr = &system->my_nbrs[i];

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
        }
    }
}


/********************* ATOM TRANSFER ***********************/

/***************** PACK & UNPACK ATOMS *********************/
static void Pack_MPI_Atom( mpi_atom * const matm, const reax_atom * const ratm, int i )
{
    matm->orig_id = ratm->orig_id;
    matm->imprt_id = i;
    matm->type = ratm->type;
    matm->num_bonds = ratm->num_bonds;
    matm->num_hbonds = ratm->num_hbonds;
    strncpy( matm->name, ratm->name, MAX_ATOM_NAME_LEN );
    rvec_Copy( matm->x, ratm->x );
    rvec_Copy( matm->v, ratm->v );
    rvec_Copy( matm->f_old, ratm->f_old );
    memcpy( matm->s, ratm->s, sizeof(rvec4) ); //rvec_Copy( matm->s, ratm->s );
    memcpy( matm->t, ratm->t, sizeof(rvec4) ); //rvec_Copy( matm->t, ratm->t );
}


static void Unpack_MPI_Atom( reax_atom * const ratm, const mpi_atom * const matm )
{
    ratm->orig_id = matm->orig_id;
    ratm->imprt_id = matm->imprt_id;
    ratm->type = matm->type;
    ratm->num_bonds = matm->num_bonds;
    ratm->num_hbonds = matm->num_hbonds;
    strncpy( ratm->name, matm->name, MAX_ATOM_NAME_LEN );
    rvec_Copy( ratm->x, matm->x );
    rvec_Copy( ratm->v, matm->v );
    rvec_Copy( ratm->f_old, matm->f_old );
    memcpy( ratm->s, matm->s, sizeof(rvec4) ); //rvec_Copy( ratm->s, matm->s );
    memcpy( ratm->t, matm->t, sizeof(rvec4) ); //rvec_Copy( ratm->t, matm->t );
}


/*********************** SORTER **************************/
static void Sort_Transfer_Atoms( reax_system * const system, int start, int end,
        int dim, mpi_out_data * const out_bufs )
{
    int i, d, out_cnt;
    reax_atom * const atoms = system->my_atoms;
    simulation_box * const my_box = &system->my_box;
    mpi_atom *out_buf;

#if defined(DEBUG)
    fprintf( stderr, "p%d sort_transfers: start=%d end=%d dim=%d starting...\n",
            system->my_rank, start, end, dim );
#endif

    /* place each atom into the appropriate outgoing list */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
        {
            if ( atoms[i].x[d] < my_box->min[d] )
            {
                out_cnt = out_bufs[2 * d].cnt++;
                out_buf = (mpi_atom *) out_bufs[2 * d].out_atoms;
                Pack_MPI_Atom( out_buf + out_cnt, atoms + i, i );
                atoms[i].orig_id = -1;
                break;
            }
            else if ( atoms[i].x[d] >= my_box->max[d] )
            {
                out_cnt = out_bufs[2 * d + 1].cnt++;
                out_buf = (mpi_atom *) out_bufs[2 * d + 1].out_atoms;
                Pack_MPI_Atom( out_buf + out_cnt, atoms + i, i );
                atoms[i].orig_id = -1;
                break;
            }
        }
    }

#if defined(DEBUG_FOCUS)
    for ( d = 2 * dim; d < 2 * dim + 2; ++d )
    {
        if ( out_bufs[d].cnt )
        {
            fprintf( stderr, "p%d to p%d(nbr%d) # of transfers = %d\n",
                    system->my_rank, system->my_nbrs[d].rank, d, out_bufs[d].cnt );

            out_buf = out_bufs[d].out_atoms;

            for ( i = 0; i < out_bufs[d].cnt; ++i )
            {
                fprintf( stderr, "p%d to p%d: transfer atom%d [%.3f %.3f %.3f]\n",
                        system->my_rank, system->my_nbrs[d].rank, out_buf[i].imprt_id,
                        out_buf[i].x[0],  out_buf[i].x[1],  out_buf[i].x[2] );
            }
        }
    //fprintf( stderr, "p%d sort_transfers: start=%d end=%d dim=%d done!\n",
    //   system->my_rank, start, end, dim );
    }
#endif
}


/*********************** UNPACKER **************************/
static void Unpack_Transfer_Message( reax_system * const system, int end, void * const dummy,
        int cnt, neighbor_proc * const nbr, int dim )
{
    int i;
    real dx;
    reax_atom * const dest = system->my_atoms + end;
    mpi_atom * const src = (mpi_atom*) dummy;

    for ( i = 0; i < cnt; ++i )
    {
        Unpack_MPI_Atom( dest + i, src + i );
    }

    /* adjust coordinates of recved atoms if nbr is a periodic one */
    if ( nbr->prdc[dim] )
    {
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];

        for ( i = 0; i < cnt; ++i )
        {
            dest[i].x[dim] += dx;
        }
    }
}


/************** EXCHANGE BOUNDARY ATOMS *****************/

/************ PACK & UNPACK BOUNDARY ATOMS **************/
static void Pack_Boundary_Atom( boundary_atom * const matm, const reax_atom * const ratm, int i )
{
    matm->orig_id = ratm->orig_id;
    matm->imprt_id = i;
    matm->type = ratm->type;
    matm->num_bonds = ratm->num_bonds;
    matm->num_hbonds = ratm->num_hbonds;
    rvec_Copy( matm->x, ratm->x );
}


static void Unpack_Boundary_Atom( reax_atom * const ratm, const boundary_atom * const matm )
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
static void Sort_Boundary_Atoms( reax_system * const system, int start, int end,
        int dim, mpi_out_data * const out_bufs )
{
    int i, d, p, out_cnt;
    const reax_atom * const atoms = system->my_atoms;
    boundary_atom *out_buf;
    neighbor_proc *nbr_pr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d sort_exchange: start=%d end=%d dim=%d starting...\n",
            system->my_rank, start, end, dim );
#endif

    /* place each atom into the appropriate outgoing list */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
        {
            for ( p = 2 * d; p < 2 * d + 2; ++p )
            {
                nbr_pr = &system->my_nbrs[p];

                if ( nbr_pr->bndry_min[d] <= atoms[i].x[d] &&
                        atoms[i].x[d] < nbr_pr->bndry_max[d] )
                {
                    out_cnt = out_bufs[p].cnt++;
                    out_bufs[p].index[out_cnt] = i;
                    out_buf = (boundary_atom *) out_bufs[p].out_atoms;
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


void Estimate_Boundary_Atoms( reax_system * const system, int start, int end,
        int d, mpi_out_data * const out_bufs )
{
    int i, p, out_cnt;
    const reax_atom * const atoms = system->my_atoms;
    boundary_atom *out_buf;
    neighbor_proc * const nbr1 = &system->my_nbrs[2 * d];
    neighbor_proc * const nbr2 = &system->my_nbrs[2 * d + 1];
    neighbor_proc *nbr_pr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d estimate_exchange: start=%d end=%d dim=%d starting.\n",
            system->my_rank, start, end, d );
#endif

    nbr1->est_send = 0;
    nbr2->est_send = 0;

    /* count the number of atoms in each processor's outgoing list */
    for ( i = 0; i < end; ++i )
    {
        if ( nbr1->bndry_min[d] <= atoms[i].x[d] && atoms[i].x[d] < nbr1->bndry_max[d] )
        {
            nbr1->est_send++;
        }
        if ( nbr2->bndry_min[d] <= atoms[i].x[d] && atoms[i].x[d] < nbr2->bndry_max[d] )
        {
            nbr2->est_send++;
        }
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
        nbr_pr = &system->my_nbrs[p];
        out_bufs[p].index = scalloc( nbr_pr->est_send, sizeof(int),
                "Estimate_Boundary_Atoms::mpibuf:index" );
        out_bufs[p].out_atoms = scalloc( nbr_pr->est_send, sizeof(boundary_atom),
                "Estimate_Boundary_Atoms::mpibuf:out_atoms" );
    }

    /* sort the atoms to their outgoing buffers */
    for ( i = 0; i < end; ++i )
    {
        /* check if atom is outbound to another processor
         * in either direction of the dimension under consideration */
        for ( p = 2 * d; p < 2 * d + 2; ++p )
        {
            nbr_pr = &system->my_nbrs[p];
            if ( nbr_pr->bndry_min[d] <= atoms[i].x[d] &&
                    atoms[i].x[d] < nbr_pr->bndry_max[d] )
            {
                out_cnt = out_bufs[p].cnt;
                out_bufs[p].index[out_cnt] = i;
                out_buf = (boundary_atom *) out_bufs[p].out_atoms;
                Pack_Boundary_Atom( out_buf + out_cnt, atoms + i, i );
                ++out_bufs[p].cnt;
            }
        }
    }

#if defined(DEBUG_FOCUS)
    for ( p = 2 * d; p < 2 * d + 2; ++p )
    {
        for ( i = 0; i < out_bufs[p].cnt; ++i )
        {
            fprintf( stderr, "p%d: out_bufs[%d].index[%d] = %d\n",
                    system->my_rank, p, i, out_bufs[p].index[i] );
            fprintf( stderr, "  p%d: atom %6d, x[0] = %10.4f, x[1] = %10.4f, x[2] = %10.4f\n",
                    system->my_rank,
                    ((boundary_atom *)(out_bufs[p].out_atoms))[i].orig_id,
                    ((boundary_atom *)(out_bufs[p].out_atoms))[i].x[0],
                    ((boundary_atom *)(out_bufs[p].out_atoms))[i].x[1],
                    ((boundary_atom *)(out_bufs[p].out_atoms))[i].x[2] );
        }
    }

fprintf( stderr, "p%d estimate_exchange: end=%d dim=%d done!\n",
         system->my_rank, end, d );
#endif
}


static void Estimate_Init_Storage( int me, neighbor_proc * const nbr1, neighbor_proc * const nbr2,
        int d, int * const max, int * const nrecv, mpi_datatypes * const mpi_data, MPI_Comm comm )
{
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    int new_max, ret;

    /* first exchange the estimates, then allocate buffers */
    ret = MPI_Irecv( &nbr1->est_recv, 1, MPI_INT, nbr1->rank, 2 * d + 1, comm, &req1 );
    Check_MPI_Error( ret, "Estimate_Init_Storage::MPI_Irecv::nbr1" );
    ret = MPI_Irecv( &nbr2->est_recv, 1, MPI_INT, nbr2->rank, 2 * d, comm, &req2 );
    Check_MPI_Error( ret, "Estimate_Init_Storage::MPI_Irecv::nbr2" );
    ret = MPI_Send( &nbr1->est_send, 1, MPI_INT, nbr1->rank, 2 * d, comm );
    Check_MPI_Error( ret, "Estimate_Init_Storage::MPI_Send::nbr1" );
    ret = MPI_Send( &nbr2->est_send, 1, MPI_INT, nbr2->rank, 2 * d + 1, comm );
    Check_MPI_Error( ret, "Estimate_Init_Storage::MPI_Send::nbr2" );
    ret = MPI_Wait( &req1, &stat1 );
    Check_MPI_Error( ret, "Estimate_Init_Storage::MPI_Wait::nbr1" );
    ret = MPI_Wait( &req2, &stat2 );
    Check_MPI_Error( ret, "Estimate_Init_Storage::MPI_Wait::nbr2" );
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

        if ( mpi_data->in1_buffer != NULL )
        {
            sfree( mpi_data->in1_buffer, "Estimate_Init_Storage::mpi_data->in1_buffer" );
        }
        if ( mpi_data->in2_buffer != NULL )
        {
            sfree( mpi_data->in2_buffer, "Estimate_Init_Storage::mpi_data->in2_buffer" );
        }

        mpi_data->in1_buffer = smalloc( sizeof(boundary_atom) * new_max,
                "Estimate_Init_Storage::mpi_data->in1_buffer" );
        mpi_data->in2_buffer = smalloc( sizeof(boundary_atom) * new_max,
                "Estimate_Init_Storage::mpi_data->in2_buffer" );
    }
}


/*********************** UNPACKER **************************/
static void Unpack_Exchange_Message( reax_system * const system, int end, void * const dummy,
        int cnt, neighbor_proc * const nbr, int dim )
{
    int i;
    real dx;
    const boundary_atom * const src = (boundary_atom *) dummy;
    reax_atom * const dest = system->my_atoms + end;

    for ( i = 0; i < cnt; ++i )
    {
        Unpack_Boundary_Atom( dest + i, src + i );
    }

#if defined(DEBUG_FOCUS)
    for ( i = end; i < end + cnt; ++i )
    {
        fprintf( stderr, "UNPACK p%d: d = %d, atom %d, x[0] = %10.4f, x[1] = %10.4f, x[2] = %10.4f\n",
              system->my_rank, dim, i,
              system->my_atoms[i].x[0],
              system->my_atoms[i].x[1],
              system->my_atoms[i].x[2] );
    }
#endif

    /* record the atoms recv'd from this nbr */
    nbr->atoms_str = end;
    nbr->atoms_cnt = cnt;

    /* update est_recv */
    nbr->est_recv = MAX( (int)(cnt * SAFER_ZONE), MIN_SEND );

    /* update max_recv to make sure that we reallocate at the right time */
    if ( cnt > system->max_recved )
    {
        system->max_recved = cnt;
    }

    /* adjust coordinates of recved atoms if nbr is a periodic one */
    if ( nbr->prdc[dim] )
    {
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "UNPACK p%d: dim = %d, dx = %f\n",
                    system->my_rank, dim, dx );
#endif

        for ( i = 0; i < cnt; ++i )
        {
            dest[i].x[dim] += dx;
        }
    }
}


void Unpack_Estimate_Message( reax_system * const system, int end, void * const dummy,
        int cnt, neighbor_proc * const nbr, int dim )
{
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d-p%d unpack_estimate: end=%d cnt=%d - unpacking\n",
             system->my_rank, nbr->rank, end, cnt );
#endif

    system->my_atoms = srealloc( system->my_atoms, sizeof(reax_atom) * (end + cnt),
            "Unpack_Estimate_Message::system:my_atoms" );

    Unpack_Exchange_Message( system, end, dummy, cnt, nbr, dim );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d-p%d unpack_estimate: end=%d cnt=%d - done\n",
             system->my_rank, nbr->rank, end, cnt );
#endif
}


/**************** UPDATE ATOM POSITIONS *******************/

/**************** PACK POSITION UPDATES *******************/
static void Sort_Position_Updates( reax_system * const system, int start, int end,
        int dim, mpi_out_data * const out_bufs )
{
    int i, p;
    const reax_atom * const atoms = system->my_atoms;
    rvec *out;

    for ( p = 2 * dim; p < 2 * dim + 2; ++p )
    {
        out = (rvec*) out_bufs[p].out_atoms;

        for ( i = 0; i < out_bufs[p].cnt; ++i )
        {
            memcpy( out[i], atoms[ out_bufs[p].index[i] ].x, sizeof(rvec) );
        }
    }
}

/*************** UNPACK POSITION UPDATES ******************/
static void Unpack_Position_Updates( reax_system * const system, int end, void * const dummy,
        int cnt, neighbor_proc * const nbr, int dim )
{
    int i;
    const int start = nbr->atoms_str;
    reax_atom * const atoms = system->my_atoms;
    real dx;
    rvec * const src = (rvec*) dummy;

    for ( i = 0; i < cnt; ++i )
    {
        memcpy( atoms[start + i].x, src[i], sizeof(rvec) );
    }

    /* adjust coordinates of recved atoms if nbr is a periodic one */
    if ( nbr->prdc[dim] )
    {
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];

        for ( i = 0; i < cnt; ++i )
        {
            atoms[start + i].x[dim] += dx;
        }
    }
}


int SendRecv( reax_system * const system, mpi_datatypes * const mpi_data, MPI_Datatype type,
        int * const nrecv, message_sorter sort_func, unpacker unpack, int clr )
{
    int d, cnt, start, end, max, est_flag, ret;
    mpi_out_data * const out_bufs = mpi_data->out_buffers;
    const MPI_Comm comm = mpi_data->comm_mesh3D;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    neighbor_proc *nbr1, *nbr2;
#if defined(DEBUG_FOCUS)
    int i, p;
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d sendrecv: entered\n", system->my_rank );
#endif

    if ( clr == TRUE )
    {
        Reset_Out_Buffers( mpi_data->out_buffers, system->num_nbrs );
    }
    start = 0;
    end = system->n;
    max = 0;
    est_flag = (mpi_data->in1_buffer == NULL) || (mpi_data->in2_buffer == NULL) ?
        TRUE : FALSE;

    for ( d = 0; d < 3; ++d )
    {
        sort_func( system, start, end, d, out_bufs );
        start = end;
        nbr1 = &system->my_nbrs[2 * d];
        nbr2 = &system->my_nbrs[2 * d + 1];

        /* for estimates in1_buffer & in2_buffer will be NULL */
        if ( est_flag == TRUE )
        {
            Estimate_Init_Storage( system->my_rank, nbr1, nbr2, d,
                    &max, nrecv, mpi_data, comm );
        }

#if defined(DEBUG_FOCUS)
        for ( p = 2 * d; p < 2 * d + 2; ++p )
        {
            for ( i = 0; i < out_bufs[p].cnt; ++i )
            {
                fprintf( stderr, "p%d: out_bufs[%d].index[%d] = %d\n",
                        system->my_rank, p, i, out_bufs[p].index[i] );
                fprintf( stderr, "  p%d: atom %6d, x[0] = %10.4f, x[1] = %10.4f, x[2] = %10.4f\n",
                        system->my_rank,
                        ((boundary_atom *)(out_bufs[p].out_atoms))[i].orig_id,
                        ((boundary_atom *)(out_bufs[p].out_atoms))[i].x[0],
                        ((boundary_atom *)(out_bufs[p].out_atoms))[i].x[1],
                        ((boundary_atom *)(out_bufs[p].out_atoms))[i].x[2] );
            }
        }
#endif

        /* initiate recvs */
        ret = MPI_Irecv( mpi_data->in1_buffer, nrecv[2 * d], type, nbr1->rank, 2 * d + 1, comm, &req1 );
        Check_MPI_Error( ret, "SendRecv::MPI_Irecv::nbr1" );
        ret = MPI_Irecv( mpi_data->in2_buffer, nrecv[2 * d + 1], type, nbr2->rank, 2 * d, comm, &req2 );
        Check_MPI_Error( ret, "SendRecv::MPI_Irecv::nbr2" );

        /* send both messages in dimension d */
        ret = MPI_Send( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt, type,
                nbr1->rank, 2 * d, comm );
        Check_MPI_Error( ret, "SendRecv::MPI_Send::nbr1" );
        ret = MPI_Send( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt, type,
                nbr2->rank, 2 * d + 1, comm );
        Check_MPI_Error( ret, "SendRecv::MPI_Send::nbr2" );

        /* recv and unpack atoms from nbr1 in dimension d */
        ret = MPI_Wait( &req1, &stat1 );
        Check_MPI_Error( ret, "SendRecv::MPI_Wait::nbr1" );
        ret = MPI_Get_count( &stat1, type, &cnt );
        Check_MPI_Error( ret, "SendRecv::MPI_Count::nbr1" );
        unpack( system, end, mpi_data->in1_buffer, cnt, nbr1, d );
        end += cnt;

#if defined(DEBUG)
        fprintf( stderr, "p%d: nbr1: d = %d, end = %d\n", system->my_rank, d, end );
#endif

        /* recv and unpack atoms from nbr2 in dimension d */
        ret = MPI_Wait( &req2, &stat2 );
        Check_MPI_Error( ret, "SendRecv::MPI_Wait::nbr2" );
        ret = MPI_Get_count( &stat2, type, &cnt );
        Check_MPI_Error( ret, "SendRecv::MPI_Count::nbr2" );
        unpack( system, end, mpi_data->in2_buffer, cnt, nbr2, d );
        end += cnt;

#if defined(DEBUG)
        fprintf( stderr, "p%d: nbr2: d = %d, end = %d\n", system->my_rank, d, end );
#endif
    }

    if ( est_flag == TRUE )
    {
        system->est_recv = max;
        system->est_trans = (max * sizeof(boundary_atom)) / sizeof(mpi_atom);
    }

    return end;
}


void Comm_Atoms( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes * const mpi_data, int renbr )
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

    if ( renbr == TRUE )
    {
        /* transfer ownership of atoms */
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nrecv[i] = system->est_trans;
        }
        system->n = SendRecv( system, mpi_data, mpi_data->mpi_atom_type, nrecv,
                Sort_Transfer_Atoms, Unpack_Transfer_Message, TRUE );
        Bin_My_Atoms( system, &workspace->realloc );
        Reorder_My_Atoms( system, workspace );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d, step %d: updated local atoms, n=%d\n",
                 system->my_rank, data->step, system->n );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        /* exchange ghost region info with neighbors */
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nrecv[i] = system->my_nbrs[i].est_recv;
        }
        system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
                Sort_Boundary_Atoms, Unpack_Exchange_Message, TRUE );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d, step %d: exchanged boundary atoms, N=%d\n",
                 system->my_rank, data->step, system->N );
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            fprintf( stderr, "p%d: nbr%d(p%d) str=%d cnt=%d end=%d\n",
                    system->my_rank, i, system->my_nbrs[i].rank,
                    system->my_nbrs[i].atoms_str,  system->my_nbrs[i].atoms_cnt,
                    system->my_nbrs[i].atoms_str + system->my_nbrs[i].atoms_cnt );
        }
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        Bin_Boundary_Atoms( system );
    }
    else
    {
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nrecv[i] = system->my_nbrs[i].atoms_cnt;
        }
        SendRecv( system, mpi_data, mpi_data->mpi_rvec, nrecv,
                Sort_Position_Updates, Unpack_Position_Updates, FALSE );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: updated positions\n", system->my_rank );
        MPI_Barrier( MPI_COMM_WORLD );
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
    fprintf( stderr, "p%d @ renbr=%d: comm_atoms done\n",
            system->my_rank, renbr );
    fprintf( stderr, "p%d: system->n = %d, system->N = %d\n",
             system->my_rank, system->n, system->N );
    //Print_My_Ext_Atoms( system );
    //Print_All_GCells( system );
    //MPI_Barrier( MPI_COMM_WORLD );
#endif
}
