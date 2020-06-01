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


#if defined(NEUTRAL_TERRITORY)
void Setup_NT_Comm( reax_system * const system, control_params * const control,
                 mpi_datatypes * const mpi_data )
{
    int i, d;
    real bndry_cut;
    neighbor_proc *nbr_pr;
    simulation_box *my_box;
    ivec nbr_coords, nbr_recv_coords;
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
        {+1, 0, 0}, // +x
        {+1, -1, 0}  // +x-y
    };
    my_box = &system->my_box;
    bndry_cut = system->bndry_cuts.ghost_cutoff;
    system->num_nt_nbrs = MAX_NT_NBRS;

    /* identify my neighbors */
    for ( i = 0; i < system->num_nt_nbrs; ++i )
    {
        nbr_pr = &system->my_nt_nbrs[i];
        ivec_Sum( nbr_coords, system->my_coords, r[i] ); /* actual nbr coords */
        MPI_Cart_rank( mpi_data->comm_mesh3D, nbr_coords, &nbr_pr->rank );
        
        /* set the rank of the neighbor processor in the receiving direction */
        ivec_Sum( nbr_recv_coords, system->my_coords, r[i + 6] ); /* actual nbr coords */
        MPI_Cart_rank( mpi_data->comm_mesh3D, nbr_recv_coords, &nbr_pr->receive_rank );

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
            {
                nbr_pr->prdc[d] = -1;
            }
            else if ( nbr_coords[d] >= control->procs_by_dim[d] )
            {
                nbr_pr->prdc[d] = 1;
            }
            else
            {
                nbr_pr->prdc[d] = 0;
            }
        }

    }
}


static int Sort_Neutral_Territory( reax_system *system, int dir, mpi_out_data *out_bufs, int write )
{
    int i, cnt;
    reax_atom *atoms;
    neighbor_proc *nbr_pr;

    cnt = 0;
    atoms = system->my_atoms;
    /* place each atom into the appropriate outgoing list */
    nbr_pr = &system->my_nt_nbrs[dir];

    for ( i = 0; i < system->n; ++i )
    {
        if ( nbr_pr->bndry_min[0] <= atoms[i].x[0]
                && atoms[i].x[0] < nbr_pr->bndry_max[0]
                && nbr_pr->bndry_min[1] <= atoms[i].x[1]
                && atoms[i].x[1] < nbr_pr->bndry_max[1]
                && nbr_pr->bndry_min[2] <= atoms[i].x[2]
                && atoms[i].x[2] < nbr_pr->bndry_max[2] )
        {
            if ( write )
            {
                out_bufs[dir].index[out_bufs[dir].cnt] = i;
                out_bufs[dir].cnt++;
            }
            else
            {
                cnt++;
            }
        }
    }

    return cnt;
}


void Estimate_NT_Atoms( reax_system * const system, mpi_datatypes * const mpi_data )
{
    int d;
    mpi_out_data *out_bufs;
    neighbor_proc *nbr;

    out_bufs = mpi_data->out_nt_buffers;

    for ( d = 0; d < 6; ++d )
    {
        /* count the number of atoms in each processor's outgoing list */
        nbr = &system->my_nt_nbrs[d];
        nbr->est_send = Sort_Neutral_Territory( system, d, out_bufs, 0 );

        /* estimate the space needed based on the count above */
        nbr->est_send = MAX( MIN_SEND, nbr->est_send * SAFER_ZONE_NT );

        /* allocate the estimated space */
        out_bufs[d].index = scalloc( 2 * nbr->est_send, sizeof(int),
                "Estimate_NT_Atoms::out_bufs[d].index", MPI_COMM_WORLD );
        out_bufs[d].out_atoms = scalloc( 2 * nbr->est_send, sizeof(real),
                "Estimate_NT_Atoms::out_bufs[d].out_atoms", MPI_COMM_WORLD );

        /* sort the atoms to their outgoing buffers */
        // TODO: to call or not to call?
        //Sort_Neutral_Territory( system, d, out_bufs, 1 );
    }
}
#endif


void Check_MPI_Error( int code, const char * const filename, int line )
{
    int len;
    char err_msg[MPI_MAX_ERROR_STRING];

    if ( code != MPI_SUCCESS )
    {
        MPI_Error_string( code, err_msg, &len );

        fprintf( stderr, "[ERROR] MPI error\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strnlen(filename, MAX_STR), filename );
        fprintf( stderr, "    [INFO] Error code %d\n", code );
        fprintf( stderr, "    [INFO] Error message: %.*s\n", len, err_msg );
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
        MPI_Cart_rank( mpi_data->comm_mesh3D, nbr_coords, &nbr_pr->rank );

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
                nbr_pr->bndry_min[d] = NEG_INF;
                nbr_pr->bndry_max[d] = NEG_INF;
            }

            /* determine if it is a periodic neighbor */
            if ( nbr_coords[d] < 0 )
            {
                nbr_pr->prdc[d] = -1;
            }
            else if ( nbr_coords[d] >= control->procs_by_dim[d] )
            {
                nbr_pr->prdc[d] = 1;
            }
            else
            {
                nbr_pr->prdc[d] = 0;
            }
        }
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
                nbr_pr->bndry_min[d] = NEG_INF;
                nbr_pr->bndry_max[d] = NEG_INF;
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
    strncpy( matm->name, ratm->name, sizeof(matm->name) - 1 );
    matm->name[sizeof(matm->name) - 1] = '\0';
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
    strncpy( ratm->name, matm->name, sizeof(ratm->name) - 1 );
    ratm->name[sizeof(ratm->name) - 1] = '\0';
    rvec_Copy( ratm->x, matm->x );
    rvec_Copy( ratm->v, matm->v );
    rvec_Copy( ratm->f_old, matm->f_old );
    memcpy( ratm->s, matm->s, sizeof(rvec4) ); //rvec_Copy( ratm->s, matm->s );
    memcpy( ratm->t, matm->t, sizeof(rvec4) ); //rvec_Copy( ratm->t, matm->t );
}


/* Count number of atoms to be placed in egress MPI buffers.
 * The ownership of these atoms is to be transferred to other MPI processes. */
static void Count_Transfer_Atoms( reax_system const * const system,
        int start, int end, int dim, mpi_out_data * const out_bufs,
        int * const cnt )
{
    int i, d;
    simulation_box const * const my_box = &system->my_box;

    /* count atoms in the appropriate outgoing lists */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
        {
            if ( system->my_atoms[i].x[d] < my_box->min[d] )
            {
                ++cnt[2 * d];
                break;
            }
            else if ( system->my_atoms[i].x[d] >= my_box->max[d] )
            {
                ++cnt[2 * d + 1];
                break;
            }
        }
    }
}


/* Populate egress MPI buffers with atoms (of reax_atom type) whose
 * ownership is being transferred to other MPI processes
 *
 * Note: the "sort" refers to dividing the atoms into their appropriate buffers */
static void Sort_Transfer_Atoms( reax_system * const system, int start, int end,
        int dim, mpi_out_data * const out_bufs, mpi_datatypes * const mpi_data )
{
    int i, d;
    simulation_box * const my_box = &system->my_box;
    mpi_atom *out_buf;

    /* place each atom into the appropriate egress buffer */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
        {
            if ( system->my_atoms[i].x[d] < my_box->min[d] )
            {
                out_buf = out_bufs[2 * d].out_atoms;
                Pack_MPI_Atom( &out_buf[out_bufs[2 * d].cnt], &system->my_atoms[i], i );
                system->my_atoms[i].orig_id = -1;
                ++out_bufs[2 * d].cnt;
                break;
            }
            else if ( system->my_atoms[i].x[d] >= my_box->max[d] )
            {
                out_buf = out_bufs[2 * d + 1].out_atoms;
                Pack_MPI_Atom( &out_buf[out_bufs[2 * d + 1].cnt], &system->my_atoms[i], i );
                system->my_atoms[i].orig_id = -1;
                ++out_bufs[2 * d + 1].cnt;
                break;
            }
        }
    }
}


/*********************** UNPACKER **************************/
static void Unpack_Transfer_Message( reax_system * const system,
        int end, void * const mpi_buffer,
        int cnt, neighbor_proc * const nbr, int dim )
{
    int i;
    real dx;
    mpi_atom * const src = (mpi_atom *) mpi_buffer;

    if ( end + cnt > system->total_cap )
    {
        /* need space for my_atoms now, other reallocations will trigger in Reallocate */
        system->my_atoms = srealloc( system->my_atoms,
                sizeof(reax_atom) * (end + cnt), "Unpack_Transfer_Message" );
    }

    /* adjust coordinates of recved atoms if nbr is a periodic one */
    if ( nbr->prdc[dim] )
    {
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];

        for ( i = 0; i < cnt; ++i )
        {
            Unpack_MPI_Atom( &system->my_atoms[end + i], &src[i] );
            system->my_atoms[end + i].x[dim] += dx;
        }
    }
    else
    {
        for ( i = 0; i < cnt; ++i )
        {
            Unpack_MPI_Atom( &system->my_atoms[end + i], &src[i] );
        }
    }
}


/************** EXCHANGE BOUNDARY ATOMS *****************/

/************ PACK & UNPACK BOUNDARY ATOMS **************/
static void Pack_Boundary_Atom( boundary_atom * const matm,
        reax_atom const * const ratm, int i )
{
    matm->orig_id = ratm->orig_id;
    matm->imprt_id = i;
    matm->type = ratm->type;
    matm->num_bonds = ratm->num_bonds;
    matm->num_hbonds = ratm->num_hbonds;
    rvec_Copy( matm->x, ratm->x );
}


static void Unpack_Boundary_Atom( reax_atom * const ratm,
        const boundary_atom * const matm )
{
    ratm->orig_id = matm->orig_id;
    ratm->imprt_id = matm->imprt_id;
    ratm->type = matm->type;
    ratm->num_bonds = matm->num_bonds;
    ratm->num_hbonds = matm->num_hbonds;
//    ratm->renumber = offset + matm->imprt_id;
    rvec_Copy( ratm->x, matm->x );
}


void Count_Boundary_Atoms( reax_system const * const system,
        int start, int end, int dim, mpi_out_data * const out_bufs,
        int * const cnt )
{
    int i, d, p;

    /* place each atom into the appropriate outgoing list */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
        {
            for ( p = 2 * d; p < 2 * d + 2; ++p )
            {
                if ( system->my_nbrs[p].bndry_min[d] <= system->my_atoms[i].x[d]
                        && system->my_atoms[i].x[d] < system->my_nbrs[p].bndry_max[d] )
                {
                    ++cnt[p];
                }
            }
        }
    }
}


void Sort_Boundary_Atoms( reax_system * const system, int start, int end,
        int dim, mpi_out_data * const out_bufs, mpi_datatypes * const mpi_data )
{
    int i, d, p;
    boundary_atom *out_buf;

    /* place each atom into the appropriate egress buffer */
    for ( i = start; i < end; ++i )
    {
        for ( d = dim; d < 3; ++d )
        {
            for ( p = 2 * d; p < 2 * d + 2; ++p )
            {
                if ( system->my_nbrs[p].bndry_min[d] <= system->my_atoms[i].x[d]
                        && system->my_atoms[i].x[d] < system->my_nbrs[p].bndry_max[d] )
                {
                    out_bufs[p].index[out_bufs[p].cnt] = i;
                    out_buf = (boundary_atom *) out_bufs[p].out_atoms;
                    Pack_Boundary_Atom( &out_buf[out_bufs[p].cnt], &system->my_atoms[i], i );
                    ++out_bufs[p].cnt;
                }
            }
        }
    }
}


/* Copy received atoms out of MPI ingress buffer */
void Unpack_Exchange_Message( reax_system * const system, int end,
        void * const mpi_buffer, int cnt,
        neighbor_proc * const nbr, int dim )
{
    int i;
    real dx;
    const boundary_atom * const src = (boundary_atom *) mpi_buffer;

    if ( end + cnt > system->total_cap )
    {
        /* need space for my_atoms now, other reallocations will trigger in Reallocate */
        system->my_atoms = srealloc( system->my_atoms,
                sizeof(reax_atom) * (end + cnt), "Unpack_Exchange_Message" );
    }

    if ( nbr->prdc[dim] )
    {
        /* adjust coordinates of recved atoms if nbr is a periodic one */
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];

        for ( i = 0; i < cnt; ++i )
        {
            Unpack_Boundary_Atom( &system->my_atoms[end + i], &src[i] );
            system->my_atoms[end + i].x[dim] += dx;
        }
    }
    else
    {
        for ( i = 0; i < cnt; ++i )
        {
            Unpack_Boundary_Atom( &system->my_atoms[end + i], &src[i] );
        }
    }

    /* record the atoms recv'd from this nbr */
    nbr->atoms_str = end;
    nbr->atoms_cnt = cnt;
}


static void Count_Position_Updates( reax_system const * const system,
        int start, int end, int dim, mpi_out_data * const out_bufs,
        int * const cnt )
{
    int p;

    /* counts set via previous calls to SendRecv in reneighbor steps
     * (during boundary atom messaging), so just copy */
    for ( p = 2 * dim; p < 2 * dim + 2; ++p )
    {
        cnt[p] = out_bufs[p].cnt;
    }
}


static void Sort_Position_Updates( reax_system * const system, int start, int end,
        int dim, mpi_out_data * const out_bufs, mpi_datatypes * const mpi_data )
{
    int i, p;
    rvec *out;

    for ( p = 2 * dim; p < 2 * dim + 2; ++p )
    {
        out = (rvec*) out_bufs[p].out_atoms;

        for ( i = 0; i < out_bufs[p].cnt; ++i )
        {
            rvec_Copy( out[i], system->my_atoms[ out_bufs[p].index[i] ].x );
        }
    }
}


static void Unpack_Position_Updates( reax_system * const system, int end,
        void * const mpi_buffer, int cnt, neighbor_proc * const nbr, int dim )
{
    int i;
    const int start = nbr->atoms_str;
    real dx;
    rvec * const src = (rvec*) mpi_buffer;

    if ( nbr->prdc[dim] )
    {
        /* adjust coordinates of recved atoms if nbr is a periodic one */
        dx = nbr->prdc[dim] * system->big_box.box_norms[dim];

        for ( i = 0; i < cnt; ++i )
        {
            rvec_Copy( system->my_atoms[start + i].x, src[i] );
            system->my_atoms[start + i].x[dim] += dx;
        }
    }
    else
    {
        for ( i = 0; i < cnt; ++i )
        {
            rvec_Copy( system->my_atoms[start + i].x, src[i] );
        }
    }
}


int SendRecv( reax_system * const system, mpi_datatypes * const mpi_data,
        MPI_Datatype type, message_counter count_func,
        message_sorter sort_func, unpacker unpack, int clr )
{
    int i, d, cnt1, cnt2, cnt[6], start, end, ret;
    mpi_out_data * const out_bufs = mpi_data->out_buffers;
    const MPI_Comm comm = mpi_data->comm_mesh3D;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    neighbor_proc *nbr1, *nbr2;
    MPI_Aint extent, lower_bound, type_size;

    ret = MPI_Type_get_extent( type, &lower_bound, &extent );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    type_size = MPI_Aint_add( lower_bound, extent );

    if ( clr == TRUE )
    {
        Reset_Out_Buffers( mpi_data->out_buffers, system->num_nbrs );
    }
    start = 0;
    end = system->n;

    for ( i = 0; i < 6; ++i )
    {
        cnt[i] = 0;
    }

    for ( d = 0; d < 3; ++d )
    {
        /* nbr1 is in the negative direction, nbr2 the positive direction */
        nbr1 = &system->my_nbrs[2 * d];
        nbr2 = &system->my_nbrs[2 * d + 1];

        count_func( system, start, end, d, out_bufs, cnt );

        for ( i = 2 * d; i < 6; ++i )
        {
            check_srealloc( &out_bufs[i].out_atoms,
                    &out_bufs[i].out_atoms_size,
                    type_size * (out_bufs[i].cnt + cnt[i]),
                    "SendRecv::mpi_data->out_atoms" );
            check_srealloc( (void **) &out_bufs[i].index,
                    &out_bufs[i].index_size,
                    sizeof(int) * (out_bufs[i].cnt + cnt[i]),
                    "SendRecv::mpi_data->index" );
        }

        sort_func( system, start, end, d, out_bufs, mpi_data );

        start = end;

        /* send both messages in dimension d */
        ret = MPI_Isend( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt,
                type, nbr1->rank, 2 * d, comm, &req1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        ret = MPI_Isend( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt,
                type, nbr2->rank, 2 * d + 1, comm, &req2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        /* recv and unpack atoms from nbr1 in dimension d */
        ret = MPI_Probe( nbr1->rank, 2 * d + 1, comm, &stat1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat1, type, &cnt1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        check_smalloc( &mpi_data->in1_buffer, &mpi_data->in1_buffer_size,
                type_size * cnt1, "SendRecv::mpi_data->in1_buffer" );

        ret = MPI_Recv( mpi_data->in1_buffer, cnt1, type,
                nbr1->rank, 2 * d + 1, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        unpack( system, end, mpi_data->in1_buffer, cnt1, nbr1, d );
        end += cnt1;

        /* recv and unpack atoms from nbr2 in dimension d */
        ret = MPI_Probe( nbr2->rank, 2 * d, comm, &stat2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat2, type, &cnt2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        check_smalloc( &mpi_data->in2_buffer, &mpi_data->in2_buffer_size,
                type_size * cnt2, "SendRecv::mpi_data->in2_buffer" );

        ret = MPI_Recv( mpi_data->in2_buffer, cnt2, type,
                nbr2->rank, 2 * d, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        unpack( system, end, mpi_data->in2_buffer, cnt2, nbr2, d );
        end += cnt2;

        ret = MPI_Wait( &req1, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Wait( &req2, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }

    return end;
}


void Comm_Atoms( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes * const mpi_data, int renbr )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    if ( renbr == TRUE )
    {
        /* transfer ownership of any previous local atoms
         * which have moved outside of this processor's local sim. box */
        system->n = SendRecv( system, mpi_data, mpi_data->mpi_atom_type,
                &Count_Transfer_Atoms, &Sort_Transfer_Atoms,
                &Unpack_Transfer_Message, TRUE );

        Bin_My_Atoms( system, workspace );
        Reorder_My_Atoms( system, workspace );

        /* exchange ghost region info with neighbors */
        system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type,
                &Count_Boundary_Atoms, &Sort_Boundary_Atoms,
                &Unpack_Exchange_Message, TRUE );

        Bin_Boundary_Atoms( system );
    }
    else
    {
        /* provide position updates of atoms to neighboring processors */
        SendRecv( system, mpi_data, mpi_data->mpi_rvec,
                &Count_Position_Updates, &Sort_Position_Updates,
                &Unpack_Position_Updates, FALSE );
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.comm );
#endif
}
