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

#if defined(PURE_REAX)
  #include "basic_comm.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_basic_comm.h"
  #include "reax_vector.h"
#endif


typedef void (*dist_packer)( void*, mpi_out_data* );
typedef void (*coll_unpacker)( void*, void*, mpi_out_data* );


static void int_packer( void *dummy, mpi_out_data *out_buf )
{
    int i;
    int *buf = (int*) dummy;
    int *out = (int*) out_buf->out_atoms;

    for ( i = 0; i < out_buf->cnt; ++i )
    {
        //if( buf[ out_buf->index[i] ] !=-1 )
        out[i] = buf[ out_buf->index[i] ];
    }
}


static void real_packer( void *dummy, mpi_out_data *out_buf )
{
    int i;
    real *buf = (real*) dummy;
    real *out = (real*) out_buf->out_atoms;

    for ( i = 0; i < out_buf->cnt; ++i )
    {
        out[i] = buf[ out_buf->index[i] ];
    }
}


static void rvec_packer( void *dummy, mpi_out_data *out_buf )
{
    int i;
    rvec *buf, *out;

    buf = (rvec*) dummy;
    out = (rvec*) out_buf->out_atoms;

    for ( i = 0; i < out_buf->cnt; ++i )
    {
        memcpy( out + i, buf + out_buf->index[i], sizeof(rvec) );
    }
}


static void rvec2_packer( void *dummy, mpi_out_data *out_buf )
{
    int i;
    rvec2 *buf, *out;

    buf = (rvec2*) dummy;
    out = (rvec2*) out_buf->out_atoms;

    for ( i = 0; i < out_buf->cnt; ++i )
    {
        memcpy( out + i, buf + out_buf->index[i], sizeof(rvec2) );
    }
}


static void int_unpacker( void *dummy_in, void *dummy_buf, mpi_out_data *out_buf )
{
        int i;
        int *in, *buf;

        in = (int*) dummy_in;
        buf = (int*) dummy_buf;

        for ( i = 0; i < out_buf->cnt; ++i )
        {
            if( buf[ out_buf->index[i] ] == -1 && in[i] != -1 )
            {
                buf[ out_buf->index[i] ] = in[i];
            }
        }
}


static void real_unpacker( void *dummy_in, void *dummy_buf, mpi_out_data *out_buf )
{
    int i;
    real *in, *buf;

    in = (real*) dummy_in;
    buf = (real*) dummy_buf;

    for ( i = 0; i < out_buf->cnt; ++i )
    {
        buf[ out_buf->index[i] ] += in[i];
    }
}


static void rvec_unpacker( void *dummy_in, void *dummy_buf, mpi_out_data *out_buf )
{
    int i;
    rvec *in, *buf;

    in = (rvec*) dummy_in;
    buf = (rvec*) dummy_buf;

    for ( i = 0; i < out_buf->cnt; ++i )
    {
        rvec_Add( buf[ out_buf->index[i] ], in[i] );

#if defined(DEBUG)
        fprintf( stderr, "rvec_unpacker: cnt=%d  i =%d  index[i]=%d\n",
                out_buf->cnt, i, out_buf->index[i] );
#endif
    }
}


static void rvec2_unpacker( void *dummy_in, void *dummy_buf, mpi_out_data *out_buf )
{
    int i;
    rvec2 *in, *buf;

    in = (rvec2*) dummy_in;
    buf = (rvec2*) dummy_buf;

    for ( i = 0; i < out_buf->cnt; ++i )
    {
        buf[ out_buf->index[i] ][0] += in[i][0];
        buf[ out_buf->index[i] ][1] += in[i][1];
    }
}


static void * Get_Buffer_Offset( const void * const buffer,
        const int offset, const int type )
{
    void * ptr;

    switch ( type )
    {
        case INT_PTR_TYPE:
            ptr = (int *) buffer + offset;
            break;

        case REAL_PTR_TYPE:
            ptr = (real *) buffer + offset;
            break;

        case RVEC_PTR_TYPE:
            ptr = (rvec *) buffer + offset;
            break;

        case RVEC2_PTR_TYPE:
            ptr = (rvec2 *) buffer + offset;
            break;

        default:
            fprintf( stderr, "[ERROR] unknown pointer type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    return ptr;
}


static dist_packer Get_Packer( const int type )
{
    dist_packer ptr;

    switch ( type )
    {
        case INT_PTR_TYPE:
            ptr = &int_packer;
            break;

        case REAL_PTR_TYPE:
            ptr = &real_packer;
            break;

        case RVEC_PTR_TYPE:
            ptr = &rvec_packer;
            break;

        case RVEC2_PTR_TYPE:
            ptr = &rvec2_packer;
            break;

        default:
            fprintf( stderr, "[ERROR] unknown pointer type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    return ptr;
}


static coll_unpacker Get_Unpacker( const int type )
{
    coll_unpacker ptr;

    switch ( type )
    {
        case INT_PTR_TYPE:
            ptr = &int_unpacker;
            break;

        case REAL_PTR_TYPE:
            ptr = &real_unpacker;
            break;

        case RVEC_PTR_TYPE:
            ptr = &rvec_unpacker;
            break;

        case RVEC2_PTR_TYPE:
            ptr = &rvec2_unpacker;
            break;

        default:
            fprintf( stderr, "[ERROR] unknown pointer type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    return ptr;
}


void Dist( const reax_system * const system, mpi_datatypes * const mpi_data,
        void *buf, int buf_type, MPI_Datatype type )
{
#if defined(NEUTRAL_TERRITORY)
    int d, count, index;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req[6];
    MPI_Status stat[6];
    dist_packer pack;

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_nt_buffers;
    pack = Get_Packer( buf_type );
    count = 0;

    /* initiate recvs */
    for ( d = 0; d < 6; ++d )
    {
        if ( system->my_nt_nbrs[d].atoms_cnt )
        {
            count++;
            MPI_Irecv( Get_Buffer_Offset( buf, system->my_nt_nbrs[d].atoms_str, buf_type ),
                    system->my_nt_nbrs[d].atoms_cnt, type,
                    system->my_nt_nbrs[d].receive_rank, d, comm, &req[d] );
        }
    }

    for ( d = 0; d < 6; ++d)
    {
        /* send both messages in dimension d */
        if ( out_bufs[d].cnt )
        {
            pack( buf, &out_bufs[d] );
            MPI_Send( out_bufs[d].out_atoms, out_bufs[d].cnt, type,
                    system->my_nt_nbrs[d].rank, d, comm );
        }
    }

    for ( d = 0; d < count; ++d )
    {
        MPI_Waitany( MAX_NT_NBRS, req, &index, stat);
    }
    
#if defined(DEBUG)
    fprintf( stderr, "p%d dist: done\n", system->my_rank );
#endif

#else
    int d;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    const neighbor_proc *nbr1, *nbr2;
    dist_packer pack;

#if defined(DEBUG)
    fprintf( stderr, "p%d dist: entered\n", system->my_rank );
#endif

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;
    pack = Get_Packer( buf_type );

    for ( d = 0; d < 3; ++d )
    {
        /* initiate recvs */
        nbr1 = &system->my_nbrs[2 * d];
        if ( nbr1->atoms_cnt )
        {
            MPI_Irecv( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    nbr1->atoms_cnt, type, nbr1->rank, 2 * d + 1, comm, &req1 );
        }

        nbr2 = &system->my_nbrs[2 * d + 1];
        if ( nbr2->atoms_cnt )
        {
            MPI_Irecv( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    nbr2->atoms_cnt, type, nbr2->rank, 2 * d, comm, &req2 );
        }

        /* send both messages in dimension d */
        if ( out_bufs[2 * d].cnt )
        {
            pack( buf, &out_bufs[2 * d] );
            MPI_Send( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt,
                    type, nbr1->rank, 2 * d, comm );
        }

        if ( out_bufs[2 * d + 1].cnt )
        {
            pack( buf, &out_bufs[2 * d + 1] );
            MPI_Send( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt,
                    type, nbr2->rank, 2 * d + 1, comm );
        }

        if( nbr1->atoms_cnt )
        {
            MPI_Wait( &req1, &stat1 );
        }
        if( nbr2->atoms_cnt )
        {
            MPI_Wait( &req2, &stat2 );
        }
    }


#if defined(DEBUG)
    fprintf( stderr, "p%d dist: done\n", system->my_rank );
#endif
#endif
}


void Dist_FS( const reax_system * const system, mpi_datatypes * const mpi_data,
        void *buf, int buf_type, MPI_Datatype type )
{
    int d;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    const neighbor_proc *nbr1, *nbr2;
    dist_packer pack;

#if defined(DEBUG)
    fprintf( stderr, "p%d dist: entered\n", system->my_rank );
#endif

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;
    pack = Get_Packer( buf_type );

    for ( d = 0; d < 3; ++d )
    {
        /* initiate recvs */
        nbr1 = &system->my_nbrs[2 * d];
        if ( nbr1->atoms_cnt )
        {
            MPI_Irecv( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    nbr1->atoms_cnt, type, nbr1->rank, 2 * d + 1, comm, &req1 );
        }

        nbr2 = &system->my_nbrs[2 * d + 1];
        if ( nbr2->atoms_cnt )
        {
            MPI_Irecv( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    nbr2->atoms_cnt, type, nbr2->rank, 2 * d, comm, &req2 );
        }

        /* send both messages in dimension d */
        if ( out_bufs[2 * d].cnt )
        {
            pack( buf, &out_bufs[2 * d] );
            MPI_Send( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt,
                    type, nbr1->rank, 2 * d, comm );
        }

        if ( out_bufs[2 * d + 1].cnt )
        {
            pack( buf, &out_bufs[2 * d + 1] );
            MPI_Send( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt,
                    type, nbr2->rank, 2 * d + 1, comm );
        }

        if( nbr1->atoms_cnt )
        {
            MPI_Wait( &req1, &stat1 );
        }
        if( nbr2->atoms_cnt )
        {
            MPI_Wait( &req2, &stat2 );
        }
    }


#if defined(DEBUG)
    fprintf( stderr, "p%d dist: done\n", system->my_rank );
#endif
}


void Coll( const reax_system * const system, mpi_datatypes * const mpi_data,
        void *buf, int buf_type, MPI_Datatype type )
{   
#if defined(NEUTRAL_TERRITORY)
    int d, count, index;
    void *in[6];
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req[6];
    MPI_Status stat[6];
    coll_unpacker unpack;

#if defined(DEBUG)
    fprintf( stderr, "p%d coll: entered\n", system->my_rank );
#endif

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_nt_buffers;
    unpack = Get_Unpacker( buf_type );
    count = 0;

    for ( d = 0; d < 6; ++d )
    {
        in[d] = mpi_data->in_nt_buffer[d];

        if ( out_bufs[d].cnt )
        {
            count++;
            MPI_Irecv( in[d], out_bufs[d].cnt, type,
                    system->my_nt_nbrs[d].rank, d, comm, &req[d] );
        }
    }

    for ( d = 0; d < 6; ++d )
    {
        /* send both messages in direction d */
        if ( system->my_nt_nbrs[d].atoms_cnt )
        {
            MPI_Send( Get_Buffer_Offset( buf, system->my_nt_nbrs[d].atoms_str, buf_type ),
                    system->my_nt_nbrs[d].atoms_cnt, type,
                    system->my_nt_nbrs[d].receive_rank, d, comm );
        }
    }
    
    for ( d = 0; d < count; ++d )
    {
        MPI_Waitany( MAX_NT_NBRS, req, &index, stat);
        unpack( in[index], buf, &out_bufs[index] );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d coll: done\n", system->my_rank );
#endif

#else
    int d;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    const neighbor_proc *nbr1, *nbr2;
    coll_unpacker unpack;

#if defined(DEBUG)
    fprintf( stderr, "p%d coll: entered\n", system->my_rank );
#endif

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;
    unpack = Get_Unpacker( buf_type );

    for ( d = 2; d >= 0; --d )
    {
        /* initiate recvs */
        nbr1 = &system->my_nbrs[2 * d];

        if ( out_bufs[2 * d].cnt )
        {
            MPI_Irecv( mpi_data->in1_buffer, out_bufs[2 * d].cnt,
                    type, nbr1->rank, 2 * d + 1, comm, &req1 );
        }

        nbr2 = &system->my_nbrs[2 * d + 1];

        if ( out_bufs[2 * d + 1].cnt )
        {

            MPI_Irecv( mpi_data->in2_buffer, out_bufs[2 * d + 1].cnt,
                    type, nbr2->rank, 2 * d, comm, &req2 );
        }
        
        /* send both messages in dimension d */
        if ( nbr1->atoms_cnt )
        {
            MPI_Send( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    nbr1->atoms_cnt, type, nbr1->rank, 2 * d, comm );
        }
        
        if ( nbr2->atoms_cnt )
        {
            MPI_Send( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    nbr2->atoms_cnt, type, nbr2->rank, 2 * d + 1, comm );
        }

#if defined(DEBUG)
        fprintf( stderr, "p%d coll[%d] nbr1: str=%d cnt=%d recv=%d\n",
                system->my_rank, d, nbr1->atoms_str, nbr1->atoms_cnt,
                out_bufs[2 * d].cnt );
        fprintf( stderr, "p%d coll[%d] nbr2: str=%d cnt=%d recv=%d\n",
                system->my_rank, d, nbr2->atoms_str, nbr2->atoms_cnt,
                out_bufs[2 * d + 1].cnt );
#endif

        if ( out_bufs[2 * d].cnt )
        {
            MPI_Wait( &req1, &stat1 );
            unpack( mpi_data->in1_buffer, buf, &out_bufs[2 * d] );
        }

        if ( out_bufs[2 * d + 1].cnt )
        {
            MPI_Wait( &req2, &stat2 );
            unpack( mpi_data->in2_buffer, buf, &out_bufs[2 * d + 1] );
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d coll: done\n", system->my_rank );
#endif
#endif
}


void Coll_FS( const reax_system * const system, mpi_datatypes * const mpi_data,
        void *buf, int buf_type, MPI_Datatype type )
{   
    int d;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    const neighbor_proc *nbr1, *nbr2;
    coll_unpacker unpack;

#if defined(DEBUG)
    fprintf( stderr, "p%d coll: entered\n", system->my_rank );
#endif

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;
    unpack = Get_Unpacker( buf_type );

    for ( d = 2; d >= 0; --d )
    {
        /* initiate recvs */
        nbr1 = &system->my_nbrs[2 * d];

        if ( out_bufs[2 * d].cnt )
        {
            MPI_Irecv( mpi_data->in1_buffer, out_bufs[2 * d].cnt,
                    type, nbr1->rank, 2 * d + 1, comm, &req1 );
        }

        nbr2 = &system->my_nbrs[2 * d + 1];

        if ( out_bufs[2 * d + 1].cnt )
        {

            MPI_Irecv( mpi_data->in2_buffer, out_bufs[2 * d + 1].cnt,
                    type, nbr2->rank, 2 * d, comm, &req2 );
        }
        
        /* send both messages in dimension d */
        if ( nbr1->atoms_cnt )
        {
            MPI_Send( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    nbr1->atoms_cnt, type, nbr1->rank, 2 * d, comm );
        }
        
        if ( nbr2->atoms_cnt )
        {
            MPI_Send( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    nbr2->atoms_cnt, type, nbr2->rank, 2 * d + 1, comm );
        }

#if defined(DEBUG)
        fprintf( stderr, "p%d coll[%d] nbr1: str=%d cnt=%d recv=%d\n",
                system->my_rank, d, nbr1->atoms_str, nbr1->atoms_cnt,
                out_bufs[2 * d].cnt );
        fprintf( stderr, "p%d coll[%d] nbr2: str=%d cnt=%d recv=%d\n",
                system->my_rank, d, nbr2->atoms_str, nbr2->atoms_cnt,
                out_bufs[2 * d + 1].cnt );
#endif

        if ( out_bufs[2 * d].cnt )
        {
            MPI_Wait( &req1, &stat1 );
            unpack( mpi_data->in1_buffer, buf, &out_bufs[2 * d] );
        }

        if ( out_bufs[2 * d + 1].cnt )
        {
            MPI_Wait( &req2, &stat2 );
            unpack( mpi_data->in2_buffer, buf, &out_bufs[2 * d + 1] );
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d coll: done\n", system->my_rank );
#endif
}


/*****************************************************************************/
real Parallel_Norm( real *v, int n, MPI_Comm comm )
{
    int i;
    real my_sum, norm_sqr;

    my_sum = 0.0;

    for ( i = 0; i < n; ++i )
    {
        my_sum += SQR( v[i] );
    }

    MPI_Allreduce( &my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, comm );

    return sqrt( norm_sqr );
}


real Parallel_Dot( real *v1, real *v2, int n, MPI_Comm comm )
{
    int  i;
    real my_dot, res;

    my_dot = 0.0;

    for ( i = 0; i < n; ++i )
    {
        my_dot += v1[i] * v2[i];
    }

    MPI_Allreduce( &my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, comm );

    return res;
}


real Parallel_Vector_Acc( real *v, int n, MPI_Comm comm )
{
    int  i;
    real my_acc, res;

    my_acc = 0;
    for ( i = 0; i < n; ++i )
        my_acc += v[i];

    MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, comm );

    return res;
}


/*****************************************************************************/
#if defined(TEST_FORCES)
void Coll_ids_at_Master( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data )
{
    int i;
    int *id_list;

    MPI_Gather( &system->n, 1, MPI_INT, workspace->rcounts, 1, MPI_INT,
            MASTER_NODE, mpi_data->world );

    if ( system->my_rank == MASTER_NODE )
    {
        workspace->displs[0] = 0;
        for ( i = 1; i < system->wsize; ++i )
            workspace->displs[i] = workspace->displs[i - 1] + workspace->rcounts[i - 1];
    }

    id_list = (int*) malloc( system->n * sizeof(int) );
    for ( i = 0; i < system->n; ++i )
        id_list[i] = system->my_atoms[i].orig_id;

    MPI_Gatherv( id_list, system->n, MPI_INT,
            workspace->id_all, workspace->rcounts, workspace->displs,
            MPI_INT, MASTER_NODE, mpi_data->world );

    sfree( id_list, "id_list" );

#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
    {
        for ( i = 0 ; i < system->bigN; ++i )
            fprintf( stderr, "id_all[%d]: %d\n", i, workspace->id_all[i] );
    }
#endif
}


void Coll_rvecs_at_Master( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data, rvec* v )
{
    MPI_Gatherv( v, system->n, mpi_data->mpi_rvec,
            workspace->f_all, workspace->rcounts, workspace->displs,
            mpi_data->mpi_rvec, MASTER_NODE, mpi_data->world );
}
#endif
