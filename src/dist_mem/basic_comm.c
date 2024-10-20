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

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "../common/include/config.h"
#endif

#if defined(PURE_REAX)
  #include "basic_comm.h"

  #include "comm_tools.h"
  #include "tool_box.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_basic_comm.h"

  #include "reax_comm_tools.h"
  #include "reax_tool_box.h"
  #include "reax_vector.h"
#endif


typedef void (*dist_packer)( void const * const, void * const, int const * const, int );
typedef void (*coll_unpacker)( void const * const, void * const, int const * const, int );


static void int_packer( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i;
    int *src_, *dest_;

    src_ = (int *) src;
    dest_ = (int *) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

        dest_[i] = src_[index[i]];
    }
}


static void real_packer( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i;
    real *src_, *dest_;

    src_ = (real *) src;
    dest_ = (real *) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

        dest_[i] = src_[index[i]];
    }
}


static void rvec_packer( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i;
    rvec *src_, *dest_;

    src_ = (rvec *) src;
    dest_ = (rvec *) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

//        memcpy( &dest_[i], &src_[index[i]], sizeof(rvec) );
        dest_[i][0] = src_[index[i]][0];
        dest_[i][1] = src_[index[i]][1];
        dest_[i][2] = src_[index[i]][2];
    }
}


static void rvec2_packer( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i;
    rvec2 *src_, *dest_;

    src_ = (rvec2 *) src;
    dest_ = (rvec2 *) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

//        memcpy( &dest_[i], &src_[index[i]], sizeof(rvec2) );
        dest_[i][0] = src_[index[i]][0];
        dest_[i][1] = src_[index[i]][1];
    }
}


static void int_unpacker( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i, *src_, *dest_;

    src_ = (int*) src;
    dest_ = (int*) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

        dest_[index[i]] = src_[i];
    }
}


static void real_unpacker( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i;
    real *src_, *dest_;

    src_ = (real*) src;
    dest_ = (real*) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

        dest_[index[i]] += src_[i];
    }
}


static void rvec_unpacker( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i;
    rvec *src_, *dest_;

    src_ = (rvec*) src;
    dest_ = (rvec*) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

        rvec_Add( dest_[index[i]], src_[i] );
    }
}


static void rvec2_unpacker( void const * const src, void * const dest,
        int const * const index, int cnt )
{
    int i;
    rvec2 *src_, *dest_;

    src_ = (rvec2*) src;
    dest_ = (rvec2*) dest;

    for ( i = 0; i < cnt; ++i )
    {
        assert( index[i] >= 0 );

        dest_[index[i]][0] += src_[i][0];
        dest_[index[i]][1] += src_[i][1];
    }
}


static void * Get_Buffer_Offset( void const * const buffer,
        int offset, int type )
{
    void * ptr;

    switch ( type )
    {
        case INT_PTR_TYPE:
            ptr = &((int *) buffer)[offset];
            break;

        case REAL_PTR_TYPE:
            ptr = &((real *) buffer)[offset];
            break;

        case RVEC_PTR_TYPE:
            ptr = &((rvec *) buffer)[offset];
            break;

        case RVEC2_PTR_TYPE:
            ptr = &((rvec2 *) buffer)[offset];
            break;

        default:
            fprintf( stderr, "[ERROR] unknown pointer type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    return ptr;
}


static dist_packer Get_Packer( int type )
{
    dist_packer func_ptr;

    switch ( type )
    {
        case INT_PTR_TYPE:
            func_ptr = &int_packer;
            break;

        case REAL_PTR_TYPE:
            func_ptr = &real_packer;
            break;

        case RVEC_PTR_TYPE:
            func_ptr = &rvec_packer;
            break;

        case RVEC2_PTR_TYPE:
            func_ptr = &rvec2_packer;
            break;

        default:
            fprintf( stderr, "[ERROR] unknown pointer type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    return func_ptr;
}


static coll_unpacker Get_Unpacker( int type )
{
    coll_unpacker func_ptr;

    switch ( type )
    {
        case INT_PTR_TYPE:
            func_ptr = &int_unpacker;
            break;

        case REAL_PTR_TYPE:
            func_ptr = &real_unpacker;
            break;

        case RVEC_PTR_TYPE:
            func_ptr = &rvec_unpacker;
            break;

        case RVEC2_PTR_TYPE:
            func_ptr = &rvec2_unpacker;
            break;

        default:
            fprintf( stderr, "[ERROR] unknown pointer type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    return func_ptr;
}


void Dist( reax_system const * const system, mpi_datatypes * const mpi_data,
        void const * const buf, int buf_type, MPI_Datatype type )
{
#if defined(NEUTRAL_TERRITORY)
    int d, index, ret;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req[6];
    MPI_Status stat[6];
    dist_packer pack;

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_nt_buffers;
    pack = Get_Packer( buf_type );

    /* initiate recvs */
    for ( d = 0; d < 6; ++d )
    {
        ret = MPI_Irecv( Get_Buffer_Offset( buf, system->my_nt_nbrs[d].atoms_str, buf_type ),
                system->my_nt_nbrs[d].atoms_cnt, type,
                system->my_nt_nbrs[d].receive_rank, d, comm, &req[d] );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }

    for ( d = 0; d < 6; ++d )
    {
        /* send both messages in dimension d */
        pack( buf, out_bufs[d].out_atoms, out_bufs[d].index, out_bufs[d].cnt );
        ret = MPI_Send( out_bufs[d].out_atoms, out_bufs[d].cnt, type,
                system->my_nt_nbrs[d].rank, d, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }

    for ( d = 0; d < 6; ++d )
    {
        ret = MPI_Waitany( MAX_NT_NBRS, req, &index, stat );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }

#else
    int d, cnt1, cnt2, ret;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    const neighbor_proc *nbr1, *nbr2;
    dist_packer pack;
    MPI_Aint extent, lower_bound;
    size_t type_size;

    ret = MPI_Type_get_extent( type, &lower_bound, &extent );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
//    type_size = MPI_Aint_add( lower_bound, extent );
    type_size = extent;

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;
    pack = Get_Packer( buf_type );

    for ( d = 0; d < 3; ++d )
    {
        nbr1 = &system->my_nbrs[2 * d];
        nbr2 = &system->my_nbrs[2 * d + 1];

        /* pack MPI buffers and initiate sends */
        smalloc_check( &out_bufs[2 * d].out_atoms,
                &out_bufs[2 * d].out_atoms_size,
                type_size * out_bufs[2 * d].cnt,
                TRUE, SAFE_ZONE, __FILE__, __LINE__ );
        srealloc_check( (void **) &out_bufs[2 * d].index,
                &out_bufs[2 * d].index_size,
                sizeof(int) * out_bufs[2 * d].cnt,
                TRUE, SAFE_ZONE, __FILE__, __LINE__ );

        pack( buf, out_bufs[2 * d].out_atoms,
                out_bufs[2 * d].index, out_bufs[2 * d].cnt );

        if ( nbr1->rank != system->my_rank )
        {
            ret = MPI_Isend( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt,
                    type, nbr1->rank, 2 * d, comm, &req1 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }

        smalloc_check( &out_bufs[2 * d + 1].out_atoms,
                &out_bufs[2 * d + 1].out_atoms_size,
                type_size * out_bufs[2 * d + 1].cnt,
                TRUE, SAFE_ZONE, __FILE__, __LINE__ );
        srealloc_check( (void **) &out_bufs[2 * d + 1].index,
                &out_bufs[2 * d + 1].index_size,
                sizeof(int) * out_bufs[2 * d + 1].cnt,
                TRUE, SAFE_ZONE, __FILE__, __LINE__ );

        pack( buf, out_bufs[2 * d + 1].out_atoms,
                out_bufs[2 * d + 1].index, out_bufs[2 * d + 1].cnt );

        if ( nbr2->rank != system->my_rank )
        {
            ret = MPI_Isend( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt,
                    type, nbr2->rank, 2 * d + 1, comm, &req2 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }

        /* recv both messages in dimension d */
        if ( nbr1->rank != system->my_rank )
        {
            ret = MPI_Probe( nbr1->rank, 2 * d + 1, comm, &stat1 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            ret = MPI_Get_count( &stat1, type, &cnt1 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );

            if ( cnt1 == MPI_UNDEFINED )
            {
                fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }
            else if ( cnt1 + nbr1->atoms_str > system->total_cap )
            {
                fprintf( stderr, "[ERROR] Dist: not enough space in recv buffer for nbr1 (dim = %d)\n", d );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }

            ret = MPI_Recv( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    cnt1, type, nbr1->rank, 2 * d + 1, comm, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        else
        {
            pack( buf, Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    out_bufs[2 * d + 1].index, out_bufs[2 * d + 1].cnt );
        }

        if ( nbr2->rank != system->my_rank )
        {
            ret = MPI_Probe( nbr2->rank, 2 * d, comm, &stat2 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            ret = MPI_Get_count( &stat2, type, &cnt2 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );

            if ( cnt2 == MPI_UNDEFINED )
            {
                fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }
            else if ( cnt2 + nbr2->atoms_str > system->total_cap )
            {
                fprintf( stderr, "[ERROR] Dist: not enough space in recv buffer for nbr2 (dim = %d)\n", d );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }

            ret = MPI_Recv( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    cnt2, type, nbr2->rank, 2 * d, comm, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        else
        {
            pack( buf, Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    out_bufs[2 * d].index, out_bufs[2 * d].cnt );
        }

        if ( nbr1->rank != system->my_rank )
        {
            ret = MPI_Wait( &req1, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        if ( nbr2->rank != system->my_rank )
        {
            ret = MPI_Wait( &req2, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
    }
#endif
}


void Coll( reax_system const * const system, mpi_datatypes * const mpi_data,
        void * const buf, int buf_type, MPI_Datatype type )
{   
#if defined(NEUTRAL_TERRITORY)
    int d, index, ret;
    void *in[6];
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req[6];
    MPI_Status stat[6];
    coll_unpacker unpack;

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_nt_buffers;
    unpack = Get_Unpacker( buf_type );

    for ( d = 0; d < 6; ++d )
    {
        in[d] = mpi_data->in_nt_buffer[d];

        ret = MPI_Irecv( in[d], out_bufs[d].cnt, type,
                system->my_nt_nbrs[d].rank, d, comm, &req[d] );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }

    for ( d = 0; d < 6; ++d )
    {
        /* send both messages in direction d */
        ret = MPI_Send( Get_Buffer_Offset( buf, system->my_nt_nbrs[d].atoms_str, buf_type ),
                system->my_nt_nbrs[d].atoms_cnt, type,
                system->my_nt_nbrs[d].receive_rank, d, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }
    
    for ( d = 0; d < 6; ++d )
    {
        ret = MPI_Waitany( MAX_NT_NBRS, req, &index, stat);
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        unpack( in[index], buf, out_bufs[index].index, out_bufs[index].cnt );
    }

#else
    int d, cnt1, cnt2, ret;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2;
    MPI_Status stat1, stat2;
    const neighbor_proc *nbr1, *nbr2;
    coll_unpacker unpack;
    MPI_Aint extent, lower_bound;
    size_t type_size;

    ret = MPI_Type_get_extent( type, &lower_bound, &extent );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
//    type_size = MPI_Aint_add( lower_bound, extent );
    type_size = extent;

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;
    unpack = Get_Unpacker( buf_type );

    for ( d = 2; d >= 0; --d )
    {
        nbr1 = &system->my_nbrs[2 * d];
        nbr2 = &system->my_nbrs[2 * d + 1];
        
        /* send both messages in dimension d */
        if ( nbr1->rank != system->my_rank )
        {
            ret = MPI_Isend( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    nbr1->atoms_cnt, type, nbr1->rank, 2 * d, comm, &req1 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
    
        if ( nbr2->rank != system->my_rank )
        {
            ret = MPI_Isend( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    nbr2->atoms_cnt, type, nbr2->rank, 2 * d + 1, comm, &req2 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }

        /* recvs and unpack messages */
        if ( nbr1->rank != system->my_rank )
        {
            ret = MPI_Probe( nbr1->rank, 2 * d + 1, comm, &stat1 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            ret = MPI_Get_count( &stat1, type, &cnt1 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );

            if ( cnt1 == MPI_UNDEFINED )
            {
                fprintf( stderr, "[ERROR] Coll: MPI_Get_count returned MPI_UNDEFINED\n" );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }
            else if ( cnt1 != out_bufs[2 * d].cnt )
            {
                fprintf( stderr, "[ERROR] Coll: counts mismatch (MPI_Get_count = %d, out_bufs[%d].cnt = %d\n",
                      cnt1, 2 * d, out_bufs[2 * d].cnt );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }

            smalloc_check( &mpi_data->in1_buffer, &mpi_data->in1_buffer_size,
                    type_size * cnt1, TRUE, SAFE_ZONE, __FILE__, __LINE__ );

            ret = MPI_Recv( mpi_data->in1_buffer, cnt1,
                    type, nbr1->rank, 2 * d + 1, comm, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }

        if ( nbr2->rank != system->my_rank )
        {
            ret = MPI_Probe( nbr2->rank, 2 * d, comm, &stat2 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            ret = MPI_Get_count( &stat2, type, &cnt2 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );

            if ( cnt2 == MPI_UNDEFINED )
            {
                fprintf( stderr, "[ERROR] Coll: MPI_Get_count returned MPI_UNDEFINED\n" );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }
            else if ( cnt2 != out_bufs[2 * d + 1].cnt )
            {
                fprintf( stderr, "[ERROR] Coll: counts mismatch (MPI_Get_count = %d, out_bufs[%d].cnt = %d\n",
                      cnt2, 2 * d + 1, out_bufs[2 * d + 1].cnt );
                MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
            }

            smalloc_check( &mpi_data->in2_buffer, &mpi_data->in2_buffer_size,
                    type_size * cnt2, TRUE, SAFE_ZONE, __FILE__, __LINE__ );

            ret = MPI_Recv( mpi_data->in2_buffer, cnt2,
                    type, nbr2->rank, 2 * d, comm, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }

        if ( nbr1->rank != system->my_rank )
        {
            ret = MPI_Wait( &req1, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        if ( nbr2->rank != system->my_rank )
        {
            ret = MPI_Wait( &req2, MPI_STATUS_IGNORE );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }

        if ( nbr1->rank != system->my_rank )
        {
            unpack( mpi_data->in1_buffer, buf, out_bufs[2 * d].index,
                    out_bufs[2 * d].cnt );
        }
        else
        {
            unpack( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                    buf, out_bufs[2 * d + 1].index, out_bufs[2 * d + 1].cnt );
        }
        if ( nbr2->rank != system->my_rank )
        {
            unpack( mpi_data->in2_buffer, buf, out_bufs[2 * d + 1].index,
                    out_bufs[2 * d + 1].cnt );
        }
        else
        {
            unpack( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                    buf, out_bufs[2 * d].index, out_bufs[2 * d].cnt );
        }
    }
#endif
}


real Parallel_Dot( real const * const v1, real const * const v2,
        const int n, MPI_Comm comm )
{
    int i, ret;
    real dot_l, dot;

    dot_l = 0.0;

    /* compute local part of inner product */
    for ( i = 0; i < n; ++i )
    {
        dot_l += v1[i] * v2[i];
    }

    ret = MPI_Allreduce( &dot_l, &dot, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    return dot;
}


/*****************************************************************************/
#if defined(TEST_FORCES)
void Coll_ids_at_Master( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data )
{
    int i, *id_list, ret;

    ret = MPI_Gather( &system->n, 1, MPI_INT, workspace->rcounts, 1, MPI_INT,
            MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        workspace->displs[0] = 0;
        for ( i = 1; i < system->nprocs; ++i )
        {
            workspace->displs[i] = workspace->displs[i - 1] + workspace->rcounts[i - 1];
        }
    }

    id_list = smalloc( system->n * sizeof(int), __FILE__, __LINE__ );
    for ( i = 0; i < system->n; ++i )
    {
        id_list[i] = system->my_atoms[i].orig_id;
    }

    ret = MPI_Gatherv( id_list, system->n, MPI_INT, workspace->id_all,
            workspace->rcounts, workspace->displs, MPI_INT,
            MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    sfree( id_list, __FILE__, __LINE__ );

#if defined(DEBUG_FOCUS)
    if ( system->my_rank == MASTER_NODE )
    {
        for ( i = 0 ; i < system->bigN; ++i )
        {
            fprintf( stderr, "id_all[%d]: %d\n", i, workspace->id_all[i] );
        }
    }
#endif
}


void Coll_rvecs_at_Master( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data, rvec* v )
{
    int ret;

    ret = MPI_Gatherv( v, system->n, mpi_data->mpi_rvec, workspace->f_all,
            workspace->rcounts, workspace->displs, mpi_data->mpi_rvec,
            MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
}

#endif
