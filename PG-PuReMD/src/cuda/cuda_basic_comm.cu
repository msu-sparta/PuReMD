
#include "cuda_basic_comm.h"

#include "cuda_utils.h"

#include "../comm_tools.h"
#include "../tool_box.h"
#include "../vector.h"


typedef void (*dist_packer)( void const * const, mpi_out_data * const );
typedef void (*coll_unpacker)( void const * const, void * const,
        mpi_out_data * const );


/* copy integer entries from buffer to MPI egress buffer
 *
 * arguments:
 *  dummy: buffer containing data to be copied
 *  index: indices for buffer to be copied into the MPI egress buffer
 *  out_atoms: MPI egress buffer
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_int_packer( int const * const dummy, int const * const index,
        int * const out_atoms, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    out_atoms[i] = dummy[ index[i] ];
}


/* copy double precision entries from buffer to MPI egress buffer
 *
 * arguments:
 *  dummy: buffer containing data to be copied
 *  index: indices for buffer to be copied into the MPI egress buffer
 *  out_atoms: MPI egress buffer
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_real_packer( real const * const dummy, int const * const index,
        real * const out_atoms, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    out_atoms[i] = dummy[ index[i] ];
}


/* copy rvec entries from buffer to MPI egress buffer
 *
 * arguments:
 *  dummy: buffer containing data to be copied
 *  index: indices for buffer to be copied into the MPI egress buffer
 *  out_atoms: MPI egress buffer
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_rvec_packer( rvec const * const dummy, int const * const index,
        rvec * const out_atoms, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    rvec_Copy( out_atoms[i], dummy[ index[i] ] );
}


/* copy rvec2 entries from buffer to MPI egress buffer
 *
 * arguments:
 *  dummy: buffer containing data to be copied
 *  index: indices for buffer to be copied into the MPI egress buffer
 *  out_atoms: MPI egress buffer
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_rvec2_packer( rvec2 const * const dummy, int const * const index,
        rvec2 * const out_atoms, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    out_atoms[i][0] = dummy[ index[i] ][0];
    out_atoms[i][1] = dummy[ index[i] ][1];
}


/* copy integer entries from MPI ingress buffer to buffer
 *
 * arguments:
 *  in: MPI ingress buffer containing data to be copied
 *  index: indices for buffer to be copied into
 *  buf: buffer to be copied into
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_int_unpacker( int const * const in, int const * const index,
        int * const buf, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    //TODO: used in SAI, purpose?
    if ( buf[ index[i] ] == -1 && in[i] != -1 )
    {
        buf[ index[i] ] = in[i];
    }
}


/* copy double precision entries from MPI ingress buffer to buffer
 *
 * arguments:
 *  in: MPI ingress buffer containing data to be copied
 *  index: indices for buffer to be copied into
 *  buf: buffer to be copied into
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_real_unpacker( real const * const in, int const * const index,
        real * const buf, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    buf[ index[i] ] = in[i];
}


/* copy rvec entries from MPI ingress buffer to buffer
 *
 * arguments:
 *  in: MPI ingress buffer containing data to be copied
 *  index: indices for buffer to be copied into
 *  buf: buffer to be copied into
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_rvec_unpacker( rvec const * const in, int const * const index,
        rvec * const buf, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    rvec_Add( buf[ index[i] ], in[i] );
}


/* copy rvec2 entries from MPI ingress buffer to buffer
 *
 * arguments:
 *  in: MPI ingress buffer containing data to be copied
 *  index: indices for buffer to be copied into
 *  buf: buffer to be copied into
 *  k: number of entries in buffer
 */
CUDA_GLOBAL void k_rvec2_unpacker( rvec2 const * const in, int const * const index,
        rvec2 * const buf, int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    buf[ index[i] ][0] += in[i][0];
    buf[ index[i] ][1] += in[i][1];
}


static void int_packer( void const * const dummy, mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_int_packer <<< blocks, DEF_BLOCK_SIZE >>>
        ( (int *) dummy, out_buf->index, (int *) out_buf->out_atoms, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void real_packer( void const * const dummy, mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_real_packer <<< blocks, DEF_BLOCK_SIZE >>>
        ( (real *) dummy, out_buf->index, (real *) out_buf->out_atoms, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void rvec_packer( void const * const dummy, mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec_packer <<< blocks, DEF_BLOCK_SIZE >>>
        ( (rvec *) dummy, out_buf->index, (rvec *) out_buf->out_atoms, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void rvec2_packer( void const * const dummy, mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec2_packer <<< blocks, DEF_BLOCK_SIZE >>>
        ( (rvec2 *) dummy, out_buf->index, (rvec2 *) out_buf->out_atoms, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void int_unpacker( void const * const dummy_in, void * const dummy_buf,
        mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_int_unpacker <<< blocks, DEF_BLOCK_SIZE >>>
        ( (int *) dummy_in, out_buf->index, (int *) dummy_buf, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void real_unpacker( void const * const dummy_in, void * const dummy_buf,
        mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_real_unpacker <<< blocks, DEF_BLOCK_SIZE >>>
        ( (real *) dummy_in, out_buf->index, (real *) dummy_buf, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void rvec_unpacker( void const * const dummy_in, void * const dummy_buf,
        mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec_unpacker <<< blocks, DEF_BLOCK_SIZE >>>
        ( (rvec *) dummy_in, out_buf->index, (rvec *) dummy_buf, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void rvec2_unpacker( void const * const dummy_in, void * const dummy_buf,
        mpi_out_data * const out_buf )
{
    int blocks;

    blocks = (out_buf->cnt / DEF_BLOCK_SIZE)
        + ((out_buf->cnt % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec2_unpacker <<< blocks, DEF_BLOCK_SIZE >>>
        ( (rvec2 *) dummy_in, out_buf->index, (rvec2 *) dummy_buf, out_buf->cnt );
    /* explicitly wait for kernel compilation as MPI calls need buffer prepared */
    cudaDeviceSynchronize( );
    cudaCheckError( );
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


void Cuda_Dist( reax_system const * const system, mpi_datatypes * const mpi_data,
        void const * const buf, int buf_type, MPI_Datatype type )
{
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
    type_size = MPI_Aint_add( lower_bound, extent );

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->d_out_buffers;
    pack = Get_Packer( buf_type );

    for ( d = 0; d < 3; ++d )
    {
        nbr1 = &system->my_nbrs[2 * d];
        nbr2 = &system->my_nbrs[2 * d + 1];

        /* pack MPI buffers and initiate sends */
        sCudaCheckMalloc( &out_bufs[2 * d].out_atoms,
                &out_bufs[2 * d].out_atoms_size,
                type_size * out_bufs[2 * d].cnt, __FILE__, __LINE__ );
        sCudaCheckMalloc( (void **) &out_bufs[2 * d].index,
                &out_bufs[2 * d].index_size,
                sizeof(int) * out_bufs[2 * d].cnt, __FILE__, __LINE__ );

        pack( buf, &out_bufs[2 * d] );

        ret = MPI_Isend( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt,
                type, nbr1->rank, 2 * d, comm, &req1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        sCudaCheckMalloc( &out_bufs[2 * d + 1].out_atoms,
                &out_bufs[2 * d + 1].out_atoms_size,
                type_size * out_bufs[2 * d + 1].cnt, __FILE__, __LINE__ );
        sCudaCheckMalloc( (void **) &out_bufs[2 * d + 1].index,
                &out_bufs[2 * d + 1].index_size,
                sizeof(int) * out_bufs[2 * d + 1].cnt, __FILE__, __LINE__ );

        pack( buf, &out_bufs[2 * d + 1] );

        ret = MPI_Isend( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt,
                type, nbr2->rank, 2 * d + 1, comm, &req2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        /* recv both messages in dimension d */
        ret = MPI_Probe( nbr1->rank, 2 * d + 1, comm, &stat1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat1, type, &cnt1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        if ( cnt1 == MPI_UNDEFINED )
        {
            fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
            MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
        }

        ret = MPI_Recv( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                cnt1, type, nbr1->rank, 2 * d + 1, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        ret = MPI_Probe( nbr2->rank, 2 * d, comm, &stat2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat2, type, &cnt2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        if ( cnt2 == MPI_UNDEFINED )
        {
            fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
            MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
        }

        ret = MPI_Recv( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                cnt2, type, nbr2->rank, 2 * d, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        ret = MPI_Wait( &req1, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Wait( &req2, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }
}


void Cuda_Dist_FS( reax_system const * const system, mpi_datatypes * const mpi_data,
        void const * const buf, int buf_type, MPI_Datatype type )
{
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
    type_size = MPI_Aint_add( lower_bound, extent );

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->d_out_buffers;
    pack = Get_Packer( buf_type );

    for ( d = 0; d < 3; ++d )
    {
        nbr1 = &system->my_nbrs[2 * d];
        nbr2 = &system->my_nbrs[2 * d + 1];

        /* pack MPI buffers and initiate sends */
        sCudaCheckMalloc( &out_bufs[2 * d].out_atoms,
                &out_bufs[2 * d].out_atoms_size,
                type_size * out_bufs[2 * d].cnt, __FILE__, __LINE__ );
        sCudaCheckMalloc( (void **) &out_bufs[2 * d].index,
                &out_bufs[2 * d].index_size,
                sizeof(int) * out_bufs[2 * d].cnt, __FILE__, __LINE__ );

        pack( buf, &out_bufs[2 * d] );

        ret = MPI_Isend( out_bufs[2 * d].out_atoms, out_bufs[2 * d].cnt,
                type, nbr1->rank, 2 * d, comm, &req1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        sCudaCheckMalloc( &out_bufs[2 * d + 1].out_atoms,
                &out_bufs[2 * d + 1].out_atoms_size,
                type_size * out_bufs[2 * d + 1].cnt, __FILE__, __LINE__ );
        sCudaCheckMalloc( (void **) &out_bufs[2 * d + 1].index,
                &out_bufs[2 * d + 1].index_size,
                sizeof(int) * out_bufs[2 * d + 1].cnt, __FILE__, __LINE__ );

        pack( buf, &out_bufs[2 * d + 1] );

        ret = MPI_Isend( out_bufs[2 * d + 1].out_atoms, out_bufs[2 * d + 1].cnt,
                type, nbr2->rank, 2 * d + 1, comm, &req2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        /* recv both messages in dimension d */
        ret = MPI_Probe( nbr1->rank, 2 * d + 1, comm, &stat1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat1, type, &cnt1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        if ( cnt1 == MPI_UNDEFINED )
        {
            fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
            MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
        }

        ret = MPI_Recv( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                cnt1, type, nbr1->rank, 2 * d + 1, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        ret = MPI_Probe( nbr2->rank, 2 * d, comm, &stat2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat2, type, &cnt2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        if ( cnt2 == MPI_UNDEFINED )
        {
            fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
            MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
        }

        ret = MPI_Recv( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                cnt2, type, nbr2->rank, 2 * d, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        ret = MPI_Wait( &req1, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Wait( &req2, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }
}


void Cuda_Coll( reax_system const * const system, mpi_datatypes * const mpi_data,
        void * const buf, int buf_type, MPI_Datatype type )
{   
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
    type_size = MPI_Aint_add( lower_bound, extent );

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->d_out_buffers;
    unpack = Get_Unpacker( buf_type );

    for ( d = 2; d >= 0; --d )
    {
        nbr1 = &system->my_nbrs[2 * d];
        nbr2 = &system->my_nbrs[2 * d + 1];
        
        /* send both messages in dimension d */
        ret = MPI_Isend( Get_Buffer_Offset( buf, nbr1->atoms_str, buf_type ),
                nbr1->atoms_cnt, type, nbr1->rank, 2 * d, comm, &req1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    
        ret = MPI_Isend( Get_Buffer_Offset( buf, nbr2->atoms_str, buf_type ),
                nbr2->atoms_cnt, type, nbr2->rank, 2 * d + 1, comm, &req2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        /* recvs and unpack messages */
        ret = MPI_Probe( nbr1->rank, 2 * d + 1, comm, &stat1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat1, type, &cnt1 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        if ( cnt1 == MPI_UNDEFINED )
        {
            fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
            MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
        }

        sCudaCheckMalloc( &mpi_data->d_in1_buffer, &mpi_data->d_in1_buffer_size,
                type_size * cnt1, __FILE__, __LINE__ );

        ret = MPI_Recv( mpi_data->d_in1_buffer, cnt1,
                type, nbr1->rank, 2 * d + 1, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        ret = MPI_Probe( nbr2->rank, 2 * d, comm, &stat2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Get_count( &stat2, type, &cnt2 );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        if ( cnt2 == MPI_UNDEFINED )
        {
            fprintf( stderr, "[ERROR] MPI_Get_count returned MPI_UNDEFINED\n" );
            MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
        }

        sCudaCheckMalloc( &mpi_data->d_in2_buffer, &mpi_data->d_in2_buffer_size,
                type_size * cnt2, __FILE__, __LINE__ );

        ret = MPI_Recv( mpi_data->d_in2_buffer, cnt2,
                type, nbr2->rank, 2 * d, comm, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        ret = MPI_Wait( &req1, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Wait( &req2, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        unpack( mpi_data->d_in1_buffer, buf, &out_bufs[2 * d] );
        unpack( mpi_data->d_in2_buffer, buf, &out_bufs[2 * d + 1] );
    }
}
