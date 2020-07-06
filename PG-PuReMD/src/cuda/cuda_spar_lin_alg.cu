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

#include "cuda_spar_lin_alg.h"

#include "cuda_dense_lin_alg.h"
#include "cuda_helpers.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "../basic_comm.h"
#include "../comm_tools.h"
#include "../tool_box.h"


/* mask used to determine which threads within a warp participate in operations */
#define FULL_MASK (0xFFFFFFFF)


/* Jacobi preconditioner computation */
CUDA_GLOBAL void k_jacobi_cm_half( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        real * const Hdia_inv, int N )
{
    int i;
    real diag;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    if ( FABS( vals[row_ptr_end[i]] ) >= 1.0e-12 )
    {
        diag = 1.0 / vals[row_ptr_end[i]];
    }
    else
    {
        diag = 1.0;
    }

    Hdia_inv[i] = diag;
}


/* Jacobi preconditioner computation */
CUDA_GLOBAL void k_jacobi_cm_full( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        real * const Hdia_inv, int N )
{
    int i, pj;
    real diag;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    for ( pj = row_ptr_start[i]; pj < row_ptr_end[i]; ++pj )
    {
        if ( col_ind[pj] == i )
        {
            if ( FABS( vals[pj] ) >= 1.0e-12 )
            {
                diag = 1.0 / vals[pj];
            }
            else
            {
                diag = 1.0;
            }

            break;
        }
    }

    __syncthreads( );

    Hdia_inv[i] = diag;
}


CUDA_GLOBAL void k_dual_jacobi_apply( real const * const Hdia_inv, rvec2 const * const y,
        rvec2 * const x, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    x[i][0] = y[i][0] * Hdia_inv[i];
    x[i][1] = y[i][1] * Hdia_inv[i];
}


CUDA_GLOBAL void k_jacobi_apply( real const * const Hdia_inv, real const * const y,
        real * const x, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    x[i] = Hdia_inv[i] * y[i];
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where one GPU thread multiplies a row
 *
 * A: symmetric (lower triangular portion only stored), square matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
CUDA_GLOBAL void k_sparse_matvec_half_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int N )
{
    int i, pj, si, ei;
    real b_local;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    si = row_ptr_start[i];
    ei = row_ptr_end[i];
    b_local = 0.0;

    for ( pj = si; pj < ei; ++pj )
    {
        b_local += vals[pj] * x[col_ind[pj]];
        /* symmetric contribution (A is symmetric, lower half stored) */
        atomicAdd( (double*) &b[col_ind[pj]], (double) (vals[pj] * x[i]) );
    }

    /* diagonal entry */
    b_local += vals[ei] * x[i];

    /* local contribution to row i for this thread */
    atomicAdd( (double*) &b[i], (double) b_local );
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where one GPU thread multiplies a row
 *
 * A: symmetric (lower triangular portion only stored), square matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
CUDA_GLOBAL void k_sparse_matvec_full_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int n )
{
    int i, pj, si, ei;
    real sum;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    sum = 0.0;
    si = row_ptr_start[i];
    ei = row_ptr_end[i];

    for ( pj = si; pj < ei; ++pj )
    {
        sum += vals[pj] * x[col_ind[pj]];
    }

    __syncthreads( );

    b[i] = sum;
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where warps of 32 threads
 * collaborate to multiply each row
 *
 * A: symmetric, full, square matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
CUDA_GLOBAL void k_sparse_matvec_full_opt_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int n )
{
    int i, pj, thread_id, warp_id, lane_id, offset;
    int si, ei;
    unsigned int mask;
    real sum;

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 
    mask = __ballot_sync( FULL_MASK, warp_id < n );

    if ( warp_id < n )
    {
        i = warp_id;
        si = row_ptr_start[i];
        ei = row_ptr_end[i];
        sum = 0.0;

        /* partial sums per thread */
        for ( pj = si + lane_id; pj < ei; pj += warpSize )
        {
            sum += vals[pj] * x[ col_ind[pj] ];
        }

        /* warp-level reduction of partial sums
         * using registers within a warp */
        for ( offset = warpSize >> 1; offset > 0; offset /= 2 )
        {
            sum += __shfl_down_sync( mask, sum, offset );
        }

        /* first thread within a warp writes sum to global memory */
        if ( lane_id == 0 )
        {
            b[i] = sum;
        }
    }
}


/* 1 thread per row implementation */
CUDA_GLOBAL void k_dual_sparse_matvec_full_csr( sparse_matrix A,
        rvec2 const * const x, rvec2 * const b, int n )
{
    int i, pj, si, ei;
    rvec2 sum;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    sum[0] = 0.0;
    sum[1] = 0.0;
    si = A.start[i];
    ei = A.end[i];

    for ( pj = si; pj < ei; ++pj )
    {
        sum[0] += A.val[pj] * x[ A.j[pj] ][0];
        sum[1] += A.val[pj] * x[ A.j[pj] ][1];
    }

    b[i][0] = sum[0];
    b[i][1] = sum[1];
}


/* sparse matrix, dense vector multiplication AX = B,
 * where warps of 32 threads
 * collaborate to multiply each row
 *
 * A: symmetric, full, square matrix,
 *    stored in CSR format
 * X: 2 dense vectors, size equal to num. columns in A
 * B (output): 2 dense vectors, size equal to num. columns in A
 * N: number of rows in A */
CUDA_GLOBAL void k_dual_sparse_matvec_full_opt_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        rvec2 const * const x, rvec2 * const b, int n )
{
    int i, pj, thread_id, warp_id, lane_id, offset;
    int si, ei;
    unsigned int mask;
    rvec2 sum;

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 
    mask = __ballot_sync( FULL_MASK, warp_id < n );

    if ( warp_id < n )
    {
        i = warp_id;
        si = row_ptr_start[i];
        ei = row_ptr_end[i];
        sum[0] = 0.0;
        sum[1] = 0.0;

        /* partial sums per thread */
        for ( pj = si + lane_id; pj < ei; pj += warpSize )
        {
            sum[0] += vals[pj] * x[ col_ind[pj] ][0];
            sum[1] += vals[pj] * x[ col_ind[pj] ][1];
        }

        /* warp-level reduction of partial sums
         * using registers within a warp */
        for ( offset = warpSize >> 1; offset > 0; offset /= 2 )
        {
            sum[0] += __shfl_down_sync( mask, sum[0], offset );
            sum[1] += __shfl_down_sync( mask, sum[1], offset );
        }

        /* first thread within a warp writes sum to global memory */
        if ( lane_id == 0 )
        {
            b[i][0] = sum[0];
            b[i][1] = sum[1];
        }
    }
}


void dual_jacobi_apply( real const * const Hdia_inv, rvec2 const * const y,
        rvec2 * const x, int n )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_dual_jacobi_apply <<< blocks, DEF_BLOCK_SIZE >>>
        ( Hdia_inv, y, x, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


void jacobi_apply( real const * const Hdia_inv, real const * const y,
        real * const x, int n )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_jacobi_apply <<< blocks, DEF_BLOCK_SIZE >>>
        ( Hdia_inv, y, x, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


/* Communications for sparse matrix-dense vector multiplication AX = B
 *
 * system:
 * control: 
 * mpi_data:
 * x: dense vector (device)
 * n: number of entries in x
 * buf_type: data structure type for x
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static void Dual_Sparse_MatVec_Comm_Part1( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, void const * const x, int n,
        int buf_type, MPI_Datatype mpi_type )
{
    rvec2 *spad;

//#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
//    Dist( system, mpi_data, x, buf_type, mpi_type );
//#else

    check_smalloc( &workspace->host_scratch, &workspace->host_scratch_size,
            sizeof(rvec2) * n, TRUE, SAFE_ZONE,
            "Dual_Sparse_MatVec_Comm_Part1::workspace->host_scratch" );
    spad = (rvec2 *) workspace->host_scratch;
    copy_host_device( spad, (void *) x, sizeof(rvec2) * n,
            cudaMemcpyDeviceToHost, "Dual_Sparse_MatVec_Comm_Part1::x" );

    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, spad, buf_type, mpi_type );

    copy_host_device( spad, (void *) x, sizeof(rvec2) * n,
            cudaMemcpyHostToDevice, "Dual_Sparse_MatVec_Comm_Part1::x" );
//#endif
}


/* Local arithmetic portion of sparse matrix-dense vector multiplication AX = B
 *
 * control:
 * A: sparse matrix, 1D partitioned row-wise
 * x: dense vector
 * b (output): dense vector
 * n: number of entries in b
 */
static void Dual_Sparse_MatVec_local( control_params const * const control,
        sparse_matrix const * const A, rvec2 const * const x,
        rvec2 * const b, int n )
{
    int blocks;

    if ( A->format == SYM_HALF_MATRIX )
    {
        /* half-format requires entries of b be initialized to zero */
        cuda_memset( b, 0, sizeof(rvec2) * n, "Dual_Sparse_MatVec_local::b" );

        //TODO: implement half-format dual SpMV
//        blocks = n * 32 / DEF_BLOCK_SIZE + 
//            (n * 32 % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        /* 1 thread per row implementation */
//        k_dual_sparse_matvec_half_csr <<< control->blocks, control->block_size >>>
//            ( A->start, A->end, A->j, A->val, x, b, n );

        /* multiple threads per row implementation,
         * with shared memory to accumulate partial row sums */
//        k_dual_sparse_matvec_half_opt_csr <<< blocks, DEF_BLOCK_SIZE >>>
//             ( A->start, A->end, A->j, A->val, x, b, n );
        cudaDeviceSynchronize( );
        cudaCheckError( );
    }
    else if ( A->format == SYM_FULL_MATRIX )
    {
        /* 1 thread per row implementation */
//        k_dual_sparse_matvec_full_csr <<< control->blocks_n, control->blocks_size_n >>>
//             ( *A, x, b, n );

        blocks = (A->n * 32 / DEF_BLOCK_SIZE)
            + ((A->n * 32 % DEF_BLOCK_SIZE) == 0 ? 0 : 1);
        
        /* multiple threads per row implementation
         * using registers to accumulate partial row sums */
        k_dual_sparse_matvec_full_opt_csr <<< blocks, DEF_BLOCK_SIZE >>>
                ( A->start, A->end, A->j, A->val, x, b, n );
        cudaDeviceSynchronize( );
        cudaCheckError( );
    }
}


/* Communications for collecting the distributed partial sums
 * in the sparse matrix-dense vector multiplication AX = B.
 * Specifically, B contains the distributed partial sums
 * (and hence has the same number of entries as X).
 *
 * system:
 * control:
 * mpi_data:
 * mat_format: storage type of sparse matrix A
 * b: dense vector (device)
 * n1: number of entries in x
 * n2: number of entries in b (at output)
 * buf_type: data structure type for b
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static void Dual_Sparse_MatVec_Comm_Part2( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, int mat_format,
        void * const b, int n1, int n2, int buf_type, MPI_Datatype mpi_type )
{
    rvec2 *spad;

    /* reduction required for symmetric half matrix */
//    if ( mat_format == SYM_HALF_MATRIX )
    {
//#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//        Coll( system, mpi_data, b, buf_type, mpi_type );
//#else

        check_smalloc( &workspace->host_scratch, &workspace->host_scratch_size,
                sizeof(rvec2) * n1, TRUE, SAFE_ZONE,
                "Dual_Sparse_MatVec_Comm_Part2::workspace->host_scratch" );
        spad = (rvec2 *) workspace->host_scratch;
        copy_host_device( spad, b, sizeof(rvec2) * n1,
                cudaMemcpyDeviceToHost, "Dual_Sparse_MatVec_Comm_Part2::b" );

        Coll( system, mpi_data, spad, buf_type, mpi_type );

        copy_host_device( spad, b, sizeof(rvec2) * n2,
                cudaMemcpyHostToDevice, "Dual_Sparse_MatVec_Comm_Part2::b" );
//#endif
    }
}


/* sparse matrix, dense vector multiplication AX = B
 *
 * system:
 * control:
 * data:
 * workspace: storage container for workspace structures
 * A: symmetric (lower triangular portion only stored), square matrix,
 *    stored in CSR format
 * X: dense vector, size equal to num. columns in A
 * n: number of rows in X
 * B (output): dense vector */
static void Dual_Sparse_MatVec( const reax_system * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data,
        sparse_matrix const * const A, rvec2 * const x,
        int n, rvec2 * const b )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Dual_Sparse_MatVec_Comm_Part1( system, control, workspace, mpi_data,
            x, n, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif

    Dual_Sparse_MatVec_local( control, A, x, b, n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_spmv );
#endif

    Dual_Sparse_MatVec_Comm_Part2( system, control, workspace, mpi_data,
            A->format, b, n, A->n, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif
}


/* Communications for sparse matrix-dense vector multiplication Ax = b
 *
 * system:
 * control: 
 * mpi_data:
 * x: dense vector (device)
 * n: number of entries in x
 * buf_type: data structure type for x
 * mpi_type: MPI_Datatype struct for communications
 */
static void Sparse_MatVec_Comm_Part1( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, void const * const x, int n,
        int buf_type, MPI_Datatype mpi_type )
{
    real *spad;

//#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
//    Dist( system, mpi_data, x, buf_type, mpi_type );
//#else

    check_smalloc( &workspace->host_scratch, &workspace->host_scratch_size,
            sizeof(real) * n, TRUE, SAFE_ZONE,
            "Sparse_MatVec_Comm_Part1::workspace->host_scratch" );
    spad = (real *) workspace->host_scratch;
    copy_host_device( spad, (void *) x, sizeof(real) * n,
            cudaMemcpyDeviceToHost, "Sparse_MatVec_Comm_Part1::x" );

    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, spad, buf_type, mpi_type );

    copy_host_device( spad, (void *) x, sizeof(real) * n,
            cudaMemcpyHostToDevice, "Sparse_MatVec_Comm_Part1::x" );
//#endif
}


/* Local arithmetic portion of sparse matrix-dense vector multiplication Ax = b
 *
 * control:
 * A: sparse matrix, 1D partitioned row-wise
 * x: dense vector
 * b (output): dense vector
 * n: number of entries in b
 */
static void Sparse_MatVec_local( control_params const * const control,
        sparse_matrix const * const A, real const * const x,
        real * const b, int n )
{
    int blocks;

    if ( A->format == SYM_HALF_MATRIX )
    {
//        blocks = (A->n * 32 / DEF_BLOCK_SIZE)
//            + ((A->n * 32 % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

        /* half-format requires entries of b be initialized to zero */
        cuda_memset( b, 0, sizeof(real) * n, "Sparse_MatVec_local::b" );

        /* 1 thread per row implementation */
        k_sparse_matvec_half_csr <<< control->blocks, control->block_size >>>
            ( A->start, A->end, A->j, A->val, x, b, n );

        /* multiple threads per row implementation,
         * with shared memory to accumulate partial row sums */
//        k_sparse_matvec_half_opt_csr <<< blocks, DEF_BLOCK_SIZE >>>
//             ( A->start, A->end, A->j, A->val, x, b, n );
        cudaDeviceSynchronize( );
        cudaCheckError( );
    }
    else if ( A->format == SYM_FULL_MATRIX )
    {
        /* 1 thread per row implementation */
//        k_sparse_matvec_full_csr <<< control->blocks, control->blocks_size >>>
//             ( A->start, A->end, A->j, A->val, x, b, n );

        blocks = (A->n * 32 / DEF_BLOCK_SIZE)
            + ((A->n * 32 % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

        /* multiple threads per row implementation
         * using registers to accumulate partial row sums */
        k_sparse_matvec_full_opt_csr <<< blocks, DEF_BLOCK_SIZE >>>
             ( A->start, A->end, A->j, A->val, x, b, n );
        cudaDeviceSynchronize( );
        cudaCheckError( );
    }
}


/* Communications for collecting the distributed partial sums
 * in the sparse matrix-dense vector multiplication Ax = b.
 * Specifically, b contains the distributed partial sums
 * (and hence has the same number of entries as x).
 *
 * system:
 * control:
 * mpi_data:
 * mat_format: storage type of sparse matrix A
 * b: dense vector (device)
 * n1: number of entries in x
 * n2: number of entries in b (at output)
 * buf_type: data structure type for b
 * mpi_type: MPI_Datatype struct for communications
 */
static void Sparse_MatVec_Comm_Part2( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, int mat_format,
        void * const b, int n1, int n2, int buf_type, MPI_Datatype mpi_type )
{
    real *spad;

    /* reduction required for symmetric half matrix */
//    if ( mat_format == SYM_HALF_MATRIX )
    {
//#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//        Coll( system, mpi_data, b, buf_type, mpi_type );
//#else

        check_smalloc( &workspace->host_scratch, &workspace->host_scratch_size,
                sizeof(real) * n1, TRUE, SAFE_ZONE,
                "Sparse_MatVec_Comm_Part2::workspace->host_scratch" );
        spad = (real *) workspace->host_scratch;
        copy_host_device( spad, b, sizeof(real) * n1,
                cudaMemcpyDeviceToHost, "Sparse_MatVec_Comm_Part2::b" );

        Coll( system, mpi_data, spad, buf_type, mpi_type );

        copy_host_device( spad, b, sizeof(real) * n2,
                cudaMemcpyHostToDevice, "Sparse_MatVec_Comm_Part2::b" );
//#endif
    }
}


/* sparse matrix, dense vector multiplication Ax = b
 *
 * system:
 * control:
 * data:
 * workspace: storage container for workspace structures
 * A: symmetric (lower triangular portion only stored), square matrix,
 *    stored in CSR format
 * x: dense vector
 * n: number of entries in x
 * b (output): dense vector */
static void Sparse_MatVec( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data,
        sparse_matrix const * const A, real const * const x,
        int n, real * const b )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Sparse_MatVec_Comm_Part1( system, control, workspace, mpi_data,
            x, n, REAL_PTR_TYPE, MPI_DOUBLE );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif

    Sparse_MatVec_local( control, A, x, b, n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_spmv );
#endif

    Sparse_MatVec_Comm_Part2( system, control, workspace, mpi_data,
            A->format, b, n, A->n, REAL_PTR_TYPE, MPI_DOUBLE );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif
}


int Cuda_dual_CG( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, rvec2 const * const b, real tol,
        rvec2 * const x, mpi_datatypes * const mpi_data, FILE *fout )
{
    unsigned int i, matvecs;
    int ret;
    rvec2 tmp, alpha, beta, r_norm, b_norm, sig_old, sig_new;
    real redux[6];
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    matvecs = 0;

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->q2 );

    Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->d_workspace->q2, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r2,
            workspace->d_workspace->d2, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Dot_local_rvec2( control, workspace, workspace->d_workspace->r2,
            workspace->d_workspace->d2, system->n, &redux[0], &redux[1] );
    Dot_local_rvec2( control, workspace, workspace->d_workspace->d2,
            workspace->d_workspace->d2, system->n, &redux[2], &redux[3] );
    Dot_local_rvec2( control, workspace, b, b, system->n, &redux[4], &redux[5] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    sig_new[0] = redux[0];
    sig_new[1] = redux[1];
    r_norm[0] = SQRT( redux[2] );
    r_norm[1] = SQRT( redux[3] );
    b_norm[0] = SQRT( redux[4] );
    b_norm[1] = SQRT( redux[5] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d2, system->N, workspace->d_workspace->q2 );

        Dot_local_rvec2( control, workspace, workspace->d_workspace->d2,
                workspace->d_workspace->q2, system->n, &redux[0], &redux[1] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        tmp[0] = redux[0];
        tmp[1] = redux[1];

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        Vector_Add_rvec2( x, alpha[0], alpha[1],
                workspace->d_workspace->d2, system->n );
        Vector_Add_rvec2( workspace->d_workspace->r2, -1.0 * alpha[0], -1.0 * alpha[1],
                workspace->d_workspace->q2, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r2,
                workspace->d_workspace->p2, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Dot_local_rvec2( control, workspace, workspace->d_workspace->r2,
                workspace->d_workspace->p2, system->n, &redux[0], &redux[1] );
        Dot_local_rvec2( control, workspace, workspace->d_workspace->p2,
                workspace->d_workspace->p2, system->n, &redux[2], &redux[3] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        sig_new[0] = redux[0];
        sig_new[1] = redux[1];
        r_norm[0] = SQRT( redux[2] );
        r_norm[1] = SQRT( redux[3] );
        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        /* d = p + beta * d */
        Vector_Sum_rvec2( workspace->d_workspace->d2,
                1.0, 1.0, workspace->d_workspace->p2,
                beta[0], beta[1], workspace->d_workspace->d2, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( r_norm[0] / b_norm[0] <= tol
            && r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n );

        matvecs = Cuda_CG( system, control, data, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n );
    }
    else if ( r_norm[1] / b_norm[1] <= tol
            && r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n );

        matvecs = Cuda_CG( system, control, data, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n );
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual CG convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error for s solve: %e\n", r_norm[0] / b_norm[0] );
        fprintf( stderr, "    [INFO] Rel. residual error for t solve: %e\n", r_norm[1] / b_norm[1] );
    }

    return (i + 1) + matvecs;
}


/* Preconditioned Conjugate Gradient Method */
int Cuda_CG( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data )
{
    unsigned int i;
    int ret;
    real tmp, alpha, beta, r_norm, b_norm;
    real sig_old, sig_new;
    real redux[3];
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->q );

    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r,
            workspace->d_workspace->d, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    redux[0] = Dot_local( workspace, workspace->d_workspace->r,
            workspace->d_workspace->d, system->n );
    redux[1] = Dot_local( workspace, workspace->d_workspace->d,
            workspace->d_workspace->d, system->n );
    redux[2] = Dot_local( workspace, b, b, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    sig_new = redux[0];
    r_norm = SQRT( redux[1] );
    b_norm = SQRT( redux[2] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d, system->N, workspace->d_workspace->q );

        tmp = Dot( workspace, workspace->d_workspace->d, workspace->d_workspace->q,
                system->n, MPI_COMM_WORLD );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d_workspace->d, system->n );
        Vector_Add( workspace->d_workspace->r, -1.0 * alpha,
                workspace->d_workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r,
                workspace->d_workspace->p, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        redux[0] = Dot_local( workspace, workspace->d_workspace->r,
                workspace->d_workspace->p, system->n );
        redux[1] = Dot_local( workspace, workspace->d_workspace->p,
                workspace->d_workspace->p, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig_old = sig_new;
        sig_new = redux[0];
        r_norm = SQRT( redux[1] );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d_workspace->d, 1.0, workspace->d_workspace->p,
                beta, workspace->d_workspace->d, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( i >= control->cm_solver_max_iters
            && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] CG convergence failed (%d iters)\n", i );
        fprintf( stderr, "    [INFO] Rel. residual error: %e\n", r_norm / b_norm );
    }

    return i;
}


/* Bi-conjugate gradient stabalized method with left preconditioning for
 * solving nonsymmetric linear systems
 *
 * system: 
 * workspace: struct containing storage for workspace for the linear solver
 * control: struct containing parameters governing the simulation and numeric methods
 * data: struct containing simulation data (e.g., atom info)
 * H: sparse, symmetric matrix, lower half stored in CSR format
 * b: right-hand side of the linear system
 * tol: tolerence compared against the relative residual for determining convergence
 * x: inital guess
 * mpi_data: 
 *
 * Reference: Netlib (in MATLAB)
 *  http://www.netlib.org/templates/matlab/bicgstab.m
 * */
int Cuda_BiCGStab( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data )
{
    unsigned int i;
    int ret;
    real tmp, alpha, beta, omega, sigma, rho, rho_old, r_norm, b_norm;
    real redux[2];
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->d );

    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->d, system->n );
    redux[0] = Dot_local( workspace, b, b, system->n );
    redux[1] = Dot_local( workspace, workspace->d_workspace->r,
            workspace->d_workspace->r, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    b_norm = SQRT( redux[0] );
    r_norm = SQRT( redux[1] );
    if ( b_norm == 0.0 )
    {
        b_norm = 1.0;
    }
    Vector_Copy( workspace->d_workspace->r_hat,
            workspace->d_workspace->r, system->n );
    omega = 1.0;
    rho = 1.0;

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        redux[0] = Dot_local( workspace, workspace->d_workspace->r_hat,
                workspace->d_workspace->r, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        rho = redux[0];
        if ( rho == 0.0 )
        {
            break;
        }
        if ( i > 0 )
        {
            beta = (rho / rho_old) * (alpha / omega);
            Vector_Sum( workspace->d_workspace->q,
                    1.0, workspace->d_workspace->p,
                    -1.0 * omega, workspace->d_workspace->z, system->n );
            Vector_Sum( workspace->d_workspace->p,
                    1.0, workspace->d_workspace->r,
                    beta, workspace->d_workspace->q, system->n );
        }
        else
        {
            Vector_Copy( workspace->d_workspace->p,
                    workspace->d_workspace->r, system->n );
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->p,
                workspace->d_workspace->d, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d, system->N, workspace->d_workspace->z );

        redux[0] = Dot_local( workspace, workspace->d_workspace->r_hat,
                workspace->d_workspace->z, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp = redux[0];
        alpha = rho / tmp;
        Vector_Sum( workspace->d_workspace->q,
                1.0, workspace->d_workspace->r,
                -1.0 * alpha, workspace->d_workspace->z, system->n );
        redux[0] = Dot_local( workspace, workspace->d_workspace->q,
                workspace->d_workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp = redux[0];
        /* early convergence check */
        if ( tmp < tol )
        {
            Vector_Add( x, alpha, workspace->d_workspace->d, system->n );
            break;
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->q,
                workspace->d_workspace->q_hat, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->q_hat, system->N, workspace->d_workspace->y );

        redux[0] = Dot_local( workspace, workspace->d_workspace->y,
                workspace->d_workspace->q, system->n );
        redux[1] = Dot_local( workspace, workspace->d_workspace->y,
                workspace->d_workspace->y, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sigma = redux[0];
        tmp = redux[1];
        omega = sigma / tmp;
        Vector_Sum( workspace->d_workspace->g,
                alpha, workspace->d_workspace->d,
                omega, workspace->d_workspace->q_hat, system->n );
        Vector_Add( x, 1.0, workspace->d_workspace->g, system->n );
        Vector_Sum( workspace->d_workspace->r,
                1.0, workspace->d_workspace->q,
                -1.0 * omega, workspace->d_workspace->y, system->n );
        redux[0] = Dot_local( workspace, workspace->d_workspace->r,
                workspace->d_workspace->r, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        r_norm = SQRT( redux[0] );
        if ( omega == 0.0 )
        {
            break;
        }
        rho_old = rho;

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( i >= control->cm_solver_max_iters
            && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] CG convergence failed (%d iters)\n", i );
        fprintf( stderr, "    [INFO] Rel. residual error: %e\n", r_norm / b_norm );
    }

    return i;
}
