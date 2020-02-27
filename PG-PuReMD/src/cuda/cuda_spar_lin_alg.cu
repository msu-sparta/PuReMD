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
#include "cuda_shuffle.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "../basic_comm.h"
#include "../tool_box.h"


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
    int i, k, si, ei;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    si = row_ptr_start[i];
    ei = row_ptr_end[i];

    for ( k = si; k < ei; ++k )
    {
        myatomicAdd( (double*) &b[i], (double) (vals[k] * x[col_ind[k]]) );
        /* symmetric contribution (A is symmetric, lower half stored) */
        myatomicAdd( (double*) &b[col_ind[k]], (double) (vals[k] * x[i]) );
    }

    /* diagonal entry */
    myatomicAdd( (double*) &b[i], (double) (vals[ei] * x[i]) );
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
        const real * const x, real * const b, int N )
{
    int i, k, si, ei;
    real sum;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    sum = 0.0;
    si = row_ptr_start[i];
    ei = row_ptr_end[i];

    for ( k = si; k < ei; ++k )
    {
        sum += vals[k] * x[col_ind[k]];
    }

    __syncthreads( );

    b[i] = sum;
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where warps of MATVEC_KER_THREADS_PER_ROW threads
 * collaborate to multiply each row
 *
 * A: symmetric, full, square matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
CUDA_GLOBAL void k_sparse_matvec_full_opt_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int N )
{
#if defined(__SM_35__)
    int c, i, pj, thread_id, warp_id, lane;
    int si, ei;
    real vals_local;

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id / MATVEC_KER_THREADS_PER_ROW;
    /* one warp per row */
    i = warp_id;
    lane = thread_id & ( MATVEC_KER_THREADS_PER_ROW - 1);

    vals_local = 0.0;

    if ( i < N )
    {
        /* compute running sum per thread */
        si = row_ptr_start[i];
        ei = row_ptr_end[i];

        for ( pj = si + lane; pj < ei; pj += MATVEC_KER_THREADS_PER_ROW )
        {
            vals_local += vals[pj] * x[ col_ind[pj] ];
        }
    }

    /* parallel reduction in shared memory:
     * SIMD instructions within a warp are synchronous,
     * so we do not need to sync here */
    for ( c = MATVEC_KER_THREADS_PER_ROW >> 1; c >= 1; c /= 2 )
    {
        vals_local += shfl( vals_local, c );
    }

    if ( lane == 0 && i < N )
    {
        b[i] = vals_local;
    }
#else
    int i, pj, thread_id, warp_id, lane;
    int si, ei;
    extern __shared__ real vals_local[];

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id / MATVEC_KER_THREADS_PER_ROW;
    /* one warp per row */
    i = warp_id;
    lane = thread_id & (MATVEC_KER_THREADS_PER_ROW - 1);

    vals_local[threadIdx.x] = 0.0;

    if ( i < N )
    {
        /* coalesce memory by loading sparse matrix column
         * indices and nonzero values together within a warp,
         * and write the warp-local running results to
         * shared memory */
        si = row_ptr_start[i];
        ei = row_ptr_end[i];

        for ( pj = si + lane; pj < ei; pj += MATVEC_KER_THREADS_PER_ROW )
        {
            vals_local[threadIdx.x] += vals[pj] * x[ col_ind[pj] ];
        }
    }

    __syncthreads( );

    /* local tree reduction (sum) on local results in shared memory:
     * SIMD instructions within a warp are synchronous,
     * so we do not need to sync here */
    if ( lane < 16 )
    {
        vals_local[threadIdx.x] += vals_local[threadIdx.x + 16];
    }
    __syncthreads( );
    if ( lane < 8 )
    {
        vals_local[threadIdx.x] += vals_local[threadIdx.x + 8];
    }
    __syncthreads( );
    if ( lane < 4 )
    {
        vals_local[threadIdx.x] += vals_local[threadIdx.x + 4];
    }
    __syncthreads( );
    if ( lane < 2 )
    {
        vals_local[threadIdx.x] += vals_local[threadIdx.x + 2];
    }
    __syncthreads( );
    if ( lane < 1 )
    {
        vals_local[threadIdx.x] += vals_local[threadIdx.x + 1];
    }
    __syncthreads( );

    /* first thread writes the result back to global memory */
    if ( lane == 0 && i < N )
    {
        b[i] = vals_local[threadIdx.x];
    }
#endif
}


/* 1 thread per row implementation */
CUDA_GLOBAL void k_dual_sparse_matvec_full_csr( sparse_matrix A, rvec2 const * const x,
        rvec2 * const b, int n )
{
    int i, c, col;
    rvec2 results_row;
    real val;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    results_row[0] = 0.0;
    results_row[1] = 0.0;

    for ( c = A.start[i]; c < A.end[i]; c++ )
    {
        col = A.j[c];
        val = A.val[c];

        results_row[0] += val * x[col][0];
        results_row[1] += val * x[col][1];
    }

    b[i][0] = results_row[0];
    b[i][1] = results_row[1];
}


CUDA_GLOBAL void k_dual_sparse_matvec_full_opt_csr( sparse_matrix A, rvec2 const * const vec,
        rvec2 * const results, int num_rows )
{
#if defined(__SM_35__)
    rvec2 rvals;
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / MATVEC_KER_THREADS_PER_ROW;
    int lane = thread_id & (MATVEC_KER_THREADS_PER_ROW - 1);
    int row_start;
    int row_end;
    // one warp per row
    int row = warp_id;

    rvals[0] = 0;
    rvals[1] = 0;

    if ( row < num_rows )
    {
        row_start = A.start[row];
        row_end = A.end[row];

        for( int jj = row_start + lane; jj < row_end; jj += MATVEC_KER_THREADS_PER_ROW )
        {
            rvals[0] += A.val[jj] * vec [ A.j[jj] ][0];
            rvals[1] += A.val[jj] * vec [ A.j[jj] ][1];
        }
    }

    for ( int s = MATVEC_KER_THREADS_PER_ROW >> 1; s >= 1; s /= 2 )
    {
        rvals[0] += shfl( rvals[0], s);
        rvals[1] += shfl( rvals[1], s);
    }

    if ( lane == 0 && row < num_rows )
    {
        results[row][0] = rvals[0];
        results[row][1] = rvals[1];
    }

#else
    extern __shared__ rvec2 rvals[];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane = thread_id & (32 - 1);
    int row_start;
    int row_end;
    // one warp per row
    //int row = warp_id;
    int row = warp_id;

    rvals[threadIdx.x][0] = 0;
    rvals[threadIdx.x][1] = 0;

    if ( row < num_rows )
    {
        row_start = A.start[row];
        row_end = A.end[row];

        // compute running sum per thread
        for ( int jj = row_start + lane; jj < row_end; jj += 32 )
        {
            rvals[threadIdx.x][0] += A.val[jj] * vec [ A.j[jj] ][0];
            rvals[threadIdx.x][1] += A.val[jj] * vec [ A.j[jj] ][1];
        }
    }

    __syncthreads( );

    // parallel reduction in shared memory
    //SIMD instructions with a WARP are synchronous -- so we do not need to synch here
    if ( lane < 16 )
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 16][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 16][1]; 
    }
    __syncthreads( );
    if ( lane < 8 )
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 8][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 8][1]; 
    }
    __syncthreads( );
    if ( lane < 4 )
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 4][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 4][1]; 
    }
    __syncthreads( );
    if ( lane < 2 )
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 2][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 2][1]; 
    }
    __syncthreads( );
    if ( lane < 1 )
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 1][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 1][1]; 
    }
    __syncthreads( );

    // first thread writes the result
    if ( lane == 0 && row < num_rows )
    {
        results[row][0] = rvals[threadIdx.x][0];
        results[row][1] = rvals[threadIdx.x][1];
    }
#endif
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
 * buf_type: data structure type for x
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static real Dual_Sparse_MatVec_Comm_Part1( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, void const * const x,
        int buf_type, MPI_Datatype mpi_type )
{
    int t_start, t_comm;
    rvec2 *spad;

    t_start = MPI_Wtime( );
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, x, buf_type, mpi_type );
#else
    spad = (rvec2 *) workspace->host_scratch;

    copy_host_device( spad, (void *)x, sizeof(rvec2) * system->total_cap,
            cudaMemcpyDeviceToHost, "Dual_Sparse_MatVec_Comm_Part1::x" );

    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, spad, buf_type, mpi_type );

    copy_host_device( spad, (void *)x, sizeof(rvec2) * system->total_cap,
            cudaMemcpyHostToDevice, "Dual_Sparse_MatVec_Comm_Part1::x" );
#endif
    t_comm = MPI_Wtime( ) - t_start;

    return t_comm;
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
//    int blocks;

    cuda_memset( b, 0, sizeof(rvec2) * n, "Dual_Sparse_MatVec_local::b" );

    if ( A->format == SYM_HALF_MATRIX )
    {
        //TODO: implement half-format dual SpMV
//        blocks = (n / DEF_BLOCK_SIZE) + 
//            (( n % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

        /* 1 thread per row implementation */
//        k_dual_sparse_matvec_half_csr <<< blocks, DEF_BLOCK_SIZE >>>
//            ( A->start, A->end, A->j, A->val, x, b, n );

        /* multiple threads per row implementation,
         * with shared memory to accumulate partial row sums */
//#if defined(__SM_35__)
//        k_dual_sparse_matvec_half_opt_csr <<< blocks, MATVEC_BLOCK_SIZE >>>
//#else
//        k_dual_sparse_matvec_half_opt_csr <<< blocks, MATVEC_BLOCK_SIZE,
//                     sizeof(real) * MATVEC_BLOCK_SIZE >>>
//#endif
//             ( A->start, A->end, A->j, A->val, x, b, n );
    }
    else if ( A->format == SYM_FULL_MATRIX )
    {
        /* 1 thread per row implementation */
//        k_dual_sparse_matvec_full_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE >>>
//             ( *A, x, b, n );
        
        /* one warp per row implementation,
         * with shared memory to accumulate partial row sums */
#if defined(__SM_35__)
        k_dual_sparse_matvec_full_opt_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE >>>
#else
        k_dual_sparse_matvec_full_opt_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE,
                          sizeof(rvec2) * MATVEC_BLOCK_SIZE >>>
#endif
                ( *A, x, b, n );
    }

    cudaDeviceSynchronize( );
    cudaCheckError( );
}


/* Communications for sparse matrix-dense vector multiplication AX = B
 *
 * system:
 * control:
 * mpi_data:
 * mat_format: storage type of sparse matrix A
 * b: dense vector (device)
 * buf_type: data structure type for b
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static real Dual_Sparse_MatVec_Comm_Part2( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, int mat_format,
        void * const b, int buf_type, MPI_Datatype mpi_type )
{
    int t_start, t_comm;
    rvec2 *spad;

    t_start = MPI_Wtime( );
    /* reduction required for symmetric half matrix */
//    if ( mat_format == SYM_HALF_MATRIX )
    {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        Coll( system, mpi_data, b, buf_type, mpi_type );
#else
        spad = (rvec2 *) workspace->host_scratch;

        copy_host_device( spad, b, sizeof(rvec2) * system->total_cap,
                cudaMemcpyDeviceToHost, "Dual_Sparse_MatVec_Comm_Part2::b" );

        Coll( system, mpi_data, spad, buf_type, mpi_type );

        copy_host_device( spad, b, sizeof(rvec2) * system->total_cap,
                cudaMemcpyHostToDevice, "Dual_Sparse_MatVec_Comm_Part2::b" );
#endif
    }
    t_comm = MPI_Wtime( ) - t_start;

    return t_comm;
}


/* sparse matrix, dense vector multiplication AX = B
 *
 * system:
 * control:
 * workspace: storage container for workspace structures
 * A: symmetric (lower triangular portion only stored), square matrix,
 *    stored in CSR format
 * X: dense vector, size equal to num. columns in A
 * B (output): dense vector, size equal to num. columns in A */
static void Dual_Sparse_MatVec( const reax_system * const system,
        control_params const * const control,
        storage * const workspace, mpi_datatypes * const mpi_data,
        sparse_matrix const * const A, rvec2 * const x,
        rvec2 * const b )
{
    real t_comm;

    t_comm = Dual_Sparse_MatVec_Comm_Part1( system, control, workspace, mpi_data,
            x, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

    Dual_Sparse_MatVec_local( control, A, x, b, system->N );

    t_comm += Dual_Sparse_MatVec_Comm_Part2( system, control, workspace, mpi_data,
            A->format, b, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
}


/* Communications for sparse matrix-dense vector multiplication Ax = b
 *
 * system:
 * control: 
 * mpi_data:
 * x: dense vector (device)
 * buf_type: data structure type for x
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static real Sparse_MatVec_Comm_Part1( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, void const * const x,
        int buf_type, MPI_Datatype mpi_type )
{
    int t_start, t_comm;
    real *spad;

    t_start = MPI_Wtime( );
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, x, buf_type, mpi_type );
#else
    spad = (real *) workspace->host_scratch;

    copy_host_device( spad, (void *)x, sizeof(real) * system->total_cap,
            cudaMemcpyDeviceToHost, "Sparse_MatVec_Comm_Part1::x" );

    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, spad, buf_type, mpi_type );

    copy_host_device( spad, (void *)x, sizeof(real) * system->total_cap,
            cudaMemcpyHostToDevice, "Sparse_MatVec_Comm_Part1::x" );
#endif
    t_comm = MPI_Wtime( ) - t_start;

    return t_comm;
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

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

    cuda_memset( b, 0, sizeof(real) * n, "Sparse_MatVec_local::b" );

    if ( A->format == SYM_HALF_MATRIX )
    {
        /* 1 thread per row implementation */
        k_sparse_matvec_half_csr <<< blocks, DEF_BLOCK_SIZE >>>
            ( A->start, A->end, A->j, A->val, x, b, n );

        /* multiple threads per row implementation,
         * with shared memory to accumulate partial row sums */
//#if defined(__SM_35__)
//        k_sparse_matvec_half_opt_csr <<< blocks, MATVEC_BLOCK_SIZE >>>
//#else
//        k_sparse_matvec_half_opt_csr <<< blocks, MATVEC_BLOCK_SIZE,
//                     sizeof(real) * MATVEC_BLOCK_SIZE >>>
//#endif
//             ( A->start, A->end, A->j, A->val, x, b, n );
    }
    else if ( A->format == SYM_FULL_MATRIX )
    {
        /* 1 thread per row implementation */
//        k_sparse_matvec_full_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE >>>
//             ( A->start, A->end, A->j, A->val, x, b, n );

        /* multiple threads per row implementation,
         * with shared memory to accumulate partial row sums */
#if defined(__SM_35__)
        k_sparse_matvec_full_opt_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE >>>
#else
        k_sparse_matvec_full_opt_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE,
                                 sizeof(real) * MATVEC_BLOCK_SIZE >>>
#endif
             ( A->start, A->end, A->j, A->val, x, b, n );
    }

    cudaDeviceSynchronize( );
    cudaCheckError( );
}


/* Communications for sparse matrix-dense vector multiplication Ax = b
 *
 * system:
 * control:
 * mpi_data:
 * mat_format: storage type of sparse matrix A
 * b: dense vector (device)
 * buf_type: data structure type for b
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static real Sparse_MatVec_Comm_Part2( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, int mat_format,
        void * const b, int buf_type, MPI_Datatype mpi_type )
{
    int t_start, t_comm;
    real *spad;

    t_start = MPI_Wtime( );
    /* reduction required for symmetric half matrix */
//    if ( mat_format == SYM_HALF_MATRIX )
    {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        Coll( system, mpi_data, b, buf_type, mpi_type );
#else
        spad = (real *) workspace->host_scratch;

        copy_host_device( spad, b, sizeof(real) * system->total_cap,
                cudaMemcpyDeviceToHost, "Cuda_CG::q" );

        Coll( system, mpi_data, spad, buf_type, mpi_type );

        copy_host_device( spad, b, sizeof(real) * system->total_cap,
                cudaMemcpyHostToDevice, "Cuda_CG::q" );
#endif
    }
    t_comm = MPI_Wtime( ) - t_start;

    return t_comm;
}


/* sparse matrix, dense vector multiplication Ax = b
 *
 * system:
 * control:
 * workspace: storage container for workspace structures
 * A: symmetric (lower triangular portion only stored), square matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A */
static void Sparse_MatVec( reax_system const * const system,
        control_params const * const control,
        storage * const workspace, mpi_datatypes * const mpi_data,
        sparse_matrix const * const A, real const * const x,
        real * const b )
{
    real t_comm;

    t_comm = Sparse_MatVec_Comm_Part1( system, control, workspace, mpi_data,
            x, REAL_PTR_TYPE, MPI_DOUBLE );

    Sparse_MatVec_local( control, A, x, b, system->N );

    t_comm += Sparse_MatVec_Comm_Part2( system, control, workspace, mpi_data,
            A->format, b, REAL_PTR_TYPE, MPI_DOUBLE );
}


int Cuda_dual_CG( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, rvec2 const * const b, real tol,
        rvec2 * const x, mpi_datatypes * const mpi_data, FILE *fout )
{
    unsigned int i, matvecs;
    rvec2 tmp, alpha, beta, norm, b_norm;
    rvec2 sig_old, sig_new;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[6];

    matvecs = 0;
    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = Get_Time( );
    Dual_Sparse_MatVec( system, control, workspace, mpi_data,
            H, x, workspace->d_workspace->q2 );
    t_spmv += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->d_workspace->q2, system->n );
    t_vops += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    dual_jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r2,
            workspace->d_workspace->d2, system->n );
    t_pa += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    Dot_local_rvec2( control, workspace, workspace->d_workspace->r2,
            workspace->d_workspace->d2, system->n, &redux[0], &redux[1] );
    Dot_local_rvec2( control, workspace, workspace->d_workspace->d2,
            workspace->d_workspace->d2, system->n, &redux[2], &redux[3] );
    Dot_local_rvec2( control, workspace, b, b, system->n, &redux[4], &redux[5] );
    t_vops += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    MPI_Allreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM, mpi_data->world );
    sig_new[0] = redux[0];
    sig_new[1] = redux[1];
    norm[0] = SQRT( redux[2] );
    norm[1] = SQRT( redux[3] );
    b_norm[0] = SQRT( redux[4] );
    b_norm[1] = SQRT( redux[5] );
    t_allreduce += Get_Timing_Info( t_start );

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( norm[0] / b_norm[0] <= tol || norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        t_start = Get_Time( );
        Dual_Sparse_MatVec( system, control, workspace, mpi_data,
                H, workspace->d_workspace->d2, workspace->d_workspace->q2 );
        t_spmv += Get_Timing_Info( t_start );

        /* dot product: d.q */
        t_start = Get_Time( );
        Dot_local_rvec2( control, workspace, workspace->d_workspace->d2,
                workspace->d_workspace->q2, system->n, &redux[0], &redux[1] );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        tmp[0] = redux[0];
        tmp[1] = redux[1];
        t_allreduce += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        Vector_Add_rvec2( x, alpha[0], alpha[1],
                workspace->d_workspace->d2, system->n );
        Vector_Add_rvec2( workspace->d_workspace->r2, -1.0 * alpha[0], -1.0 * alpha[1],
                workspace->d_workspace->q2, system->n );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        dual_jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r2,
                workspace->d_workspace->p2, system->n );
        t_pa += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        /* dot products: r.p and p.p */
        Dot_local_rvec2( control, workspace, workspace->d_workspace->r2,
                workspace->d_workspace->p2, system->n, &redux[0], &redux[1] );
        Dot_local_rvec2( control, workspace, workspace->d_workspace->p2,
                workspace->d_workspace->p2, system->n, &redux[2], &redux[3] );
        t_vops += Get_Timing_Info( t_start );

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = Get_Time( );
        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        sig_new[0] = redux[0];
        sig_new[1] = redux[1];
        norm[0] = SQRT( redux[2] );
        norm[1] = SQRT( redux[3] );
        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        /* d = p + beta * d */
        Vector_Sum_rvec2( workspace->d_workspace->d2,
                1.0, 1.0, workspace->d_workspace->p2,
                beta[0], beta[1], workspace->d_workspace->d2, system->n );
        t_vops += Get_Timing_Info( t_start );
    }

    if ( norm[0] / b_norm[0] <= tol
            && norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n );

        matvecs = Cuda_CG( system, control, data, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n );
    }
    else if ( norm[1] / b_norm[1] <= tol
            && norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n );

        matvecs = Cuda_CG( system, control, data, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n );
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual CG convergence failed! (%d steps)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] s lin solve error: %f\n", norm[0] / b_norm[0] );
        fprintf( stderr, "    [INFO] t lin solve error: %f\n", norm[1] / b_norm[1] );
    }

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( fout, "QEq %d + %d iters. matvecs: %f  dot: %f\n",
                i + 1, matvecs, matvec_time, dot_time );
    }
#endif

    return (i + 1) + matvecs;
}


int Cuda_CG( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data )
{
    unsigned int i;
    real tmp, alpha, beta, norm, b_norm;
    real sig_old, sig_new;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[3];

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = Get_Time( );
    Sparse_MatVec( system, control, workspace, mpi_data,
            H, x, workspace->d_workspace->q );
    t_spmv += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->q, system->n );
    t_vops += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r,
            workspace->d_workspace->d, system->n );
    t_pa += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    redux[0] = Dot_local( workspace, workspace->d_workspace->r,
            workspace->d_workspace->d, system->n );
    redux[1] = Dot_local( workspace, workspace->d_workspace->d,
            workspace->d_workspace->d, system->n );
    redux[2] = Dot_local( workspace, b, b, system->n );
    t_vops += Get_Timing_Info( t_start );

    t_start = Get_Time( );
    MPI_Allreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM, mpi_data->world );
    sig_new = redux[0];
    norm = SQRT( redux[1] );
    b_norm = SQRT( redux[2] );
    t_allreduce += Get_Timing_Info( t_start );

    for ( i = 0; i < control->cm_solver_max_iters && norm / b_norm > tol; ++i )
    {
        t_start = Get_Time( );
        Sparse_MatVec( system, control, workspace, mpi_data,
                H, workspace->d_workspace->d, workspace->d_workspace->q );
        t_spmv += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        tmp = Dot( workspace, workspace->d_workspace->d, workspace->d_workspace->q,
                system->n, mpi_data->world );
        t_allreduce += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d_workspace->d, system->n );
        Vector_Add( workspace->d_workspace->r, -1.0 * alpha, workspace->d_workspace->q, system->n );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        jacobi_apply( workspace->d_workspace->Hdia_inv, workspace->d_workspace->r,
                workspace->d_workspace->p, system->n );
        t_pa += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        redux[0] = Dot_local( workspace, workspace->d_workspace->r, workspace->d_workspace->p, system->n );
        redux[1] = Dot_local( workspace, workspace->d_workspace->p, workspace->d_workspace->p, system->n );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        sig_old = sig_new;
        sig_new = redux[0];
        norm = SQRT( redux[1] );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d_workspace->d, 1.0, workspace->d_workspace->p,
                beta, workspace->d_workspace->d, system->n );
        t_vops += Get_Timing_Info( t_start );
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] CG convergence failed!\n" );
        fprintf( stderr, "    [INFO] lin solve error: %f\n", SQRT(sig_new) / b_norm );
    }

    return i;
}
