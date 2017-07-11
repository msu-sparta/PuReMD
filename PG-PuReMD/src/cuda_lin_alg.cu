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

#include "cuda_lin_alg.h"

#include "reax_types.h"

#include "cuda_shuffle.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"


//one thread per row
CUDA_GLOBAL void k_matvec( sparse_matrix H, real *vec, real *results,
        int rows )
{
    int i, col;
    real results_row;
    real val;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= rows )
    {
        return;
    }

    results_row = 0;

    for (int c = H.start[i]; c < H.end[i]; c++)
    {
        col = H.entries [c].j;
        val = H.entries[c].val;

        results_row += val * vec[col];
    }

    results[i] = results_row;
}


//32 thread warp per matrix row.
//invoked as follows
// <<< system->N, 32 >>>
//CUDA_GLOBAL void __launch_bounds__(384, 16) k_matvec_csr(sparse_matrix H, real *vec, real *results, int num_rows)
CUDA_GLOBAL void k_matvec_csr( sparse_matrix H, real *vec, real *results,
        int num_rows )
{
#if defined(__SM_35__)
    real vals;
    int x;
#else
    extern __shared__ real vals[];
#endif
    int jj;
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / MATVEC_KER_THREADS_PER_ROW;
    int lane = thread_id & ( MATVEC_KER_THREADS_PER_ROW - 1);
    int row_start;
    int row_end;
    // one warp per row
    int row = warp_id;
    
#if defined(__SM_35__)
    vals = 0;
#else
    vals[threadIdx.x] = 0;
#endif

    if (row < num_rows)
    {
        row_start = H.start[row];
        row_end = H.end[row];

        // compute running sum per thread
        for ( jj = row_start + lane; jj < row_end;
                jj += MATVEC_KER_THREADS_PER_ROW )
#if defined(__SM_35__)
        {
            vals += H.entries[jj].val * vec[ H.entries[jj].j ];
        }
    }
#else
        {
            vals[threadIdx.x] += H.entries[jj].val * vec[ H.entries[jj].j ];
        }
    }

    __syncthreads( );
#endif

    // parallel reduction in shared memory
    //SIMD instructions with a WARP are synchronous -- so we do not need to synch here
#if defined(__SM_35__)
    for (x = MATVEC_KER_THREADS_PER_ROW >> 1; x >= 1; x/=2)
    {
        vals += shfl( vals, x );
    }

    if (lane == 0 && row < num_rows)
    {
        results[row] = vals;
    }
#else
    if (lane < 16)
    {
        vals[threadIdx.x] += vals[threadIdx.x + 16];
    }
    __syncthreads( );
    if (lane < 8)
    {
        vals[threadIdx.x] += vals[threadIdx.x + 8];
    }
    __syncthreads( );
    if (lane < 4)
    {
        vals[threadIdx.x] += vals[threadIdx.x + 4];
    }
    __syncthreads( );
    if (lane < 2)
    {
        vals[threadIdx.x] += vals[threadIdx.x + 2];
    }
    __syncthreads( );
    if (lane < 1)
    {
        vals[threadIdx.x] += vals[threadIdx.x + 1];
    }
    __syncthreads( );

    // first thread writes the result
    if (lane == 0 && row < num_rows)
    {
        results[row] = vals[threadIdx.x];
    }
#endif
}


//one thread per row
CUDA_GLOBAL void k_dual_matvec( sparse_matrix H, rvec2 *vec, rvec2 *results,
        int rows )
{
    int i, c, col;
    rvec2 results_row;
    real val;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= rows)
    {
        return;
    }

    results_row[0] = 0.0;
    results_row[1] = 0.0;

    for (c = H.start[i]; c < H.end[i]; c++)
    {
        col = H.entries [c].j;
        val = H.entries[c].val;

        results_row[0] += val * vec [col][0];
        results_row[1] += val * vec [col][1];
    }

    results[i][0] = results_row[0];
    results[i][1] = results_row[1];
}


//32 thread warp per matrix row.
//invoked as follows
// <<< system->N, 32 >>>
//CUDA_GLOBAL void __launch_bounds__(384, 8) k_dual_matvec_csr(sparse_matrix H, rvec2 *vec, rvec2 *results, int num_rows)
CUDA_GLOBAL void  k_dual_matvec_csr( sparse_matrix H, rvec2 *vec,
        rvec2 *results, int num_rows )
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

    if (row < num_rows)
    {
        row_start = H.start[row];
        row_end = H.end[row];

        for(int jj = row_start + lane; jj < row_end; jj += MATVEC_KER_THREADS_PER_ROW)
        {
            rvals[0] += H.entries[jj].val * vec [ H.entries[jj].j ][0];
            rvals[1] += H.entries[jj].val * vec [ H.entries[jj].j ][1];
        }
    }

    for (int s = MATVEC_KER_THREADS_PER_ROW >> 1; s >= 1; s /= 2)
    {
        rvals[0] += shfl( rvals[0], s);
        rvals[1] += shfl( rvals[1], s);
    }

    if (lane == 0 && row < num_rows)
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

    if (row < num_rows)
    {
        row_start = H.start[row];
        row_end = H.end[row];

        // compute running sum per thread
        for(int jj = row_start + lane; jj < row_end; jj += 32)
        {
            rvals[threadIdx.x][0] += H.entries[jj].val * vec [ H.entries[jj].j ][0];
            rvals[threadIdx.x][1] += H.entries[jj].val * vec [ H.entries[jj].j ][1];
        }
    }

    __syncthreads( );

    // parallel reduction in shared memory
    //SIMD instructions with a WARP are synchronous -- so we do not need to synch here
    if (lane < 16)
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 16][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 16][1]; 
    }
    __syncthreads( );
    if (lane < 8)
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 8][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 8][1]; 
    }
    __syncthreads( );
    if (lane < 4)
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 4][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 4][1]; 
    }
    __syncthreads( );
    if (lane < 2)
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 2][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 2][1]; 
    }
    __syncthreads( );
    if (lane < 1)
    {
        rvals[threadIdx.x][0] += rvals[threadIdx.x + 1][0]; 
        rvals[threadIdx.x][1] += rvals[threadIdx.x + 1][1]; 
    }
    __syncthreads( );

    // first thread writes the result
    if (lane == 0 && row < num_rows)
    {
        results[row][0] = rvals[threadIdx.x][0];
        results[row][1] = rvals[threadIdx.x][1];
    }

#endif
}


void Cuda_Vector_Sum( real *res, real a, real *x, real b, real *y, int count )
{
    //res = ax + by
    //use the cublas here
    int blocks;

    blocks = (count / DEF_BLOCK_SIZE) + 
        ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_sum <<< blocks, DEF_BLOCK_SIZE >>>
        ( res, a, x, b, y, count );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


void Cuda_CG_Preconditioner( real *res, real *a, real *b, int count )
{
    //res = a*b - vector multiplication
    //use the cublas here.
    int blocks;

    blocks = (count / DEF_BLOCK_SIZE) + 
        ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_mul <<< blocks, DEF_BLOCK_SIZE >>>
        ( res, a, b, count );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_diagonal_preconditioner(storage p_workspace, rvec2 *b, int n)
{
    storage *workspace;
    int j;
   
    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    workspace = &( p_workspace );

    //for( j = 0; j < system->n; ++j ) {
    // residual 
    workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
    workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];

    // apply diagonal pre-conditioner
    workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j]; 
    workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j]; 
    //}
}


void Cuda_CG_Diagonal_Preconditioner( storage *workspace, rvec2 *b, int n )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_diagonal_preconditioner <<< blocks, DEF_BLOCK_SIZE >>>
        (*workspace, b, n);

    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_dual_cg_preconditioner( storage p_workspace, rvec2 *x, 
        real alpha_0, real alpha_1, int n, rvec2 *my_dot )
{
    storage *workspace;
    rvec2 alpha;
    int j;
   
    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    workspace = &( p_workspace );
    alpha[0] = alpha_0;
    alpha[1] = alpha_1;
    my_dot[j][0] = my_dot[j][1] = 0.0;

    //for( j = 0; j < system->n; ++j ) {
    // update x 
    x[j][0] += alpha[0] * workspace->d2[j][0];
    x[j][1] += alpha[1] * workspace->d2[j][1];      

    // update residual 
    workspace->r2[j][0] -= alpha[0] * workspace->q2[j][0]; 
    workspace->r2[j][1] -= alpha[1] * workspace->q2[j][1]; 

    // apply diagonal pre-conditioner 
    workspace->p2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
    workspace->p2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];

    // dot product: r.p 
    my_dot[j][0] = workspace->r2[j][0] * workspace->p2[j][0];
    my_dot[j][1] = workspace->r2[j][1] * workspace->p2[j][1];
    //}
}


void Cuda_DualCG_Preconditioner( storage *workspace, rvec2 *x, rvec2 alpha,
        int n, rvec2 result )
{
    int blocks;
    rvec2 *tmp = (rvec2 *) scratch;

    cuda_memset( tmp, 0, sizeof(rvec2) * ( 2 * n + 1),
            "cuda_dualcg_preconditioner" );
    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_dual_cg_preconditioner <<< blocks, DEF_BLOCK_SIZE >>>
        (*workspace, x, alpha[0], alpha[1], n, tmp);

    cudaThreadSynchronize( );
    cudaCheckError( );

    //Reduction to calculate my_dot
    k_reduction_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * DEF_BLOCK_SIZE >>>
        ( tmp, tmp + n, n);

    cudaThreadSynchronize( );
    cudaCheckError( );

    k_reduction_rvec2 <<< 1, BLOCKS_POW_2, sizeof(rvec2) * BLOCKS_POW_2 >>>
        ( tmp + n, tmp + 2*n, blocks);

    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( result, (tmp + 2*n), sizeof(rvec2),
            cudaMemcpyDeviceToHost, "my_dot" );
}


void Cuda_Norm( rvec2 *arr, int n, rvec2 result )
{
    int blocks;
    rvec2 *tmp = (rvec2 *) scratch;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_norm_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * DEF_BLOCK_SIZE >>>
        (arr, tmp, n, INITIAL);
    cudaThreadSynchronize( );
    cudaCheckError( );

    k_norm_rvec2 <<< 1, BLOCKS_POW_2, sizeof(rvec2) * BLOCKS_POW_2 >>>
        (tmp, tmp + BLOCKS_POW_2, blocks, FINAL );
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( result, tmp + BLOCKS_POW_2, sizeof(rvec2), 
            cudaMemcpyDeviceToHost, "cuda_norm_rvec2" );
}


void Cuda_Dot( rvec2 *a, rvec2 *b, rvec2 result, int n )
{
    int blocks;
    rvec2 *tmp = (rvec2 *) scratch;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_dot_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * DEF_BLOCK_SIZE >>>
        ( a, b, tmp, n );
    cudaThreadSynchronize( );
    cudaCheckError( );

    k_norm_rvec2 <<< 1, BLOCKS_POW_2, sizeof(rvec2) * BLOCKS_POW_2 >>> 
    //k_norm_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * BLOCKS_POW_2 >>> 
        ( tmp, tmp + BLOCKS_POW_2, blocks, FINAL );
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( result, tmp + BLOCKS_POW_2, sizeof(rvec2), 
            cudaMemcpyDeviceToHost, "cuda_dot" );
}


void Cuda_Vector_Sum_Rvec2(rvec2 *x, rvec2 *a, rvec2 b, rvec2 *c, int n)
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec2_pbetad <<< blocks, DEF_BLOCK_SIZE >>> 
        ( x, a, b[0], b[1], c, n);

    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_rvec2_to_real_copy( real *dst, rvec2 *src, int index, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
    {
        return;
    }

    dst[i] = src[i][index];
}


void Cuda_RvecCopy_From( real *dst, rvec2 *src, int index, int n )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec2_to_real_copy <<< blocks, DEF_BLOCK_SIZE >>>
        ( dst, src, index, n);
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_real_to_rvec2_copy( rvec2 *dst, real *src, int index, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
    {
        return;
    }

    dst[i][index] = src[i];
}


void Cuda_RvecCopy_To(rvec2 *dst, real *src, int index, int n)
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_real_to_rvec2_copy <<< blocks, DEF_BLOCK_SIZE >>>
        ( dst, src, index, n);

    cudaThreadSynchronize( );
    cudaCheckError( );
}


void Cuda_Dual_Matvec( sparse_matrix *H, rvec2 *a, rvec2 *b, int n, int size )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

    cuda_memset( b, 0, sizeof(rvec2) * size, "dual_matvec:result" );

    //One thread per row implementation
    //k_dual_matvec <<< blocks, DEF_BLOCK_SIZE >>>
    //        (*H, a, b, n);
    //cudaThreadSynchronize ();
    //cudaCheckError ();

    //One warp per row implementation
#if defined(__SM_35__)
    k_dual_matvec_csr <<< MATVEC_BLOCKS, MATVEC_BLOCK_SIZE >>>
#else
    k_dual_matvec_csr <<< MATVEC_BLOCKS, MATVEC_BLOCK_SIZE,
                      sizeof(rvec2) * MATVEC_BLOCK_SIZE >>>
#endif
            ( *H, a, b, n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


void Cuda_Matvec( sparse_matrix *H, real *a, real *b, int n, int size )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE) + 
        (( n % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

    cuda_memset( b, 0, sizeof(real) * size, "dual_matvec:result" );

    //one thread per row implementation
    //k_matvec <<< blocks, DEF_BLOCK_SIZE >>>
    //        (*H, a, b, n);
    //cudaThreadSynchronize ();
    //cudaCheckError ();

#if defined(__SM_35__)
    k_matvec_csr <<< MATVEC_BLOCKS, MATVEC_BLOCK_SIZE >>>
#else
    k_matvec_csr <<< MATVEC_BLOCKS, MATVEC_BLOCK_SIZE,
                 sizeof(real) * MATVEC_BLOCK_SIZE>>>
#endif
                     (*H, a, b, n);

    cudaThreadSynchronize( );
    cudaCheckError( );
}
