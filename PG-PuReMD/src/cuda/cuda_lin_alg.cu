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

#include "cuda_shuffle.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "../basic_comm.h"


//one thread per row
CUDA_GLOBAL void k_matvec( sparse_matrix H, real *vec, real *results,
        int rows )
{
    int i, c, col;
    real results_row;
    real val;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= rows )
    {
        return;
    }

    results_row = 0;

    for ( c = H.start[i]; c < H.end[i]; c++ )
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

    if ( i >= rows )
    {
        return;
    }

    results_row[0] = 0.0;
    results_row[1] = 0.0;

    for ( c = H.start[i]; c < H.end[i]; c++ )
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

    if ( row < num_rows )
    {
        row_start = H.start[row];
        row_end = H.end[row];

        for( int jj = row_start + lane; jj < row_end; jj += MATVEC_KER_THREADS_PER_ROW )
        {
            rvals[0] += H.entries[jj].val * vec [ H.entries[jj].j ][0];
            rvals[1] += H.entries[jj].val * vec [ H.entries[jj].j ][1];
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
        row_start = H.start[row];
        row_end = H.end[row];

        // compute running sum per thread
        for ( int jj = row_start + lane; jj < row_end; jj += 32 )
        {
            rvals[threadIdx.x][0] += H.entries[jj].val * vec [ H.entries[jj].j ][0];
            rvals[threadIdx.x][1] += H.entries[jj].val * vec [ H.entries[jj].j ][1];
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


void Cuda_Vector_Sum( real *res, real a, real *x, real b, real *y, int count )
{
    //res = ax + by
    //use the cublas here
    int blocks;

    blocks = (count / DEF_BLOCK_SIZE)
        + ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_sum <<< blocks, DEF_BLOCK_SIZE >>>
        ( res, a, x, b, y, count );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


void Cuda_CG_Preconditioner( real *res, real *a, real *b, int count )
{
    //res = a*b - vector multiplication
    //use the cublas here.
    int blocks;

    blocks = (count / DEF_BLOCK_SIZE)
        + ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_mul <<< blocks, DEF_BLOCK_SIZE >>>
        ( res, a, b, count );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_diagonal_preconditioner( storage p_workspace, rvec2 *b, int n )
{
    storage *workspace;
    int j;
   
    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    workspace = &p_workspace;

    /* compute residuals */
    workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
    workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];

    /* apply diagonal preconditioner to residuals */
    workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j]; 
    workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j]; 
}


void Cuda_CG_Diagonal_Preconditioner( storage *workspace, rvec2 *b, int n )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + (( n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_diagonal_preconditioner <<< blocks, DEF_BLOCK_SIZE >>>
        ( *workspace, b, n );

    cudaDeviceSynchronize( );
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

    workspace = &p_workspace;
    alpha[0] = alpha_0;
    alpha[1] = alpha_1;
    my_dot[j][0] = my_dot[j][1] = 0.0;

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
}


void Cuda_DualCG_Preconditioner( control_params *control, storage *workspace, rvec2 *x, rvec2 alpha,
        int n, rvec2 result )
{
    int blocks;
    rvec2 *tmp = (rvec2 *) workspace->scratch;

    cuda_memset( tmp, 0, sizeof(rvec2) * (2 * n + 1),
            "Cuda_DualCG_Preconditioner::tmp" );

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_dual_cg_preconditioner <<< blocks, DEF_BLOCK_SIZE >>>
        ( *(workspace->d_workspace), x, alpha[0], alpha[1], n, tmp );

    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* reduction to calculate my_dot */
    k_reduction_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * DEF_BLOCK_SIZE >>>
        ( tmp, tmp + n, n);

    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_reduction_rvec2 <<< 1, control->blocks_pow_2, sizeof(rvec2) * control->blocks_pow_2 >>>
        ( tmp + n, tmp + 2*n, blocks);

    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( result, (tmp + 2*n), sizeof(rvec2),
            cudaMemcpyDeviceToHost, "my_dot" );
}


void Cuda_Norm( control_params *control, storage *workspace,
        rvec2 *arr, int n, rvec2 result )
{
    int blocks;
    rvec2 *tmp = (rvec2 *) workspace->scratch;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_norm_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * DEF_BLOCK_SIZE >>>
        (arr, tmp, n, INITIAL);
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_norm_rvec2 <<< 1, control->blocks_pow_2, sizeof(rvec2) * control->blocks_pow_2 >>>
        ( tmp, tmp + control->blocks_pow_2, blocks, FINAL );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( result, tmp + control->blocks_pow_2, sizeof(rvec2), 
            cudaMemcpyDeviceToHost, "cuda_norm_rvec2" );
}


void Cuda_Dot( control_params *control, storage *workspace,
        rvec2 *a, rvec2 *b, rvec2 result, int n )
{
    int blocks;
    rvec2 *tmp = (rvec2 *) workspace->scratch;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_dot_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * DEF_BLOCK_SIZE >>>
        ( a, b, tmp, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_norm_rvec2 <<< 1, control->blocks_pow_2, sizeof(rvec2) * control->blocks_pow_2 >>> 
    //k_norm_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * control->blocks_pow_2 >>> 
        ( tmp, tmp + control->blocks_pow_2, blocks, FINAL );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( result, tmp + control->blocks_pow_2, sizeof(rvec2), 
            cudaMemcpyDeviceToHost, "cuda_dot" );
}


void Cuda_Vector_Sum_Rvec2(rvec2 *x, rvec2 *a, rvec2 b, rvec2 *c, int n)
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec2_pbetad <<< blocks, DEF_BLOCK_SIZE >>> 
        ( x, a, b[0], b[1], c, n);

    cudaDeviceSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_rvec2_to_real_copy( real *dst, rvec2 *src, int index, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    dst[i] = src[i][index];
}


void Cuda_RvecCopy_From( real *dst, rvec2 *src, int index, int n )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_rvec2_to_real_copy <<< blocks, DEF_BLOCK_SIZE >>>
        ( dst, src, index, n);
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_real_to_rvec2_copy( rvec2 *dst, real *src, int index, int n)
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    dst[i][index] = src[i];
}


void Cuda_RvecCopy_To(rvec2 *dst, real *src, int index, int n)
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_real_to_rvec2_copy <<< blocks, DEF_BLOCK_SIZE >>>
        ( dst, src, index, n);

    cudaDeviceSynchronize( );
    cudaCheckError( );
}


void Cuda_Dual_Matvec( control_params *control, sparse_matrix *H,
        rvec2 *a, rvec2 *b, int n, int size )
{
//    int blocks;

//    blocks = (n / DEF_BLOCK_SIZE)
//        + ((n % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

    cuda_memset( b, 0, sizeof(rvec2) * size, "dual_matvec:result" );

    /* one thread per row implementation */
//    k_dual_matvec <<< blocks, DEF_BLOCK_SIZE >>>
//        ( *H, a, b, n );
//    cudaDeviceSynchronize( );
//    cudaCheckError( );

    //One warp per row implementation
#if defined(__SM_35__)
    k_dual_matvec_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE >>>
#else
    k_dual_matvec_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE,
                      sizeof(rvec2) * MATVEC_BLOCK_SIZE >>>
#endif
            ( *H, a, b, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


void Cuda_Matvec( control_params *control, sparse_matrix *H,
        real *a, real *b, int n, int size )
{
//    int blocks;

//    blocks = (n / DEF_BLOCK_SIZE) + 
//        (( n % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

    cuda_memset( b, 0, sizeof(real) * size, "dual_matvec:result" );

    /* one thread per row implementation */
//    k_matvec <<< blocks, DEF_BLOCK_SIZE >>>
//        ( *H, a, b, n );
//    cudaDeviceSynchronize( );
//    cudaCheckError( );

#if defined(__SM_35__)
    k_matvec_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE >>>
#else
    k_matvec_csr <<< control->matvec_blocks, MATVEC_BLOCK_SIZE,
                 sizeof(real) * MATVEC_BLOCK_SIZE >>>
#endif
         ( *H, a, b, n );

    cudaDeviceSynchronize( );
    cudaCheckError( );
}


int Cuda_dual_CG( reax_system *system, control_params *control, storage *workspace,
        sparse_matrix *H, rvec2 *b, real tol, rvec2 *x, mpi_datatypes* mpi_data,
        FILE *fout, simulation_data *data )
{
    unsigned int i;
    int n, matvecs;
//    int j, N;
    rvec2 tmp, alpha, beta;
    rvec2 my_sum, norm_sqr, b_norm, my_dot;
    rvec2 sig_old, sig_new;
    MPI_Comm comm;
    rvec2 *spad = (rvec2 *) workspace->host_scratch;

    n = system->n;
//    N = system->N;
    comm = mpi_data->world;
    matvecs = 0;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        matvecs = 0;
        t_start = matvec_time = dot_time = 0;
        t_start = Get_Time( );
    }
#endif

    copy_host_device( spad, x, sizeof(rvec2) * system->total_cap,
            cudaMemcpyDeviceToHost, "Cuda_dual_CG:x:get" );
    Dist( system, mpi_data, spad, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
    copy_host_device( spad, x, sizeof(rvec2) * system->total_cap,
            cudaMemcpyHostToDevice, "Cuda_dual_CG:x:put" );

    //originally we were using only H->n which was system->n (init_md.c)
    //Cuda_Dual_Matvec( control, H, x, workspace->d_workspace->q2, H->n, system->total_cap );
    Cuda_Dual_Matvec( control, H, x, workspace->d_workspace->q2, system->N, system->total_cap );

    copy_host_device( spad, workspace->d_workspace->q2, sizeof(rvec2) * system->total_cap,
            cudaMemcpyDeviceToHost, "Cuda_dual_CG:q2:get" );
    Coll( system, mpi_data, spad, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
    copy_host_device( spad, workspace->d_workspace->q2, sizeof(rvec2) * system->total_cap,
            cudaMemcpyHostToDevice,"Cuda_dual_CG:q2:put" );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &matvec_time );
    }
#endif

    Cuda_CG_Diagonal_Preconditioner( workspace->d_workspace, b, system->n );

    my_sum[0] = 0.0;
    my_sum[1] = 0.0;
    Cuda_Norm( control, workspace, b, n, my_sum );

    MPI_Allreduce( &my_sum, &norm_sqr, 2, MPI_DOUBLE, MPI_SUM, comm );
    b_norm[0] = SQRT( norm_sqr[0] );
    b_norm[1] = SQRT( norm_sqr[1] );

    /* dot product: r.d */
    my_dot[0] = 0.0;
    my_dot[1] = 0.0;
    Cuda_Dot( control, workspace, workspace->d_workspace->r2,
            workspace->d_workspace->d2, my_dot, n );
    
    MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < control->cm_solver_max_iters; ++i )
    {
        copy_host_device( spad, workspace->d_workspace->d2, sizeof(rvec2) * system->total_cap,
                cudaMemcpyDeviceToHost, "Cuda_dual_CG::d2:get" );
        Dist( system, mpi_data, spad, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        copy_host_device( spad, workspace->d_workspace->d2, sizeof(rvec2) * system->total_cap,
                cudaMemcpyHostToDevice, "Cuda_dual_CG::d2:put" );

        Cuda_Dual_Matvec( control, H, workspace->d_workspace->d2, workspace->d_workspace->q2, system->N,
                system->total_cap );

        copy_host_device( spad, workspace->d_workspace->q2, sizeof(rvec2) * system->total_cap,
                cudaMemcpyDeviceToHost, "Cuda_dual_CG::q2:get" );
        Coll( system, mpi_data, spad, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        copy_host_device( spad, workspace->d_workspace->q2, sizeof(rvec2) * system->total_cap,
                cudaMemcpyHostToDevice, "Cuda_dual_CG::q2:put" );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        /* dot product: d.q */
        my_dot[0] = 0.0;
        my_dot[1] = 0.0;
        Cuda_Dot( control, workspace, workspace->d_workspace->d2,
                workspace->d_workspace->q2, my_dot, n );

        MPI_Allreduce( &my_dot, &tmp, 2, MPI_DOUBLE, MPI_SUM, comm );

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        my_dot[0] = 0.0;
        my_dot[1] = 0.0;

        Cuda_DualCG_Preconditioner( control, workspace,
                x, alpha, system->n, my_dot );

        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif

        if ( SQRT(sig_new[0]) / b_norm[0] <= tol || SQRT(sig_new[1]) / b_norm[1] <= tol )
        {
            break;
        }

        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];

        Cuda_Vector_Sum_Rvec2( workspace->d_workspace->d2,
                workspace->d_workspace->p2, beta,
                workspace->d_workspace->d2, system->n );
    }

    if ( SQRT(sig_new[0]) / b_norm[0] <= tol )
    {
        Cuda_RvecCopy_From( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n );

        matvecs = Cuda_CG( system, control, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data );

        Cuda_RvecCopy_To( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n );
    }
    else if ( SQRT(sig_new[1]) / b_norm[1] <= tol )
    {
        Cuda_RvecCopy_From( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n );

        matvecs = Cuda_CG( system, control, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data );

        Cuda_RvecCopy_To( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n );
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual CG convergence failed! (%d steps)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] s lin solve error: %f\n", SQRT(sig_new[0]) / b_norm[0] );
        fprintf( stderr, "    [INFO] t lin solve error: %f\n", SQRT(sig_new[1]) / b_norm[1] );
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


int Cuda_CG( reax_system *system, control_params *control, storage *workspace,
        sparse_matrix *H, real *b, real tol, real *x, mpi_datatypes* mpi_data )
{
    unsigned int i;
//    int j;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new;
    real *spad = (real *) workspace->host_scratch;

    memset( spad, 0, sizeof(real) * system->total_cap );
    copy_host_device( spad, x, sizeof(real) * system->total_cap,
            cudaMemcpyDeviceToHost, "Cuda_CG::x:get" );
    Dist( system, mpi_data, spad, REAL_PTR_TYPE, MPI_DOUBLE );

    copy_host_device( spad, x, sizeof(real) * system->total_cap,
            cudaMemcpyHostToDevice, "Cuda_CG::x:put" );
    Cuda_Matvec( control, H, x, workspace->d_workspace->q, system->N, system->total_cap );

    copy_host_device( spad, workspace->d_workspace->q, sizeof(real) * system->total_cap,
            cudaMemcpyDeviceToHost, "Cuda_CG::q:get" );
    Coll( system, mpi_data, spad, REAL_PTR_TYPE, MPI_DOUBLE );

    copy_host_device( spad, workspace->d_workspace->q, sizeof(real) * system->total_cap,
            cudaMemcpyHostToDevice, "Cuda_CG::q:put" );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &matvec_time );
    }
#endif

    Cuda_Vector_Sum( workspace->d_workspace->r , 1.0,  b, -1.0,
            workspace->d_workspace->q, system->n );

    Cuda_CG_Preconditioner( workspace->d_workspace->d, workspace->d_workspace->r,
            workspace->d_workspace->Hdia_inv, system->n );

    //TODO do the parallel_norm on the device for the local sum
    copy_host_device( spad, b, sizeof(real) * system->n,
            cudaMemcpyDeviceToHost, "Cuda_CG::b:get" );
    b_norm = Parallel_Norm( spad, system->n, mpi_data->world );

    //TODO do the parallel dot on the device for the local sum
    copy_host_device( spad, workspace->d_workspace->r, sizeof(real) * system->total_cap,
            cudaMemcpyDeviceToHost, "Cuda_CG::r:get" );
    copy_host_device( spad + system->total_cap, workspace->d_workspace->d, sizeof(real) * system->total_cap,
            cudaMemcpyDeviceToHost, "Cuda_CG::d:get" );
    sig_new = Parallel_Dot( spad, spad + system->total_cap, system->n,
            mpi_data->world );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < control->cm_solver_max_iters && SQRT(sig_new) / b_norm > tol; ++i )
    {
        copy_host_device( spad, workspace->d_workspace->d, sizeof(real) * system->total_cap,
                cudaMemcpyDeviceToHost, "Cuda_CG::d:get" );
        Dist( system, mpi_data, spad, REAL_PTR_TYPE, MPI_DOUBLE );
        copy_host_device( spad, workspace->d_workspace->d, sizeof(real) * system->total_cap,
                cudaMemcpyHostToDevice, "Cuda_CG::d:put" );

        Cuda_Matvec( control, H, workspace->d_workspace->d, workspace->d_workspace->q, system->N, system->total_cap );

        copy_host_device( spad, workspace->d_workspace->q, sizeof(real) * system->total_cap,
                cudaMemcpyDeviceToHost, "Cuda_CG::q:get" );
        Coll( system, mpi_data, spad, REAL_PTR_TYPE, MPI_DOUBLE );
        copy_host_device( spad, workspace->d_workspace->q, sizeof(real) * system->total_cap,
                cudaMemcpyHostToDevice, "Cuda_CG::q:get" );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        //TODO do the parallel dot on the device for the local sum
        copy_host_device( spad, workspace->d_workspace->d, sizeof(real) * system->n,
                cudaMemcpyDeviceToHost, "Cuda_CG::d:get" );
        copy_host_device( spad + system->n, workspace->d_workspace->q, sizeof(real) * system->n,
                cudaMemcpyDeviceToHost, "Cuda_CG::q:get" );
        tmp = Parallel_Dot( spad, spad + system->n, system->n, mpi_data->world );

        alpha = sig_new / tmp;
        //Cuda_Vector_Add( x, alpha, workspace->d_workspace->d, system->n );
        Cuda_Vector_Sum( x, alpha, workspace->d_workspace->d, 1.0, x, system->n );

        //Cuda_Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        Cuda_Vector_Sum( workspace->d_workspace->r, -alpha, workspace->d_workspace->q, 1.0,
                workspace->d_workspace->r, system->n );

        Cuda_CG_Preconditioner( workspace->d_workspace->p, workspace->d_workspace->r,
                workspace->d_workspace->Hdia_inv, system->n );

        sig_old = sig_new;

        //TODO do the parallel dot on the device for the local sum
        copy_host_device( spad, workspace->d_workspace->r, sizeof(real) * system->n,
                cudaMemcpyDeviceToHost, "Cuda_CG::r:get" );
        copy_host_device( spad + system->n, workspace->d_workspace->p, sizeof(real) * system->n,
                cudaMemcpyDeviceToHost, "Cuda_CG::p:get" );
        sig_new = Parallel_Dot( spad , spad + system->n, system->n, mpi_data->world );

        beta = sig_new / sig_old;
        Cuda_Vector_Sum( workspace->d_workspace->d, 1., workspace->d_workspace->p, beta,
                workspace->d_workspace->d, system->n );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif
    }

    return i;
}
