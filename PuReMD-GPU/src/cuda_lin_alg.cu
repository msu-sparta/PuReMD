/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
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

#include "list.h"
#include "vector.h"
#include "index_utils.h"

#include "cuda_copy.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"
#include "system_props.h"

#include "cublas_v2.h"
#include "cusparse_v2.h"


//one thread per row
GLOBAL void Cuda_Matvec (sparse_matrix H, real *vec, real *results, int rows)
{
    real results_row = 0;
    int col;
    real val;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= rows) return;

    for (int c = H.start[i]; c < H.end[i]; c++)
    {
        col = H.entries [c].j;
        val = H.entries[c].val;

        results_row += val * vec [col];
    }

    results [i] = results_row;
}


//32 thread warp per matrix row.
//invoked as follows
// <<< system->N, 32 >>>
GLOBAL void Cuda_Matvec_csr (sparse_matrix H, real *vec, real *results, int num_rows)
{
    extern __shared__ real vals [];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane = thread_id & (32 - 1);

    int row_start;
    int row_end;

    // one warp per row
    //int row = warp_id;
    int row = warp_id;
    //if (row < num_rows)
    {
        vals[threadIdx.x] = 0;

        if (row < num_rows) {
            row_start = H.start[row];
            row_end = H.end[row];

            // compute running sum per thread
            for(int jj = row_start + lane; jj < row_end; jj += 32)
                vals[threadIdx.x] += H.entries[jj].val * vec [ H.entries[jj].j ];
            //vals[threadIdx.x] += H.val[jj] * vec [ H.j[jj] ];
        }

        __syncthreads ();

        // parallel reduction in shared memory
        //SIMD instructions with a WARP are synchronous -- so we do not need to synch here
        if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16]; __syncthreads();
        if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8]; __syncthreads ();
        if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4]; __syncthreads ();
        if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2]; __syncthreads ();
        if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1]; __syncthreads ();

        // first thread writes the result
        if (lane == 0 && row < num_rows)
            results[row] = vals[threadIdx.x];
    }
}


GLOBAL void GMRES_Diagonal_Preconditioner (real *b_proc, real *b, real *Hdia_inv, int entries)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= entries) return;

    b_proc [i] = b[i] * Hdia_inv[i];
}


GLOBAL void GMRES_Givens_Rotation (int j, real *h, real *hc, real *hs, real g_j, real *output)
{
    real tmp1, tmp2, cc;

    for( int i = 0; i <= j; i++ )    {
        if( i == j ) {
            cc = SQRT( SQR(h[ index_wkspace_res (j,j) ])+SQR(h[ index_wkspace_res (j+1,j) ]) );
            hc[j] = h[ index_wkspace_res (j,j) ] / cc;
            hs[j] = h[ index_wkspace_res (j+1,j) ] / cc;
        }

        tmp1 =  hc[i] * h[ index_wkspace_res (i,j) ] + hs[i] * h[ index_wkspace_res (i+1,j) ];
        tmp2 = -hs[i] * h[ index_wkspace_res (i,j) ] + hc[i] * h[ index_wkspace_res (i+1,j) ];

        h[ index_wkspace_res (i,j) ] = tmp1;
        h[ index_wkspace_res (i+1,j) ] = tmp2;
    } 

    /* apply Givens rotations to the rhs as well */
    tmp1 =  hc[j] * g_j;
    tmp2 = -hs[j] * g_j;

    output[0] = tmp1;
    output[1] = tmp2;
}


GLOBAL void GMRES_BackSubstitution (int j, real *g, real *h, real *y)
{
    real temp;
    for( int i = j-1; i >= 0; i-- ) {
        temp = g[i];      
        for( int k = j-1; k > i; k-- )
            temp -= h[ index_wkspace_res (i,k) ] * y[k];

        y[i] = temp / h[ index_wkspace_res (i,i) ];
    }
}


int Cuda_GMRES( static_storage *workspace, real *b, real tol, real *x )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v_add_tmp;
    sparse_matrix *H = &workspace->H;
    real t_start, t_elapsed;
    real *spad = (real *)scratch;
    real *g = (real *) calloc ((RESTART+1), REAL_SIZE);

    N = H->n;

    cuda_memset(spad, 0, REAL_SIZE * H->n * 2, RES_SCRATCH );

    Cuda_Norm <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>>
        (b, spad, H->n, INITIAL);
    cudaThreadSynchronize();
    cudaCheckError();

    Cuda_Norm <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>>
        (spad, spad + BLOCKS_POW_2, BLOCKS_POW_2, FINAL);
    cudaThreadSynchronize();
    cudaCheckError();

    copy_host_device( &bnorm, spad + BLOCKS_POW_2, REAL_SIZE,
            cudaMemcpyDeviceToHost, __LINE__);

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Norm of the array is %e \n", bnorm );
#endif

    /* apply the diagonal pre-conditioner to rhs */
    GMRES_Diagonal_Preconditioner <<<BLOCKS, BLOCK_SIZE>>>
        (workspace->b_prc, b, workspace->Hdia_inv, N);
    cudaThreadSynchronize();
    cudaCheckError();

    /* GMRES outer-loop */
    for( itr = 0; itr < MAX_ITR; ++itr ) {
        /* calculate r0 */
        //Sparse_MatVec( H, x, workspace->b_prm );      
        Cuda_Matvec_csr <<<MATVEC_BLOCKS, MATVEC_BLOCK_SIZE, REAL_SIZE * MATVEC_BLOCK_SIZE>>>
            ( *H, x, workspace->b_prm, N );
        cudaThreadSynchronize();
        cudaCheckError();

        GMRES_Diagonal_Preconditioner <<< BLOCKS, BLOCK_SIZE >>>
            (workspace->b_prm, workspace->b_prm, workspace->Hdia_inv, N);
        cudaThreadSynchronize();
        cudaCheckError();

        Cuda_Vector_Sum <<< BLOCKS, BLOCK_SIZE >>>
            (&workspace->v[ index_wkspace_sys (0,0,N) ], 1.,
             workspace->b_prc, -1., workspace->b_prm, N);
        cudaThreadSynchronize();
        cudaCheckError ();

        //workspace->g[0] = Norm( &workspace->v[index_wkspace_sys (0,0,system->N)], N );
        {
            cuda_memset( spad, 0, REAL_SIZE * H->n * 2, RES_SCRATCH );

            Cuda_Norm <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>> 
                (&workspace->v [index_wkspace_sys (0, 0, N)], spad, N, INITIAL);
            cudaThreadSynchronize();
            cudaCheckError();

            Cuda_Norm <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>>
                (spad, &workspace->g[0], BLOCKS_POW_2, FINAL);
            cudaThreadSynchronize();
            cudaCheckError();

            copy_host_device( g, workspace->g, REAL_SIZE,
                    cudaMemcpyDeviceToHost, RES_STORAGE_G);
        }

        Cuda_Vector_Scale<<< BLOCKS, BLOCK_SIZE >>>
            ( &workspace->v[ index_wkspace_sys (0,0,N) ], 1.0/g[0],
              &workspace->v[index_wkspace_sys(0,0,N)], N );
        cudaThreadSynchronize();
        cudaCheckError();

        /* GMRES inner-loop */
#ifdef __DEBUG_CUDA__
        fprintf( stderr,
                " Inner loop inputs bnorm : %f , tol : %f g[j] : %f \n", bnorm,
                tol, g[0] );
#endif

        for( j = 0; j < RESTART && fabs(g[j]) / bnorm > tol; j++ ) {
            /* matvec */
            //Sparse_MatVec( H, &workspace->v[index_wkspace_sys(j,0,system->N)], &workspace->v[index_wkspace_sys(j+1,0,system->N)] );
            Cuda_Matvec_csr<<<MATVEC_BLOCKS, MATVEC_BLOCK_SIZE, REAL_SIZE * MATVEC_BLOCK_SIZE>>> 
                ( *H, &workspace->v[ index_wkspace_sys (j, 0, N)],
                  &workspace->v[ index_wkspace_sys (j+1, 0, N) ], N );
            cudaThreadSynchronize();
            cudaCheckError();

            GMRES_Diagonal_Preconditioner<<<BLOCKS, BLOCK_SIZE>>>
                (&workspace->v[ index_wkspace_sys (j+1,0,N) ],
                 &workspace->v[ index_wkspace_sys( j+1,0,N) ],
                 workspace->Hdia_inv, N );
            cudaThreadSynchronize();
            cudaCheckError();


            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for( i = 0; i <= j; i++ )
            {
                Cuda_Dot <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>>
                    (&workspace->v[index_wkspace_sys(i,0,N)],
                     &workspace->v[index_wkspace_sys(j+1,0,N)], spad, N);
                cudaThreadSynchronize();
                cudaCheckError();

                Cuda_reduction<<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>>
                    (spad, &workspace->h[ index_wkspace_res (i,j) ], BLOCKS_POW_2);
                cudaThreadSynchronize();
                cudaCheckError();

                copy_host_device (&v_add_tmp, &workspace->h[ index_wkspace_res (i,j)], REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);

                Cuda_Vector_Add<<< BLOCKS, BLOCK_SIZE >>>
                    ( &workspace->v[index_wkspace_sys(j+1,0,N)], 
                      -v_add_tmp, &workspace->v[index_wkspace_sys(i,0,N)], N );
                cudaThreadSynchronize();
                cudaCheckError();
            }

            //workspace->h[ index_wkspace_res (j+1,j) ] = Norm( &workspace->v[index_wkspace_sys(j+1,0,system->N)], N );
            cuda_memset(spad, 0, REAL_SIZE * N * 2, RES_SCRATCH );

            Cuda_Norm<<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>>
                (&workspace->v[index_wkspace_sys(j+1,0,N)], spad, N, INITIAL);
            cudaThreadSynchronize();
            cudaCheckError();

            Cuda_Norm<<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>>
                (spad, &workspace->h[ index_wkspace_res (j+1,j) ], BLOCKS_POW_2, FINAL);
            cudaThreadSynchronize();
            cudaCheckError();

            copy_host_device(&v_add_tmp,
                    &workspace->h[ index_wkspace_res (j+1,j) ], REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);

            Cuda_Vector_Scale<<< BLOCKS, BLOCK_SIZE >>>
                ( &workspace->v[index_wkspace_sys(j+1,0,N)], 
                  1. / v_add_tmp, &workspace->v[index_wkspace_sys(j+1,0,N)], N );
            cudaThreadSynchronize();
            cudaCheckError();

            /* Givens rotations on the upper-Hessenberg matrix to make it U */
            GMRES_Givens_Rotation<<<1, 1>>>
                (j, workspace->h, workspace->hc, workspace->hs, g[j], spad);
            cudaThreadSynchronize();
            cudaCheckError();
            copy_host_device(&g[j], spad, 2 * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
        }

        copy_host_device(g, workspace->g, (RESTART+1)*REAL_SIZE,
                cudaMemcpyHostToDevice, __LINE__);

        /* solve Hy = g.
           H is now upper-triangular, do back-substitution */
        copy_host_device(g, spad, (RESTART+1) * REAL_SIZE,
                cudaMemcpyHostToDevice, RES_STORAGE_G);
        GMRES_BackSubstitution<<<1, 1>>>
            (j, spad, workspace->h, workspace->y);
        cudaThreadSynchronize();
        cudaCheckError();

        /* update x = x_0 + Vy */
        for( i = 0; i < j; i++ )
        {
            copy_host_device(&v_add_tmp, &workspace->y[i], REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
            Cuda_Vector_Add <<<BLOCKS, BLOCK_SIZE>>>
                ( x, v_add_tmp, &workspace->v[index_wkspace_sys(i,0,N)], N );
            cudaThreadSynchronize ();
            cudaCheckError();
        }

        /* stopping condition */
        if( fabs(g[j]) / bnorm <= tol )
        {
            break;
        }
    }

    if( itr >= MAX_ITR ) {
        fprintf( stderr, "GMRES convergence failed\n" );
        return itr * (RESTART+1) + j + 1;
    }

#ifdef __DEBUG_CUDA__
    fprintf (stderr, " GPU values itr : %d, RESTART: %d, j: %d \n", itr, RESTART, j);
#endif

    return itr * (RESTART+1) + j + 1;
}


int Cublas_GMRES(reax_system *system, static_storage *workspace, real *b, real tol, real *x )
{

    real CSR_ALPHA = 1, CSR_BETA = 0;

    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v_add_tmp;
    sparse_matrix *H = &workspace->H;
    real t_start, t_elapsed;
    real *spad = (real *)scratch;
    real *g = (real *) calloc ((RESTART+1), REAL_SIZE);

    N = H->n;

    cuda_memset (spad, 0, REAL_SIZE * H->n * 2, RES_SCRATCH );

    /*
       Cuda_Norm <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>> (b, spad, H->n, INITIAL);
       cudaThreadSynchronize ();
       cudaCheckError ();

       Cuda_Norm <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> (spad, spad + BLOCKS_POW_2, BLOCKS_POW_2, FINAL);
       cudaThreadSynchronize ();
       cudaCheckError ();

       copy_host_device ( &bnorm, spad + BLOCKS_POW_2, REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
     */

    cublasCheckError( cublasDnrm2( cublasHandle, N, b, 1, &bnorm ) );

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Norm of the array is %e \n", bnorm );
#endif

    /* apply the diagonal pre-conditioner to rhs */
    GMRES_Diagonal_Preconditioner <<<BLOCKS, BLOCK_SIZE>>>
        (workspace->b_prc, b, workspace->Hdia_inv, N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    /* GMRES outer-loop */
    for( itr = 0; itr < MAX_ITR; ++itr ) {
        /* calculate r0 */
        //Sparse_MatVec( H, x, workspace->b_prm );      
        Cuda_Matvec_csr <<<MATVEC_BLOCKS, MATVEC_BLOCK_SIZE, REAL_SIZE * MATVEC_BLOCK_SIZE>>> ( *H, x, workspace->b_prm, N );
        cudaThreadSynchronize ();
        cudaCheckError ();

        GMRES_Diagonal_Preconditioner <<< BLOCKS, BLOCK_SIZE >>>
            (workspace->b_prm, workspace->b_prm, workspace->Hdia_inv, N);
        cudaThreadSynchronize ();
        cudaCheckError ();

        /*
           Cuda_Vector_Sum <<< BLOCKS, BLOCK_SIZE >>>
           (&workspace->v[ index_wkspace_sys (0,0,N) ], 1.,workspace->b_prc, -1., workspace->b_prm, N);
           cudaThreadSynchronize ();
           cudaCheckError ();
         */
        cuda_memset (workspace->v, 0, REAL_SIZE * (RESTART+1) * N, RES_STORAGE_V);

        double D_ONE = 1.;
        double D_MINUS_ONE = -1.;
        cublasCheckError (cublasDaxpy (cublasHandle, N, &D_ONE, workspace->b_prc, 1, &workspace->v[ index_wkspace_sys (0,0,N) ], 1));
        cublasCheckError (cublasDaxpy (cublasHandle, N, &D_MINUS_ONE, workspace->b_prm, 1, &workspace->v[ index_wkspace_sys (0,0,N) ], 1));

        //workspace->g[0] = Norm( &workspace->v[index_wkspace_sys (0,0,system->N)], N );
        {
            /*
               cuda_memset (spad, 0, REAL_SIZE * H->n * 2, RES_SCRATCH );

               Cuda_Norm <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>> 
               (&workspace->v [index_wkspace_sys (0, 0, N)], spad, N, INITIAL);
               cudaThreadSynchronize ();
               cudaCheckError ();

               Cuda_Norm <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> (spad, &workspace->g[0], BLOCKS_POW_2, FINAL);
               cudaThreadSynchronize ();
               cudaCheckError ();

               copy_host_device( g, workspace->g, REAL_SIZE, cudaMemcpyDeviceToHost, RES_STORAGE_G);
             */

            cublasCheckError (cublasDnrm2 ( cublasHandle, N, &workspace->v [index_wkspace_sys (0, 0, N)], 1, g ));
            copy_host_device( g, workspace->g, REAL_SIZE, cudaMemcpyHostToDevice, RES_STORAGE_G);
        }

        /*
           Cuda_Vector_Scale <<< BLOCKS, BLOCK_SIZE >>>
           ( &workspace->v[ index_wkspace_sys (0,0,N) ], 1.0/g[0], &workspace->v[index_wkspace_sys(0,0,N)], N );
           cudaThreadSynchronize ();
           cudaCheckError ();
         */

        double D_SCALE = 1.0 / g[0];
        cublasCheckError (cublasDscal (cublasHandle, N, &D_SCALE, &workspace->v[ index_wkspace_sys (0,0,N) ], 1));


        /* GMRES inner-loop */
#ifdef __DEBUG_CUDA__
        fprintf (stderr, " Inner loop inputs bnorm : %f , tol : %f g[j] : %f \n", bnorm, tol, g[0] );
#endif
        for( j = 0; j < RESTART && fabs(g[j]) / bnorm > tol; j++ ) {
            /* matvec */
            Cuda_Matvec_csr 
                <<<MATVEC_BLOCKS, MATVEC_BLOCK_SIZE, REAL_SIZE * MATVEC_BLOCK_SIZE>>> 
                ( *H, &workspace->v[ index_wkspace_sys (j, 0, N)], &workspace->v[ index_wkspace_sys (j+1, 0, N) ], N );
            cudaThreadSynchronize ();
            cudaCheckError ();

            GMRES_Diagonal_Preconditioner <<<BLOCKS, BLOCK_SIZE>>>
                (&workspace->v[ index_wkspace_sys (j+1,0,N) ], &workspace->v[ index_wkspace_sys (j+1,0,N) ], workspace->Hdia_inv, N);
            cudaThreadSynchronize ();
            cudaCheckError ();


            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for( i = 0; i <= j; i++ ) {

                /*
                   Cuda_Dot <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>>
                   (&workspace->v[index_wkspace_sys(i,0,N)], &workspace->v[index_wkspace_sys(j+1,0,N)], spad, N);
                   cudaThreadSynchronize ();
                   cudaCheckError ();

                   Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> (spad, &workspace->h[ index_wkspace_res (i,j) ], BLOCKS_POW_2);
                   cudaThreadSynchronize ();
                   cudaCheckError ();

                   copy_host_device (&v_add_tmp, &workspace->h[ index_wkspace_res (i,j)], REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
                 */

                cublasCheckError (cublasDdot (cublasHandle, N, &workspace->v[index_wkspace_sys(i,0,N)], 1, 
                            &workspace->v[index_wkspace_sys(j+1,0,N)], 1, 
                            &v_add_tmp));
                copy_host_device (&v_add_tmp, &workspace->h[ index_wkspace_res (i,j)], REAL_SIZE, cudaMemcpyHostToDevice, __LINE__);

                /*
                   Cuda_Vector_Add <<< BLOCKS, BLOCK_SIZE >>>
                   ( &workspace->v[index_wkspace_sys(j+1,0,N)], 
                   -v_add_tmp, &workspace->v[index_wkspace_sys(i,0,N)], N );
                   cudaThreadSynchronize ();
                   cudaCheckError ();
                 */

                double NEG_V_ADD_TMP = -v_add_tmp;
                cublasCheckError (cublasDaxpy (cublasHandle, N, &NEG_V_ADD_TMP, &workspace->v[index_wkspace_sys(i,0,N)], 1, 
                            &workspace->v[index_wkspace_sys(j+1,0,N)], 1 ));
            }


            //workspace->h[ index_wkspace_res (j+1,j) ] = Norm( &workspace->v[index_wkspace_sys(j+1,0,system->N)], N );
            /*
               cuda_memset (spad, 0, REAL_SIZE * N * 2, RES_SCRATCH );

               Cuda_Norm <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>> (&workspace->v[index_wkspace_sys(j+1,0,N)], spad, N, INITIAL);
               cudaThreadSynchronize ();
               cudaCheckError ();

               Cuda_Norm <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> (spad, &workspace->h[ index_wkspace_res (j+1,j) ], BLOCKS_POW_2, FINAL);
               cudaThreadSynchronize ();
               cudaCheckError ();

               copy_host_device (&v_add_tmp, &workspace->h[ index_wkspace_res (j+1,j) ], REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
             */
            cublasCheckError (cublasDnrm2 ( cublasHandle, N, &workspace->v [index_wkspace_sys (j+1, 0, N)], 1, &v_add_tmp ));
            copy_host_device (&v_add_tmp, &workspace->h[ index_wkspace_res (j+1,j) ], REAL_SIZE, cudaMemcpyHostToDevice, __LINE__);


            /*
               Cuda_Vector_Scale <<< BLOCKS, BLOCK_SIZE >>>
               ( &workspace->v[index_wkspace_sys(j+1,0,N)], 
               1. / v_add_tmp, &workspace->v[index_wkspace_sys(j+1,0,N)], N );
               cudaThreadSynchronize ();
               cudaCheckError ();
             */
            double REC_V_ADD_TMP = 1. / v_add_tmp;
            cublasCheckError (cublasDscal (cublasHandle, N, &REC_V_ADD_TMP,  &workspace->v[index_wkspace_sys(j+1,0,N)], 1));



            /* Givens rotations on the upper-Hessenberg matrix to make it U */
            GMRES_Givens_Rotation <<<1, 1>>>
                (j, workspace->h, workspace->hc, workspace->hs, g[j], spad);
            cudaThreadSynchronize ();
            cudaCheckError ();
            copy_host_device (&g[j], spad, 2 * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
        }

        copy_host_device (g, workspace->g, (RESTART+1)*REAL_SIZE, cudaMemcpyHostToDevice, __LINE__);

        /* solve Hy = g.
           H is now upper-triangular, do back-substitution */
        copy_host_device (g, spad, (RESTART+1) * REAL_SIZE, cudaMemcpyHostToDevice, RES_STORAGE_G);
        GMRES_BackSubstitution <<<1, 1>>>
            (j, spad, workspace->h, workspace->y);
        cudaThreadSynchronize ();
        cudaCheckError ();

        /* update x = x_0 + Vy */
        for( i = 0; i < j; i++ )
        {
            /*
               copy_host_device (&v_add_tmp, &workspace->y[i], REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
               Cuda_Vector_Add <<<BLOCKS, BLOCK_SIZE>>>
               ( x, v_add_tmp, &workspace->v[index_wkspace_sys(i,0,N)], N );
               cudaThreadSynchronize ();
               cudaCheckError ();
             */

            copy_host_device (&v_add_tmp, &workspace->y[i], REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
            cublasCheckError (cublasDaxpy (cublasHandle, N, &v_add_tmp, &workspace->v[index_wkspace_sys(i,0,N)], 1, 
                        x, 1));
        }

        /* stopping condition */
        if( fabs(g[j]) / bnorm <= tol )
            break;
    }

    if( itr >= MAX_ITR ) {
        fprintf( stderr, "GMRES convergence failed\n" );
        return itr * (RESTART+1) + j + 1;
    }

#ifdef __DEBUG_CUDA__
    fprintf (stderr, " GPU values itr : %d, RESTART: %d, j: %d \n", itr, RESTART, j);
#endif

    return itr * (RESTART+1) + j + 1;
}


