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

#include "GMRES.h"
#include "list.h"
#include "vector.h"
#include "index_utils.h"

#include "cuda_copy.h"
#include "cuda_utils.h"
#include "reduction.h"
#include "matvec.h"
#include "system_props.h"

#include "cublas_v2.h"
#include "cusparse_v2.h"

void Sparse_MatVec( sparse_matrix *A, real *x, real *b )
{
    int i, j, k, n, si, ei;
    real H;

    n = A->n;
    for( i = 0; i < n; ++i )
        b[i] = 0;

    for( i = 0; i < n; ++i ) {
        si = A->start[i];
        ei = A->start[i+1]-1;

        for( k = si; k < ei; ++k ) {
            j = A->entries[k].j;
            H = A->entries[k].val;
            b[j] += H * x[i]; 
            b[i] += H * x[j];
        }

        // the diagonal entry is the last one in
        b[i] += A->entries[k].val * x[i]; 
    }
}


void Forward_Subs( sparse_matrix *L, real *b, real *y )
{
    int i, pj, j, si, ei;
    real val;

    for( i = 0; i < L->n; ++i ) {
        y[i] = b[i];
        si = L->start[i];
        ei = L->start[i+1];
        for( pj = si; pj < ei-1; ++pj ){
            j = L->entries[pj].j;
            val = L->entries[pj].val;
            y[i] -= val * y[j];
        }
        y[i] /= L->entries[pj].val;
    }
}


void Backward_Subs( sparse_matrix *U, real *y, real *x )
{
    int i, pj, j, si, ei;
    real val;

    for( i = U->n-1; i >= 0; --i ) {
        x[i] = y[i];
        si = U->start[i];
        ei = U->start[i+1];
        for( pj = si+1; pj < ei; ++pj ){
            j = U->entries[pj].j;
            val = U->entries[pj].val;
            x[i] -= val * x[j];
        }
        x[i] /= U->entries[si].val;
    }
}


int GMRES( static_storage *workspace, sparse_matrix *H, 
        real *b, real tol, real *x, FILE *fout, reax_system* system)
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;

    N = H->n;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];  

    /* GMRES outer-loop */
    for( itr = 0; itr < MAX_ITR; ++itr ) {
        /* calculate r0 */
        Sparse_MatVec( H, x, workspace->b_prm );      

        for( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; /* pre-conditioner */    


        Vector_Sum(&workspace->v[ index_wkspace_sys (0,0,system) ], 1.,workspace->b_prc, -1., workspace->b_prm, N);
        workspace->g[0] = Norm( &workspace->v[index_wkspace_sys (0,0,system)], N );
        Vector_Scale( &workspace->v[ index_wkspace_sys (0,0,system) ], 1.0/workspace->g[0], &workspace->v[index_wkspace_sys(0,0,system)], N );

        /* GMRES inner-loop */
        for( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ ) {
            /* matvec */
            Sparse_MatVec( H, &workspace->v[index_wkspace_sys(j,0,system)], &workspace->v[index_wkspace_sys(j+1,0,system)] );

            for( k = 0; k < N; ++k )  
                workspace->v[ index_wkspace_sys (j+1,k,system)] *= workspace->Hdia_inv[k]; /*pre-conditioner*/ 

            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for( i = 0; i <= j; i++ ) {
                workspace->h[ index_wkspace_res (i,j) ] = Dot( &workspace->v[index_wkspace_sys(i,0,system)], &workspace->v[index_wkspace_sys(j+1,0,system)], N );
                Vector_Add( &workspace->v[index_wkspace_sys(j+1,0,system)], 
                        -workspace->h[index_wkspace_res (i,j) ], &workspace->v[index_wkspace_sys(i,0,system)], N );
            }


            workspace->h[ index_wkspace_res (j+1,j) ] = Norm( &workspace->v[index_wkspace_sys(j+1,0,system)], N );
            Vector_Scale( &workspace->v[index_wkspace_sys(j+1,0,system)], 
                    1. / workspace->h[ index_wkspace_res (j+1,j) ], &workspace->v[index_wkspace_sys(j+1,0,system)], N );
            // fprintf( stderr, "%d-%d: orthogonalization completed.\n", itr, j );


            /* Givens rotations on the upper-Hessenberg matrix to make it U */
            for( i = 0; i <= j; i++ )    {
                if( i == j ) {
                    cc = SQRT( SQR(workspace->h[ index_wkspace_res (j,j) ])+SQR(workspace->h[ index_wkspace_res (j+1,j) ]) );
                    workspace->hc[j] = workspace->h[ index_wkspace_res (j,j) ] / cc;
                    workspace->hs[j] = workspace->h[ index_wkspace_res (j+1,j) ] / cc;
                }

                tmp1 =  workspace->hc[i] * workspace->h[ index_wkspace_res (i,j) ] + 
                    workspace->hs[i] * workspace->h[ index_wkspace_res (i+1,j) ];
                tmp2 = -workspace->hs[i] * workspace->h[ index_wkspace_res (i,j) ] + 
                    workspace->hc[i] * workspace->h[ index_wkspace_res (i+1,j) ];

                workspace->h[ index_wkspace_res (i,j) ] = tmp1;
                workspace->h[ index_wkspace_res (i+1,j) ] = tmp2;
            } 

            /* apply Givens rotations to the rhs as well */
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j+1] = tmp2;

            // fprintf( stderr, "h: " );
            // for( i = 0; i <= j+1; ++i )
            //  fprintf( stderr, "%.6f ", workspace->h[i][j] );
            // fprintf( stderr, "\n" );
            //fprintf( stderr, "res: %.15e\n", workspace->g[j+1] );
        }


        /* solve Hy = g.
           H is now upper-triangular, do back-substitution */
        for( i = j-1; i >= 0; i-- ) {
            temp = workspace->g[i];      
            for( k = j-1; k > i; k-- )
                temp -= workspace->h[ index_wkspace_res (i,k) ] * workspace->y[k];

            workspace->y[i] = temp / workspace->h[ index_wkspace_res (i,i) ];
        }


        /* update x = x_0 + Vy */
        for( i = 0; i < j; i++ )
            Vector_Add( x, workspace->y[i], &workspace->v[index_wkspace_sys(i,0,system)], N );

        /* stopping condition */
        if( fabs(workspace->g[j]) / bnorm <= tol )
            break;
    }

    // Sparse_MatVec( H, x, workspace->b_prm );
    // for( i = 0; i < N; ++i )
    // workspace->b_prm[i] *= workspace->Hdia_inv[i];    
    // fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
    // for( i = 0; i < N; ++i )
    // fprintf( fout, "%10.5f%15.12f%15.12f\n", 
    // workspace->b_prc[i], workspace->b_prm[i], x[i] );*/

    // fprintf(fout,"GMRES outer:%d, inner:%d iters - residual norm: %25.20f\n", 
    //          itr, j, fabs( workspace->g[j] ) / bnorm );
    // data->timing.matvec += itr * RESTART + j;

    if( itr >= MAX_ITR ) {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART+1) + j + 1;
    }

    return itr * (RESTART+1) + j + 1;
}


/////////////////////////////////////////////////////////////////
//Cuda Functions for GMRES implementation
/////////////////////////////////////////////////////////////////
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

        //workspace->g[0] = Norm( &workspace->v[index_wkspace_sys (0,0,system)], N );
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
            //Sparse_MatVec( H, &workspace->v[index_wkspace_sys(j,0,system)], &workspace->v[index_wkspace_sys(j+1,0,system)] );
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


            //workspace->h[ index_wkspace_res (j+1,j) ] = Norm( &workspace->v[index_wkspace_sys(j+1,0,system)], N );
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

    cublasCheckError (cublasDnrm2 ( cublasHandle, N, b, 1, &bnorm ));

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

        //workspace->g[0] = Norm( &workspace->v[index_wkspace_sys (0,0,system)], N );
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


            //workspace->h[ index_wkspace_res (j+1,j) ] = Norm( &workspace->v[index_wkspace_sys(j+1,0,system)], N );
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


int GMRES_HouseHolder( static_storage *workspace, sparse_matrix *H, 
        real *b, real tol, real *x, FILE *fout, reax_system *system)
{
    int  i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[RESTART+2][10000], w[RESTART+2];
    real u[RESTART+2][10000];

    N = H->n;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];  

    // memset( x, 0, sizeof(real) * N );

    /* GMRES outer-loop */
    for( itr = 0; itr < MAX_ITR; ++itr ) {
        /* compute z = r0 */
        Sparse_MatVec( H, x, workspace->b_prm );      
        for( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; /* pre-conditioner */
        Vector_Sum( z[0], 1.,  workspace->b_prc, -1., workspace->b_prm, N );

        Vector_MakeZero( w, RESTART+1 );
        w[0] = Norm( z[0], N );

        Vector_Copy( u[0], z[0], N );
        u[0][0] += ( u[0][0] < 0.0 ? -1 : 1 ) * w[0];
        Vector_Scale( u[0], 1 / Norm( u[0], N ), u[0], N );

        w[0]    *= ( u[0][0] < 0.0 ?  1 :-1 );
        // fprintf( stderr, "\n\n%12.6f\n", w[0] );

        /* GMRES inner-loop */
        for( j = 0; j < RESTART && fabs( w[j] ) / bnorm > tol; j++ ) {
            /* compute v_j */
            Vector_Scale( z[j], -2 * u[j][j], u[j], N );
            z[j][j] += 1.; /* due to e_j */

            for( i = j-1; i >= 0; --i )
                Vector_Add( z[j]+i, -2 * Dot( u[i]+i, z[j]+i, N-i ), u[i]+i, N-i );


            /* matvec */
            Sparse_MatVec( H, z[j], v );

            for( k = 0; k < N; ++k )
                v[k] *= workspace->Hdia_inv[k]; /* pre-conditioner */

            for( i = 0; i <= j; ++i )
                Vector_Add( v+i, -2 * Dot( u[i]+i, v+i, N-i ), u[i]+i, N-i );


            if( !Vector_isZero( v + (j+1), N - (j+1) ) ) {
                /* compute the HouseHolder unit vector u_j+1 */
                for( i = 0; i <= j; ++i )  
                    u[j+1][i] = 0;

                Vector_Copy( u[j+1] + (j+1), v + (j+1), N - (j+1) );

                u[j+1][j+1] += ( v[j+1]<0.0 ? -1:1 ) * Norm( v+(j+1), N-(j+1) );

                Vector_Scale( u[j+1], 1 / Norm( u[j+1], N ), u[j+1], N );

                /* overwrite v with P_m+1 * v */
                v[j+1] -= 2 * Dot( u[j+1]+(j+1), v+(j+1), N-(j+1) ) * u[j+1][j+1];
                Vector_MakeZero( v + (j+2), N - (j+2) );
                // Vector_Add( v, -2 * Dot( u[j+1], v, N ), u[j+1], N );
            }


            /* prev Givens rots on the upper-Hessenberg matrix to make it U */
            for( i = 0; i < j; i++ ) {
                tmp1 =  workspace->hc[i] * v[i] + workspace->hs[i] * v[i+1];
                tmp2 = -workspace->hs[i] * v[i] + workspace->hc[i] * v[i+1];

                v[i]   = tmp1;
                v[i+1] = tmp2;
            }

            /* apply the new Givens rotation to H and right-hand side */
            if( fabs(v[j+1]) >= ALMOST_ZERO )    {
                cc = SQRT( SQR( v[j] ) + SQR( v[j+1] ) );
                workspace->hc[j] = v[j] / cc;
                workspace->hs[j] = v[j+1] / cc;

                tmp1 =  workspace->hc[j] * v[j] + workspace->hs[j] * v[j+1];
                tmp2 = -workspace->hs[j] * v[j] + workspace->hc[j] * v[j+1];

                v[j]   = tmp1;
                v[j+1] = tmp2;

                /* Givens rotations to rhs */
                tmp1 =  workspace->hc[j] * w[j];
                tmp2 = -workspace->hs[j] * w[j];
                w[j]   = tmp1;
                w[j+1] = tmp2;
            }

            /* extend R */
            for( i = 0; i <= j; ++i )
                workspace->h[ index_wkspace_res (i,j) ] = v[i];


            // fprintf( stderr, "h:" );
            // for( i = 0; i <= j+1 ; ++i )
            // fprintf( stderr, "%.6f ", h[i][j] );
            // fprintf( stderr, "\n" );
            // fprintf( stderr, "%12.6f\n", w[j+1] );
        }


        /* solve Hy = w.
           H is now upper-triangular, do back-substitution */
        for( i = j-1; i >= 0; i-- ) {
            temp = w[i];      
            for( k = j-1; k > i; k-- )
                temp -= workspace->h[ index_wkspace_res (i,k) ] * workspace->y[k];

            workspace->y[i] = temp / workspace->h[ index_wkspace_res (i,i) ];
        }

        // fprintf( stderr, "y: " );
        // for( i = 0; i < RESTART+1; ++i )
        //   fprintf( stderr, "%8.3f ", workspace->y[i] );


        /* update x = x_0 + Vy */
        // memset( z, 0, sizeof(real) * N );
        // for( i = j-1; i >= 0; i-- )
        //   {
        //     Vector_Copy( v, z, N );
        //     v[i] += workspace->y[i];
        //    
        //     Vector_Sum( z, 1., v, -2 * Dot( u[i], v, N ), u[i], N );
        //   }      
        //
        // fprintf( stderr, "\nz: " );
        // for( k = 0; k < N; ++k )
        // fprintf( stderr, "%6.2f ", z[k] );

        // fprintf( stderr, "\nx_bef: " );
        // for( i = 0; i < N; ++i )
        //   fprintf( stderr, "%6.2f ", x[i] );

        // Vector_Add( x, 1, z, N );
        for( i = j-1; i >= 0; i-- )
            Vector_Add( x, workspace->y[i], z[i], N );

        // fprintf( stderr, "\nx_aft: " );
        // for( i = 0; i < N; ++i )
        //   fprintf( stderr, "%6.2f ", x[i] );

        /* stopping condition */
        if( fabs( w[j] ) / bnorm <= tol )
            break;
    }

    // Sparse_MatVec( H, x, workspace->b_prm );
    // for( i = 0; i < N; ++i )
    // workspace->b_prm[i] *= workspace->Hdia_inv[i];

    // fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
    // for( i = 0; i < N; ++i )
    // fprintf( fout, "%10.5f%15.12f%15.12f\n", 
    // workspace->b_prc[i], workspace->b_prm[i], x[i] );

    //fprintf( fout,"GMRES outer:%d, inner:%d iters - residual norm: %15.10f\n", 
    //         itr, j, fabs( workspace->g[j] ) / bnorm );

    if( itr >= MAX_ITR ) {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART+1) + j + 1;
    }

    return itr * (RESTART+1) + j + 1;
}


int PGMRES( static_storage *workspace, sparse_matrix *H, real *b, real tol, 
        sparse_matrix *L, sparse_matrix *U, real *x, FILE *fout, reax_system *system )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;

    N = H->n;
    bnorm = Norm( b, N );

    /* GMRES outer-loop */
    for( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* calculate r0 */
        Sparse_MatVec( H, x, workspace->b_prm );      
        Vector_Sum( &workspace->v[index_wkspace_sys(0,0,system)], 1., b, -1., workspace->b_prm, N );
        Forward_Subs( L, &workspace->v[index_wkspace_sys(0,0,system)], &workspace->v[index_wkspace_sys(0,0,system)] );
        Backward_Subs( U, &workspace->v[index_wkspace_sys(0,0,system)], &workspace->v[index_wkspace_sys(0,0,system)] );
        workspace->g[0] = Norm( &workspace->v[index_wkspace_sys(0,0,system)], N );
        Vector_Scale( &workspace->v[index_wkspace_sys(0,0,system)], 1. / workspace->g[0], &workspace->v[index_wkspace_sys (0,0,system)], N );
        //fprintf( stderr, "res: %.15e\n", workspace->g[0] );

        /* GMRES inner-loop */
        for( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            /* matvec */
            Sparse_MatVec( H, &workspace->v[index_wkspace_sys (j,0,system)], &workspace->v[index_wkspace_sys (j+1,0,system)] );
            Forward_Subs( L, &workspace->v[index_wkspace_sys(j+1,0,system)], &workspace->v[index_wkspace_sys(j+1,0,system)] );
            Backward_Subs( U, &workspace->v[index_wkspace_sys(j+1,0,system)], &workspace->v[index_wkspace_sys(j+1,0,system)] );

            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for( i = 0; i < j-1; i++ )
            {
                workspace->h[ index_wkspace_res (i,j)] = 0;
            }

            //for( i = 0; i <= j; i++ ) {
            for( i = MAX(j-1,0); i <= j; i++ ) {
                workspace->h[index_wkspace_res (i,j)] = Dot( &workspace->v[index_wkspace_sys (i,0,system)], &workspace->v[index_wkspace_sys(j+1,0,system)], N );
                Vector_Add( &workspace->v[index_wkspace_sys(j+1,0,system)],-workspace->h[ index_wkspace_res (i,j) ], &workspace->v[index_wkspace_sys(i,0,system)], N );
            }

            workspace->h[index_wkspace_res (j+1,j) ] = Norm( &workspace->v[index_wkspace_sys (j+1,0,system)], N );
            Vector_Scale( &workspace->v[index_wkspace_sys(j+1,0,system)], 
                    1. / workspace->h[ index_wkspace_res (j+1,j)], &workspace->v[index_wkspace_sys(j+1,0,system)], N );
            // fprintf( stderr, "%d-%d: orthogonalization completed.\n", itr, j );

            /* Givens rotations on the upper-Hessenberg matrix to make it U */
            for( i = MAX(j-1,0); i <= j; i++ )
            {
                if( i == j )
                {
                    cc = SQRT( SQR(workspace->h[ index_wkspace_res (j,j) ])+SQR(workspace->h[ index_wkspace_res (j+1,j) ]) );
                    workspace->hc[j] = workspace->h[ index_wkspace_res (j,j) ] / cc;
                    workspace->hs[j] = workspace->h[ index_wkspace_res (j+1,j) ] / cc;
                }

                tmp1 =  workspace->hc[i] * workspace->h[ index_wkspace_res (i,j) ] + 
                    workspace->hs[i] * workspace->h[index_wkspace_res (i+1,j) ];
                tmp2 = -workspace->hs[i] * workspace->h[index_wkspace_res (i,j)] + 
                    workspace->hc[i] * workspace->h[index_wkspace_res (i+1,j) ];

                workspace->h[ index_wkspace_res (i,j) ] = tmp1;
                workspace->h[ index_wkspace_res (i+1,j) ] = tmp2;
            } 

            /* apply Givens rotations to the rhs as well */
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j+1] = tmp2;

            //fprintf( stderr, "h: " );
            //for( i = 0; i <= j+1; ++i )
            //fprintf( stderr, "%.6f ", workspace->h[i][j] );
            //fprintf( stderr, "\n" );
            //fprintf( stderr, "res: %.15e\n", workspace->g[j+1] );
        }


        /* solve Hy = g: H is now upper-triangular, do back-substitution */
        for( i = j-1; i >= 0; i-- )
        {
            temp = workspace->g[i];      
            for( k = j-1; k > i; k-- )
            {
                temp -= workspace->h[ index_wkspace_res (i,k) ] * workspace->y[k];
            }

            workspace->y[i] = temp / workspace->h[index_wkspace_res (i,i)];
        }

        /* update x = x_0 + Vy */
        Vector_MakeZero( workspace->p, N );
        for( i = 0; i < j; i++ )
            Vector_Add( workspace->p, workspace->y[i], &workspace->v[index_wkspace_sys(i,0,system)], N );
        //Backward_Subs( U, workspace->p, workspace->p );
        //Forward_Subs( L, workspace->p, workspace->p );
        Vector_Add( x, 1., workspace->p, N );

        /* stopping condition */
        if( fabs(workspace->g[j]) / bnorm <= tol )
        {
            break;
        }
    }

    // Sparse_MatVec( H, x, workspace->b_prm );
    // for( i = 0; i < N; ++i )
    // workspace->b_prm[i] *= workspace->Hdia_inv[i];    
    // fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
    // for( i = 0; i < N; ++i )
    // fprintf( fout, "%10.5f%15.12f%15.12f\n", 
    // workspace->b_prc[i], workspace->b_prm[i], x[i] );*/

    // fprintf(fout,"GMRES outer:%d, inner:%d iters - residual norm: %25.20f\n", 
    //          itr, j, fabs( workspace->g[j] ) / bnorm );
    // data->timing.matvec += itr * RESTART + j;

    if( itr >= MAX_ITR ) {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART+1) + j + 1;
    }

    return itr * (RESTART+1) + j + 1;
}



int PCG( static_storage *workspace, sparse_matrix *A, real *b, real tol, 
        sparse_matrix *L, sparse_matrix *U, real *x, FILE *fout, reax_system* system )
{
    int  i, N;
    real tmp, alpha, beta, b_norm, r_norm;
    real sig0, sig_old, sig_new;

    N = A->n;
    b_norm = Norm( b, N );
    //fprintf( stderr, "b_norm: %.15e\n", b_norm );

    Sparse_MatVec( A, x, workspace->q );
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, N );
    r_norm = Norm(workspace->r, N);
    //Print_Soln( workspace, x, q, b, N );
    //fprintf( stderr, "res: %.15e\n", r_norm );

    Forward_Subs( L, workspace->r, workspace->d );
    Backward_Subs( U, workspace->d, workspace->p );
    sig_new = Dot( workspace->r, workspace->p, N );
    sig0 = sig_new;

    for( i = 0; i < 200 && r_norm/b_norm > tol; ++i )
    {
        //for( i = 0; i < 200 && sig_new > SQR(tol) * sig0; ++i ) {
        Sparse_MatVec( A, workspace->p, workspace->q );
        tmp = Dot( workspace->q, workspace->p, N );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->p, N );
        //fprintf( stderr, "iter%d: |p|=%.15e |q|=%.15e tmp=%.15e\n",
        //     i+1, Norm(workspace->p,N), Norm(workspace->q,N), tmp );

        Vector_Add( workspace->r, -alpha, workspace->q, N );
        r_norm = Norm(workspace->r, N);
        //fprintf( stderr, "res: %.15e\n", r_norm );

        Forward_Subs( L, workspace->r, workspace->d );
        Backward_Subs( U, workspace->d, workspace->d );
        sig_old = sig_new;
        sig_new = Dot( workspace->r, workspace->d, N );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->p, 1., workspace->d, beta, workspace->p, N );
    }

    //fprintf( fout, "CG took %d iterations\n", i );
    if( i >= 200 ) {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}


int CG( static_storage *workspace, sparse_matrix *H, 
        real *b, real tol, real *x, FILE *fout, reax_system *system)
{
    int  i, j, N;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

    N = H->n;
    b_norm = Norm( b, N );
    //fprintf( stderr, "b_norm: %10.6f\n", b_norm );

    Sparse_MatVec( H, x, workspace->q );
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, N );
    for( j = 0; j < N; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];

    sig_new = Dot( workspace->r, workspace->d, N );
    sig0 = sig_new;
    //Print_Soln( workspace, x, q, b, N );
    //fprintf( stderr, "sig_new: %24.15e, d_norm:%24.15e, q_norm:%24.15e\n", 
    // sqrt(sig_new), Norm(workspace->d,N), Norm(workspace->q,N) );
    //fprintf( stderr, "sig_new: %f\n", sig_new );

    for( i = 0; i < 300 && SQRT(sig_new) / b_norm > tol; ++i ) {
        //for( i = 0; i < 300 && sig_new > SQR(tol)*sig0; ++i ) {
        Sparse_MatVec( H, workspace->d, workspace->q );
        tmp = Dot( workspace->d, workspace->q, N );
        //fprintf( stderr, "tmp: %f\n", tmp );
        alpha = sig_new / tmp;    
        Vector_Add( x, alpha, workspace->d, N );
        //fprintf( stderr, "d_norm:%24.15e, q_norm:%24.15e, tmp:%24.15e\n",
        //     Norm(workspace->d,N), Norm(workspace->q,N), tmp );

        Vector_Add( workspace->r, -alpha, workspace->q, N );    
        for( j = 0; j < N; ++j )
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];

        sig_old = sig_new;
        sig_new = Dot( workspace->r, workspace->p, N );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, N );
        //fprintf( stderr, "sig_new: %f\n", sig_new );
    }

    fprintf( stderr, "CG took %d iterations\n", i );

    if( i >= 300 ) {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}



/* Steepest Descent */
int SDM( static_storage *workspace, sparse_matrix *H, 
        real *b, real tol, real *x, FILE *fout )
{
    int  i, j, N;
    real tmp, alpha, beta, b_norm;
    real sig0, sig;

    N = H->n;
    b_norm = Norm( b, N );
    //fprintf( stderr, "b_norm: %10.6f\n", b_norm );

    Sparse_MatVec( H, x, workspace->q );
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, N );
    for( j = 0; j < N; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];

    sig = Dot( workspace->r, workspace->d, N );
    sig0 = sig;

    for( i = 0; i < 300 && SQRT(sig) / b_norm > tol; ++i ) {
        Sparse_MatVec( H, workspace->d, workspace->q );

        sig = Dot( workspace->r, workspace->d, N );
        tmp = Dot( workspace->d, workspace->q, N );
        alpha = sig / tmp;    

        Vector_Add( x, alpha, workspace->d, N );
        Vector_Add( workspace->r, -alpha, workspace->q, N );
        for( j = 0; j < N; ++j )
            workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];

        //fprintf( stderr, "d_norm:%24.15e, q_norm:%24.15e, tmp:%24.15e\n",
        //     Norm(workspace->d,N), Norm(workspace->q,N), tmp );
    }

    fprintf( stderr, "SDM took %d iterations\n", i );

    if( i >= 300 ) {
        fprintf( stderr, "SDM convergence failed!\n" );
        return i;
    }

    return i;
}
