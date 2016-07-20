/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

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

#include "allocate.h"
#include "GMRES.h"
#include "list.h"
#include "vector.h"

#include <omp.h>


typedef enum
{
    LOWER = 0,
    UPPER = 1,
} TRIANGULARITY;


/* sparse matrix-vector product Ax=b
 * where:
 *   A: lower triangular matrix
 *   x: vector
 *   b: vector (result) */
static void Sparse_MatVec( const sparse_matrix * const A,
                           const real * const x, real * const b )
{
    int i, j, k, n, si, ei;
    real H;
#ifdef _OPENMP
    static real *b_local;
    unsigned int tid;
#endif

    n = A->n;
    Vector_MakeZero( b, n );

    #pragma omp parallel \
        default(none) shared(n, b_local) private(si, ei, H, i, j, k, tid)
    {
#ifdef _OPENMP
        tid = omp_get_thread_num();

        #pragma omp master
        {
            /* keep b_local for program duration to avoid allocate/free
             * overhead per Sparse_MatVec call*/
            if ( b_local == NULL )
            {
                if ( (b_local = (real*) malloc( omp_get_num_threads() * n * sizeof(real))) == NULL )
		{
                    exit( INSUFFICIENT_SPACE );
		}
            }

            Vector_MakeZero( (real * const)b_local, omp_get_num_threads() * n );
        }
        #pragma omp barrier

#endif
        #pragma omp for schedule(guided)
        for ( i = 0; i < n; ++i )
        {
            si = A->start[i];
            ei = A->start[i + 1] - 1;

            for ( k = si; k < ei; ++k )
            {
                j = A->entries[k].j;
                H = A->entries[k].val;
#ifdef _OPENMP
                b_local[tid * n + j] += H * x[i];
                b_local[tid * n + i] += H * x[j];
#else
                b[j] += H * x[i];
                b[i] += H * x[j];
#endif
            }

            // the diagonal entry is the last one in
#ifdef _OPENMP
            b_local[tid * n + i] += A->entries[k].val * x[i];
#else
            b[i] += A->entries[k].val * x[i];
#endif
        }
#ifdef _OPENMP
        #pragma omp for schedule(guided)
        for ( i = 0; i < n; ++i )
        {
            for ( j = 0; j < omp_get_num_threads(); ++j )
            {
                b[i] += b_local[j * n + i];
            }
        }
#endif
    }

}


/* sparse matrix-vector product Gx=b (for Jacobi iteration),
 * followed by vector addition of D^{-1}b,
 * where G = (I - D^{-1}R) (G not explicitly computed and stored)
 *   R: strictly triangular matrix (diagonals not used)
 *   tri: triangularity of A (lower/upper)
 *   D^{-1} (D_inv): inverse of the diagonal of R
 *   x: vector
 *   b: vector (result)
 *   D^{-1}b (Dinv_b): precomputed vector-vector product */
static void Sparse_MatVec_Vector_Add( const sparse_matrix * const R,
                            const TRIANGULARITY tri, const real * const Dinv,
                            const real * const x, real * const b, const real * const Dinv_b)
{
    int i, k, si = 0, ei = 0;
#ifdef _OPENMP
    static real *b_local;
    unsigned int tid;
#endif

    Vector_MakeZero( b, R->n );

    #pragma omp parallel \
        default(none) shared(b_local) private(si, ei, i, k, tid)
    {
#ifdef _OPENMP
        tid = omp_get_thread_num();

        #pragma omp master
        {
            /* keep b_local for program duration to avoid allocate/free
             * overhead per Sparse_MatVec call*/
            if ( b_local == NULL )
            {
                if ( (b_local = (real*) malloc( omp_get_num_threads() * R->n * sizeof(real))) == NULL )
                {
                    exit( INSUFFICIENT_SPACE );
                }
	    }

	    Vector_MakeZero( b_local, omp_get_num_threads() * R->n );
	}
        #pragma omp barrier

#endif
        #pragma omp for schedule(guided)
        for ( i = 0; i < R->n; ++i )
        {
            if (tri == LOWER)
            {
                si = R->start[i];
                ei = R->start[i + 1] - 1;
            }
            else if (tri == UPPER)
            {

                si = R->start[i] + 1;
                ei = R->start[i + 1];
            }

            for ( k = si; k < ei; ++k )
            {
#ifdef _OPENMP
                b_local[tid * R->n + i] += R->entries[k].val * x[R->entries[k].j];
#else
                b[i] += R->entries[k].val * x[R->entries[k].j];
#endif
            }
#ifdef _OPENMP
            b_local[tid * R->n + i] *= -Dinv[i];
#else
            b[i] *= -Dinv[i];
#endif
        }
#ifdef _OPENMP
        #pragma omp for schedule(guided)
        for ( i = 0; i < R->n; ++i )
        {
            for ( k = 0; k < omp_get_num_threads(); ++k )
            {
                b[i] += b_local[k * R->n + i];
            }

	    b[i] += Dinv_b[i];
        }
#endif
    }
}


/* solve sparse lower triangular linear system using forward substitution */
static void Forward_Subs( const sparse_matrix * const L, const real * const b, real * const y )
{
    int i, pj, j, si, ei;
    real val;

    for ( i = 0; i < L->n; ++i )
    {
        y[i] = b[i];
        si = L->start[i];
        ei = L->start[i + 1];
        for ( pj = si; pj < ei - 1; ++pj )
        {
            // TODO: remove assignments? compiler optimizes away?
            j = L->entries[pj].j;
            val = L->entries[pj].val;
            y[i] -= val * y[j];
        }
        y[i] /= L->entries[pj].val;
    }
}


/* solve sparse upper triangular linear system using backward substitution */
static void Backward_Subs( const sparse_matrix * const U, const real * const y, real * const x )
{
    int i, pj, j, si, ei;
    real val;

    for ( i = U->n - 1; i >= 0; --i )
    {
        x[i] = y[i];
        si = U->start[i];
        ei = U->start[i + 1];
        for ( pj = si + 1; pj < ei; ++pj )
        {
            // TODO: remove assignments? compiler optimizes away?
            j = U->entries[pj].j;
            val = U->entries[pj].val;
            x[i] -= val * x[j];
        }
        x[i] /= U->entries[si].val;
    }
}


/* Jacobi iteration using truncated Neumann series: x_{k+1} = Gx_k + D^{-1}b
 * where:
 *   G = I - D^{-1}R
 *   R = triangular matrix
 *   D = diagonal matrix, diagonals from R
 *
 * Note: used during the backsolves when applying preconditioners with
 * triangular factors in iterative linear solvers
 *
 * Note: Newmann series arises from series expansion of the inverse of
 * the coefficient matrix in the triangular system */
static void Jacobi_Iter( const sparse_matrix * const R, const TRIANGULARITY tri,
                         const real * const Dinv, const unsigned int n,
                         const real * const b, real * const x, const unsigned int maxiter )
{
    unsigned int i, k, si = 0, ei = 0, iter = 0;
#ifdef _OPENMP
    static real *b_local;
    unsigned int tid;
#endif
    static real *Dinv_b, *rp, *rp2, *rp3;

    #pragma omp parallel \
        default(none) shared(b_local, Dinv_b, rp, rp2, rp3, iter, stderr) private(si, ei, i, k, tid)
    {
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
    
        #pragma omp master
        {
            /* keep b_local for program duration to avoid allocate/free
             * overhead per Sparse_MatVec call*/
            if ( Dinv_b == NULL )
            {
                if ( (Dinv_b = (real*) malloc(sizeof(real) * n)) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit(INSUFFICIENT_SPACE);
                }
            }
            if ( rp == NULL )
            {
                if ( (rp = (real*) malloc(sizeof(real) * n)) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit(INSUFFICIENT_SPACE);
                }
            }
            if ( rp2 == NULL )
            {
                    if ( (rp2 = (real*) malloc(sizeof(real) * n)) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit(INSUFFICIENT_SPACE);
                }
            }
#ifdef _OPENMP
            if ( b_local == NULL )
            {
                if ( (b_local = (real*) malloc( omp_get_num_threads() * R->n * sizeof(real))) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit( INSUFFICIENT_SPACE );
                }
	    }
#endif
    
            Vector_MakeZero( rp, n );
	}
    
        #pragma omp barrier
    
        /* precompute and cache, as invariant in loop below */
        #pragma omp for schedule(guided)
        for ( i = 0; i < n; ++i )
        {
            Dinv_b[i] = Dinv[i] * b[i];
        }

        #pragma omp barrier

        do
        {
            // x_{k+1} = G*x_{k} + Dinv*b;
	    //Sparse_MatVec_Vector_Add( (sparse_matrix*)R, tri, Dinv, rp, rp2, Dinv_b );
            #pragma omp master
            {
                Vector_MakeZero( rp2, R->n );
#ifdef _OPENMP
    	        Vector_MakeZero( b_local, omp_get_num_threads() * R->n );
#endif
            }
    
            #pragma omp barrier

            #pragma omp for schedule(guided)
            for ( i = 0; i < R->n; ++i )
            {
                if (tri == LOWER)
                {
                    si = R->start[i];
                    ei = R->start[i + 1] - 1;
                }
                else if (tri == UPPER)
                {
    
                    si = R->start[i] + 1;
                    ei = R->start[i + 1];
                }
    
                for ( k = si; k < ei; ++k )
                {
#ifdef _OPENMP
                    b_local[tid * R->n + i] += R->entries[k].val * rp[R->entries[k].j];
#else
                    rp2[i] += R->entries[k].val * rp[R->entries[k].j];
#endif
                }
#ifdef _OPENMP
                b_local[tid * R->n + i] *= -Dinv[i];
#else
                rp2[i] *= -Dinv[i];
#endif
            }
	    
            #pragma omp barrier

            #pragma omp for schedule(guided)
            for ( i = 0; i < R->n; ++i )
            {
#ifdef _OPENMP
                for ( k = 0; k < omp_get_num_threads(); ++k )
                {
                    rp2[i] += b_local[k * R->n + i];
                }
#endif

    	        rp2[i] += Dinv_b[i];
            }

            #pragma omp master
            {
                rp3 = rp;
                rp = rp2;
                rp2 = rp3;
                ++iter;
            }

            #pragma omp barrier
        }
        while ( iter < maxiter );
    }

    Vector_Copy( x, rp, n );
}


/* generalized minimual residual iterative solver for sparse linear systems,
 * diagonal preconditioner */
int GMRES( static_storage *workspace, sparse_matrix *H,
           real *b, real tol, real *x, FILE *fout, real *time, real *spmv_time )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    struct timeval start, stop;

    N = H->n;
    bnorm = Norm( b, N );
    /* apply the diagonal pre-conditioner to rhs */
    gettimeofday( &start, NULL );
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];
    gettimeofday( &stop, NULL );
    *time += (stop.tv_sec + stop.tv_usec / 1000000.0)
             - (start.tv_sec + start.tv_usec / 1000000.0);

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* calculate r0 */
        gettimeofday( &start, NULL );
        Sparse_MatVec( H, x, workspace->b_prm );
        gettimeofday( &stop, NULL );
        *spmv_time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                 - (start.tv_sec + start.tv_usec / 1000000.0);
        for ( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; /* pre-conditioner */

        Vector_Sum(workspace->v[0], 1., workspace->b_prc, -1., workspace->b_prm, N);
        workspace->g[0] = Norm( workspace->v[0], N );
        Vector_Scale( workspace->v[0], 1. / workspace->g[0], workspace->v[0], N );
        //fprintf( stderr, "res: %.15e\n", workspace->g[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            /* matvec */
            gettimeofday( &start, NULL );
            Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1] );
            gettimeofday( &stop, NULL );
            *spmv_time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                     - (start.tv_sec + start.tv_usec / 1000000.0);
            /*pre-conditioner*/
            gettimeofday( &start, NULL );
            for ( k = 0; k < N; ++k )
                workspace->v[j + 1][k] *= workspace->Hdia_inv[k];
            gettimeofday( &stop, NULL );
            *time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                     - (start.tv_sec + start.tv_usec / 1000000.0);
            //fprintf( stderr, "%d-%d: matvec done.\n", itr, j );

            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for ( i = 0; i <= j; i++ )
            {
                workspace->h[i][j] = Dot( workspace->v[i], workspace->v[j + 1], N );
                Vector_Add( workspace->v[j + 1],
                            -workspace->h[i][j], workspace->v[i], N );
            }

            workspace->h[j + 1][j] = Norm( workspace->v[j + 1], N );
            Vector_Scale( workspace->v[j + 1],
                          1. / workspace->h[j + 1][j], workspace->v[j + 1], N );
            // fprintf( stderr, "%d-%d: orthogonalization completed.\n", itr, j );


            /* Givens rotations on the upper-Hessenberg matrix to make it U */
            for ( i = 0; i <= j; i++ )
            {
                if ( i == j )
                {
                    cc = SQRT( SQR(workspace->h[j][j]) + SQR(workspace->h[j + 1][j]) );
                    workspace->hc[j] = workspace->h[j][j] / cc;
                    workspace->hs[j] = workspace->h[j + 1][j] / cc;
                }

                tmp1 =  workspace->hc[i] * workspace->h[i][j] +
                        workspace->hs[i] * workspace->h[i + 1][j];
                tmp2 = -workspace->hs[i] * workspace->h[i][j] +
                       workspace->hc[i] * workspace->h[i + 1][j];

                workspace->h[i][j] = tmp1;
                workspace->h[i + 1][j] = tmp2;
            }

            /* apply Givens rotations to the rhs as well */
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j + 1] = tmp2;

            // fprintf( stderr, "h: " );
            // for( i = 0; i <= j+1; ++i )
            //  fprintf( stderr, "%.6f ", workspace->h[i][j] );
            // fprintf( stderr, "\n" );
            //fprintf( stderr, "res: %.15e\n", workspace->g[j+1] );
        }


        /* solve Hy = g.
           H is now upper-triangular, do back-substitution */
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = workspace->g[i];
            for ( k = j - 1; k > i; k-- )
                temp -= workspace->h[i][k] * workspace->y[k];

            workspace->y[i] = temp / workspace->h[i][i];
        }

        /* update x = x_0 + Vy */
        for ( i = 0; i < j; i++ )
            Vector_Add( x, workspace->y[i], workspace->v[i], N );

        /* stopping condition */
        if ( fabs(workspace->g[j]) / bnorm <= tol )
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

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART + 1) + j + 1;
    }

    return itr * (RESTART + 1) + j + 1;
}



int GMRES_HouseHolder( static_storage *workspace, sparse_matrix *H,
                       real *b, real tol, real *x, FILE *fout)
{
    int  i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[RESTART + 2][10000], w[RESTART + 2];
    real u[RESTART + 2][10000];

    N = H->n;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];

    // memset( x, 0, sizeof(real) * N );

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* compute z = r0 */
        Sparse_MatVec( H, x, workspace->b_prm );
        for ( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; /* pre-conditioner */
        Vector_Sum( z[0], 1.,  workspace->b_prc, -1., workspace->b_prm, N );

        Vector_MakeZero( w, RESTART + 1 );
        w[0] = Norm( z[0], N );

        Vector_Copy( u[0], z[0], N );
        u[0][0] += ( u[0][0] < 0.0 ? -1 : 1 ) * w[0];
        Vector_Scale( u[0], 1 / Norm( u[0], N ), u[0], N );

        w[0]    *= ( u[0][0] < 0.0 ?  1 : -1 );
        // fprintf( stderr, "\n\n%12.6f\n", w[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs( w[j] ) / bnorm > tol; j++ )
        {
            /* compute v_j */
            Vector_Scale( z[j], -2 * u[j][j], u[j], N );
            z[j][j] += 1.; /* due to e_j */

            for ( i = j - 1; i >= 0; --i )
                Vector_Add( z[j] + i, -2 * Dot( u[i] + i, z[j] + i, N - i ), u[i] + i, N - i );


            /* matvec */
            Sparse_MatVec( H, z[j], v );

            for ( k = 0; k < N; ++k )
                v[k] *= workspace->Hdia_inv[k]; /* pre-conditioner */

            for ( i = 0; i <= j; ++i )
                Vector_Add( v + i, -2 * Dot( u[i] + i, v + i, N - i ), u[i] + i, N - i );


            if ( !Vector_isZero( v + (j + 1), N - (j + 1) ) )
            {
                /* compute the HouseHolder unit vector u_j+1 */
                for ( i = 0; i <= j; ++i )
                    u[j + 1][i] = 0;

                Vector_Copy( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) );

                u[j + 1][j + 1] += ( v[j + 1] < 0.0 ? -1 : 1 ) * Norm( v + (j + 1), N - (j + 1) );

                Vector_Scale( u[j + 1], 1 / Norm( u[j + 1], N ), u[j + 1], N );

                /* overwrite v with P_m+1 * v */
                v[j + 1] -= 2 * Dot( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) ) * u[j + 1][j + 1];
                Vector_MakeZero( v + (j + 2), N - (j + 2) );
                // Vector_Add( v, -2 * Dot( u[j+1], v, N ), u[j+1], N );
            }


            /* prev Givens rots on the upper-Hessenberg matrix to make it U */
            for ( i = 0; i < j; i++ )
            {
                tmp1 =  workspace->hc[i] * v[i] + workspace->hs[i] * v[i + 1];
                tmp2 = -workspace->hs[i] * v[i] + workspace->hc[i] * v[i + 1];

                v[i]   = tmp1;
                v[i + 1] = tmp2;
            }

            /* apply the new Givens rotation to H and right-hand side */
            if ( fabs(v[j + 1]) >= ALMOST_ZERO )
            {
                cc = SQRT( SQR( v[j] ) + SQR( v[j + 1] ) );
                workspace->hc[j] = v[j] / cc;
                workspace->hs[j] = v[j + 1] / cc;

                tmp1 =  workspace->hc[j] * v[j] + workspace->hs[j] * v[j + 1];
                tmp2 = -workspace->hs[j] * v[j] + workspace->hc[j] * v[j + 1];

                v[j]   = tmp1;
                v[j + 1] = tmp2;

                /* Givens rotations to rhs */
                tmp1 =  workspace->hc[j] * w[j];
                tmp2 = -workspace->hs[j] * w[j];
                w[j]   = tmp1;
                w[j + 1] = tmp2;
            }

            /* extend R */
            for ( i = 0; i <= j; ++i )
                workspace->h[i][j] = v[i];


            // fprintf( stderr, "h:" );
            // for( i = 0; i <= j+1 ; ++i )
            // fprintf( stderr, "%.6f ", h[i][j] );
            // fprintf( stderr, "\n" );
            // fprintf( stderr, "%12.6f\n", w[j+1] );
        }


        /* solve Hy = w.
           H is now upper-triangular, do back-substitution */
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = w[i];
            for ( k = j - 1; k > i; k-- )
                temp -= workspace->h[i][k] * workspace->y[k];

            workspace->y[i] = temp / workspace->h[i][i];
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
        for ( i = j - 1; i >= 0; i-- )
            Vector_Add( x, workspace->y[i], z[i], N );

        // fprintf( stderr, "\nx_aft: " );
        // for( i = 0; i < N; ++i )
        //   fprintf( stderr, "%6.2f ", x[i] );

        /* stopping condition */
        if ( fabs( w[j] ) / bnorm <= tol )
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

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART + 1) + j + 1;
    }

    return itr * (RESTART + 1) + j + 1;
}


/* generalized minimual residual iterative solver for sparse linear systems,
 * with preconditioner using factors LU \approx H
 * and forward / backward substitution */
int PGMRES( static_storage *workspace, sparse_matrix *H, real *b, real tol,
            sparse_matrix *L, sparse_matrix *U, real *x, FILE *fout, real *time, real *spmv_time )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    struct timeval start, stop;

    N = H->n;
    bnorm = Norm( b, N );

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* calculate r0 */
        gettimeofday( &start, NULL );
        Sparse_MatVec( H, x, workspace->b_prm );
        gettimeofday( &stop, NULL );
        *spmv_time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                 - (start.tv_sec + start.tv_usec / 1000000.0);
        Vector_Sum( workspace->v[0], 1., b, -1., workspace->b_prm, N );
        gettimeofday( &start, NULL );
        Forward_Subs( L, workspace->v[0], workspace->v[0] );
        Backward_Subs( U, workspace->v[0], workspace->v[0] );
        gettimeofday( &stop, NULL );
        *time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                 - (start.tv_sec + start.tv_usec / 1000000.0);
        workspace->g[0] = Norm( workspace->v[0], N );
        Vector_Scale( workspace->v[0], 1. / workspace->g[0], workspace->v[0], N );
        //fprintf( stderr, "res: %.15e\n", workspace->g[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            /* matvec */
            gettimeofday( &start, NULL );
            Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1] );
            gettimeofday( &stop, NULL );
            *spmv_time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                     - (start.tv_sec + start.tv_usec / 1000000.0);
            gettimeofday( &start, NULL );
            Forward_Subs( L, workspace->v[j + 1], workspace->v[j + 1] );
            Backward_Subs( U, workspace->v[j + 1], workspace->v[j + 1] );
            gettimeofday( &stop, NULL );
            *time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                     - (start.tv_sec + start.tv_usec / 1000000.0);

            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for ( i = 0; i < j - 1; i++ ) workspace->h[i][j] = 0;

            //for( i = 0; i <= j; i++ ) {
            for ( i = MAX(j - 1, 0); i <= j; i++ )
            {
                workspace->h[i][j] = Dot( workspace->v[i], workspace->v[j + 1], N );
                Vector_Add( workspace->v[j + 1], -workspace->h[i][j], workspace->v[i], N );
            }

            workspace->h[j + 1][j] = Norm( workspace->v[j + 1], N );
            Vector_Scale( workspace->v[j + 1],
                          1. / workspace->h[j + 1][j], workspace->v[j + 1], N );
            // fprintf( stderr, "%d-%d: orthogonalization completed.\n", itr, j );

            /* Givens rotations on the upper-Hessenberg matrix to make it U */
            for ( i = MAX(j - 1, 0); i <= j; i++ )
            {
                if ( i == j )
                {
                    cc = SQRT( SQR(workspace->h[j][j]) + SQR(workspace->h[j + 1][j]) );
                    workspace->hc[j] = workspace->h[j][j] / cc;
                    workspace->hs[j] = workspace->h[j + 1][j] / cc;
                }

                tmp1 =  workspace->hc[i] * workspace->h[i][j] +
                        workspace->hs[i] * workspace->h[i + 1][j];
                tmp2 = -workspace->hs[i] * workspace->h[i][j] +
                       workspace->hc[i] * workspace->h[i + 1][j];

                workspace->h[i][j] = tmp1;
                workspace->h[i + 1][j] = tmp2;
            }

            /* apply Givens rotations to the rhs as well */
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j + 1] = tmp2;

            //fprintf( stderr, "h: " );
            //for( i = 0; i <= j+1; ++i )
            //fprintf( stderr, "%.6f ", workspace->h[i][j] );
            //fprintf( stderr, "\n" );
            //fprintf( stderr, "res: %.15e\n", workspace->g[j+1] );
        }


        /* solve Hy = g: H is now upper-triangular, do back-substitution */
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = workspace->g[i];
            for ( k = j - 1; k > i; k-- )
                temp -= workspace->h[i][k] * workspace->y[k];

            workspace->y[i] = temp / workspace->h[i][i];
        }

        /* update x = x_0 + Vy */
        Vector_MakeZero( workspace->p, N );
        for ( i = 0; i < j; i++ )
            Vector_Add( workspace->p, workspace->y[i], workspace->v[i], N );
        //Backward_Subs( U, workspace->p, workspace->p );
        //Forward_Subs( L, workspace->p, workspace->p );
        Vector_Add( x, 1., workspace->p, N );

        /* stopping condition */
        if ( fabs(workspace->g[j]) / bnorm <= tol )
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

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART + 1) + j + 1;
    }

    return itr * (RESTART + 1) + j + 1;
}


/* generalized minimual residual iterative solver for sparse linear systems,
 * with preconditioner using factors LU \approx H
 * and Jacobi iteration for approximate factor application */
int PGMRES_Jacobi( static_storage *workspace, sparse_matrix *H, real *b, real tol,
                   sparse_matrix *L, sparse_matrix *U, real *x, unsigned int iters,
		   FILE *fout, real *time, real *spmv_time )
{
    int i, j, k, itr, N, si;
    real cc, tmp1, tmp2, temp, bnorm;
    real *Dinv_L, *Dinv_U;
    struct timeval start, stop;

    N = H->n;
    bnorm = Norm( b, N );

    /* Compute Jacobi iteration matrices from
     * truncated Newmann series: x_{k+1} = Gx_k + D^{-1}b
     * where:
     *   G = I - D^{-1}R
     *   R = triangular matrix
     *   D = diagonal matrix, diagonals from R */
    if ( (Dinv_L = (real*) malloc(sizeof(real) * N)) == NULL
            || (Dinv_U = (real*) malloc(sizeof(real) * N)) == NULL )
    {
        fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
        exit(INSUFFICIENT_SPACE);
    }

    /* construct D^{-1}_L and D^{-1}_U */
    for ( i = 0; i < N; ++i )
    {
        si = L->start[i + 1] - 1;
        Dinv_L[i] = 1. / L->entries[si].val;

        si = U->start[i];
        Dinv_U[i] = 1. / U->entries[si].val;
    }

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* calculate r0 */
        gettimeofday( &start, NULL );
        Sparse_MatVec( H, x, workspace->b_prm );
        gettimeofday( &stop, NULL );
        *spmv_time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                 - (start.tv_sec + start.tv_usec / 1000000.0);
        Vector_Sum( workspace->v[0], 1., b, -1., workspace->b_prm, N );
        gettimeofday( &start, NULL );
        Jacobi_Iter( L, LOWER, Dinv_L, N, workspace->v[0], workspace->v[0], iters );
        Jacobi_Iter( U, UPPER, Dinv_U, N, workspace->v[0], workspace->v[0], iters );
        gettimeofday( &stop, NULL );
        *time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                 - (start.tv_sec + start.tv_usec / 1000000.0);
        workspace->g[0] = Norm( workspace->v[0], N );
        Vector_Scale( workspace->v[0], 1. / workspace->g[0], workspace->v[0], N );
        //fprintf( stderr, "res: %.15e\n", workspace->g[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            /* matvec */
            gettimeofday( &start, NULL );
            Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1] );
            gettimeofday( &stop, NULL );
            *spmv_time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                     - (start.tv_sec + start.tv_usec / 1000000.0);
            gettimeofday( &start, NULL );
            Jacobi_Iter( L, LOWER, Dinv_L, N, workspace->v[j + 1], workspace->v[j + 1], iters );
            Jacobi_Iter( U, UPPER, Dinv_U, N, workspace->v[j + 1], workspace->v[j + 1], iters );
            gettimeofday( &stop, NULL );
            *time += (stop.tv_sec + stop.tv_usec / 1000000.0)
                     - (start.tv_sec + start.tv_usec / 1000000.0);

            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for ( i = 0; i < j - 1; i++ )
            {
                workspace->h[i][j] = 0;
            }

            //for( i = 0; i <= j; i++ ) {
            for ( i = MAX(j - 1, 0); i <= j; i++ )
            {
                workspace->h[i][j] = Dot( workspace->v[i], workspace->v[j + 1], N );
                Vector_Add( workspace->v[j + 1], -workspace->h[i][j], workspace->v[i], N );
            }

            workspace->h[j + 1][j] = Norm( workspace->v[j + 1], N );
            Vector_Scale( workspace->v[j + 1],
                          1. / workspace->h[j + 1][j], workspace->v[j + 1], N );
            // fprintf( stderr, "%d-%d: orthogonalization completed.\n", itr, j );

            /* Givens rotations on the upper-Hessenberg matrix to make it U */
            for ( i = MAX(j - 1, 0); i <= j; i++ )
            {
                if ( i == j )
                {
                    cc = SQRT( SQR(workspace->h[j][j]) + SQR(workspace->h[j + 1][j]) );
                    workspace->hc[j] = workspace->h[j][j] / cc;
                    workspace->hs[j] = workspace->h[j + 1][j] / cc;
                }

                tmp1 =  workspace->hc[i] * workspace->h[i][j] +
                        workspace->hs[i] * workspace->h[i + 1][j];
                tmp2 = -workspace->hs[i] * workspace->h[i][j] +
                       workspace->hc[i] * workspace->h[i + 1][j];

                workspace->h[i][j] = tmp1;
                workspace->h[i + 1][j] = tmp2;
            }

            /* apply Givens rotations to the rhs as well */
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j + 1] = tmp2;

            //fprintf( stderr, "h: " );
            //for( i = 0; i <= j+1; ++i )
            //fprintf( stderr, "%.6f ", workspace->h[i][j] );
            //fprintf( stderr, "\n" );
            //fprintf( stderr, "res: %.15e\n", workspace->g[j+1] );
        }


        /* TODO: solve using Jacobi iteration? */
        /* solve Hy = g: H is now upper-triangular, do back-substitution */
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = workspace->g[i];
            for ( k = j - 1; k > i; k-- )
            {
                temp -= workspace->h[i][k] * workspace->y[k];
            }

            workspace->y[i] = temp / workspace->h[i][i];
        }

        /* update x = x_0 + Vy */
        Vector_MakeZero( workspace->p, N );
        for ( i = 0; i < j; i++ )
        {
            Vector_Add( workspace->p, workspace->y[i], workspace->v[i], N );
        }
        //Backward_Subs( U, workspace->p, workspace->p );
        //Forward_Subs( L, workspace->p, workspace->p );
        Vector_Add( x, 1., workspace->p, N );

        /* stopping condition */
        if ( fabs(workspace->g[j]) / bnorm <= tol )
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

    free( Dinv_U );
    free( Dinv_L );

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART + 1) + j + 1;
    }

    return itr * (RESTART + 1) + j + 1;
}


int PCG( static_storage *workspace, sparse_matrix *A, real *b, real tol,
         sparse_matrix *L, sparse_matrix *U, real *x, FILE *fout )
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

    for ( i = 0; i < 200 && r_norm / b_norm > tol; ++i )
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
    if ( i >= 200 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}


int CG( static_storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, FILE *fout )
{
    int  i, j, N;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

    N = H->n;
    b_norm = Norm( b, N );
    //fprintf( stderr, "b_norm: %10.6f\n", b_norm );

    Sparse_MatVec( H, x, workspace->q );
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, N );
    for ( j = 0; j < N; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];

    sig_new = Dot( workspace->r, workspace->d, N );
    sig0 = sig_new;
    //Print_Soln( workspace, x, q, b, N );
    //fprintf( stderr, "sig_new: %24.15e, d_norm:%24.15e, q_norm:%24.15e\n",
    // sqrt(sig_new), Norm(workspace->d,N), Norm(workspace->q,N) );
    //fprintf( stderr, "sig_new: %f\n", sig_new );

    for ( i = 0; i < 300 && SQRT(sig_new) / b_norm > tol; ++i )
    {
        //for( i = 0; i < 300 && sig_new > SQR(tol)*sig0; ++i ) {
        Sparse_MatVec( H, workspace->d, workspace->q );
        tmp = Dot( workspace->d, workspace->q, N );
        //fprintf( stderr, "tmp: %f\n", tmp );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, N );
        //fprintf( stderr, "d_norm:%24.15e, q_norm:%24.15e, tmp:%24.15e\n",
        //     Norm(workspace->d,N), Norm(workspace->q,N), tmp );

        Vector_Add( workspace->r, -alpha, workspace->q, N );
        for ( j = 0; j < N; ++j )
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];

        sig_old = sig_new;
        sig_new = Dot( workspace->r, workspace->p, N );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, N );
        //fprintf( stderr, "sig_new: %f\n", sig_new );
    }

    fprintf( stderr, "CG took %d iterations\n", i );

    if ( i >= 300 )
    {
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
    for ( j = 0; j < N; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];

    sig = Dot( workspace->r, workspace->d, N );
    sig0 = sig;

    for ( i = 0; i < 300 && SQRT(sig) / b_norm > tol; ++i )
    {
        Sparse_MatVec( H, workspace->d, workspace->q );

        sig = Dot( workspace->r, workspace->d, N );
        tmp = Dot( workspace->d, workspace->q, N );
        alpha = sig / tmp;

        Vector_Add( x, alpha, workspace->d, N );
        Vector_Add( workspace->r, -alpha, workspace->q, N );
        for ( j = 0; j < N; ++j )
            workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];

        //fprintf( stderr, "d_norm:%24.15e, q_norm:%24.15e, tmp:%24.15e\n",
        //     Norm(workspace->d,N), Norm(workspace->q,N), tmp );
    }

    fprintf( stderr, "SDM took %d iterations\n", i );

    if ( i >= 300 )
    {
        fprintf( stderr, "SDM convergence failed!\n" );
        return i;
    }

    return i;
}


real condest( const sparse_matrix * const L, const sparse_matrix * const U )
{
    unsigned int i, N;
    real *e, c;

    N = L->n;

    if ( (e = (real*) malloc(sizeof(real) * N)) == NULL )
    {
        fprintf( stderr, "Not enough memory for condest. Terminating.\n" );
        exit(INSUFFICIENT_SPACE);
    }

    for ( i = 0; i < N; ++i )
    {
        e[i] = 1.;
    }

    Forward_Subs( L, e, e );
    Backward_Subs( U, e, e );

    c = fabs(e[0]);
    for ( i = 1; i < N; ++i)
    {
        if ( fabs(e[i]) > c )
        {
            c = fabs(e[i]);
        }

    }

    free( e );

    return c;
}
