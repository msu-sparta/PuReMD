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

#include "GMRES.h"
#include "allocate.h"
#include "list.h"
#include "tool_box.h"
#include "vector.h"


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
                    exit( INSUFFICIENT_MEMORY );
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
                j = A->j[k];
                H = A->val[k];
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
            b_local[tid * n + i] += A->val[k] * x[i];
#else
            b[i] += A->val[k] * x[i];
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
                    exit( INSUFFICIENT_MEMORY );
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
                b_local[tid * R->n + i] += R->val[k] * x[R->j[k]];
#else
                b[i] += R->val[k] * x[R->j[k]];
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


static void diag_pre_app( const real * const Hdia_inv, const real * const y,
        real * const x, const int N )
{
    unsigned int i;

    #pragma omp parallel for schedule(guided) \
        default(none) private(i)
    for ( i = 0; i < N; ++i )
    {
        x[i] = y[i] * Hdia_inv[i];
    }
}


/* Solve triangular system LU*x = y using level scheduling
 *
 * LU: lower/upper triangular, stored in CSR
 * y: constants in linear system (RHS)
 * x: solution
 * tri: triangularity of LU (lower/upper)
 * 
 * Assumptions:
 *   LU has non-zero diagonals
 *   Each row of LU has at least one non-zero (i.e., no rows with all zeros) */
static void tri_solve( const sparse_matrix * const LU, const real * const y,
        real * const x, const TRIANGULARITY tri )
{
    int i, pj, j, si, ei;
    real val;

    if ( tri == LOWER )
    {
        for ( i = 0; i < LU->n; ++i )
        {
            x[i] = y[i];
            si = LU->start[i];
            ei = LU->start[i + 1];
            for ( pj = si; pj < ei - 1; ++pj )
            {
                j = LU->j[pj];
                val = LU->val[pj];
                x[i] -= val * x[j];
            }
            x[i] /= LU->val[pj];
        }
    }
    else
    {
        for ( i = LU->n - 1; i >= 0; --i )
        {
            x[i] = y[i];
            si = LU->start[i];
            ei = LU->start[i + 1];
            for ( pj = si + 1; pj < ei; ++pj )
            {
                j = LU->j[pj];
                val = LU->val[pj];
                x[i] -= val * x[j];
            }
            x[i] /= LU->val[si];
        }
    }
}


/* Solve triangular system LU*x = y using level scheduling
 *
 * LU: lower/upper triangular, stored in CSR
 * y: constants in linear system (RHS)
 * x: solution
 * tri: triangularity of LU (lower/upper)
 * find_levels: perform level search if positive, otherwise reuse existing levels
 * 
 * Assumptions:
 *   LU has non-zero diagonals
 *   Each row of LU has at least one non-zero (i.e., no rows with all zeros) */
static void tri_solve_level_sched( const sparse_matrix * const LU, const real * const y,
        real * const x, const TRIANGULARITY tri, int find_levels )
{
    int i, j, pj, local_row, local_level, levels;
    static int levels_L = 1, levels_U = 1;
    static unsigned int *row_levels_L = NULL, *level_rows_L = NULL, *level_rows_cnt_L = NULL;
    static unsigned int *row_levels_U = NULL, *level_rows_U = NULL, *level_rows_cnt_U = NULL;
    unsigned int *row_levels, *level_rows, *level_rows_cnt;

    if ( tri == LOWER )
    {
        row_levels = row_levels_L;
        level_rows = level_rows_L;
        level_rows_cnt = level_rows_cnt_L;
        levels = levels_L;
    }
    else
    {
        row_levels = row_levels_U;
        level_rows = level_rows_U;
        level_rows_cnt = level_rows_cnt_U;
        levels = levels_U;
    }

    if ( row_levels == NULL || level_rows == NULL || level_rows_cnt == NULL )
    {
        if ( (row_levels = (unsigned int*) malloc(LU->n * sizeof(unsigned int))) == NULL
            || (level_rows = (unsigned int*) malloc(LU->n * LU->n * sizeof(unsigned int))) == NULL
            || (level_rows_cnt = (unsigned int*) malloc(LU->n * sizeof(unsigned int))) == NULL )
        {
            fprintf( stderr, "Not enough space for triangular solve via level scheduling. Terminating...\n" );
            exit( INSUFFICIENT_MEMORY );
        }
    }

    /* find levels (row dependencies in substitutions) */
    if ( find_levels )
    {
        memset( row_levels, 0, LU->n * sizeof( unsigned int) );
        memset( level_rows_cnt, 0, LU->n * sizeof( unsigned int) );

        if ( tri == LOWER )
        {
            for ( i = 0; i < LU->n; ++i )
            {
                local_level = 0;
                for ( pj = LU->start[i]; pj < LU->start[i + 1] - 1; ++pj )
                {
                    local_level = MAX( local_level, row_levels[LU->j[pj]] + 1 );
                }
        
                levels = MAX( levels, local_level + 1 );
                row_levels[i] = local_level;
                level_rows[local_level * LU->n + level_rows_cnt[local_level]] = i;
                ++level_rows_cnt[local_level];
            }
        }
        else
        {
            for ( i = LU->n - 1; i >= 0; --i )
            {
                local_level = 0;
                for ( pj = LU->start[i] + 1; pj < LU->start[i + 1]; ++pj )
                {
                    local_level = MAX( local_level, row_levels[LU->j[pj]] + 1 );
                }
        
                levels = MAX( levels, local_level + 1 );
                row_levels[i] = local_level;
                level_rows[local_level * LU->n + level_rows_cnt[local_level]] = i;
                ++level_rows_cnt[local_level];
            }
        }
    }

    /* perform substitutions by level */
    if ( tri == LOWER )
    {
        for ( i = 0; i < levels; ++i )
        {
            #pragma omp parallel for schedule(guided) \
                default(none) private(j, pj, local_row) shared(stderr, i, level_rows_cnt, level_rows)
            for ( j = 0; j < level_rows_cnt[i]; ++j )
            {
                local_row = level_rows[i * LU->n + j];
                x[local_row] = y[local_row];
                for ( pj = LU->start[local_row]; pj < LU->start[local_row + 1] - 1; ++pj )
                {
                    x[local_row] -= LU->val[pj] * x[LU->j[pj]];
    
                }
                x[local_row] /= LU->val[pj];
            }
        }
    }
    else
    {
        for ( i = 0; i < levels; ++i )
        {
            #pragma omp parallel for schedule(guided) \
                default(none) private(j, pj, local_row) shared(i, level_rows_cnt, level_rows)
            for ( j = 0; j < level_rows_cnt[i]; ++j )
            {
                local_row = level_rows[i * LU->n + j];
                x[local_row] = y[local_row];
                for ( pj = LU->start[local_row] + 1; pj < LU->start[local_row + 1]; ++pj )
                {
                    x[local_row] -= LU->val[pj] * x[LU->j[pj]];
    
                }
                x[local_row] /= LU->val[LU->start[local_row]];
            }
        }
    }

    /* save level info for re-use if performing repeated triangular solves via preconditioning */
    if ( tri == LOWER )
    {
        row_levels_L = row_levels;
        level_rows_L = level_rows;
        level_rows_cnt_L = level_rows_cnt;
        levels_L = levels;
    }
    else
    {
        row_levels_U = row_levels;
        level_rows_U = level_rows;
        level_rows_cnt_U = level_rows_cnt;
        levels_U = levels;
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
static void jacobi_iter( const sparse_matrix * const R, const real * const Dinv,
        const real * const b, real * const x, const TRIANGULARITY tri,
        const unsigned int maxiter )
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
                if ( (Dinv_b = (real*) malloc(sizeof(real) * R->n)) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            if ( rp == NULL )
            {
                if ( (rp = (real*) malloc(sizeof(real) * R->n)) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            if ( rp2 == NULL )
            {
                    if ( (rp2 = (real*) malloc(sizeof(real) * R->n)) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
#ifdef _OPENMP
            if ( b_local == NULL )
            {
                if ( (b_local = (real*) malloc( omp_get_num_threads() * R->n * sizeof(real))) == NULL )
                {
                    fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
	    }
#endif
    
            Vector_MakeZero( rp, R->n );
	}
    
        #pragma omp barrier
    
        /* precompute and cache, as invariant in loop below */
        #pragma omp for schedule(guided)
        for ( i = 0; i < R->n; ++i )
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
                else
                {
    
                    si = R->start[i] + 1;
                    ei = R->start[i + 1];
                }
    
                for ( k = si; k < ei; ++k )
                {
#ifdef _OPENMP
                    b_local[tid * R->n + i] += R->val[k] * rp[R->j[k]];
#else
                    rp2[i] += R->val[k] * rp[R->j[k]];
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

    Vector_Copy( x, rp, R->n );
}


/* Solve triangular system LU*x = y using level scheduling
 *
 * workspace: data struct containing matrices, lower/upper triangular, stored in CSR
 * control: data struct containing parameters
 * y: constants in linear system (RHS)
 * x: solution
 * fresh_pre: parameter indicating if this is a newly computed (fresh) preconditioner
 * 
 * Assumptions:
 *   Matrices have non-zero diagonals
 *   Each row of a matrix has at least one non-zero (i.e., no rows with all zeros) */
void apply_preconditioner( const static_storage * const workspace, const control_params * const control,
        const real * const y, real * const x, const int fresh_pre )
{
    int i, si;
    static real *Dinv_L = NULL, *Dinv_U = NULL;

    switch ( control->pre_app_type )
    {
        case NONE_PA:
            break;
        case TRI_SOLVE_PA:
            switch ( control->pre_comp_type )
            {
                case DIAG_PC:
                    diag_pre_app( workspace->Hdia_inv, y, x, workspace->H->n );
                    break;
                case ICHOLT_PC:
                case ILU_PAR_PC:
                case ILUT_PAR_PC:
                    tri_solve( workspace->L, y, x, LOWER );
                    tri_solve( workspace->U, y, x, UPPER );
                    break;
                default:
                    fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                    exit( INVALID_INPUT );
                    break;
            }
            break;
        case TRI_SOLVE_LEVEL_SCHED_PA:
            switch ( control->pre_comp_type )
            {
                case DIAG_PC:
                    diag_pre_app( workspace->Hdia_inv, y, x, workspace->H->n );
                    break;
                case ICHOLT_PC:
                case ILU_PAR_PC:
                case ILUT_PAR_PC:
                    tri_solve_level_sched( workspace->L, y, x, LOWER, fresh_pre );
                    tri_solve_level_sched( workspace->U, y, x, UPPER, fresh_pre );
                    break;
                default:
                    fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                    exit( INVALID_INPUT );
                    break;
            }
            break;
        case JACOBI_ITER_PA:
            switch ( control->pre_comp_type )
            {
                case DIAG_PC:
                    fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                    exit( INVALID_INPUT );
                    break;
                case ICHOLT_PC:
                case ILU_PAR_PC:
                case ILUT_PAR_PC:
                    if ( Dinv_L == NULL )
                    {
                        if ( (Dinv_L = (real*) malloc(sizeof(real) * workspace->L->n)) == NULL )
                        {
                            fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                            exit( INSUFFICIENT_MEMORY );
                        }
                    }

                    /* construct D^{-1}_L */
                    if ( fresh_pre )
                    {
                        for ( i = 0; i < workspace->L->n; ++i )
                        {
                            si = workspace->L->start[i + 1] - 1;
                            Dinv_L[i] = 1. / workspace->L->val[si];
                        }
                    }

                    jacobi_iter( workspace->L, Dinv_L, y, x, LOWER, control->pre_app_jacobi_iters );

                    if ( Dinv_U == NULL )
                    {
                        if ( (Dinv_U = (real*) malloc(sizeof(real) * workspace->U->n)) == NULL )
                        {
                            fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                            exit( INSUFFICIENT_MEMORY );
                        }
                    }

                    /* construct D^{-1}_U */
                    if ( fresh_pre )
                    {
                        for ( i = 0; i < workspace->U->n; ++i )
                        {
                            si = workspace->U->start[i];
                            Dinv_U[i] = 1. / workspace->U->val[si];
                        }
                    }

                    jacobi_iter( workspace->U, Dinv_U, y, x, UPPER, control->pre_app_jacobi_iters );
                    break;
                default:
                    fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                    exit( INVALID_INPUT );
                    break;
            }
            break;
        default:
            fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

    }

    return;
}


/* generalized minimual residual iterative solver for sparse linear systems */
int GMRES( const static_storage * const workspace, const control_params * const control,
        const sparse_matrix * const H, const real * const b, real tol, real * const x,
        const FILE * const fout, real * const time, real * const spmv_time, const int fresh_pre )
{
    int i, j, k, itr, N, si;
    real cc, tmp1, tmp2, temp, bnorm, time_start;

    N = H->n;
    bnorm = Norm( b, N );

    if ( control->pre_comp_type == DIAG_PC )
    {
        /* apply preconditioner to RHS */
        apply_preconditioner( workspace, control, b, workspace->b_prc, fresh_pre );
    }

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* calculate r0 */
        time_start = Get_Time( );
        Sparse_MatVec( H, x, workspace->b_prm );
        *spmv_time += Get_Timing_Info( time_start );

        if ( control->pre_comp_type == DIAG_PC )
        {
            time_start = Get_Time( );
            apply_preconditioner( workspace, control, workspace->b_prm, workspace->b_prm, fresh_pre );
            *time += Get_Timing_Info( time_start );
        }


        if ( control->pre_comp_type == DIAG_PC )
        {
            Vector_Sum( workspace->v[0], 1., workspace->b_prc, -1., workspace->b_prm, N );
        }
        else
        {
            Vector_Sum( workspace->v[0], 1., b, -1., workspace->b_prm, N );
        }

        if ( control->pre_comp_type != DIAG_PC )
        {
            time_start = Get_Time( );
            apply_preconditioner( workspace, control, workspace->v[0], workspace->v[0],
                    itr == 0 ? fresh_pre : 0 );
            *time += Get_Timing_Info( time_start );
        }

        workspace->g[0] = Norm( workspace->v[0], N );
        Vector_Scale( workspace->v[0], 1. / workspace->g[0], workspace->v[0], N );
        //fprintf( stderr, "res: %.15e\n", workspace->g[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            /* matvec */
            time_start = Get_Time( );
            Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1] );
            *spmv_time += Get_Timing_Info( time_start );

            time_start = Get_Time( );
            apply_preconditioner( workspace, control, workspace->v[j + 1], workspace->v[j + 1], 0 );
            *time += Get_Timing_Info( time_start );

            if ( control->pre_comp_type == DIAG_PC )
            {
                /* apply modified Gram-Schmidt to orthogonalize the new residual */
                for( i = 0; i <= j; i++ )
                {
                    workspace->h[i][j] = Dot( workspace->v[i], workspace->v[j+1], N );
                    Vector_Add( workspace->v[j+1], -workspace->h[i][j], workspace->v[i], N );
                }
                
                workspace->h[j+1][j] = Norm( workspace->v[j+1], N );
                Vector_Scale( workspace->v[j+1], 1. / workspace->h[j+1][j], workspace->v[j+1], N );
                // fprintf( stderr, "%d-%d: orthogonalization completed.\n", itr, j );
                
                /* Givens rotations on the upper-Hessenberg matrix to make it U */
                for( i = 0; i <= j; i++ )
                {
                    if( i == j )
                    {
                        cc = SQRT( SQR(workspace->h[j][j])+SQR(workspace->h[j+1][j]) );
                        workspace->hc[j] = workspace->h[j][j] / cc;
                        workspace->hs[j] = workspace->h[j+1][j] / cc;
                    }
                
                    tmp1 =  workspace->hc[i] * workspace->h[i][j] +
                    workspace->hs[i] * workspace->h[i+1][j];
                    tmp2 = -workspace->hs[i] * workspace->h[i][j] +
                    workspace->hc[i] * workspace->h[i+1][j];
                    
                    workspace->h[i][j] = tmp1;
                    workspace->h[i+1][j] = tmp2;
                }
                
                /* apply Givens rotations to the rhs as well */
                tmp1 =  workspace->hc[j] * workspace->g[j];
                tmp2 = -workspace->hs[j] * workspace->g[j];
                workspace->g[j] = tmp1;
                workspace->g[j+1] = tmp2;
            }
	    else
            {
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
            }

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

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return itr * (RESTART + 1) + j + 1;
    }

    return itr * (RESTART + 1) + j + 1;
}


int GMRES_HouseHolder( const static_storage * const workspace, const control_params * const control,
        const sparse_matrix * const H, const real * const b, real tol, real * const x,
        const FILE * const fout, real * const time, real * const spmv_time, const int fresh_pre )
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


/* Preconditioned Conjugate Gradient */
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

    tri_solve( L, workspace->r, workspace->d, LOWER );
    tri_solve( U, workspace->d, workspace->p, UPPER );
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

        tri_solve( L, workspace->r, workspace->d, LOWER );
        tri_solve( U, workspace->d, workspace->d, UPPER );
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


/* Conjugate Gradient */
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
    {
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
    }

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
    {
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
    }

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
        {
            workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }

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


/* Estimate the stability of a 2-side preconditioning scheme
 * using the factorization A \approx LU. Specifically, estimate the 1-norm of A^{-1}
 * using the 1-norm of (LU)^{-1}e, with e = [1 1 ... 1]^T through 2 triangular solves:
 *   1) Ly = e
 *   2) Ux = y where y = Ux
 * That is, we seek to solve e = LUx for unknown x 
 *
 * Reference: Incomplete LU Preconditioning with the Multilevel Fast Multipole Algorithm
 *   for Electromagnetic Scattering, SIAM J. Sci. Computing, 2007 */
real condest( const sparse_matrix * const L, const sparse_matrix * const U )
{
    unsigned int i, N;
    real *e, c;

    N = L->n;

    if ( (e = (real*) malloc(sizeof(real) * N)) == NULL )
    {
        fprintf( stderr, "Not enough memory for condest. Terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    for ( i = 0; i < N; ++i )
    {
        e[i] = 1.;
    }

    tri_solve( L, e, e, LOWER );
    tri_solve( U, e, e, UPPER );

    /* compute 1-norm of vector e */
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
