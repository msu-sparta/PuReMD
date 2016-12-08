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

#include "lin_alg.h"

#include "allocate.h"
#include "list.h"
#include "print_utils.h"
#include "tool_box.h"
#include "vector.h"


typedef enum
{
    LOWER = 0,
    UPPER = 1,
} TRIANGULARITY;


/* global to make OpenMP shared (Sparse_MatVec) */
#ifdef _OPENMP
real *b_local = NULL;
#endif
/* global to make OpenMP shared (apply_preconditioner) */
real *Dinv_L = NULL, *Dinv_U = NULL;
/* global to make OpenMP shared (tri_solve_level_sched) */
int levels = 1;
int levels_L = 1, levels_U = 1;
unsigned int *row_levels_L = NULL, *level_rows_L = NULL, *level_rows_cnt_L = NULL;
unsigned int *row_levels_U = NULL, *level_rows_U = NULL, *level_rows_cnt_U = NULL;
unsigned int *row_levels, *level_rows, *level_rows_cnt;
unsigned int *top = NULL;
/* global to make OpenMP shared (graph_coloring) */
unsigned int *color = NULL;
unsigned int *to_color = NULL;
unsigned int *conflict = NULL;
unsigned int *temp_ptr;
unsigned int *recolor = NULL;
unsigned int recolor_cnt;
unsigned int *color_top = NULL;
/* global to make OpenMP shared (sort_colors) */
unsigned int *permuted_row_col = NULL;
unsigned int *permuted_row_col_inv = NULL;
real *y_p = NULL;
/* global to make OpenMP shared (permute_vector) */
real *x_p = NULL;
unsigned int *mapping = NULL;
sparse_matrix *H_full;
sparse_matrix *H_p;
/* global to make OpenMP shared (jacobi_iter) */
real *Dinv_b = NULL, *rp = NULL, *rp2 = NULL, *rp3 = NULL;


/* sparse matrix-vector product Ax=b
 * where:
 *   A: lower triangular matrix, stored in CSR format
 *   x: vector
 *   b: vector (result) */
static void Sparse_MatVec( const sparse_matrix * const A,
        const real * const x, real * const b )
{
    int i, j, k, n, si, ei;
    real H;
#ifdef _OPENMP
    unsigned int tid;
#endif

    n = A->n;
    Vector_MakeZero( b, n );

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
    }

    #pragma omp barrier

    Vector_MakeZero( (real * const)b_local, omp_get_num_threads() * n );

#endif
    #pragma omp for schedule(static)
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
    #pragma omp for schedule(static)
    for ( i = 0; i < n; ++i )
    {
        for ( j = 0; j < omp_get_num_threads(); ++j )
        {
            b[i] += b_local[j * n + i];
        }
    }
#endif

}


/* Transpose A and copy into A^T
 *
 * A: stored in CSR
 * A_t: stored in CSR
 */
void Transpose( const sparse_matrix const *A, sparse_matrix const *A_t )
{
    unsigned int i, j, pj, *A_t_top;

    if ( (A_t_top = (unsigned int*) calloc( A->n + 1, sizeof(unsigned int))) == NULL )
    {
        fprintf( stderr, "Not enough space for matrix tranpose. Terminating...\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    memset( A_t->start, 0, (A->n + 1) * sizeof(unsigned int) );

    /* count nonzeros in each column of A^T, store one row greater (see next loop) */
    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            ++A_t->start[A->j[pj] + 1];
        }
    }

    /* setup the row pointers for A^T */
    for ( i = 1; i <= A->n; ++i )
    {
        A_t_top[i] = A_t->start[i] = A_t->start[i] + A_t->start[i - 1];
    }

    /* fill in A^T */
    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            j = A->j[pj];
            A_t->j[A_t_top[j]] = i;
            A_t->val[A_t_top[j]] = A->val[pj];
            ++A_t_top[j];
        }
    }

    free( A_t_top );
}


/* Transpose A in-place
 *
 * A: stored in CSR
 */
void Transpose_I( sparse_matrix * const A )
{
    sparse_matrix * A_t;

    if ( Allocate_Matrix( A_t, A->n, A->m ) == FAILURE )
    {
        fprintf( stderr, "not enough memory for transposing matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    Transpose( A, A_t );

    memcpy( A->start, A_t->start, sizeof(int) * (A_t->n + 1) );
    memcpy( A->j, A_t->j, sizeof(int) * (A_t->start[A_t->n]) );
    memcpy( A->val, A_t->val, sizeof(real) * (A_t->start[A_t->n]) );

    Deallocate_Matrix( A_t );
}


/* Apply diagonal inverse (Jacobi) preconditioner to system residual
 *
 * Hdia_inv: diagonal inverse preconditioner (constructed using H)
 * y: current residual
 * x: preconditioned residual
 * N: length of preconditioner and vectors (# rows in H)
 */
static void diag_pre_app( const real * const Hdia_inv, const real * const y,
                          real * const x, const int N )
{
    unsigned int i;

    #pragma omp for schedule(static)
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

    #pragma omp master
    {
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
    int i, j, pj, local_row, local_level;

    #pragma omp master
    {
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
            if ( (row_levels = (unsigned int*) malloc((size_t)LU->n * sizeof(unsigned int))) == NULL
                    || (level_rows = (unsigned int*) malloc((size_t)LU->n * sizeof(unsigned int))) == NULL
                    || (level_rows_cnt = (unsigned int*) malloc((size_t)(LU->n + 1) * sizeof(unsigned int))) == NULL )
            {
                fprintf( stderr, "Not enough space for triangular solve via level scheduling. Terminating...\n" );
                exit( INSUFFICIENT_MEMORY );
            }
        }

        if ( top == NULL )
        {
            if ( (top = (unsigned int*) malloc((size_t)(LU->n + 1) * sizeof(unsigned int))) == NULL )
            {
                fprintf( stderr, "Not enough space for triangular solve via level scheduling. Terminating...\n" );
                exit( INSUFFICIENT_MEMORY );
            }
        }

        /* find levels (row dependencies in substitutions) */
        if ( find_levels == TRUE )
        {
            memset( row_levels, 0, LU->n * sizeof(unsigned int) );
            memset( level_rows_cnt, 0, LU->n * sizeof(unsigned int) );
            memset( top, 0, LU->n * sizeof(unsigned int) );
            levels = 1;

            if ( tri == LOWER )
            {
                for ( i = 0; i < LU->n; ++i )
                {
                    local_level = 1;
                    for ( pj = LU->start[i]; pj < LU->start[i + 1] - 1; ++pj )
                    {
                        local_level = MAX( local_level, row_levels[LU->j[pj]] + 1 );
                    }

                    levels = MAX( levels, local_level );
                    row_levels[i] = local_level;
                    ++level_rows_cnt[local_level];
                }

//#if defined(DEBUG)
                fprintf(stderr, "levels(L): %d\n", levels);
                fprintf(stderr, "NNZ(L): %d\n", LU->start[LU->n]);
//#endif
            }
            else
            {
                for ( i = LU->n - 1; i >= 0; --i )
                {
                    local_level = 1;
                    for ( pj = LU->start[i] + 1; pj < LU->start[i + 1]; ++pj )
                    {
                        local_level = MAX( local_level, row_levels[LU->j[pj]] + 1 );
                    }

                    levels = MAX( levels, local_level );
                    row_levels[i] = local_level;
                    ++level_rows_cnt[local_level];
                }

//#if defined(DEBUG)
                fprintf(stderr, "levels(U): %d\n", levels);
                fprintf(stderr, "NNZ(U): %d\n", LU->start[LU->n]);
//#endif
            }

            for ( i = 1; i < levels + 1; ++i )
            {
                level_rows_cnt[i] += level_rows_cnt[i - 1];
                top[i] = level_rows_cnt[i];
            }

            for ( i = 0; i < LU->n; ++i )
            {
                level_rows[top[row_levels[i] - 1]] = i;
                ++top[row_levels[i] - 1];
            }
        }
    }

    #pragma omp barrier

    /* perform substitutions by level */
    if ( tri == LOWER )
    {
        for ( i = 0; i < levels; ++i )
        {
            #pragma omp for schedule(static)
            for ( j = level_rows_cnt[i]; j < level_rows_cnt[i + 1]; ++j )
            {
                local_row = level_rows[j];
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
            #pragma omp for schedule(static)
            for ( j = level_rows_cnt[i]; j < level_rows_cnt[i + 1]; ++j )
            {
                local_row = level_rows[j];
                x[local_row] = y[local_row];
                for ( pj = LU->start[local_row] + 1; pj < LU->start[local_row + 1]; ++pj )
                {
                    x[local_row] -= LU->val[pj] * x[LU->j[pj]];

                }
                x[local_row] /= LU->val[LU->start[local_row]];
            }
        }
    }

    #pragma omp master
    {
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

    #pragma omp barrier
}


static void compute_H_full( const sparse_matrix * const H )
{
    int count, i, pj;
    sparse_matrix *H_t;

    if ( Allocate_Matrix( H_t, H->n, H->m ) == FAILURE )
    {
        fprintf( stderr, "not enough memory for full H. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    /* Set up the sparse matrix data structure for A. */
    Transpose( H, H_t );

    count = 0;
    for ( i = 0; i < H->n; ++i )
    {
        H_full->start[i] = count;

        /* H: symmetric, lower triangular portion only stored */
        for ( pj = H->start[i]; pj < H->start[i + 1]; ++pj )
        {
            H_full->val[count] = H->val[pj];
            H_full->j[count] = H->j[pj];
            ++count;
        }
        /* H^T: symmetric, upper triangular portion only stored; 
         * skip diagonal from H^T, as included from H above */
        for ( pj = H_t->start[i] + 1; pj < H_t->start[i + 1]; ++pj )
        {
            H_full->val[count] = H_t->val[pj];
            H_full->j[count] = H_t->j[pj];
            ++count;
        }
    }
    H_full->start[i] = count;

    Deallocate_Matrix( H_t );
}


/* Iterative greedy shared-memory parallel graph coloring
 *
 * A: matrix to use for coloring, stored in CSR format;
 *   rows represent vertices, columns of entries within a row represent adjacent vertices
 *   (i.e., dependent rows for elimination during LU factorization)
 * tri: triangularity of LU (lower/upper)
 * color: vertex color (1-based)
 *
 * Reference:
 * Umit V. Catalyurek et al.
 * Graph Coloring Algorithms for Multi-core 
 *  and Massively Threaded Architectures
 * Parallel Computing, 2012
 */
void graph_coloring( const sparse_matrix * const A, const TRIANGULARITY tri )
{
    #pragma omp parallel
    {
#define MAX_COLOR (500)
        int i, pj, v;
        unsigned int temp;
        int *fb_color;

        #pragma omp master
        {
            memset( color, 0, sizeof(unsigned int) * A->n );
            recolor_cnt = A->n;
        }

        /* ordering of vertices to color depends on triangularity of factor
         * for which coloring is to be used for */
        if ( tri == LOWER )
        {
            #pragma omp for schedule(static)
            for ( i = 0; i < A->n; ++i )
            {
                to_color[i] = i;
            }
        }
        else
        {
            #pragma omp for schedule(static)
            for ( i = 0; i < A->n; ++i )
            {
                to_color[i] = A->n - 1 - i;
            }
        }

        if ( (fb_color = (int*) malloc(sizeof(int) * MAX_COLOR)) == NULL )
        {
            fprintf( stderr, "not enough memory for graph coloring. terminating.\n" );
            exit( INSUFFICIENT_MEMORY );
        }

        #pragma omp barrier

        while ( recolor_cnt > 0 )
        {
            memset( fb_color, -1, sizeof(int) * MAX_COLOR );

            /* color vertices */
            #pragma omp for schedule(static)
            for ( i = 0; i < recolor_cnt; ++i )
            {
                v = to_color[i];

                /* colors of adjacent vertices are forbidden */
                for ( pj = A->start[v]; pj < A->start[v + 1]; ++pj )
                {
                    if ( v != A->j[pj] )
                    {
                        fb_color[color[A->j[pj]]] = v;
                    }
                }

                /* search for min. color which is not in conflict with adjacent vertices;
                 * start at 1 since 0 is default (invalid) color for all vertices */
                for ( pj = 1; fb_color[pj] == v; ++pj );

                /* assign discovered color (no conflict in neighborhood of adjacent vertices) */
                color[v] = pj;
            }

            /* determine if recoloring required */
            //TODO: switch to reduction on recolor_cnt (+) via parallel scan through recolor
            #pragma omp master
            {
                temp = recolor_cnt;
                recolor_cnt = 0;

                for ( i = 0; i < temp; ++i )
                {
                    v = to_color[i];

                    /* search for color conflicts with adjacent vertices */
                    for ( pj = A->start[v]; pj < A->start[v + 1]; ++pj )
                    {
                        if ( color[v] == color[A->j[pj]] && v > A->j[pj] )
                        {
                            conflict[recolor_cnt] = v;
                            color[v] = 0;
                            ++recolor_cnt;
                            break;
                        }
                    }
                }

                temp_ptr = to_color;
                to_color = conflict;
                conflict = temp_ptr;
            }

            #pragma omp barrier
        }

        free( fb_color );

//#if defined(DEBUG)
//    #pragma omp master
//    {
//        for ( i = 0; i < A->n; ++i )
//            printf("Vertex: %5d, Color: %5d\n", i, color[i] );
//    }
//#endif

        #pragma omp barrier
    }
}


/* Sort coloring
 *
 * n: number of entries in coloring
 * tri: coloring to triangular factor to use (lower/upper)
 */
void sort_colors( const unsigned int n, const TRIANGULARITY tri )
{
    unsigned int i;

    memset( color_top, 0, sizeof(unsigned int) * (n + 1) );

    /* sort vertices by color (ascending within a color)
     *  1) count colors
     *  2) determine offsets of color ranges 
     *  3) sort by color
     *
     *  note: color is 1-based */
    for ( i = 0; i < n; ++i )
    {
        ++color_top[color[i]];
    }
    for ( i = 1; i < n + 1; ++i )
    {
        color_top[i] += color_top[i - 1];
    }
    for ( i = 0; i < n; ++i )
    {
        permuted_row_col[color_top[color[i] - 1]] = i;
        ++color_top[color[i] - 1];
    }

    /* invert mapping to get map from current row/column to permuted (new) row/column */
    for ( i = 0; i < n; ++i )
    {
        permuted_row_col_inv[permuted_row_col[i]] = i;
    }
}


/* Apply permutation Q^T*x or Q*x based on graph coloring
 *
 * color: vertex color (1-based); vertices represent matrix rows/columns
 * x: vector to permute (in-place)
 * n: number of entries in x
 * invert_map: if TRUE, use Q^T, otherwise use Q
 * tri: coloring to triangular factor to use (lower/upper)
 */
static void permute_vector( real * const x, const unsigned int n, const int invert_map,
       const TRIANGULARITY tri )
{
    unsigned int i;

    #pragma omp master
    {
        if ( x_p == NULL )
        {
            if ( (x_p = (real*) malloc(sizeof(real) * n)) == NULL )
            {
                fprintf( stderr, "not enough memory for permuting vector. terminating.\n" );
                exit( INSUFFICIENT_MEMORY );
            }
        }

        if ( invert_map == TRUE )
        {
            mapping = permuted_row_col_inv;
        }
        else
        {
            mapping = permuted_row_col;
        }
    }

    #pragma omp barrier

    #pragma omp for schedule(static)
    for ( i = 0; i < n; ++i )
    {
        x_p[i] = x[mapping[i]];
    }

    #pragma omp master
    {
        memcpy( x, x_p, sizeof(real) * n );
    }

    #pragma omp barrier
}


/* Apply permutation Q^T*(LU)*Q based on graph coloring
 *
 * color: vertex color (1-based); vertices represent matrix rows/columns
 * LU: matrix to permute, stored in CSR format
 * tri: triangularity of LU (lower/upper)
 */
void permute_matrix( sparse_matrix * const LU, const TRIANGULARITY tri )
{
    int i, pj, nr, nc;
    sparse_matrix *LUtemp;

    if ( Allocate_Matrix( LUtemp, LU->n, LU->m ) == FAILURE )
    {
        fprintf( stderr, "Not enough space for graph coloring (factor permutation). Terminating...\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    /* count nonzeros in each row of permuted factor (re-use color_top for counting) */
    memset( color_top, 0, sizeof(unsigned int) * (LU->n + 1) );

    if ( tri == LOWER )
    {
        for ( i = 0; i < LU->n; ++i )
        {
            nr = permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = permuted_row_col_inv[LU->j[pj]];

                if ( nc <= nr )
                {
                    ++color_top[nr + 1];
                }
                /* correct entries to maintain triangularity (lower) */
                else
                {
                    ++color_top[nc + 1];
                }
            }
        }
    }
    else
    {
        for ( i = LU->n - 1; i >= 0; --i )
        {
            nr = permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = permuted_row_col_inv[LU->j[pj]];

                if ( nc >= nr )
                {
                    ++color_top[nr + 1];
                }
                /* correct entries to maintain triangularity (upper) */
                else
                {
                    ++color_top[nc + 1];
                }
            }
        }
    }

    for ( i = 1; i < LU->n + 1; ++i )
    {
        color_top[i] += color_top[i - 1];
    }

    memcpy( LUtemp->start, color_top, sizeof(unsigned int) * (LU->n + 1) );

    /* permute factor */
    if ( tri == LOWER )
    {
        for ( i = 0; i < LU->n; ++i )
        {
            nr = permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = permuted_row_col_inv[LU->j[pj]];

                if ( nc <= nr )
                {
                    LUtemp->j[color_top[nr]] = nc;
                    LUtemp->val[color_top[nr]] = LU->val[pj];
                    ++color_top[nr];
                }
                /* correct entries to maintain triangularity (lower) */
                else
                {
                    LUtemp->j[color_top[nc]] = nr;
                    LUtemp->val[color_top[nc]] = LU->val[pj];
                    ++color_top[nc];
                }
            }
        }
    }
    else
    {
        for ( i = LU->n - 1; i >= 0; --i )
        {
            nr = permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = permuted_row_col_inv[LU->j[pj]];

                if ( nc >= nr )
                {
                    LUtemp->j[color_top[nr]] = nc;
                    LUtemp->val[color_top[nr]] = LU->val[pj];
                    ++color_top[nr];
                }
                /* correct entries to maintain triangularity (upper) */
                else
                {
                    LUtemp->j[color_top[nc]] = nr;
                    LUtemp->val[color_top[nc]] = LU->val[pj];
                    ++color_top[nc];
                }
            }
        }
    }

    memcpy( LU->start, LUtemp->start, sizeof(unsigned int) * (LU->n + 1) );
    memcpy( LU->j, LUtemp->j, sizeof(unsigned int) * LU->start[LU->n] );
    memcpy( LU->val, LUtemp->val, sizeof(real) * LU->start[LU->n] );

    Deallocate_Matrix( LUtemp );
}


/* Setup routines to build permuted QEq matrix H (via graph coloring),
 *  used for preconditioning (incomplete factorizations computed based on
 *  permuted H)
 *
 * H: symmetric, lower triangular portion only, stored in CSR format;
 *  H is permuted in-place
 */
sparse_matrix * setup_graph_coloring( sparse_matrix * const H )
{
    if ( color == NULL )
    {
        /* internal storage for graph coloring (global to facilitate simultaneous access to OpenMP threads) */
        if ( (color = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (to_color =(unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (conflict = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (recolor = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (color_top = (unsigned int*) malloc(sizeof(unsigned int) * (H->n + 1))) == NULL ||
                (permuted_row_col = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (permuted_row_col_inv = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (y_p = (real*) malloc(sizeof(real) * H->n)) == NULL ||
                (Allocate_Matrix( H_p, H->n, H->m ) == FAILURE ) ||
                (Allocate_Matrix( H_full, H->n, 2 * H->m - H->n ) == FAILURE ) )
        {
            fprintf( stderr, "not enough memory for graph coloring. terminating.\n" );
            exit( INSUFFICIENT_MEMORY );
        }
    }

    compute_H_full( H );

    graph_coloring( H_full, LOWER );
    sort_colors( H_full->n, LOWER );
    
    memcpy( H_p->start, H->start, sizeof(int) * (H->n + 1) );
    memcpy( H_p->j, H->j, sizeof(int) * (H->start[H->n]) );
    memcpy( H_p->val, H->val, sizeof(real) * (H->start[H->n]) );
    permute_matrix( H_p, LOWER );

    return H_p;
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
        const real * const b, real * const x, const TRIANGULARITY tri, const
        unsigned int maxiter )
{
    unsigned int i, k, si = 0, ei = 0, iter;

    iter = 0;

    #pragma omp master
    {
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
    }

    #pragma omp barrier

    Vector_MakeZero( rp, R->n );

    /* precompute and cache, as invariant in loop below */
    #pragma omp for schedule(static)
    for ( i = 0; i < R->n; ++i )
    {
        Dinv_b[i] = Dinv[i] * b[i];
    }

    do
    {
        // x_{k+1} = G*x_{k} + Dinv*b;
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

            rp2[i] = 0.;

            for ( k = si; k < ei; ++k )
            {
                rp2[i] += R->val[k] * rp[R->j[k]];
            }

            rp2[i] *= -Dinv[i];
            rp2[i] += Dinv_b[i];
        }

        #pragma omp master
        {
            rp3 = rp;
            rp = rp2;
            rp2 = rp3;
        }

        #pragma omp barrier

        ++iter;
    }
    while ( iter < maxiter );

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
static void apply_preconditioner( const static_storage * const workspace,
        const control_params * const control, const real * const y,
        real * const x, const int fresh_pre )
{
    int i, si;

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
            tri_solve( workspace->U, x, x, UPPER );
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
            tri_solve_level_sched( workspace->U, x, x, UPPER, fresh_pre );
            break;
        default:
            fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
        }
        break;
    case TRI_SOLVE_GC_PA:
        switch ( control->pre_comp_type )
        {
        case DIAG_PC:
            fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
        case ICHOLT_PC:
        case ILU_PAR_PC:
        case ILUT_PAR_PC:
            #pragma omp master
            {
                memcpy( y_p, y, sizeof(real) * workspace->H->n );
            }

            #pragma omp barrier

            permute_vector( y_p, workspace->H->n, FALSE, LOWER );
            tri_solve_level_sched( workspace->L, y_p, x, LOWER, fresh_pre );
            tri_solve_level_sched( workspace->U, x, x, UPPER, fresh_pre );
            permute_vector( x, workspace->H->n, TRUE, UPPER );
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
            #pragma omp master
            {
                if ( Dinv_L == NULL )
                {
                    if ( (Dinv_L = (real*) malloc(sizeof(real) * workspace->L->n)) == NULL )
                    {
                        fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                        exit( INSUFFICIENT_MEMORY );
                    }
                }
            }

            #pragma omp barrier

            /* construct D^{-1}_L */
            if ( fresh_pre == TRUE )
            {
                #pragma omp for schedule(static)
                for ( i = 0; i < workspace->L->n; ++i )
                {
                    si = workspace->L->start[i + 1] - 1;
                    Dinv_L[i] = 1. / workspace->L->val[si];
                }
            }

            jacobi_iter( workspace->L, Dinv_L, y, x, LOWER, control->pre_app_jacobi_iters );

            #pragma omp master
            {
                if ( Dinv_U == NULL )
                {
                    if ( (Dinv_U = (real*) malloc(sizeof(real) * workspace->U->n)) == NULL )
                    {
                        fprintf( stderr, "not enough memory for Jacobi iteration matrices. terminating.\n" );
                        exit( INSUFFICIENT_MEMORY );
                    }
                }
            }

            #pragma omp barrier

            /* construct D^{-1}_U */
            if ( fresh_pre == TRUE )
            {
                #pragma omp for schedule(static)
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
           simulation_data * const data, const sparse_matrix * const H,
           const real * const b, const real tol, real * const x,
           const FILE * const fout, const int fresh_pre )
{
    int i, j, k, itr, N, g_j, g_itr;
    real cc, tmp1, tmp2, temp, ret_temp, bnorm, time_start;

    N = H->n;

    #pragma omp parallel default(none) private(i, j, k, itr, bnorm, ret_temp) \
        shared(N, cc, tmp1, tmp2, temp, time_start, g_itr, g_j, stderr)
    {
        #pragma omp master
        {
            time_start = Get_Time( );
        }
        bnorm = Norm( b, N );
        #pragma omp master
        {
            data->timing.solver_vector_ops += Get_Timing_Info( time_start );
        }

        if ( control->pre_comp_type == DIAG_PC )
        {
            /* apply preconditioner to RHS */
            #pragma omp master
            {
                time_start = Get_Time( );
            }
            apply_preconditioner( workspace, control, b, workspace->b_prc, fresh_pre );
            #pragma omp master
            {
                data->timing.pre_app += Get_Timing_Info( time_start );
            }
        }

        /* GMRES outer-loop */
        for ( itr = 0; itr < MAX_ITR; ++itr )
        {
            /* calculate r0 */
            #pragma omp master
            {
                time_start = Get_Time( );
            }
            Sparse_MatVec( H, x, workspace->b_prm );
            #pragma omp master
            {
                data->timing.solver_spmv += Get_Timing_Info( time_start );
            }

            if ( control->pre_comp_type == DIAG_PC )
            {
                #pragma omp master
                {
                    time_start = Get_Time( );
                }
                apply_preconditioner( workspace, control, workspace->b_prm, workspace->b_prm, FALSE );
                #pragma omp master
                {
                    data->timing.pre_app += Get_Timing_Info( time_start );
                }
            }

            if ( control->pre_comp_type == DIAG_PC )
            {
                #pragma omp master
                {
                    time_start = Get_Time( );
                }
                Vector_Sum( workspace->v, 1., workspace->b_prc, -1., workspace->b_prm, N );
                #pragma omp master
                {
                    data->timing.solver_vector_ops += Get_Timing_Info( time_start );
                }
            }
            else
            {
                #pragma omp master
                {
                    time_start = Get_Time( );
                }
                Vector_Sum( workspace->v, 1., b, -1., workspace->b_prm, N );
                #pragma omp master
                {
                    data->timing.solver_vector_ops += Get_Timing_Info( time_start );
                }
            }

            if ( control->pre_comp_type != DIAG_PC )
            {
                #pragma omp master
                {
                    time_start = Get_Time( );
                }
                apply_preconditioner( workspace, control, workspace->v, workspace->v,
                        itr == 0 ? fresh_pre : FALSE );
                #pragma omp master
                {
                    data->timing.pre_app += Get_Timing_Info( time_start );
                }
            }

            #pragma omp master
            {
                time_start = Get_Time( );
            }
            ret_temp = Norm( workspace->v, N );
            #pragma omp single
            {
                workspace->g[0] = ret_temp;
            }
            Vector_Scale( workspace->v, 1. / workspace->g[0], workspace->v, N );
            #pragma omp master
            {
                data->timing.solver_vector_ops += Get_Timing_Info( time_start );
            }

            /* GMRES inner-loop */
            for ( j = 0; j < RESTART && FABS(workspace->g[j]) / bnorm > tol; j++ )
            {
                /* matvec */
                #pragma omp master
                {
                    time_start = Get_Time( );
                }
                Sparse_MatVec( H, workspace->v + j * N, workspace->v + (j + 1) * N );
                #pragma omp master
                {
                    data->timing.solver_spmv += Get_Timing_Info( time_start );
                }

                #pragma omp master
                {
                    time_start = Get_Time( );
                }
                apply_preconditioner( workspace, control,
                        workspace->v + (j + 1) * N, workspace->v + (j + 1) * N, FALSE );
                #pragma omp master
                {
                    data->timing.pre_app += Get_Timing_Info( time_start );
                }

                if ( control->pre_comp_type == DIAG_PC )
                {
                    /* apply modified Gram-Schmidt to orthogonalize the new residual */
                    #pragma omp master
                    {
                        time_start = Get_Time( );
                    }
                    for ( i = 0; i <= j; i++ )
                    {
                        workspace->h[(RESTART + 1) * i + j] =
                            Dot( workspace->v + i * N, workspace->v + (j + 1) * N, N );
                        Vector_Add( workspace->v + (j + 1) * N, -workspace->h[(RESTART + 1) * i + j],
                                workspace->v + i * N, N );
                    }
                    #pragma omp master
                    {
                        data->timing.solver_vector_ops += Get_Timing_Info( time_start );
                    }
                }
                else
                {
                    //TODO: investigate correctness of not explicitly orthogonalizing first few vectors
                    /* apply modified Gram-Schmidt to orthogonalize the new residual */
                    #pragma omp master
                    {
                        time_start = Get_Time( );
                        for ( i = 0; i < j - 1; i++ )
                        {
                            workspace->h[(RESTART + 1) * i + j] = 0;
                        }
                    }

                    for ( i = MAX(j - 1, 0); i <= j; i++ )
                    {
                        ret_temp = Dot( workspace->v + i * N, workspace->v + (j + 1) * N, N );
                        #pragma omp single
                        {
                            workspace->h[(RESTART + 1) * i + j] = ret_temp;
                        }
                        Vector_Add( workspace->v + (j + 1) * N,
                                -workspace->h[(RESTART + 1) * i + j], workspace->v + i * N, N );
                    }
                    #pragma omp master
                    {
                        data->timing.solver_vector_ops += Get_Timing_Info( time_start );
                    }
                }

                #pragma omp master
                {
                    time_start = Get_Time( );
                }
                ret_temp = Norm( workspace->v + (j + 1) * N, N );
                #pragma omp single
                {
                    workspace->h[(RESTART + 1) * (j + 1) + j] = ret_temp;
                }
                Vector_Scale( workspace->v + (j + 1) * N,
                              1. / workspace->h[(RESTART + 1) * (j + 1) + j],
                              workspace->v + (j + 1) * N, N );
                #pragma omp master
                {
                    data->timing.solver_vector_ops += Get_Timing_Info( time_start );
                }
#if defined(DEBUG)
                fprintf( stderr, "%d-%d: orthogonalization completed.\n", itr, j );
#endif

                #pragma omp master
                {
                    time_start = Get_Time( );
                    if ( control->pre_comp_type == DIAG_PC )
                    {
                        /* Givens rotations on the upper-Hessenberg matrix to make it U */
                        for ( i = 0; i <= j; i++ )
                        {
                            if ( i == j )
                            {
                                cc = SQRT( SQR(workspace->h[(RESTART + 1) * j + j])
                                        + SQR(workspace->h[(RESTART + 1) * (j + 1) + j]) );
                                workspace->hc[j] = workspace->h[(RESTART + 1) * j + j] / cc;
                                workspace->hs[j] = workspace->h[(RESTART + 1) * (j + 1) + j] / cc;
                            }

                            tmp1 =  workspace->hc[i] * workspace->h[(RESTART + 1) * i + j] +
                                workspace->hs[i] * workspace->h[(RESTART + 1) * (i + 1) + j];
                            tmp2 = -workspace->hs[i] * workspace->h[(RESTART + 1) * i + j] +
                                workspace->hc[i] * workspace->h[(RESTART + 1) * (i + 1) + j];

                            workspace->h[(RESTART + 1) * i + j] = tmp1;
                            workspace->h[(RESTART + 1) * (i + 1) + j] = tmp2;
                        }
                    }
                    else
                    {
                        //TODO: investigate correctness of not explicitly orthogonalizing first few vectors
                        /* Givens rotations on the upper-Hessenberg matrix to make it U */
                        for ( i = MAX(j - 1, 0); i <= j; i++ )
                        {
                            if ( i == j )
                            {
                                cc = SQRT( SQR(workspace->h[(RESTART + 1) * j + j])
                                        + SQR(workspace->h[(RESTART + 1) * (j + 1) + j]) );
                                workspace->hc[j] = workspace->h[(RESTART + 1) * j + j] / cc;
                                workspace->hs[j] = workspace->h[(RESTART + 1) * (j + 1) + j] / cc;
                            }

                            tmp1 =  workspace->hc[i] * workspace->h[(RESTART + 1) * i + j] +
                                    workspace->hs[i] * workspace->h[(RESTART + 1) * (i + 1) + j];
                            tmp2 = -workspace->hs[i] * workspace->h[(RESTART + 1) * i + j] +
                                   workspace->hc[i] * workspace->h[(RESTART + 1) * (i + 1) + j];

                            workspace->h[(RESTART + 1) * i + j] = tmp1;
                            workspace->h[(RESTART + 1) * (i + 1) + j] = tmp2;
                        }
                    }

                    /* apply Givens rotations to the rhs as well */
                    tmp1 =  workspace->hc[j] * workspace->g[j];
                    tmp2 = -workspace->hs[j] * workspace->g[j];
                    workspace->g[j] = tmp1;
                    workspace->g[j + 1] = tmp2;
                    data->timing.solver_orthog += Get_Timing_Info( time_start );
                }

                #pragma omp barrier

                //fprintf( stderr, "h: " );
                //for( i = 0; i <= j+1; ++i )
                //fprintf( stderr, "%.6f ", workspace->h[i][j] );
                //fprintf( stderr, "\n" );
                //fprintf( stderr, "res: %.15e\n", workspace->g[j+1] );
            }

            /* solve Hy = g: H is now upper-triangular, do back-substitution */
            #pragma omp master
            {
                time_start = Get_Time( );
                for ( i = j - 1; i >= 0; i-- )
                {
                    temp = workspace->g[i];
                    for ( k = j - 1; k > i; k-- )
                    {
                        temp -= workspace->h[(RESTART + 1) * i + k] * workspace->y[k];
                    }

                    workspace->y[i] = temp / workspace->h[(RESTART + 1) * i + i];
                }
                data->timing.solver_tri_solve += Get_Timing_Info( time_start );

                /* update x = x_0 + Vy */
                time_start = Get_Time( );
            }
            Vector_MakeZero( workspace->p, N );
            for ( i = 0; i < j; i++ )
            {
                Vector_Add( workspace->p, workspace->y[i], workspace->v + i * N, N );
            }

            Vector_Add( x, 1., workspace->p, N );
            #pragma omp master
            {
                data->timing.solver_vector_ops += Get_Timing_Info( time_start );
            }

            /* stopping condition */
            if ( FABS(workspace->g[j]) / bnorm <= tol )
            {
                break;
            }
        }

        #pragma omp master
        {
            g_itr = itr;
            g_j = j;
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
    // data->timing.solver_iters += itr * RESTART + j;

    if ( g_itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        // return -1;
        return g_itr * (RESTART + 1) + g_j + 1;
    }

    return g_itr * (RESTART + 1) + g_j + 1;
}


int GMRES_HouseHolder( const static_storage * const workspace, const control_params * const control,
                       simulation_data * const data, const sparse_matrix * const H,
                       const real * const b, real tol, real * const x,
                       const FILE * const fout, const int fresh_pre )
{
    int  i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[RESTART + 2][10000], w[RESTART + 2];
    real u[RESTART + 2][10000];

    N = H->n;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for ( i = 0; i < N; ++i )
    {
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];
    }

    // memset( x, 0, sizeof(real) * N );

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* compute z = r0 */
        Sparse_MatVec( H, x, workspace->b_prm );
        for ( i = 0; i < N; ++i )
        {
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; /* pre-conditioner */
        }
        Vector_Sum( z[0], 1.,  workspace->b_prc, -1., workspace->b_prm, N );

        Vector_MakeZero( w, RESTART + 1 );
        w[0] = Norm( z[0], N );

        Vector_Copy( u[0], z[0], N );
        u[0][0] += ( u[0][0] < 0.0 ? -1 : 1 ) * w[0];
        Vector_Scale( u[0], 1 / Norm( u[0], N ), u[0], N );

        w[0] *= ( u[0][0] < 0.0 ?  1 : -1 );
        // fprintf( stderr, "\n\n%12.6f\n", w[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs( w[j] ) / bnorm > tol; j++ )
        {
            /* compute v_j */
            Vector_Scale( z[j], -2 * u[j][j], u[j], N );
            z[j][j] += 1.; /* due to e_j */

            for ( i = j - 1; i >= 0; --i )
            {
                Vector_Add( z[j] + i, -2 * Dot( u[i] + i, z[j] + i, N - i ), u[i] + i, N - i );
            }

            /* matvec */
            Sparse_MatVec( H, z[j], v );

            for ( k = 0; k < N; ++k )
            {
                v[k] *= workspace->Hdia_inv[k]; /* pre-conditioner */
            }

            for ( i = 0; i <= j; ++i )
            {
                Vector_Add( v + i, -2 * Dot( u[i] + i, v + i, N - i ), u[i] + i, N - i );
            }

            if ( !Vector_isZero( v + (j + 1), N - (j + 1) ) )
            {
                /* compute the HouseHolder unit vector u_j+1 */
                for ( i = 0; i <= j; ++i )
                {
                    u[j + 1][i] = 0;
                }

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
            {
                workspace->h[(RESTART + 1) * i + j] = v[i];
            }


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
            {
                temp -= workspace->h[(RESTART + 1) * i + k] * workspace->y[k];
            }

            workspace->y[i] = temp / workspace->h[(RESTART + 1) * i + i];
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
        {
            Vector_Add( x, workspace->y[i], z[i], N );
        }

        // fprintf( stderr, "\nx_aft: " );
        // for( i = 0; i < N; ++i )
        //   fprintf( stderr, "%6.2f ", x[i] );

        /* stopping condition */
        if ( fabs( w[j] ) / bnorm <= tol )
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
        {
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }

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

    memset( e, 1., N * sizeof(real) );

    tri_solve( L, e, e, LOWER );
    tri_solve( U, e, e, UPPER );

    /* compute 1-norm of vector e */
    c = FABS(e[0]);
    for ( i = 1; i < N; ++i)
    {
        if ( FABS(e[i]) > c )
        {
            c = FABS(e[i]);
        }

    }

    free( e );

    return c;
}
