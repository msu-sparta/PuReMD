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
#include "tool_box.h"
#include "vector.h"

#include "print_utils.h"

/* Intel MKL */
#if defined(HAVE_LAPACKE_MKL)
#include "mkl.h"
/* reference LAPACK */
#elif defined(HAVE_LAPACKE)
#include "lapacke.h"
#endif

typedef struct
{
    unsigned int j;
    real val;
} sparse_matrix_entry;


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
unsigned int *conflict_cnt = NULL;
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


#if defined(TEST_MAT)
static sparse_matrix * create_test_mat( void )
{
    unsigned int i, n;
    sparse_matrix *H_test;

    if ( Allocate_Matrix( &H_test, 3, 6 ) == FAILURE )
    {
        fprintf( stderr, "not enough memory for test matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    //3x3, SPD, store lower half
    i = 0;
    n = 0;
    H_test->start[n] = i;
    H_test->j[i] = 0;
    H_test->val[i] = 4.;
    ++i;
    ++n;
    H_test->start[n] = i;
    H_test->j[i] = 0;
    H_test->val[i] = 12.;
    ++i;
    H_test->j[i] = 1;
    H_test->val[i] = 37.;
    ++i;
    ++n;
    H_test->start[n] = i;
    H_test->j[i] = 0;
    H_test->val[i] = -16.;
    ++i;
    H_test->j[i] = 1;
    H_test->val[i] = -43.;
    ++i;
    H_test->j[i] = 2;
    H_test->val[i] = 98.;
    ++i;
    ++n;
    H_test->start[n] = i;

    return H_test;
}
#endif


/* Routine used with qsort for sorting nonzeros within a sparse matrix row
 *
 * v1/v2: pointers to column indices of nonzeros within a row (unsigned int)
 */
static int compare_matrix_entry(const void *v1, const void *v2)
{
    /* larger element has larger column index */
    return ((sparse_matrix_entry *)v1)->j - ((sparse_matrix_entry *)v2)->j;
}


/* Routine used for sorting nonzeros within a sparse matrix row;
 *  internally, a combination of qsort and manual sorting is utilized
 *  (parallel calls to qsort when multithreading, rows mapped to threads)
 *
 * A: sparse matrix for which to sort nonzeros within a row, stored in CSR format
 */
void Sort_Matrix_Rows( sparse_matrix * const A )
{
    unsigned int i, j, si, ei;
    sparse_matrix_entry *temp;

#ifdef _OPENMP
    //    #pragma omp parallel default(none) private(i, j, si, ei, temp) shared(stderr)
#endif
    {
        if ( ( temp = (sparse_matrix_entry *) malloc( A->n * sizeof(sparse_matrix_entry)) ) == NULL )
        {
            fprintf( stderr, "Not enough space for matrix row sort. Terminating...\n" );
            exit( INSUFFICIENT_MEMORY );
        }

        /* sort each row of A using column indices */
#ifdef _OPENMP
        //        #pragma omp for schedule(guided)
#endif
        for ( i = 0; i < A->n; ++i )
        {
            si = A->start[i];
            ei = A->start[i + 1];

            for ( j = 0; j < (ei - si); ++j )
            {
                (temp + j)->j = A->j[si + j];
                (temp + j)->val = A->val[si + j];
            }

            /* polymorphic sort in standard C library using column indices */
            qsort( temp, ei - si, sizeof(sparse_matrix_entry), compare_matrix_entry );

            for ( j = 0; j < (ei - si); ++j )
            {
                A->j[si + j] = (temp + j)->j;
                A->val[si + j] = (temp + j)->val;
            }
        }

        sfree( temp, "Sort_Matrix_Rows::temp" );
    }
}


/* Convert a symmetric, half-sored sparse matrix into
 * a full-stored sparse matrix
 *
 * A: symmetric sparse matrix, lower half stored in CSR
 * A_full: resultant full sparse matrix in CSR
 *   If A_full is NULL, allocate space, otherwise do not
 *
 * Assumptions:
 *   A has non-zero diagonals
 *   Each row of A has at least one non-zero (i.e., no rows with all zeros) */
static void compute_full_sparse_matrix( const sparse_matrix * const A,
        sparse_matrix ** A_full )
{
    int count, i, pj;
    sparse_matrix *A_t;

    if ( *A_full == NULL )
    {
        if ( Allocate_Matrix( A_full, A->n, 2 * A->m - A->n ) == FAILURE )
        {
            fprintf( stderr, "not enough memory for full A. terminating.\n" );
            exit( INSUFFICIENT_MEMORY );
        }
    }


    if ( Allocate_Matrix( &A_t, A->n, A->m ) == FAILURE )
    {
        fprintf( stderr, "not enough memory for full A. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    /* Set up the sparse matrix data structure for A. */
    Transpose( A, A_t );

    count = 0;
    for ( i = 0; i < A->n; ++i )
    {

        if((*A_full)->start == NULL){
        }
        (*A_full)->start[i] = count;

        /* A: symmetric, lower triangular portion only stored */
        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            (*A_full)->val[count] = A->val[pj];
            (*A_full)->j[count] = A->j[pj];
            ++count;
        }
        /* A^T: symmetric, upper triangular portion only stored;
         * skip diagonal from A^T, as included from A above */
        for ( pj = A_t->start[i] + 1; pj < A_t->start[i + 1]; ++pj )
        {
            (*A_full)->val[count] = A_t->val[pj];
            (*A_full)->j[count] = A_t->j[pj];
            ++count;
        }
    }
    (*A_full)->start[i] = count;

    Deallocate_Matrix( A_t );
}


/* Setup routines for sparse approximate inverse preconditioner
 *
 * A: symmetric sparse matrix, lower half stored in CSR
 * filter: 
 * A_spar_patt: 
 *
 * Assumptions:
 *   A has non-zero diagonals
 *   Each row of A has at least one non-zero (i.e., no rows with all zeros) */
void Setup_Sparsity_Pattern( const sparse_matrix * const A,
        const real filter, sparse_matrix ** A_spar_patt )
{
    int i, pj, size;
    real min, max, threshold, val;

    min = 0.0;
    max = 0.0;

    if ( *A_spar_patt == NULL )
    {
        if ( Allocate_Matrix( A_spar_patt, A->n, A->m ) == FAILURE )
        {
            fprintf( stderr, "[SAI] Not enough memory for preconditioning matrices. terminating.\n" );
            exit( INSUFFICIENT_MEMORY );
        }
    }
    else if ( ((*A_spar_patt)->m) < (A->m) )
    {
        Deallocate_Matrix( *A_spar_patt );
        if ( Allocate_Matrix( A_spar_patt, A->n, A->m ) == FAILURE )
        {
            fprintf( stderr, "[SAI] Not enough memory for preconditioning matrices. terminating.\n" );
            exit( INSUFFICIENT_MEMORY );
        }
    }

    // find min and max element of the matrix
    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            val = A->val[pj];
            if ( pj == 0 )
            {
                min = val;
                max = val;
            }
            else
            {
                if ( min > val )
                {
                    min = val;
                }
                if ( max < val )
                {
                    max = val;
                }
            }
        }
    }

    threshold = min + (max-min)*(1.0-filter);
    // calculate the nnz of the sparsity pattern
    //    for ( size = 0, i = 0; i < A->n; ++i )
    //    {
    //        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
    //        {
    //            if ( threshold <= A->val[pj] )
    //                size++;
    //        }
    //    }
    //
    //    if ( Allocate_Matrix( A_spar_patt, A->n, size ) == NULL )
    //    {
    //        fprintf( stderr, "[SAI] Not enough memory for preconditioning matrices. terminating.\n" );
    //        exit( INSUFFICIENT_MEMORY );
    //    }

    //(*A_spar_patt)->start[(*A_spar_patt)->n] = size;

    // fill the sparsity pattern
    for ( size = 0, i = 0; i < A->n; ++i )
    {
        (*A_spar_patt)->start[i] = size;

        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            if ( ( A->val[pj] >= threshold )  || ( A->j[pj]==i ) )
            {
                (*A_spar_patt)->val[size] = A->val[pj];
                (*A_spar_patt)->j[size] = A->j[pj];
                size++;
            }
        }
    }
    (*A_spar_patt)->start[A->n] = size;
}

void Calculate_Droptol( const sparse_matrix * const A,
        real * const droptol, const real dtol )
{
    int i, j, k;
    real val;
#ifdef _OPENMP
    static real *droptol_local;
    unsigned int tid;
#endif

#ifdef _OPENMP
#pragma omp parallel default(none) private(i, j, k, val, tid), shared(droptol_local, stderr)
#endif
    {
#ifdef _OPENMP
        tid = omp_get_thread_num();

#pragma omp master
        {
            /* keep b_local for program duration to avoid allocate/free
             * overhead per Sparse_MatVec call*/
            if ( droptol_local == NULL )
            {
                if ( (droptol_local = (real*) malloc( omp_get_num_threads() * A->n * sizeof(real))) == NULL )
                {
                    fprintf( stderr, "Not enough space for droptol. Terminating...\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
        }

#pragma omp barrier
#endif

        /* init droptol to 0 */
        for ( i = 0; i < A->n; ++i )
        {
#ifdef _OPENMP
            droptol_local[tid * A->n + i] = 0.0;
#else
            droptol[i] = 0.0;
#endif
        }

#ifdef _OPENMP
#pragma omp barrier
#endif

        /* calculate sqaure of the norm of each row */
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for ( i = 0; i < A->n; ++i )
        {
            for ( k = A->start[i]; k < A->start[i + 1] - 1; ++k )
            {
                j = A->j[k];
                val = A->val[k];

#ifdef _OPENMP
                droptol_local[tid * A->n + i] += val * val;
                droptol_local[tid * A->n + j] += val * val;
#else
                droptol[i] += val * val;
                droptol[j] += val * val;
#endif
            }

            // diagonal entry
            val = A->val[k];
#ifdef _OPENMP
            droptol_local[tid * A->n + i] += val * val;
#else
            droptol[i] += val * val;
#endif
        }

#ifdef _OPENMP
#pragma omp barrier

#pragma omp for schedule(static)
        for ( i = 0; i < A->n; ++i )
        {
            droptol[i] = 0.0;
            for ( k = 0; k < omp_get_num_threads(); ++k )
            {
                droptol[i] += droptol_local[k * A->n + i];
            }
        }

#pragma omp barrier
#endif

        /* calculate local droptol for each row */
        //fprintf( stderr, "droptol: " );
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for ( i = 0; i < A->n; ++i )
        {
            //fprintf( stderr, "%f-->", droptol[i] );
            droptol[i] = SQRT( droptol[i] ) * dtol;
            //fprintf( stderr, "%f  ", droptol[i] );
        }
        //fprintf( stderr, "\n" );
    }
}


int Estimate_LU_Fill( const sparse_matrix * const A, const real * const droptol )
{
    int i, pj;
    int fillin;
    real val;

    fillin = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) private(i, pj, val) reduction(+: fillin)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            val = A->val[pj];

            if ( FABS(val) > droptol[i] )
            {
                ++fillin;
            }
        }
    }

    return fillin + A->n;
}


#if defined(HAVE_SUPERLU_MT)
real SuperLU_Factorize( const sparse_matrix * const A,
        sparse_matrix * const L, sparse_matrix * const U )
{
    unsigned int i, pj, count, *Ltop, *Utop, r;
    sparse_matrix *A_t;
    SuperMatrix A_S, AC_S, L_S, U_S;
    NCformat *A_S_store;
    SCPformat *L_S_store;
    NCPformat *U_S_store;
    superlumt_options_t superlumt_options;
    pxgstrf_shared_t pxgstrf_shared;
    pdgstrf_threadarg_t *pdgstrf_threadarg;
    int_t nprocs;
    fact_t fact;
    trans_t trans;
    yes_no_t refact, usepr;
    real u, drop_tol;
    real *a, *at;
    int_t *asub, *atsub, *xa, *xat;
    int_t *perm_c; /* column permutation vector */
    int_t *perm_r; /* row permutations from partial pivoting */
    void *work;
    int_t info, lwork;
    int_t permc_spec, panel_size, relax;
    Gstat_t Gstat;
    flops_t flopcnt;

    /* Default parameters to control factorization. */
#ifdef _OPENMP
    //TODO: set as global parameter and use
#pragma omp parallel \
    default(none) shared(nprocs)
    {
#pragma omp master
        {
            /* SuperLU_MT spawns threads internally, so set and pass parameter */
            nprocs = omp_get_num_threads();
        }
    }
#else
    nprocs = 1;
#endif

    //    fact = EQUILIBRATE; /* equilibrate A (i.e., scale rows & cols to have unit norm), then factorize */
    fact = DOFACT; /* factor from scratch */
    trans = NOTRANS;
    refact = NO; /* first time factorization */
    //TODO: add to control file and use the value there to set these
    panel_size = sp_ienv(1); /* # consec. cols treated as unit task */
    relax = sp_ienv(2); /* # cols grouped as relaxed supernode */
    u = 1.0; /* diagonal pivoting threshold */
    usepr = NO;
    drop_tol = 0.0;
    work = NULL;
    lwork = 0;

#if defined(DEBUG)
    fprintf( stderr, "nprocs = %d\n", nprocs );
    fprintf( stderr, "Panel size = %d\n", panel_size );
    fprintf( stderr, "Relax = %d\n", relax );
#endif

    if ( !(perm_r = intMalloc(A->n)) )
    {
        SUPERLU_ABORT("Malloc fails for perm_r[].");
    }
    if ( !(perm_c = intMalloc(A->n)) )
    {
        SUPERLU_ABORT("Malloc fails for perm_c[].");
    }
    if ( !(superlumt_options.etree = intMalloc(A->n)) )
    {
        SUPERLU_ABORT("Malloc fails for etree[].");
    }
    if ( !(superlumt_options.colcnt_h = intMalloc(A->n)) )
    {
        SUPERLU_ABORT("Malloc fails for colcnt_h[].");
    }
    if ( !(superlumt_options.part_super_h = intMalloc(A->n)) )
    {
        SUPERLU_ABORT("Malloc fails for part_super__h[].");
    }
    if ( ( (a = (real*) malloc( (2 * A->start[A->n] - A->n) * sizeof(real))) == NULL )
            || ( (asub = (int_t*) malloc( (2 * A->start[A->n] - A->n) * sizeof(int_t))) == NULL )
            || ( (xa = (int_t*) malloc( (A->n + 1) * sizeof(int_t))) == NULL )
            || ( (Ltop = (unsigned int*) malloc( (A->n + 1) * sizeof(unsigned int))) == NULL )
            || ( (Utop = (unsigned int*) malloc( (A->n + 1) * sizeof(unsigned int))) == NULL ) )
    {
        fprintf( stderr, "Not enough space for SuperLU factorization. Terminating...\n" );
        exit( INSUFFICIENT_MEMORY );
    }
    if ( Allocate_Matrix( &A_t, A->n, A->m ) == FAILURE )
    {
        fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    /* Set up the sparse matrix data structure for A. */
    Transpose( A, A_t );

    count = 0;
    for ( i = 0; i < A->n; ++i )
    {
        xa[i] = count;
        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            a[count] = A->entries[pj].val;
            asub[count] = A->entries[pj].j;
            ++count;
        }
        for ( pj = A_t->start[i] + 1; pj < A_t->start[i + 1]; ++pj )
        {
            a[count] = A_t->entries[pj].val;
            asub[count] = A_t->entries[pj].j;
            ++count;
        }
    }
    xa[i] = count;

    dCompRow_to_CompCol( A->n, A->n, 2 * A->start[A->n] - A->n, a, asub, xa,
            &at, &atsub, &xat );

    for ( i = 0; i < (2 * A->start[A->n] - A->n); ++i )
        fprintf( stderr, "%6d", asub[i] );
    fprintf( stderr, "\n" );
    for ( i = 0; i < (2 * A->start[A->n] - A->n); ++i )
        fprintf( stderr, "%6.1f", a[i] );
    fprintf( stderr, "\n" );
    for ( i = 0; i <= A->n; ++i )
        fprintf( stderr, "%6d", xa[i] );
    fprintf( stderr, "\n" );
    for ( i = 0; i < (2 * A->start[A->n] - A->n); ++i )
        fprintf( stderr, "%6d", atsub[i] );
    fprintf( stderr, "\n" );
    for ( i = 0; i < (2 * A->start[A->n] - A->n); ++i )
        fprintf( stderr, "%6.1f", at[i] );
    fprintf( stderr, "\n" );
    for ( i = 0; i <= A->n; ++i )
        fprintf( stderr, "%6d", xat[i] );
    fprintf( stderr, "\n" );

    A_S.Stype = SLU_NC; /* column-wise, no supernode */
    A_S.Dtype = SLU_D; /* double-precision */
    A_S.Mtype = SLU_GE; /* full (general) matrix -- required for parallel factorization */
    A_S.nrow = A->n;
    A_S.ncol = A->n;
    A_S.Store = (void *) SUPERLU_MALLOC( sizeof(NCformat) );
    A_S_store = (NCformat *) A_S.Store;
    A_S_store->nnz = 2 * A->start[A->n] - A->n;
    A_S_store->nzval = at;
    A_S_store->rowind = atsub;
    A_S_store->colptr = xat;

    /* ------------------------------------------------------------
       Allocate storage and initialize statistics variables.
       ------------------------------------------------------------*/
    StatAlloc( A->n, nprocs, panel_size, relax, &Gstat );
    StatInit( A->n, nprocs, &Gstat );

    /* ------------------------------------------------------------
       Get column permutation vector perm_c[], according to permc_spec:
       permc_spec = 0: natural ordering
       permc_spec = 1: minimum degree ordering on structure of A'*A
       permc_spec = 2: minimum degree ordering on structure of A'+A
       permc_spec = 3: approximate minimum degree for unsymmetric matrices
       ------------------------------------------------------------*/
    permc_spec = 0;
    get_perm_c( permc_spec, &A_S, perm_c );

    /* ------------------------------------------------------------
       Initialize the option structure superlumt_options using the
       user-input parameters;
       Apply perm_c to the columns of original A to form AC.
       ------------------------------------------------------------*/
    pdgstrf_init( nprocs, fact, trans, refact, panel_size, relax,
            u, usepr, drop_tol, perm_c, perm_r,
            work, lwork, &A_S, &AC_S, &superlumt_options, &Gstat );

    for ( i = 0; i < ((NCPformat*)AC_S.Store)->nnz; ++i )
        fprintf( stderr, "%6.1f", ((real*)(((NCPformat*)AC_S.Store)->nzval))[i] );
    fprintf( stderr, "\n" );

    /* ------------------------------------------------------------
       Compute the LU factorization of A.
       The following routine will create nprocs threads.
       ------------------------------------------------------------*/
    pdgstrf( &superlumt_options, &AC_S, perm_r, &L_S, &U_S, &Gstat, &info );

    fprintf( stderr, "INFO: %d\n", info );

    flopcnt = 0;
    for (i = 0; i < nprocs; ++i)
    {
        flopcnt += Gstat.procstat[i].fcops;
    }
    Gstat.ops[FACT] = flopcnt;

#if defined(DEBUG)
    printf("\n** Result of sparse LU **\n");
    L_S_store = (SCPformat *) L_S.Store;
    U_S_store = (NCPformat *) U_S.Store;
    printf( "No of nonzeros in factor L = " IFMT "\n", L_S_store->nnz );
    printf( "No of nonzeros in factor U = " IFMT "\n", U_S_store->nnz );
    fflush( stdout );
#endif

    /* convert L and R from SuperLU formats to CSR */
    memset( Ltop, 0, (A->n + 1) * sizeof(int) );
    memset( Utop, 0, (A->n + 1) * sizeof(int) );
    memset( L->start, 0, (A->n + 1) * sizeof(int) );
    memset( U->start, 0, (A->n + 1) * sizeof(int) );

    for ( i = 0; i < 2 * L_S_store->nnz; ++i )
        fprintf( stderr, "%6.1f", ((real*)(L_S_store->nzval))[i] );
    fprintf( stderr, "\n" );
    for ( i = 0; i < 2 * U_S_store->nnz; ++i )
        fprintf( stderr, "%6.1f", ((real*)(U_S_store->nzval))[i] );
    fprintf( stderr, "\n" );

    printf( "No of supernodes in factor L = " IFMT "\n", L_S_store->nsuper );
    for ( i = 0; i < A->n; ++i )
    {
        fprintf( stderr, "nzval_col_beg[%5d] = %d\n", i, L_S_store->nzval_colbeg[i] );
        fprintf( stderr, "nzval_col_end[%5d] = %d\n", i, L_S_store->nzval_colend[i] );
        //TODO: correct for SCPformat for L?
        //for( pj = L_S_store->rowind_colbeg[i]; pj < L_S_store->rowind_colend[i]; ++pj )
        //        for( pj = 0; pj < L_S_store->rowind_colend[i] - L_S_store->rowind_colbeg[i]; ++pj )
        //        {
        //            ++Ltop[L_S_store->rowind[L_S_store->rowind_colbeg[i] + pj] + 1];
        //        }
        fprintf( stderr, "col_beg[%5d] = %d\n", i, U_S_store->colbeg[i] );
        fprintf( stderr, "col_end[%5d] = %d\n", i, U_S_store->colend[i] );
        for ( pj = U_S_store->colbeg[i]; pj < U_S_store->colend[i]; ++pj )
        {
            ++Utop[U_S_store->rowind[pj] + 1];
            fprintf( stderr, "Utop[%5d]     = %d\n", U_S_store->rowind[pj] + 1, Utop[U_S_store->rowind[pj] + 1] );
        }
    }
    for ( i = 1; i <= A->n; ++i )
    {
        //        Ltop[i] = L->start[i] = Ltop[i] + Ltop[i - 1];
        Utop[i] = U->start[i] = Utop[i] + Utop[i - 1];
        //        fprintf( stderr, "Utop[%5d]     = %d\n", i, Utop[i] );
        //        fprintf( stderr, "U->start[%5d] = %d\n", i, U->start[i] );
    }
    for ( i = 0; i < A->n; ++i )
    {
        //        for( pj = 0; pj < L_S_store->nzval_colend[i] - L_S_store->nzval_colbeg[i]; ++pj )
        //        {
        //            r = L_S_store->rowind[L_S_store->rowind_colbeg[i] + pj];
        //            L->entries[Ltop[r]].j = r;
        //            L->entries[Ltop[r]].val = ((real*)L_S_store->nzval)[L_S_store->nzval_colbeg[i] + pj];
        //            ++Ltop[r];
        //        }
        for ( pj = U_S_store->colbeg[i]; pj < U_S_store->colend[i]; ++pj )
        {
            r = U_S_store->rowind[pj];
            U->entries[Utop[r]].j = i;
            U->entries[Utop[r]].val = ((real*)U_S_store->nzval)[pj];
            ++Utop[r];
        }
    }

    /* ------------------------------------------------------------
       Deallocate storage after factorization.
       ------------------------------------------------------------*/
    pxgstrf_finalize( &superlumt_options, &AC_S );
    Deallocate_Matrix( A_t );
    sfree( xa, "SuperLU_Factorize::xa" );
    sfree( asub, "SuperLU_Factorize::asub" );
    sfree( a, "SuperLU_Factorize::a" );
    SUPERLU_FREE( perm_r );
    SUPERLU_FREE( perm_c );
    SUPERLU_FREE( ((NCformat *)A_S.Store)->rowind );
    SUPERLU_FREE( ((NCformat *)A_S.Store)->colptr );
    SUPERLU_FREE( ((NCformat *)A_S.Store)->nzval );
    SUPERLU_FREE( A_S.Store );
    if ( lwork == 0 )
    {
        Destroy_SuperNode_SCP(&L_S);
        Destroy_CompCol_NCP(&U_S);
    }
    else if ( lwork > 0 )
    {
        SUPERLU_FREE(work);
    }
    StatFree(&Gstat);

    sfree( Utop, "SuperLU_Factorize::Utop" );
    sfree( Ltop, "SuperLU_Factorize::Ltop" );

    //TODO: return iters
    return 0.;
}
#endif


/* Diagonal (Jacobi) preconditioner computation */
real diag_pre_comp( const sparse_matrix * const H, real * const Hdia_inv )
{
    unsigned int i;
    real start;

    start = Get_Time( );

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) private(i)
#endif
    for ( i = 0; i < H->n; ++i )
    {
        if ( H->val[H->start[i + 1] - 1] != 0.0 )
        {
            Hdia_inv[i] = 1.0 / H->val[H->start[i + 1] - 1];
        }
        else
        {
            Hdia_inv[i] = 1.0;
        }
    }

    return Get_Timing_Info( start );
}


/* Incomplete Cholesky factorization with dual thresholding */
real ICHOLT( const sparse_matrix * const A, const real * const droptol,
        sparse_matrix * const L, sparse_matrix * const U )
{
    int *tmp_j;
    real *tmp_val;
    int i, j, pj, k1, k2, tmptop, Ltop;
    real val, start;
    unsigned int *Utop;

    start = Get_Time( );

    if ( ( Utop = (unsigned int*) malloc((A->n + 1) * sizeof(unsigned int)) ) == NULL ||
            ( tmp_j = (int*) malloc(A->n * sizeof(int)) ) == NULL ||
            ( tmp_val = (real*) malloc(A->n * sizeof(real)) ) == NULL )
    {
        fprintf( stderr, "[ICHOLT] Not enough memory for preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    // clear variables
    Ltop = 0;
    tmptop = 0;
    memset( L->start, 0, (A->n + 1) * sizeof(unsigned int) );
    memset( U->start, 0, (A->n + 1) * sizeof(unsigned int) );
    memset( Utop, 0, A->n * sizeof(unsigned int) );

    for ( i = 0; i < A->n; ++i )
    {
        L->start[i] = Ltop;
        tmptop = 0;

        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            j = A->j[pj];
            val = A->val[pj];

            if ( FABS(val) > droptol[i] )
            {
                k1 = 0;
                k2 = L->start[j];
                while ( k1 < tmptop && k2 < L->start[j + 1] )
                {
                    if ( tmp_j[k1] < L->j[k2] )
                    {
                        ++k1;
                    }
                    else if ( tmp_j[k1] > L->j[k2] )
                    {
                        ++k2;
                    }
                    else
                    {
                        val -= (tmp_val[k1++] * L->val[k2++]);
                    }
                }

                // L matrix is lower triangular,
                // so right before the start of next row comes jth diagonal
                val /= L->val[L->start[j + 1] - 1];

                tmp_j[tmptop] = j;
                tmp_val[tmptop] = val;
                ++tmptop;
            }
        }

        // sanity check
        if ( A->j[pj] != i )
        {
            fprintf( stderr, "[ICHOLT] badly built A matrix!\n (i = %d) ", i );
            exit( NUMERIC_BREAKDOWN );
        }

        // compute the ith diagonal in L
        val = A->val[pj];
        for ( k1 = 0; k1 < tmptop; ++k1 )
        {
            val -= (tmp_val[k1] * tmp_val[k1]);
        }

#if defined(DEBUG)
        if ( val < 0.0 )
        {
            fprintf( stderr, "[ICHOLT] Numeric breakdown (SQRT of negative on diagonal i = %d). Terminating.\n", i );
            exit( NUMERIC_BREAKDOWN );

        }
#endif

        tmp_j[tmptop] = i;
        tmp_val[tmptop] = SQRT( val );

        // apply the dropping rule once again
        //fprintf( stderr, "row%d: tmptop: %d\n", i, tmptop );
        //for( k1 = 0; k1<= tmptop; ++k1 )
        //  fprintf( stderr, "%d(%f)  ", tmp[k1].j, tmp[k1].val );
        //fprintf( stderr, "\n" );
        //fprintf( stderr, "row(%d): droptol=%.4f\n", i+1, droptol[i] );
        for ( k1 = 0; k1 < tmptop; ++k1 )
        {
            if ( FABS(tmp_val[k1]) > droptol[i] / tmp_val[tmptop] )
            {
                L->j[Ltop] = tmp_j[k1];
                L->val[Ltop] = tmp_val[k1];
                U->start[tmp_j[k1] + 1]++;
                ++Ltop;
                //fprintf( stderr, "%d(%.4f)  ", tmp[k1].j+1, tmp[k1].val );
            }
        }
        // keep the diagonal in any case
        L->j[Ltop] = tmp_j[k1];
        L->val[Ltop] = tmp_val[k1];
        ++Ltop;
        //fprintf( stderr, "%d(%.4f)\n", tmp[k1].j+1,  tmp[k1].val );
    }

    L->start[i] = Ltop;
    //    fprintf( stderr, "nnz(L): %d, max: %d\n", Ltop, L->n * 50 );

    /* U = L^T (Cholesky factorization) */
    Transpose( L, U );
    //    for ( i = 1; i <= U->n; ++i )
    //    {
    //        Utop[i] = U->start[i] = U->start[i] + U->start[i - 1] + 1;
    //    }
    //    for ( i = 0; i < L->n; ++i )
    //    {
    //        for ( pj = L->start[i]; pj < L->start[i + 1]; ++pj )
    //        {
    //            j = L->j[pj];
    //            U->j[Utop[j]] = i;
    //            U->val[Utop[j]] = L->val[pj];
    //            Utop[j]++;
    //        }
    //    }

    //    fprintf( stderr, "nnz(U): %d, max: %d\n", Utop[U->n], U->n * 50 );

    sfree( tmp_val, "ICHOLT::tmp_val" );
    sfree( tmp_j, "ICHOLT::tmp_j" );
    sfree( Utop, "ICHOLT::Utop" );

    return Get_Timing_Info( start );
}


/* Fine-grained (parallel) incomplete Cholesky factorization
 *
 * Reference:
 * Edmond Chow and Aftab Patel
 * Fine-Grained Parallel Incomplete LU Factorization
 * SIAM J. Sci. Comp. */
#if defined(TESTING)
real ICHOL_PAR( const sparse_matrix * const A, const unsigned int sweeps,
        sparse_matrix * const U_t, sparse_matrix * const U )
{
    unsigned int i, j, k, pj, x = 0, y = 0, ei_x, ei_y;
    real *D, *D_inv, sum, start;
    sparse_matrix *DAD;
    int *Utop;

    start = Get_Time( );

    if ( Allocate_Matrix( &DAD, A->n, A->m ) == FAILURE ||
            ( D = (real*) malloc(A->n * sizeof(real)) ) == NULL ||
            ( D_inv = (real*) malloc(A->n * sizeof(real)) ) == NULL ||
            ( Utop = (int*) malloc((A->n + 1) * sizeof(int)) ) == NULL )
    {
        fprintf( stderr, "not enough memory for ICHOL_PAR preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) shared(D_inv, D) private(i)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        D_inv[i] = SQRT( A->val[A->start[i + 1] - 1] );
        D[i] = 1. / D_inv[i];
    }

    memset( U->start, 0, sizeof(unsigned int) * (A->n + 1) );
    memset( Utop, 0, sizeof(unsigned int) * (A->n + 1) );

    /* to get convergence, A must have unit diagonal, so apply
     * transformation DAD, where D = D(1./SQRT(D(A))) */
    memcpy( DAD->start, A->start, sizeof(int) * (A->n + 1) );
#ifdef _OPENMP
#pragma omp parallel for schedule(guided) \
    default(none) shared(DAD, D_inv, D) private(i, pj)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        /* non-diagonals */
        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            DAD->j[pj] = A->j[pj];
            DAD->val[pj] = A->val[pj] * D[i] * D[A->j[pj]];
        }
        /* diagonal */
        DAD->j[pj] = A->j[pj];
        DAD->val[pj] = 1.;
    }

    /* initial guesses for U^T,
     * assume: A and DAD symmetric and stored lower triangular */
    memcpy( U_t->start, DAD->start, sizeof(int) * (DAD->n + 1) );
    memcpy( U_t->j, DAD->j, sizeof(int) * (DAD->m) );
    memcpy( U_t->val, DAD->val, sizeof(real) * (DAD->m) );

    for ( i = 0; i < sweeps; ++i )
    {
        /* for each nonzero */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        default(none) shared(DAD, stderr) private(sum, ei_x, ei_y, k) firstprivate(x, y)
#endif
        for ( j = 0; j < A->start[A->n]; ++j )
        {
            sum = ZERO;

            /* determine row bounds of current nonzero */
            x = 0;
            ei_x = 0;
            for ( k = 0; k <= A->n; ++k )
            {
                if ( U_t->start[k] > j )
                {
                    x = U_t->start[k - 1];
                    ei_x = U_t->start[k];
                    break;
                }
            }
            /* column bounds of current nonzero */
            y = U_t->start[U_t->j[j]];
            ei_y = U_t->start[U_t->j[j] + 1];

            /* sparse dot product: dot( U^T(i,1:j-1), U^T(j,1:j-1) ) */
            while ( U_t->j[x] < U_t->j[j] &&
                    U_t->j[y] < U_t->j[j] &&
                    x < ei_x && y < ei_y )
            {
                if ( U_t->j[x] == U_t->j[y] )
                {
                    sum += (U_t->val[x] * U_t->val[y]);
                    ++x;
                    ++y;
                }
                else if ( U_t->j[x] < U_t->j[y] )
                {
                    ++x;
                }
                else
                {
                    ++y;
                }
            }

            sum = DAD->val[j] - sum;

            /* diagonal entries */
            if ( (k - 1) == U_t->j[j] )
            {
                /* sanity check */
                if ( sum < ZERO )
                {
                    fprintf( stderr, "Numeric breakdown in ICHOL_PAR. Terminating.\n");
#if defined(DEBUG_FOCUS)
                    fprintf( stderr, "A(%5d,%5d) = %10.3f\n",
                            k - 1, A->entries[j].j, A->entries[j].val );
                    fprintf( stderr, "sum = %10.3f\n", sum);
#endif
                    exit(NUMERIC_BREAKDOWN);
                }

                U_t->val[j] = SQRT( sum );
            }
            /* non-diagonal entries */
            else
            {
                U_t->val[j] = sum / U_t->val[ei_y - 1];
            }
        }
    }

    /* apply inverse transformation D^{-1}U^{T},
     * since DAD \approx U^{T}U, so
     * D^{-1}DADD^{-1} = A \approx D^{-1}U^{T}UD^{-1} */
#ifdef _OPENMP
#pragma omp parallel for schedule(guided) \
    default(none) shared(D_inv) private(i, pj)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            U_t->val[pj] *= D_inv[i];
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nnz(L): %d, max: %d\n", U_t->start[U_t->n], U_t->n * 50 );
#endif

    /* transpose U^{T} and copy into U */
    Transpose( U_t, U );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nnz(U): %d, max: %d\n", Utop[U->n], U->n * 50 );
#endif

    Deallocate_Matrix( DAD );
    sfree( D_inv, "ICHOL_PAR::D_inv" );
    sfree( D, "ICHOL_PAR::D" );
    sfree( Utop, "ICHOL_PAR::Utop" );

    return Get_Timing_Info( start );
}
#endif


/* Fine-grained (parallel) incomplete LU factorization
 *
 * Reference:
 * Edmond Chow and Aftab Patel
 * Fine-Grained Parallel Incomplete LU Factorization
 * SIAM J. Sci. Comp.
 *
 * A: symmetric, half-stored (lower triangular), CSR format
 * sweeps: number of loops over non-zeros for computation
 * L / U: factorized triangular matrices (A \approx LU), CSR format */
real ILU_PAR( const sparse_matrix * const A, const unsigned int sweeps,
        sparse_matrix * const L, sparse_matrix * const U )
{
    unsigned int i, j, k, pj, x, y, ei_x, ei_y;
    real *D, *D_inv, sum, start;
    sparse_matrix *DAD;

    start = Get_Time( );

    if ( Allocate_Matrix( &DAD, A->n, A->m ) == FAILURE ||
            ( D = (real*) malloc(A->n * sizeof(real)) ) == NULL ||
            ( D_inv = (real*) malloc(A->n * sizeof(real)) ) == NULL )
    {
        fprintf( stderr, "[ILU_PAR] Not enough memory for preconditioning matrices. Terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) shared(D, D_inv) private(i)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        D_inv[i] = SQRT( FABS( A->val[A->start[i + 1] - 1] ) );
        D[i] = 1.0 / D_inv[i];
        //        printf( "A->val[%8d] = %f, D[%4d] = %f, D_inv[%4d] = %f\n", A->start[i + 1] - 1, A->val[A->start[i + 1] - 1], i, D[i], i, D_inv[i] );
    }

    /* to get convergence, A must have unit diagonal, so apply
     * transformation DAD, where D = D(1./SQRT(abs(D(A)))) */
    memcpy( DAD->start, A->start, sizeof(int) * (A->n + 1) );
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) shared(DAD, D) private(i, pj)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        /* non-diagonals */
        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            DAD->j[pj] = A->j[pj];
            DAD->val[pj] = D[i] * A->val[pj] * D[A->j[pj]];
        }
        /* diagonal */
        DAD->j[pj] = A->j[pj];
        DAD->val[pj] = 1.0;
    }

    /* initial guesses for L and U,
     * assume: A and DAD symmetric and stored lower triangular */
    memcpy( L->start, DAD->start, sizeof(int) * (DAD->n + 1) );
    memcpy( L->j, DAD->j, sizeof(int) * (DAD->start[DAD->n]) );
    memcpy( L->val, DAD->val, sizeof(real) * (DAD->start[DAD->n]) );
    /* store U^T in CSR for row-wise access and tranpose later */
    memcpy( U->start, DAD->start, sizeof(int) * (DAD->n + 1) );
    memcpy( U->j, DAD->j, sizeof(int) * (DAD->start[DAD->n]) );
    memcpy( U->val, DAD->val, sizeof(real) * (DAD->start[DAD->n]) );

    /* L has unit diagonal, by convention */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) private(i)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        L->val[L->start[i + 1] - 1] = 1.0;
    }

    for ( i = 0; i < sweeps; ++i )
    {
        /* for each nonzero in L */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        default(none) shared(DAD) private(j, k, x, y, ei_x, ei_y, sum)
#endif
        for ( j = 0; j < DAD->start[DAD->n]; ++j )
        {
            sum = ZERO;

            /* determine row bounds of current nonzero */
            x = 0;
            ei_x = 0;
            for ( k = 1; k <= DAD->n; ++k )
            {
                if ( DAD->start[k] > j )
                {
                    x = DAD->start[k - 1];
                    ei_x = DAD->start[k];
                    break;
                }
            }
            /* determine column bounds of current nonzero */
            y = DAD->start[DAD->j[j]];
            ei_y = DAD->start[DAD->j[j] + 1];

            /* sparse dot product:
             *   dot( L(i,1:j-1), U(1:j-1,j) ) */
            while ( L->j[x] < L->j[j] &&
                    L->j[y] < L->j[j] &&
                    x < ei_x && y < ei_y )
            {
                if ( L->j[x] == L->j[y] )
                {
                    sum += (L->val[x] * U->val[y]);
                    ++x;
                    ++y;
                }
                else if ( L->j[x] < L->j[y] )
                {
                    ++x;
                }
                else
                {
                    ++y;
                }
            }

            if ( j != ei_x - 1 )
            {
                L->val[j] = ( DAD->val[j] - sum ) / U->val[ei_y - 1];
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        default(none) shared(DAD) private(j, k, x, y, ei_x, ei_y, sum)
#endif
        for ( j = 0; j < DAD->start[DAD->n]; ++j )
        {
            sum = ZERO;

            /* determine row bounds of current nonzero */
            x = 0;
            ei_x = 0;
            for ( k = 1; k <= DAD->n; ++k )
            {
                if ( DAD->start[k] > j )
                {
                    x = DAD->start[k - 1];
                    ei_x = DAD->start[k];
                    break;
                }
            }
            /* determine column bounds of current nonzero */
            y = DAD->start[DAD->j[j]];
            ei_y = DAD->start[DAD->j[j] + 1];

            /* sparse dot product:
             *   dot( L(i,1:i-1), U(1:i-1,j) ) */
            while ( U->j[x] < U->j[j] &&
                    U->j[y] < U->j[j] &&
                    x < ei_x && y < ei_y )
            {
                if ( U->j[x] == U->j[y] )
                {
                    sum += (L->val[y] * U->val[x]);
                    ++x;
                    ++y;
                }
                else if ( U->j[x] < U->j[y] )
                {
                    ++x;
                }
                else
                {
                    ++y;
                }
            }

            U->val[j] = DAD->val[j] - sum;
        }
    }

    /* apply inverse transformation:
     * since DAD \approx LU, then
     * D^{-1}DADD^{-1} = A \approx D^{-1}LUD^{-1} */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) shared(DAD, D_inv) private(i, pj)
#endif
    for ( i = 0; i < DAD->n; ++i )
    {
        for ( pj = DAD->start[i]; pj < DAD->start[i + 1]; ++pj )
        {
            L->val[pj] = D_inv[i] * L->val[pj];
            /* currently storing U^T, so use row index instead of column index */
            U->val[pj] = U->val[pj] * D_inv[i];
        }
    }

    Transpose_I( U );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nnz(L): %d, max: %d\n", L->start[L->n], L->n * 50 );
    fprintf( stderr, "nnz(U): %d, max: %d\n", Utop[U->n], U->n * 50 );
#endif

    Deallocate_Matrix( DAD );
    sfree( D_inv, "ILU_PAR::D_inv" );
    sfree( D, "ILU_PAR::D_inv" );

    return Get_Timing_Info( start );
}


/* Fine-grained (parallel) incomplete LU factorization with thresholding
 *
 * Reference:
 * Edmond Chow and Aftab Patel
 * Fine-Grained Parallel Incomplete LU Factorization
 * SIAM J. Sci. Comp.
 *
 * A: symmetric, half-stored (lower triangular), CSR format
 * droptol: row-wise tolerances used for dropping
 * sweeps: number of loops over non-zeros for computation
 * L / U: factorized triangular matrices (A \approx LU), CSR format */
real ILUT_PAR( const sparse_matrix * const A, const real * droptol,
        const unsigned int sweeps, sparse_matrix * const L, sparse_matrix * const U )
{
    unsigned int i, j, k, pj, x, y, ei_x, ei_y, Ltop, Utop;
    real *D, *D_inv, sum, start;
    sparse_matrix *DAD, *L_temp, *U_temp;

    start = Get_Time( );

    if ( Allocate_Matrix( &DAD, A->n, A->m ) == FAILURE ||
            Allocate_Matrix( &L_temp, A->n, A->m ) == FAILURE ||
            Allocate_Matrix( &U_temp, A->n, A->m ) == FAILURE )
    {
        fprintf( stderr, "not enough memory for ILUT_PAR preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    if ( ( D = (real*) malloc(A->n * sizeof(real)) ) == NULL ||
            ( D_inv = (real*) malloc(A->n * sizeof(real)) ) == NULL )
    {
        fprintf( stderr, "not enough memory for ILUT_PAR preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) shared(D, D_inv) private(i)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        D_inv[i] = SQRT( FABS( A->val[A->start[i + 1] - 1] ) );
        D[i] = 1.0 / D_inv[i];
    }

    /* to get convergence, A must have unit diagonal, so apply
     * transformation DAD, where D = D(1./SQRT(D(A))) */
    memcpy( DAD->start, A->start, sizeof(int) * (A->n + 1) );
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) shared(DAD, D) private(i, pj)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        /* non-diagonals */
        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            DAD->j[pj] = A->j[pj];
            DAD->val[pj] = D[i] * A->val[pj] * D[A->j[pj]];
        }
        /* diagonal */
        DAD->j[pj] = A->j[pj];
        DAD->val[pj] = 1.0;
    }

    /* initial guesses for L and U,
     * assume: A and DAD symmetric and stored lower triangular */
    memcpy( L_temp->start, DAD->start, sizeof(int) * (DAD->n + 1) );
    memcpy( L_temp->j, DAD->j, sizeof(int) * (DAD->start[DAD->n]) );
    memcpy( L_temp->val, DAD->val, sizeof(real) * (DAD->start[DAD->n]) );
    /* store U^T in CSR for row-wise access and tranpose later */
    memcpy( U_temp->start, DAD->start, sizeof(int) * (DAD->n + 1) );
    memcpy( U_temp->j, DAD->j, sizeof(int) * (DAD->start[DAD->n]) );
    memcpy( U_temp->val, DAD->val, sizeof(real) * (DAD->start[DAD->n]) );

    /* L has unit diagonal, by convention */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) private(i) shared(L_temp)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        L_temp->val[L_temp->start[i + 1] - 1] = 1.0;
    }

    for ( i = 0; i < sweeps; ++i )
    {
        /* for each nonzero in L */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        default(none) shared(DAD, L_temp, U_temp) private(j, k, x, y, ei_x, ei_y, sum)
#endif
        for ( j = 0; j < DAD->start[DAD->n]; ++j )
        {
            sum = ZERO;

            /* determine row bounds of current nonzero */
            x = 0;
            ei_x = 0;
            for ( k = 1; k <= DAD->n; ++k )
            {
                if ( DAD->start[k] > j )
                {
                    x = DAD->start[k - 1];
                    ei_x = DAD->start[k];
                    break;
                }
            }
            /* determine column bounds of current nonzero */
            y = DAD->start[DAD->j[j]];
            ei_y = DAD->start[DAD->j[j] + 1];

            /* sparse dot product:
             *   dot( L(i,1:j-1), U(1:j-1,j) ) */
            while ( L_temp->j[x] < L_temp->j[j] &&
                    L_temp->j[y] < L_temp->j[j] &&
                    x < ei_x && y < ei_y )
            {
                if ( L_temp->j[x] == L_temp->j[y] )
                {
                    sum += (L_temp->val[x] * U_temp->val[y]);
                    ++x;
                    ++y;
                }
                else if ( L_temp->j[x] < L_temp->j[y] )
                {
                    ++x;
                }
                else
                {
                    ++y;
                }
            }

            if ( j != ei_x - 1 )
            {
                L_temp->val[j] = ( DAD->val[j] - sum ) / U_temp->val[ei_y - 1];
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
        default(none) shared(DAD, L_temp, U_temp) private(j, k, x, y, ei_x, ei_y, sum)
#endif
        for ( j = 0; j < DAD->start[DAD->n]; ++j )
        {
            sum = ZERO;

            /* determine row bounds of current nonzero */
            x = 0;
            ei_x = 0;
            for ( k = 1; k <= DAD->n; ++k )
            {
                if ( DAD->start[k] > j )
                {
                    x = DAD->start[k - 1];
                    ei_x = DAD->start[k];
                    break;
                }
            }
            /* determine column bounds of current nonzero */
            y = DAD->start[DAD->j[j]];
            ei_y = DAD->start[DAD->j[j] + 1];

            /* sparse dot product:
             *   dot( L(i,1:i-1), U(1:i-1,j) ) */
            while ( U_temp->j[x] < U_temp->j[j] &&
                    U_temp->j[y] < U_temp->j[j] &&
                    x < ei_x && y < ei_y )
            {
                if ( U_temp->j[x] == U_temp->j[y] )
                {
                    sum += (L_temp->val[y] * U_temp->val[x]);
                    ++x;
                    ++y;
                }
                else if ( U_temp->j[x] < U_temp->j[y] )
                {
                    ++x;
                }
                else
                {
                    ++y;
                }
            }

            U_temp->val[j] = DAD->val[j] - sum;
        }
    }

    /* apply inverse transformation:
     * since DAD \approx LU, then
     * D^{-1}DADD^{-1} = A \approx D^{-1}LUD^{-1} */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    default(none) shared(DAD, L_temp, U_temp, D_inv) private(i, pj)
#endif
    for ( i = 0; i < DAD->n; ++i )
    {
        for ( pj = DAD->start[i]; pj < DAD->start[i + 1]; ++pj )
        {
            L_temp->val[pj] = D_inv[i] * L_temp->val[pj];
            /* currently storing U^T, so use row index instead of column index */
            U_temp->val[pj] = U_temp->val[pj] * D_inv[i];
        }
    }

    /* apply the dropping rule */
    Ltop = 0;
    Utop = 0;
    for ( i = 0; i < DAD->n; ++i )
    {
        L->start[i] = Ltop;
        U->start[i] = Utop;

        for ( pj = L_temp->start[i]; pj < L_temp->start[i + 1] - 1; ++pj )
        {
            if ( FABS( L_temp->val[pj] ) > FABS( droptol[i] / L_temp->val[L_temp->start[i + 1] - 1] ) )
            {
                L->j[Ltop] = L_temp->j[pj];
                L->val[Ltop] = L_temp->val[pj];
                ++Ltop;
            }
        }

        /* diagonal */
        L->j[Ltop] = L_temp->j[pj];
        L->val[Ltop] = L_temp->val[pj];
        ++Ltop;

        for ( pj = U_temp->start[i]; pj < U_temp->start[i + 1] - 1; ++pj )
        {
            if ( FABS( U_temp->val[pj] ) > FABS( droptol[i] / U_temp->val[U_temp->start[i + 1] - 1] ) )
            {
                U->j[Utop] = U_temp->j[pj];
                U->val[Utop] = U_temp->val[pj];
                ++Utop;
            }
        }

        /* diagonal */
        U->j[Utop] = U_temp->j[pj];
        U->val[Utop] = U_temp->val[pj];
        ++Utop;
    }

    L->start[i] = Ltop;
    U->start[i] = Utop;

    Transpose_I( U );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nnz(L): %d\n", L->start[L->n] );
    fprintf( stderr, "nnz(U): %d\n", U->start[U->n] );
#endif

    Deallocate_Matrix( U_temp );
    Deallocate_Matrix( L_temp );
    Deallocate_Matrix( DAD );
    sfree( D_inv, "ILUT_PAR::D_inv" );
    sfree( D, "ILUT_PAR::D_inv" );

    return Get_Timing_Info( start );
}


#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
real Sparse_Approx_Inverse( const sparse_matrix * const A,
        const sparse_matrix * const A_spar_patt,
        sparse_matrix ** A_app_inv )
{
    //trial
    int i, k, pj, j_temp, identity_pos;
    int N, M, d_i, d_j;
    lapack_int m, n, nrhs, lda, ldb, info;
    int *pos_x, *pos_y;
    real start;
    real *e_j, *dense_matrix;
    sparse_matrix *A_full = NULL, *A_spar_patt_full = NULL;
    char *X, *Y;

    start = Get_Time( );

    X = (char *) smalloc( sizeof(char) * A->n, "Sparse_Approx_Inverse::X" );
    Y = (char *) smalloc( sizeof(char) * A->n, "Sparse_Approx_Inverse::Y" );
    pos_x = (int *) smalloc( sizeof(int) * A->n, "Sparse_Approx_Inverse::pos_x" );
    pos_y = (int *) smalloc( sizeof(int) * A->n, "Sparse_Approx_Inverse::pos_y" );

    for ( i = 0; i < A->n; ++i )
    {
        X[i] = 0;
        Y[i] = 0;
        pos_x[i] = 0;
        pos_y[i] = 0;
    }

    // Get A and A_spar_patt full matrices
    compute_full_sparse_matrix( A, &A_full );
    compute_full_sparse_matrix( A_spar_patt, &A_spar_patt_full );

    // char file[100] = "H_spar_patt_full";
    // Print_Sparse_Matrix2(A_spar_patt_full, file, NULL);
    // A_app_inv will be the same as A_spar_patt_full except the val array
    if ( Allocate_Matrix( A_app_inv, A_spar_patt_full->n, A_spar_patt_full->m ) == FAILURE )
    {
        fprintf( stderr, "not enough memory for approximate inverse matrix. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }


    (*A_app_inv)->start[(*A_app_inv)->n] = A_spar_patt_full->start[A_spar_patt_full->n];

    // For each row of full A_spar_patt
    for ( i = 0; i < A_spar_patt_full->n; ++i )
    {
        N = 0;
        M = 0;

        // find column indices of nonzeros (which will be the columns indices of the dense matrix)
        for ( pj = A_spar_patt_full->start[i]; pj < A_spar_patt_full->start[i + 1]; ++pj )
        {

            j_temp = A_spar_patt_full->j[pj];

            Y[j_temp] = 1;
            pos_y[j_temp] = N;
            ++N;

            // for each of those indices
            // search through the row of full A of that index
            for ( k = A_full->start[j_temp]; k < A_full->start[j_temp + 1]; ++k )
            {
                // and accumulate the nonzero column indices to serve as the row indices of the dense matrix
                X[A_full->j[k]] = 1;
            }
        }

        // enumerate the row indices from 0 to (# of nonzero rows - 1) for the dense matrix
        identity_pos = M;
        for ( k = 0; k < A_full->n; k++)
        {
            if ( X[k] != 0 )
            {
                pos_x[M] = k;
                if ( k == i )
                {
                    identity_pos = M;
                }
                ++M;
            }
        }

        // allocate memory for NxM dense matrix
        dense_matrix = (real *) smalloc( sizeof(real) * N * M,
                "Sparse_Approx_Inverse::dense_matrix" );

        // fill in the entries of dense matrix
        for ( d_i = 0; d_i < M; ++d_i)
        {
            // all rows are initialized to zero
            for ( d_j = 0; d_j < N; ++d_j )
            {
                dense_matrix[d_i * N + d_j] = 0.0;
            }
            // change the value if any of the column indices is seen
            for ( d_j = A_full->start[pos_x[d_i]];
                    d_j < A_full->start[pos_x[d_i + 1]]; ++d_j )
            {
                if ( Y[A_full->j[d_j]] == 1 )
                {
                    dense_matrix[d_i * N + pos_y[A_full->j[d_j]]] = A_full->val[d_j];
                }
            }

        }

        /* create the right hand side of the linear equation
           that is the full column of the identity matrix*/
        e_j = (real *) smalloc( sizeof(real) * M,
                "Sparse_Approx_Inverse::e_j" );

        for ( k = 0; k < M; ++k )
        {
            e_j[k] = 0.0;
        }
        e_j[identity_pos] = 1.0;

        // call QR-decompostion from LAPACK
        m = M;
        n = N;
        nrhs = 1;
        lda = N;
        ldb = 1;
        /* Executable statements */
        // printf( "LAPACKE_dgels (row-major, high-level) Example Program Results\n" );
        /* Solve the equations A*X = B */

        info = LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, dense_matrix, lda,
                e_j, ldb );
        /* Check for the full rank */
        if ( info > 0 )
        {
            fprintf( stderr, "The diagonal element %i of the triangular factor ", info );
            fprintf( stderr, "of A is zero, so that A does not have full rank;\n" );
            fprintf( stderr, "the least squares solution could not be computed.\n" );
            exit( INVALID_INPUT );
        }
        /* Print least squares solution */
        // print_matrix( "Least squares solution", n, nrhs, b, ldb );

        // accumulate the resulting vector to build A_app_inv
        (*A_app_inv)->start[i] = A_spar_patt_full->start[i];
        for ( k = A_spar_patt_full->start[i]; k < A_spar_patt_full->start[i + 1]; ++k)
        {
            (*A_app_inv)->j[k] = A_spar_patt_full->j[k];
            (*A_app_inv)->val[k] = e_j[k - A_spar_patt_full->start[i]];
        }

        //empty variables that will be used next iteration
        sfree( dense_matrix, "Sparse_Approx_Inverse::dense_matrix" );
        sfree( e_j, "Sparse_Approx_Inverse::e_j"  );
        for ( k = 0; k < A->n; ++k )
        {
            X[k] = 0;
            Y[k] = 0;
            pos_x[k] = 0;
            pos_y[k] = 0;
        }
    }

    // Deallocate
    Deallocate_Matrix( A_full );
    Deallocate_Matrix( A_spar_patt_full );
    sfree( X, "Sparse_Approx_Inverse::X" );
    sfree( Y, "Sparse_Approx_Inverse::Y" );
    sfree( pos_x, "Sparse_Approx_Inverse::pos_x" );
    sfree( pos_y, "Sparse_Approx_Inverse::pos_y" );

    return Get_Timing_Info( start );
}
#endif


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
    tid = omp_get_thread_num( );

#pragma omp single
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

    Vector_MakeZero( (real * const)b_local, omp_get_num_threads() * n );

#pragma omp for schedule(static)
#endif
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
void Transpose( const sparse_matrix * const A, sparse_matrix * const A_t )
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

    sfree( A_t_top, "Transpose::A_t_top" );
}


/* Transpose A in-place
 *
 * A: stored in CSR
 */
void Transpose_I( sparse_matrix * const A )
{
    sparse_matrix * A_t;

    if ( Allocate_Matrix( &A_t, A->n, A->m ) == FAILURE )
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
 * N: dimensions of preconditioner and vectors (# rows in H)
 */
static void diag_pre_app( const real * const Hdia_inv, const real * const y,
        real * const x, const int N )
{
    unsigned int i;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
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
 * N: dimensions of matrix and vectors
 * tri: triangularity of LU (lower/upper)
 *
 * Assumptions:
 *   LU has non-zero diagonals
 *   Each row of LU has at least one non-zero (i.e., no rows with all zeros) */
void tri_solve( const sparse_matrix * const LU, const real * const y,
        real * const x, const int N, const TRIANGULARITY tri )
{
    int i, pj, j, si, ei;
    real val;

#ifdef _OPENMP
#pragma omp single
#endif
    {
        if ( tri == LOWER )
        {
            for ( i = 0; i < N; ++i )
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
            for ( i = N - 1; i >= 0; --i )
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
 * N: dimensions of matrix and vectors
 * tri: triangularity of LU (lower/upper)
 * find_levels: perform level search if positive, otherwise reuse existing levels
 *
 * Assumptions:
 *   LU has non-zero diagonals
 *   Each row of LU has at least one non-zero (i.e., no rows with all zeros) */
void tri_solve_level_sched( const sparse_matrix * const LU,
        const real * const y, real * const x, const int N,
        const TRIANGULARITY tri, int find_levels )
{
    int i, j, pj, local_row, local_level;

#ifdef _OPENMP
#pragma omp single
#endif
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
            if ( (row_levels = (unsigned int*) malloc((size_t)N * sizeof(unsigned int))) == NULL
                    || (level_rows = (unsigned int*) malloc((size_t)N * sizeof(unsigned int))) == NULL
                    || (level_rows_cnt = (unsigned int*) malloc((size_t)(N + 1) * sizeof(unsigned int))) == NULL )
            {
                fprintf( stderr, "Not enough space for triangular solve via level scheduling. Terminating...\n" );
                exit( INSUFFICIENT_MEMORY );
            }
        }

        if ( top == NULL )
        {
            if ( (top = (unsigned int*) malloc((size_t)(N + 1) * sizeof(unsigned int))) == NULL )
            {
                fprintf( stderr, "Not enough space for triangular solve via level scheduling. Terminating...\n" );
                exit( INSUFFICIENT_MEMORY );
            }
        }

        /* find levels (row dependencies in substitutions) */
        if ( find_levels == TRUE )
        {
            memset( row_levels, 0, N * sizeof(unsigned int) );
            memset( level_rows_cnt, 0, N * sizeof(unsigned int) );
            memset( top, 0, N * sizeof(unsigned int) );
            levels = 1;

            if ( tri == LOWER )
            {
                for ( i = 0; i < N; ++i )
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
                fprintf(stderr, "NNZ(L): %d\n", LU->start[N]);
                //#endif
            }
            else
            {
                for ( i = N - 1; i >= 0; --i )
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
                fprintf(stderr, "NNZ(U): %d\n", LU->start[N]);
                //#endif
            }

            for ( i = 1; i < levels + 1; ++i )
            {
                level_rows_cnt[i] += level_rows_cnt[i - 1];
                top[i] = level_rows_cnt[i];
            }

            for ( i = 0; i < N; ++i )
            {
                level_rows[top[row_levels[i] - 1]] = i;
                ++top[row_levels[i] - 1];
            }
        }
    }

    /* perform substitutions by level */
    if ( tri == LOWER )
    {
        for ( i = 0; i < levels; ++i )
        {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
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
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
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

#ifdef _OPENMP
#pragma omp single
#endif
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
}


static void compute_H_full( const sparse_matrix * const H )
{
    int count, i, pj;
    sparse_matrix *H_t;

    if ( Allocate_Matrix( &H_t, H->n, H->m ) == FAILURE )
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
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#define MAX_COLOR (500)
        int i, pj, v;
        unsigned int temp, recolor_cnt_local, *conflict_local;
        int tid, num_thread, *fb_color;

#ifdef _OPENMP
        tid = omp_get_thread_num();
        num_thread = omp_get_num_threads();
#else
        tid = 0;
        num_thread = 1;
#endif

#ifdef _OPENMP
#pragma omp single
#endif
        {
            memset( color, 0, sizeof(unsigned int) * A->n );
            recolor_cnt = A->n;
        }

        /* ordering of vertices to color depends on triangularity of factor
         * for which coloring is to be used for */
        if ( tri == LOWER )
        {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for ( i = 0; i < A->n; ++i )
            {
                to_color[i] = i;
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for ( i = 0; i < A->n; ++i )
            {
                to_color[i] = A->n - 1 - i;
            }
        }

        if ( (fb_color = (int*) malloc(sizeof(int) * MAX_COLOR)) == NULL ||
                (conflict_local = (unsigned int*) malloc(sizeof(unsigned int) * A->n)) == NULL )
        {
            fprintf( stderr, "not enough memory for graph coloring. terminating.\n" );
            exit( INSUFFICIENT_MEMORY );
        }

#ifdef _OPENMP
#pragma omp barrier
#endif

        while ( recolor_cnt > 0 )
        {
            memset( fb_color, -1, sizeof(int) * MAX_COLOR );

            /* color vertices */
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
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
            temp = recolor_cnt;
            recolor_cnt_local = 0;

#ifdef _OPENMP
#pragma omp single
#endif
            {
                recolor_cnt = 0;
            }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for ( i = 0; i < temp; ++i )
            {
                v = to_color[i];

                /* search for color conflicts with adjacent vertices */
                for ( pj = A->start[v]; pj < A->start[v + 1]; ++pj )
                {
                    if ( color[v] == color[A->j[pj]] && v > A->j[pj] )
                    {
                        conflict_local[recolor_cnt_local] = v;
                        ++recolor_cnt_local;
                        break;
                    }
                }
            }

            /* count thread-local conflicts and compute offsets for copying into shared buffer */
            conflict_cnt[tid + 1] = recolor_cnt_local;

#ifdef _OPENMP
#pragma omp barrier

#pragma omp master
#endif
            {
                conflict_cnt[0] = 0;
                for ( i = 1; i < num_thread + 1; ++i )
                {
                    conflict_cnt[i] += conflict_cnt[i - 1];
                }
                recolor_cnt = conflict_cnt[num_thread];
            }

#ifdef _OPENMP
#pragma omp barrier
#endif

            /* copy thread-local conflicts into shared buffer */
            for ( i = 0; i < recolor_cnt_local; ++i )
            {
                conflict[conflict_cnt[tid] + i] = conflict_local[i];
                color[conflict_local[i]] = 0;
            }

#ifdef _OPENMP
#pragma omp barrier

#pragma omp single
#endif
            {
                temp_ptr = to_color;
                to_color = conflict;
                conflict = temp_ptr;
            }
        }

        sfree( conflict_local, "graph_coloring::conflict_local" );
        sfree( fb_color, "graph_coloring::fb_color" );

        //#if defined(DEBUG)
        //#ifdef _OPENMP
        //    #pragma omp master
        //#endif
        //    {
        //        for ( i = 0; i < A->n; ++i )
        //            printf("Vertex: %5d, Color: %5d\n", i, color[i] );
        //    }
        //#endif

#ifdef _OPENMP
#pragma omp barrier
#endif
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

#ifdef _OPENMP
#pragma omp single
#endif
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

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for ( i = 0; i < n; ++i )
    {
        x_p[i] = x[mapping[i]];
    }

#ifdef _OPENMP
#pragma omp single
#endif
    {
        memcpy( x, x_p, sizeof(real) * n );
    }
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

    if ( Allocate_Matrix( &LUtemp, LU->n, LU->m ) == FAILURE )
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
    int num_thread;

    if ( color == NULL )
    {
#ifdef _OPENMP
#pragma omp parallel
        {
            num_thread = omp_get_num_threads();
        }
#else
        num_thread = 1;
#endif

        /* internal storage for graph coloring (global to facilitate simultaneous access to OpenMP threads) */
        if ( (color = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (to_color = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (conflict = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (conflict_cnt = (unsigned int*) malloc(sizeof(unsigned int) * (num_thread + 1))) == NULL ||
                (recolor = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (color_top = (unsigned int*) malloc(sizeof(unsigned int) * (H->n + 1))) == NULL ||
                (permuted_row_col = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (permuted_row_col_inv = (unsigned int*) malloc(sizeof(unsigned int) * H->n)) == NULL ||
                (y_p = (real*) malloc(sizeof(real) * H->n)) == NULL ||
                (Allocate_Matrix( &H_p, H->n, H->m ) == FAILURE ) ||
                (Allocate_Matrix( &H_full, H->n, 2 * H->m - H->n ) == FAILURE ) )
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
void jacobi_iter( const sparse_matrix * const R, const real * const Dinv,
        const real * const b, real * const x, const TRIANGULARITY tri, const
        unsigned int maxiter )
{
    unsigned int i, k, si = 0, ei = 0, iter;

    iter = 0;

#ifdef _OPENMP
#pragma omp single
#endif
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

    Vector_MakeZero( rp, R->n );

    /* precompute and cache, as invariant in loop below */
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for ( i = 0; i < R->n; ++i )
    {
        Dinv_b[i] = Dinv[i] * b[i];
    }

    do
    {
        // x_{k+1} = G*x_{k} + Dinv*b;
#ifdef _OPENMP
#pragma omp for schedule(guided)
#endif
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

#ifdef _OPENMP
#pragma omp single
#endif
        {
            rp3 = rp;
            rp = rp2;
            rp2 = rp3;
        }

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
static void apply_preconditioner( const static_storage * const workspace, const control_params * const control,
        const real * const y, real * const x, const int fresh_pre )
{
    int i, si;

    /* no preconditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        Vector_Copy( x, y, workspace->H->n );
    }
    else
    {
        switch ( control->cm_solver_pre_app_type )
        {
            case TRI_SOLVE_PA:
                switch ( control->cm_solver_pre_comp_type )
                {
                    case DIAG_PC:
                        diag_pre_app( workspace->Hdia_inv, y, x, workspace->H->n );
                        break;
                    case ICHOLT_PC:
                    case ILU_PAR_PC:
                    case ILUT_PAR_PC:
                        tri_solve( workspace->L, y, x, workspace->L->n, LOWER );
                        tri_solve( workspace->U, x, x, workspace->U->n, UPPER );
                        break;
                    case SAI_PC:
                        //TODO: add code to compute SAI first
                        //                Sparse_MatVec( SAI, y, x );
                        break;
                    default:
                        fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;
                }
                break;
            case TRI_SOLVE_LEVEL_SCHED_PA:
                switch ( control->cm_solver_pre_comp_type )
                {
                    case DIAG_PC:
                        diag_pre_app( workspace->Hdia_inv, y, x, workspace->H->n );
                        break;
                    case ICHOLT_PC:
                    case ILU_PAR_PC:
                    case ILUT_PAR_PC:
                        tri_solve_level_sched( workspace->L, y, x, workspace->L->n, LOWER, fresh_pre );
                        tri_solve_level_sched( workspace->U, x, x, workspace->U->n, UPPER, fresh_pre );
                        break;
                    case SAI_PC:
                        //TODO: add code to compute SAI first
                        //                Sparse_MatVec( SAI, y, x );
                    default:
                        fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;
                }
                break;
            case TRI_SOLVE_GC_PA:
                switch ( control->cm_solver_pre_comp_type )
                {
                    case DIAG_PC:
                    case SAI_PC:
                        fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;
                    case ICHOLT_PC:
                    case ILU_PAR_PC:
                    case ILUT_PAR_PC:
#ifdef _OPENMP
#pragma omp single
#endif
                        {
                            memcpy( y_p, y, sizeof(real) * workspace->H->n );
                        }

                        permute_vector( y_p, workspace->H->n, FALSE, LOWER );
                        tri_solve_level_sched( workspace->L, y_p, x, workspace->L->n, LOWER, fresh_pre );
                        tri_solve_level_sched( workspace->U, x, x, workspace->U->n, UPPER, fresh_pre );
                        permute_vector( x, workspace->H->n, TRUE, UPPER );
                        break;
                    default:
                        fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;
                }
                break;
            case JACOBI_ITER_PA:
                switch ( control->cm_solver_pre_comp_type )
                {
                    case DIAG_PC:
                    case SAI_PC:
                        fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;
                    case ICHOLT_PC:
                    case ILU_PAR_PC:
                    case ILUT_PAR_PC:
#ifdef _OPENMP
#pragma omp single
#endif
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

                        /* construct D^{-1}_L */
                        if ( fresh_pre == TRUE )
                        {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                            for ( i = 0; i < workspace->L->n; ++i )
                            {
                                si = workspace->L->start[i + 1] - 1;
                                Dinv_L[i] = 1. / workspace->L->val[si];
                            }
                        }

                        jacobi_iter( workspace->L, Dinv_L, y, x, LOWER, control->cm_solver_pre_app_jacobi_iters );

#ifdef _OPENMP
#pragma omp single
#endif
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

                        /* construct D^{-1}_U */
                        if ( fresh_pre == TRUE )
                        {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                            for ( i = 0; i < workspace->U->n; ++i )
                            {
                                si = workspace->U->start[i];
                                Dinv_U[i] = 1. / workspace->U->val[si];
                            }
                        }

                        jacobi_iter( workspace->U, Dinv_U, y, x, UPPER, control->cm_solver_pre_app_jacobi_iters );
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
    }
}


/* generalized minimual residual iterative solver for sparse linear systems */
int GMRES( const static_storage * const workspace, const control_params * const control,
        simulation_data * const data, const sparse_matrix * const H, const real * const b,
        const real tol, real * const x, const int fresh_pre )
{
    int i, j, k, itr, N, g_j, g_itr;
    real cc, tmp1, tmp2, temp, ret_temp, bnorm, time_start;

    N = H->n;

#ifdef _OPENMP
#pragma omp parallel default(none) private(i, j, k, itr, bnorm, ret_temp) \
    shared(N, cc, tmp1, tmp2, temp, time_start, g_itr, g_j, stderr)
#endif
    {
        j = 0;
        itr = 0;

#ifdef _OPENMP
#pragma omp master
#endif
        {
            time_start = Get_Time( );
        }
        bnorm = Norm( b, N );
#ifdef _OPENMP
#pragma omp master
#endif
        {
            data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
        }

        if ( control->cm_solver_pre_comp_type == DIAG_PC )
        {
            /* apply preconditioner to residual */
#ifdef _OPENMP
#pragma omp master
#endif
            {
                time_start = Get_Time( );
            }
            apply_preconditioner( workspace, control, b, workspace->b_prc, fresh_pre );
#ifdef _OPENMP
#pragma omp master
#endif
            {
                data->timing.cm_solver_pre_app += Get_Timing_Info( time_start );
            }
        }

        /* GMRES outer-loop */
        for ( itr = 0; itr < control->cm_solver_max_iters; ++itr )
        {
            /* calculate r0 */
#ifdef _OPENMP
#pragma omp master
#endif
            {
                time_start = Get_Time( );
            }
            Sparse_MatVec( H, x, workspace->b_prm );
#ifdef _OPENMP
#pragma omp master
#endif
            {
                data->timing.cm_solver_spmv += Get_Timing_Info( time_start );
            }

            if ( control->cm_solver_pre_comp_type == DIAG_PC )
            {
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                apply_preconditioner( workspace, control, workspace->b_prm, workspace->b_prm, FALSE );
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_pre_app += Get_Timing_Info( time_start );
                }
            }

            if ( control->cm_solver_pre_comp_type == DIAG_PC )
            {
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                Vector_Sum( workspace->v[0], 1., workspace->b_prc, -1., workspace->b_prm, N );
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
                }
            }
            else
            {
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                Vector_Sum( workspace->v[0], 1., b, -1., workspace->b_prm, N );
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
                }
            }

            if ( control->cm_solver_pre_comp_type != DIAG_PC )
            {
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                apply_preconditioner( workspace, control, workspace->v[0], workspace->v[0],
                        itr == 0 ? fresh_pre : FALSE );
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_pre_app += Get_Timing_Info( time_start );
                }
            }

#ifdef _OPENMP
#pragma omp master
#endif
            {
                time_start = Get_Time( );
            }
            ret_temp = Norm( workspace->v[0], N );
#ifdef _OPENMP
#pragma omp single
#endif
            {
                workspace->g[0] = ret_temp;
            }

            Vector_Scale( workspace->v[0], 1. / workspace->g[0], workspace->v[0], N );
#ifdef _OPENMP
#pragma omp master
#endif
            {
                data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
            }

            /* GMRES inner-loop */
            for ( j = 0; j < control->cm_solver_restart && FABS(workspace->g[j]) / bnorm > tol; j++ )
            {
                /* matvec */
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1] );

#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_spmv += Get_Timing_Info( time_start );
                }

#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                apply_preconditioner( workspace, control, workspace->v[j + 1], workspace->v[j + 1], FALSE );

#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_pre_app += Get_Timing_Info( time_start );
                }

                //                if ( control->cm_solver_pre_comp_type == DIAG_PC )
                //                {
                /* apply modified Gram-Schmidt to orthogonalize the new residual */
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                for ( i = 0; i <= j; i++ )
                {
                    ret_temp = Dot( workspace->v[i], workspace->v[j + 1], N );

#ifdef _OPENMP
#pragma omp single
#endif
                    {
                        workspace->h[i][j] = ret_temp;
                    }

                    Vector_Add( workspace->v[j + 1], -workspace->h[i][j], workspace->v[i], N );

                }
#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
                }
                //                }
                //                else
                //                {
                //                    //TODO: investigate correctness of not explicitly orthogonalizing first few vectors
                //                    /* apply modified Gram-Schmidt to orthogonalize the new residual */
                //#ifdef _OPENMP
                //                    #pragma omp master
                //#endif
                //                    {
                //                        time_start = Get_Time( );
                //                    }
                //#ifdef _OPENMP
                //                    #pragma omp single
                //#endif
                //                    {
                //                        for ( i = 0; i < j - 1; i++ )
                //                        {
                //                            workspace->h[i][j] = 0.0;
                //                        }
                //                    }
                //
                //                    for ( i = MAX(j - 1, 0); i <= j; i++ )
                //                    {
                //                        ret_temp = Dot( workspace->v[i], workspace->v[j + 1], N );
                //#ifdef _OPENMP
                //                        #pragma omp single
                //#endif
                //                        {
                //                            workspace->h[i][j] = ret_temp;
                //                        }
                //
                //                        Vector_Add( workspace->v[j + 1], -workspace->h[i][j], workspace->v[i], N );
                //                    }
                //#ifdef _OPENMP
                //                    #pragma omp master
                //#endif
                //                    {
                //                        data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
                //                    }
                //                }

#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                }
                ret_temp = Norm( workspace->v[j + 1], N );
#ifdef _OPENMP
#pragma omp single
#endif
                {
                    workspace->h[j + 1][j] = ret_temp;
                }

                Vector_Scale( workspace->v[j + 1],
                        1.0 / workspace->h[j + 1][j], workspace->v[j + 1], N );

#ifdef _OPENMP
#pragma omp master
#endif
                {
                    data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
                }

#ifdef _OPENMP
#pragma omp master
#endif
                {
                    time_start = Get_Time( );
                    //                    if ( control->cm_solver_pre_comp_type == NONE_PC ||
                    //                            control->cm_solver_pre_comp_type == DIAG_PC )
                    //                    {
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
                    //                    }
                    //                    else
                    //                    {
                    //                        //TODO: investigate correctness of not explicitly orthogonalizing first few vectors
                    //                        /* Givens rotations on the upper-Hessenberg matrix to make it U */
                    //                        for ( i = MAX(j - 1, 0); i <= j; i++ )
                    //                        {
                    //                            if ( i == j )
                    //                            {
                    //                                cc = SQRT( SQR(workspace->h[j][j]) + SQR(workspace->h[j + 1][j]) );
                    //                                workspace->hc[j] = workspace->h[j][j] / cc;
                    //                                workspace->hs[j] = workspace->h[j + 1][j] / cc;
                    //                            }
                    //
                    //                            tmp1 =  workspace->hc[i] * workspace->h[i][j] +
                    //                                    workspace->hs[i] * workspace->h[i + 1][j];
                    //                            tmp2 = -workspace->hs[i] * workspace->h[i][j] +
                    //                                   workspace->hc[i] * workspace->h[i + 1][j];
                    //
                    //                            workspace->h[i][j] = tmp1;
                    //                            workspace->h[i + 1][j] = tmp2;
                    //                        }
                    //                    }

                    /* apply Givens rotations to the rhs as well */
                    tmp1 =  workspace->hc[j] * workspace->g[j];
                    tmp2 = -workspace->hs[j] * workspace->g[j];
                    workspace->g[j] = tmp1;
                    workspace->g[j + 1] = tmp2;

                    data->timing.cm_solver_orthog += Get_Timing_Info( time_start );
                }

#ifdef _OPENMP
#pragma omp barrier
#endif
            }

            /* solve Hy = g: H is now upper-triangular, do back-substitution */
#ifdef _OPENMP
#pragma omp master
#endif
            {
                time_start = Get_Time( );

                for ( i = j - 1; i >= 0; i-- )
                {
                    temp = workspace->g[i];
                    for ( k = j - 1; k > i; k-- )
                    {
                        temp -= workspace->h[i][k] * workspace->y[k];
                    }

                    workspace->y[i] = temp / workspace->h[i][i];
                }

                data->timing.cm_solver_tri_solve += Get_Timing_Info( time_start );

                /* update x = x_0 + Vy */
                time_start = Get_Time( );
            }

            Vector_MakeZero( workspace->p, N );

            for ( i = 0; i < j; i++ )
            {
                Vector_Add( workspace->p, workspace->y[i], workspace->v[i], N );
            }

            Vector_Add( x, 1., workspace->p, N );

#ifdef _OPENMP
#pragma omp master
#endif
            {
                data->timing.cm_solver_vector_ops += Get_Timing_Info( time_start );
            }

            /* stopping condition */
            if ( FABS(workspace->g[j]) / bnorm <= tol )
            {
                break;
            }
        }

#ifdef _OPENMP
#pragma omp master
#endif
        {
            g_itr = itr;
            g_j = j;
        }
    }

    if ( g_itr >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        return g_itr * (control->cm_solver_restart + 1) + g_j + 1;
    }

    return g_itr * (control->cm_solver_restart + 1) + g_j + 1;
}


int GMRES_HouseHolder( const static_storage * const workspace,
        const control_params * const control, simulation_data * const data,
        const sparse_matrix * const H, const real * const b, real tol,
        real * const x, const int fresh_pre )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[control->cm_solver_restart + 2][10000], w[control->cm_solver_restart + 2];
    real u[control->cm_solver_restart + 2][10000];

    j = 0;
    N = H->n;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for ( i = 0; i < N; ++i )
    {
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];
    }

    // memset( x, 0, sizeof(real) * N );

    /* GMRES outer-loop */
    for ( itr = 0; itr < control->cm_solver_max_iters; ++itr )
    {
        /* compute z = r0 */
        Sparse_MatVec( H, x, workspace->b_prm );
        for ( i = 0; i < N; ++i )
        {
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; /* pre-conditioner */
        }
        Vector_Sum( z[0], 1.,  workspace->b_prc, -1., workspace->b_prm, N );

        Vector_MakeZero( w, control->cm_solver_restart + 1 );
        w[0] = Norm( z[0], N );

        Vector_Copy( u[0], z[0], N );
        u[0][0] += ( u[0][0] < 0.0 ? -1 : 1 ) * w[0];
        Vector_Scale( u[0], 1 / Norm( u[0], N ), u[0], N );

        w[0] *= ( u[0][0] < 0.0 ?  1 : -1 );
        // fprintf( stderr, "\n\n%12.6f\n", w[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < control->cm_solver_restart && FABS( w[j] ) / bnorm > tol; j++ )
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
            if ( FABS(v[j + 1]) >= ALMOST_ZERO )
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
                workspace->h[i][j] = v[i];
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
                temp -= workspace->h[i][k] * workspace->y[k];
            }

            workspace->y[i] = temp / workspace->h[i][i];
        }

        // fprintf( stderr, "y: " );
        // for( i = 0; i < control->cm_solver_restart+1; ++i )
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

        /* stopping condition */
        if ( FABS( w[j] ) / bnorm <= tol )
        {
            break;
        }
    }

    if ( itr >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        return itr * (control->cm_solver_restart + 1) + j + 1;
    }

    return itr * (control->cm_solver_restart + 1) + j + 1;
}


/* Conjugate Gradient */
int CG( const static_storage * const workspace, const control_params * const control,
        const sparse_matrix * const H, const real * const b, const real tol,
        real * const x, const int fresh_pre )
{
    int i, itr, N;
    real tmp, alpha, beta, b_norm, r_norm;
    real *d, *r, *p, *z;
    real sig_old, sig_new;

    N = H->n;
    d = workspace->d;
    r = workspace->r;
    p = workspace->q;
    z = workspace->p;

#ifdef _OPENMP
#pragma omp parallel default(none) private(i, tmp, alpha, beta, b_norm, r_norm, sig_old, sig_new) \
    shared(itr, N, d, r, p, z)
#endif
    {
        b_norm = Norm( b, N );

        Sparse_MatVec( H, x, d );
        Vector_Sum( r, 1.0,  b, -1.0, d, N );
        r_norm = Norm( r, N );

        apply_preconditioner( workspace, control, r, z, fresh_pre );
        Vector_Copy( p, z, N );

        sig_new = Dot( r, z, N );

        for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
        {
            Sparse_MatVec( H, p, d );

            tmp = Dot( d, p, N );
            alpha = sig_new / tmp;
            Vector_Add( x, alpha, p, N );

            Vector_Add( r, -alpha, d, N );
            r_norm = Norm( r, N );

            apply_preconditioner( workspace, control, r, z, FALSE );

            sig_old = sig_new;
            sig_new = Dot( r, z, N );

            beta = sig_new / sig_old;
            Vector_Sum( p, 1., z, beta, p, N );
        }

#ifdef _OPENMP
#pragma omp single
#endif
        itr = i;
    }

    if ( itr >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] CG convergence failed (%d iters)\n", itr );
        return itr;
    }

    return itr;
}


/* Steepest Descent */
int SDM( const static_storage * const workspace, const control_params * const control,
        const sparse_matrix * const H, const real * const b, const real tol,
        real * const x, const int fresh_pre )
{
    int i, itr, N;
    real tmp, alpha, b_norm;
    real sig;

    N = H->n;

#ifdef _OPENMP
#pragma omp parallel default(none) private(i, tmp, alpha, b_norm, sig) \
    shared(itr, N)
#endif
    {
        b_norm = Norm( b, N );

        Sparse_MatVec( H, x, workspace->q );
        Vector_Sum( workspace->r, 1.0,  b, -1.0, workspace->q, N );

        apply_preconditioner( workspace, control, workspace->r, workspace->d, fresh_pre );

        sig = Dot( workspace->r, workspace->d, N );

        for ( i = 0; i < control->cm_solver_max_iters && SQRT(sig) / b_norm > tol; ++i )
        {
            Sparse_MatVec( H, workspace->d, workspace->q );

            sig = Dot( workspace->r, workspace->d, N );

            /* ensure each thread gets a local copy of
             * the function return value before proceeding
             * (Dot function has persistent state in the form
             * of a shared global variable for the OpenMP version) */
#ifdef _OPENMP
#pragma omp barrier
#endif

            tmp = Dot( workspace->d, workspace->q, N );
            alpha = sig / tmp;

            Vector_Add( x, alpha, workspace->d, N );
            Vector_Add( workspace->r, -alpha, workspace->q, N );

            apply_preconditioner( workspace, control, workspace->r, workspace->d, FALSE );
        }

#ifdef _OPENMP
#pragma omp single
#endif
        itr = i;
    }

    if ( itr >= control->cm_solver_max_iters  )
    {
        fprintf( stderr, "[WARNING] SDM convergence failed (%d iters)\n", itr );
        return itr;
    }

    return itr;
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

    tri_solve( L, e, e, L->n, LOWER );
    tri_solve( U, e, e, U->n, UPPER );

    /* compute 1-norm of vector e */
    c = FABS(e[0]);
    for ( i = 1; i < N; ++i)
    {
        if ( FABS(e[i]) > c )
        {
            c = FABS(e[i]);
        }

    }

    sfree( e, "condest::e" );

    return c;
}
