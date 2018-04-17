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


enum preconditioner_type
{
    LEFT = 0,
    RIGHT = 1,
};


#if defined(TEST_MAT)
static sparse_matrix * create_test_mat( void )
{
    unsigned int i, n;
    sparse_matrix *H_test;

    Allocate_Matrix( &H_test, 3, 6 );

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
        temp = (sparse_matrix_entry *) smalloc( A->n * sizeof(sparse_matrix_entry),
                                                "Sort_Matrix_Rows::temp" );

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
        Allocate_Matrix( A_full, A->n, 2 * A->m - A->n );
    }
    else if ( (*A_full)->m < 2 * A->m - A->n )
    {
        Deallocate_Matrix( *A_full );
        Allocate_Matrix( A_full, A->n, 2 * A->m - A->n );
    }

    Allocate_Matrix( &A_t, A->n, A->m );

    /* Set up the sparse matrix data structure for A. */
    Transpose( A, A_t );

    count = 0;
    for ( i = 0; i < A->n; ++i )
    {

        if ((*A_full)->start == NULL)
        {
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
void setup_sparse_approx_inverse( const sparse_matrix * const A, sparse_matrix ** A_full,
              sparse_matrix ** A_spar_patt, sparse_matrix **A_spar_patt_full,
                    sparse_matrix ** A_app_inv, const real filter )
{
    int i, pj, size;
    int left, right, k, p, turn;
    real pivot, tmp;
    real threshold;
    real *list;

    if ( *A_spar_patt == NULL )
    {
        Allocate_Matrix( A_spar_patt, A->n, A->m );
    }
    else if ( ((*A_spar_patt)->m) < (A->m) )
    {
        Deallocate_Matrix( *A_spar_patt );
        Allocate_Matrix( A_spar_patt, A->n, A->m );
    }


    /* quick-select algorithm for finding the kth greatest element in the matrix*/
    /* list: values from the matrix*/
    /* left-right: search space of the quick-select */

    list = (real *) smalloc( sizeof(real) * (A->start[A->n]),"setup_sparse_approx_inverse::list" );

    left = 0;
    right = A->start[A->n] - 1;
    k = (int)( (A->start[A->n])*filter );
    threshold = 0.0;

    for( i = left; i <= right ; ++i )
    {
        list[i] = A->val[i];
        if(list[i] < 0.0)
        {
            list[i] = -list[i];
        }
    }

    turn = 0;
    while( k ) {

        p  = left;
        turn = 1 - turn;
        if( turn == 1)
        {
            pivot = list[right];
        }
        else
        {
            pivot = list[left];
        }
        for( i = left + 1 - turn; i <= right-turn; ++i )
        {
            if( list[i] > pivot )
            {
                tmp = list[i];
                list[i] = list[p];
                list[p] = tmp;
                p++;
            }
        }
        if(turn == 1)
        {
            tmp = list[p];
            list[p] = list[right];
            list[right] = tmp;
        }
        else
        {
            tmp = list[p];
            list[p] = list[left];
            list[left] = tmp;
        }

        if( p == k - 1)
        {
            threshold = list[p];
            break;
        }
        else if( p > k - 1 )
        {
            right = p - 1;
        }
        else
        {
            left = p + 1;
        }
    }

    if(threshold < 1.000000)
    {
        threshold = 1.000001;
    }

    sfree( list, "setup_sparse_approx_inverse::list" );

    /* fill sparsity pattern */
    /* diagonal entries are always included */
    for ( size = 0, i = 0; i < A->n; ++i )
    {
        (*A_spar_patt)->start[i] = size;

        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            if ( ( A->val[pj] >= threshold )  || ( A->j[pj] == i ) )
            {
                (*A_spar_patt)->val[size] = A->val[pj];
                (*A_spar_patt)->j[size] = A->j[pj];
                size++;
            }
        }
    }
    (*A_spar_patt)->start[A->n] = size;

    compute_full_sparse_matrix( A, A_full );
    compute_full_sparse_matrix( *A_spar_patt, A_spar_patt_full );

    if ( *A_app_inv == NULL )
    {
        /* A_app_inv has the same sparsity pattern
         * * as A_spar_patt_full (omit non-zero values) */
        Allocate_Matrix( A_app_inv, (*A_spar_patt_full)->n, (*A_spar_patt_full)->m );
    }
    else if ( ((*A_app_inv)->m) < (A->m) )
    {
        Deallocate_Matrix( *A_app_inv );

        /* A_app_inv has the same sparsity pattern
         * * as A_spar_patt_full (omit non-zero values) */
        Allocate_Matrix( A_app_inv, (*A_spar_patt_full)->n, (*A_spar_patt_full)->m );
    }
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
            if ( droptol_local == NULL )
            {
                droptol_local = (real*) smalloc( omp_get_num_threads() * A->n * sizeof(real),
                                                 "Calculate_Droptol::droptol_local" );
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
    a = (real*) smalloc( (2 * A->start[A->n] - A->n) * sizeof(real),
                         "SuperLU_Factorize::a" );
    asub = (int_t*) smalloc( (2 * A->start[A->n] - A->n) * sizeof(int_t),
                             "SuperLU_Factorize::asub" );
    xa = (int_t*) smalloc( (A->n + 1) * sizeof(int_t),
                           "SuperLU_Factorize::xa" );
    Ltop = (unsigned int*) smalloc( (A->n + 1) * sizeof(unsigned int),
                                    "SuperLU_Factorize::Ltop" );
    Utop = (unsigned int*) smalloc( (A->n + 1) * sizeof(unsigned int),
                                    "SuperLU_Factorize::Utop" );
    Allocate_Matrix( &A_t, A->n, A->m );

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

    Utop = (unsigned int*) smalloc( (A->n + 1) * sizeof(unsigned int),
                                    "ICHOLT::Utop" );
    tmp_j = (int*) smalloc( A->n * sizeof(int),
                            "ICHOLT::Utop" );
    tmp_val = (real*) smalloc( A->n * sizeof(real),
                               "ICHOLT::Utop" );

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

    D = (real*) smalloc( A->n * sizeof(real), "ICHOL_PAR::D" );
    D_inv = (real*) smalloc( A->n * sizeof(real), "ICHOL_PAR::D_inv" );
    Utop = (int*) smalloc( (A->n + 1) * sizeof(int), "ICHOL_PAR::Utop" );
    Allocate_Matrix( &DAD, A->n, A->m );

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

    D = (real*) smalloc( A->n * sizeof(real), "ILU_PAR::D" );
    D_inv = (real*) smalloc( A->n * sizeof(real), "ILU_PAR::D_inv" );
    Allocate_Matrix( &DAD, A->n, A->m );

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

    Allocate_Matrix( &DAD, A->n, A->m );
    Allocate_Matrix( &L_temp, A->n, A->m );
    Allocate_Matrix( &U_temp, A->n, A->m );

    D = (real*) smalloc( A->n * sizeof(real), "ILUT_PAR::D" );
    D_inv = (real*) smalloc( A->n * sizeof(real), "ILUT_PAR::D_inv" );

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
/* Compute M^{1} \approx A which minimizes
 *  \sum_{j=1}^{N} ||m_j^T*A - e_j^T||_2^2
 *  where: e_j^T is the j-th row of the NxN identify matrix,
 *         m_j^T is the j-th row of the NxN approximate sparse matrix M
 * 
 * Internally, use LAPACKE to solve the independent least-squares problems.
 * Furthermore, internally solve the related problem
 *  \sum_{j=1}^{N} ||A*m_j - e_j||_2^2
 * for j-th columns m_j and e_j, but store the transpose of M to solve
 * the original problem. That is, for the problems
 *  min ||M_1*A - I||_F^2 and min ||A*M_2 - I||_F^2,
 * it can be shown that if A = A^T, M_1 = M_2^T.
 * Hence, we solve for M_2, and stores its transpose as the result
 * (more efficient for for CSR matrices, row-major storage).
 *
 * A: symmetric, sparse matrix, stored in full CSR format
 * A_spar_patt: sparse matrix used as template sparsity pattern
 *   for approximating the inverse, stored in full CSR format
 * A_app_inv: approximate inverse to A, stored in full CSR format (result)
 *
 * Reference:
 * Michele Benzi et al.
 * A Comparative Study of Sparse Approximate Inverse
 *  Preconditioners
 * Applied Numerical Mathematics 30, 1999
 * */
real sparse_approx_inverse( const sparse_matrix * const A,
                            const sparse_matrix * const A_spar_patt,
                            sparse_matrix ** A_app_inv )
{
    int i, k, pj, j_temp, identity_pos;
    int N, M, d_i, d_j;
    lapack_int m, n, nrhs, lda, ldb, info;
    int *pos_x, *pos_y;
    real start;
    real *e_j, *dense_matrix;
    char *X, *Y;

    start = Get_Time( );

    (*A_app_inv)->start[(*A_app_inv)->n] = A_spar_patt->start[A_spar_patt->n];

#ifdef _OPENMP
    #pragma omp parallel default(none) \
    private(i, k, pj, j_temp, identity_pos, N, M, d_i, d_j, m, n, \
            nrhs, lda, ldb, info, X, Y, pos_x, pos_y, e_j, dense_matrix) \
    shared(A_app_inv, stderr)
#endif
    {
        X = (char *) smalloc( sizeof(char) * A->n, "sparse_approx_inverse::X" );
        Y = (char *) smalloc( sizeof(char) * A->n, "sparse_approx_inverse::Y" );
        pos_x = (int *) smalloc( sizeof(int) * A->n, "sparse_approx_inverse::pos_x" );
        pos_y = (int *) smalloc( sizeof(int) * A->n, "sparse_approx_inverse::pos_y" );

        for ( i = 0; i < A->n; ++i )
        {
            X[i] = 0;
            Y[i] = 0;
            pos_x[i] = 0;
            pos_y[i] = 0;
        }

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for ( i = 0; i < A_spar_patt->n; ++i )
        {
            N = 0;
            M = 0;

            // find column indices of nonzeros (which will be the columns indices of the dense matrix)
            for ( pj = A_spar_patt->start[i]; pj < A_spar_patt->start[i + 1]; ++pj )
            {

                j_temp = A_spar_patt->j[pj];

                Y[j_temp] = 1;
                pos_y[j_temp] = N;
                ++N;

                // for each of those indices
                // search through the row of full A of that index
                for ( k = A->start[j_temp]; k < A->start[j_temp + 1]; ++k )
                {
                    // and accumulate the nonzero column indices to serve as the row indices of the dense matrix
                    X[A->j[k]] = 1;
                }
            }

            // enumerate the row indices from 0 to (# of nonzero rows - 1) for the dense matrix
            identity_pos = M;
            for ( k = 0; k < A->n; k++)
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
                                             "sparse_approx_inverse::dense_matrix" );

            // fill in the entries of dense matrix
            for ( d_i = 0; d_i < M; ++d_i)
            {
                // all rows are initialized to zero
                for ( d_j = 0; d_j < N; ++d_j )
                {
                    dense_matrix[d_i * N + d_j] = 0.0;
                }
                // change the value if any of the column indices is seen
                for ( d_j = A->start[pos_x[d_i]];
                        d_j < A->start[pos_x[d_i] + 1]; ++d_j )
                {
                    if ( Y[A->j[d_j]] == 1 )
                    {
                        dense_matrix[d_i * N + pos_y[A->j[d_j]]] = A->val[d_j];
                    }
                }

            }

            /* create the right hand side of the linear equation
               that is the full column of the identity matrix*/
            e_j = (real *) smalloc( sizeof(real) * M,
                                    "sparse_approx_inverse::e_j" );

            for ( k = 0; k < M; ++k )
            {
                e_j[k] = 0.0;
            }
            e_j[identity_pos] = 1.0;

            /* Solve the overdetermined system AX = B through the least-squares problem:
             * min ||B - AX||_2 */
            m = M;
            n = N;
            nrhs = 1;
            lda = N;
            ldb = nrhs;
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
            (*A_app_inv)->start[i] = A_spar_patt->start[i];
            for ( k = A_spar_patt->start[i]; k < A_spar_patt->start[i + 1]; ++k)
            {
                (*A_app_inv)->j[k] = A_spar_patt->j[k];
                (*A_app_inv)->val[k] = e_j[k - A_spar_patt->start[i]];
            }

            //empty variables that will be used next iteration
            sfree( dense_matrix, "sparse_approx_inverse::dense_matrix" );
            sfree( e_j, "sparse_approx_inverse::e_j"  );
            for ( k = 0; k < A->n; ++k )
            {
                X[k] = 0;
                Y[k] = 0;
                pos_x[k] = 0;
                pos_y[k] = 0;
            }
        }

        sfree( pos_y, "sparse_approx_inverse::pos_y" );
        sfree( pos_x, "sparse_approx_inverse::pos_x" );
        sfree( Y, "sparse_approx_inverse::Y" );
        sfree( X, "sparse_approx_inverse::X" );
    }

    return Get_Timing_Info( start );
}
#endif


/* sparse matrix-vector product Ax = b
 *
 * workspace: storage container for workspace structures
 * A: lower triangular matrix, stored in CSR format
 * x: vector
 * b (output): vector */
static void Sparse_MatVec( const static_storage * const workspace,
        const sparse_matrix * const A, const real * const x, real * const b )
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

    Vector_MakeZero( workspace->b_local, omp_get_num_threads() * n );

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
            workspace->b_local[tid * n + j] += H * x[i];
            workspace->b_local[tid * n + i] += H * x[j];
#else
            b[j] += H * x[i];
            b[i] += H * x[j];
#endif
        }

        // the diagonal entry is the last one in
#ifdef _OPENMP
        workspace->b_local[tid * n + i] += A->val[k] * x[i];
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
            b[i] += workspace->b_local[j * n + i];
        }
    }
#endif
}


/* sparse matrix-vector product Ax = b
 * where:
 *   A: matrix, stored in CSR format
 *   x: vector
 *   b: vector (result) */
static void Sparse_MatVec_full( const sparse_matrix * const A,
                                const real * const x, real * const b )
{
    int i, pj;

    Vector_MakeZero( b, A->n );

#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
        {
            b[i] += A->val[pj] * x[A->j[pj]];
        }
    }
}


/* Transpose A and copy into A^T
 *
 * A: stored in CSR
 * A_t: stored in CSR
 */
void Transpose( const sparse_matrix * const A, sparse_matrix * const A_t )
{
    unsigned int i, j, pj, *A_t_top;

    A_t_top = (unsigned int*) scalloc( A->n + 1, sizeof(unsigned int),
                                       "Transpose::A_t_top" );

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

    Allocate_Matrix( &A_t, A->n, A->m );

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
 * workspace: storage container for workspace structures
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
void tri_solve_level_sched( static_storage * workspace,
        const sparse_matrix * const LU,
        const real * const y, real * const x, const int N,
        const TRIANGULARITY tri, int find_levels )
{
    int i, j, pj, local_row, local_level;
    unsigned int *row_levels, *level_rows, *level_rows_cnt;
    int levels;

    if ( tri == LOWER )
    {
        row_levels = workspace->row_levels_L;
        level_rows = workspace->level_rows_L;
        level_rows_cnt = workspace->level_rows_cnt_L;
    }
    else
    {
        row_levels = workspace->row_levels_U;
        level_rows = workspace->level_rows_U;
        level_rows_cnt = workspace->level_rows_cnt_U;
    }

#ifdef _OPENMP
    #pragma omp single
#endif
    {
        /* find levels (row dependencies in substitutions) */
        if ( find_levels == TRUE )
        {
            memset( row_levels, 0, N * sizeof(unsigned int) );
            memset( level_rows_cnt, 0, N * sizeof(unsigned int) );
            memset( workspace->top, 0, N * sizeof(unsigned int) );
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

                workspace->levels_L = levels;

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

                workspace->levels_U = levels;

                //#if defined(DEBUG)
                fprintf(stderr, "levels(U): %d\n", levels);
                fprintf(stderr, "NNZ(U): %d\n", LU->start[N]);
                //#endif
            }

            for ( i = 1; i < levels + 1; ++i )
            {
                level_rows_cnt[i] += level_rows_cnt[i - 1];
                workspace->top[i] = level_rows_cnt[i];
            }

            for ( i = 0; i < N; ++i )
            {
                level_rows[workspace->top[row_levels[i] - 1]] = i;
                ++workspace->top[row_levels[i] - 1];
            }
        }
    }

    if ( tri == LOWER )
    {
        levels = workspace->levels_L;
    }
    else
    {
        levels = workspace->levels_U;
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
}


/* Iterative greedy shared-memory parallel graph coloring
 *
 * control: container for control info
 * workspace: storage container for workspace structures
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
void graph_coloring( const control_params * const control,
        static_storage * workspace,
        const sparse_matrix * const A, const TRIANGULARITY tri )
{
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int i, pj, v;
        unsigned int temp, recolor_cnt_local, *conflict_local;
        int tid, *fb_color;
        unsigned int *p_to_color, *p_conflict, *p_temp;

#ifdef _OPENMP
        tid = omp_get_thread_num( );
#else
        tid = 0;
#endif
        p_to_color = workspace->to_color;
        p_conflict = workspace->conflict;

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for ( i = 0; i < A->n; ++i )
        {
            workspace->color[i] = 0;
        }

#ifdef _OPENMP
        #pragma omp single
#endif
        {
            workspace->recolor_cnt = A->n;
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
                p_to_color[i] = i;
            }
        }
        else
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for ( i = 0; i < A->n; ++i )
            {
                p_to_color[i] = A->n - 1 - i;
            }
        }

        fb_color = (int*) smalloc( sizeof(int) * A->n,
                "graph_coloring::fb_color" );
        conflict_local = (unsigned int*) smalloc( sizeof(unsigned int) * A->n,
                "graph_coloring::fb_color" );

        while ( workspace->recolor_cnt > 0 )
        {
            memset( fb_color, -1, sizeof(int) * A->n );

            /* color vertices */
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for ( i = 0; i < workspace->recolor_cnt; ++i )
            {
                v = p_to_color[i];

                /* colors of adjacent vertices are forbidden */
                for ( pj = A->start[v]; pj < A->start[v + 1]; ++pj )
                {
                    if ( v != A->j[pj] )
                    {
                        fb_color[workspace->color[A->j[pj]]] = v;
                    }
                }

                /* search for min. color which is not in conflict with adjacent vertices;
                 * start at 1 since 0 is default (invalid) color for all vertices */
                for ( pj = 1; fb_color[pj] == v; ++pj )
                    ;

                /* assign discovered color (no conflict in neighborhood of adjacent vertices) */
                workspace->color[v] = pj;
            }

            /* determine if recoloring required */
            temp = workspace->recolor_cnt;
            recolor_cnt_local = 0;

#ifdef _OPENMP
            #pragma omp barrier

            #pragma omp single
#endif
            {
                workspace->recolor_cnt = 0;
            }

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for ( i = 0; i < temp; ++i )
            {
                v = p_to_color[i];

                /* search for color conflicts with adjacent vertices */
                for ( pj = A->start[v]; pj < A->start[v + 1]; ++pj )
                {
                    if ( workspace->color[v] == workspace->color[A->j[pj]] && v > A->j[pj] )
                    {
                        conflict_local[recolor_cnt_local] = v;
                        ++recolor_cnt_local;
                        break;
                    }
                }
            }

            /* count thread-local conflicts and compute offsets for copying into shared buffer */
            workspace->conflict_cnt[tid + 1] = recolor_cnt_local;

#ifdef _OPENMP
            #pragma omp barrier

            #pragma omp single
#endif
            {
                workspace->conflict_cnt[0] = 0;
                for ( i = 1; i < control->num_threads + 1; ++i )
                {
                    workspace->conflict_cnt[i] += workspace->conflict_cnt[i - 1];
                }
                workspace->recolor_cnt = workspace->conflict_cnt[control->num_threads];
            }

            /* copy thread-local conflicts into shared buffer */
            for ( i = 0; i < recolor_cnt_local; ++i )
            {
                p_conflict[workspace->conflict_cnt[tid] + i] = conflict_local[i];
                workspace->color[conflict_local[i]] = 0;
            }

#ifdef _OPENMP
            #pragma omp barrier
#endif
            p_temp = p_to_color;
            p_to_color = p_conflict;
            p_conflict = p_temp;
        }

        sfree( conflict_local, "graph_coloring::conflict_local" );
        sfree( fb_color, "graph_coloring::fb_color" );
    }
}


/* Sort rows by coloring
 *
 * workspace: storage container for workspace structures
 * n: number of entries in coloring
 * tri: coloring to triangular factor to use (lower/upper)
 */
void sort_rows_by_colors( const static_storage * const workspace,
        const unsigned int n, const TRIANGULARITY tri )
{
    unsigned int i;

    memset( workspace->color_top, 0, sizeof(unsigned int) * (n + 1) );

    /* sort vertices by color (ascending within a color)
     *  1) count colors
     *  2) determine offsets of color ranges
     *  3) sort rows by color
     *
     *  note: color is 1-based */
    for ( i = 0; i < n; ++i )
    {
        ++workspace->color_top[workspace->color[i]];
    }
    for ( i = 1; i < n + 1; ++i )
    {
        workspace->color_top[i] += workspace->color_top[i - 1];
    }
    for ( i = 0; i < n; ++i )
    {
        workspace->permuted_row_col[workspace->color_top[workspace->color[i] - 1]] = i;
        ++workspace->color_top[workspace->color[i] - 1];
    }

    /* invert mapping to get map from current row/column to permuted (new) row/column */
    for ( i = 0; i < n; ++i )
    {
        workspace->permuted_row_col_inv[workspace->permuted_row_col[i]] = i;
    }
}


/* Apply permutation Q^T*x or Q*x based on graph coloring
 *
 * workspace: storage container for workspace structures
 * color: vertex color (1-based); vertices represent matrix rows/columns
 * x: vector to permute (in-place)
 * n: number of entries in x
 * invert_map: if TRUE, use Q^T, otherwise use Q
 * tri: coloring to triangular factor to use (lower/upper)
 */
static void permute_vector( const static_storage * const workspace,
        real * const x, const unsigned int n, const int invert_map,
        const TRIANGULARITY tri )
{
    unsigned int i;
    unsigned int *mapping;

    if ( invert_map == TRUE )
    {
        mapping = workspace->permuted_row_col_inv;
    }
    else
    {
        mapping = workspace->permuted_row_col;
    }

#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( i = 0; i < n; ++i )
    {
        workspace->x_p[i] = x[mapping[i]];
    }

#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( i = 0; i < n; ++i )
    {
        x[i] = workspace->x_p[i];
    }
}


/* Apply permutation Q^T*(LU)*Q based on graph coloring
 *
 * workspace: storage container for workspace structures
 * color: vertex color (1-based); vertices represent matrix rows/columns
 * LU: matrix to permute, stored in CSR format
 * tri: triangularity of LU (lower/upper)
 */
void permute_matrix( const static_storage * const workspace,
        sparse_matrix * const LU, const TRIANGULARITY tri )
{
    int i, pj, nr, nc;
    sparse_matrix *LUtemp;

    Allocate_Matrix( &LUtemp, LU->n, LU->m );

    /* count nonzeros in each row of permuted factor (re-use color_top for counting) */
    memset( workspace->color_top, 0, sizeof(unsigned int) * (LU->n + 1) );

    if ( tri == LOWER )
    {
        for ( i = 0; i < LU->n; ++i )
        {
            nr = workspace->permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = workspace->permuted_row_col_inv[LU->j[pj]];

                if ( nc <= nr )
                {
                    ++workspace->color_top[nr + 1];
                }
                /* correct entries to maintain triangularity (lower) */
                else
                {
                    ++workspace->color_top[nc + 1];
                }
            }
        }
    }
    else
    {
        for ( i = LU->n - 1; i >= 0; --i )
        {
            nr = workspace->permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = workspace->permuted_row_col_inv[LU->j[pj]];

                if ( nc >= nr )
                {
                    ++workspace->color_top[nr + 1];
                }
                /* correct entries to maintain triangularity (upper) */
                else
                {
                    ++workspace->color_top[nc + 1];
                }
            }
        }
    }

    for ( i = 1; i < LU->n + 1; ++i )
    {
        workspace->color_top[i] += workspace->color_top[i - 1];
    }

    memcpy( LUtemp->start, workspace->color_top, sizeof(unsigned int) * (LU->n + 1) );

    /* permute factor */
    if ( tri == LOWER )
    {
        for ( i = 0; i < LU->n; ++i )
        {
            nr = workspace->permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = workspace->permuted_row_col_inv[LU->j[pj]];

                if ( nc <= nr )
                {
                    LUtemp->j[workspace->color_top[nr]] = nc;
                    LUtemp->val[workspace->color_top[nr]] = LU->val[pj];
                    ++workspace->color_top[nr];
                }
                /* correct entries to maintain triangularity (lower) */
                else
                {
                    LUtemp->j[workspace->color_top[nc]] = nr;
                    LUtemp->val[workspace->color_top[nc]] = LU->val[pj];
                    ++workspace->color_top[nc];
                }
            }
        }
    }
    else
    {
        for ( i = LU->n - 1; i >= 0; --i )
        {
            nr = workspace->permuted_row_col_inv[i];

            for ( pj = LU->start[i]; pj < LU->start[i + 1]; ++pj )
            {
                nc = workspace->permuted_row_col_inv[LU->j[pj]];

                if ( nc >= nr )
                {
                    LUtemp->j[workspace->color_top[nr]] = nc;
                    LUtemp->val[workspace->color_top[nr]] = LU->val[pj];
                    ++workspace->color_top[nr];
                }
                /* correct entries to maintain triangularity (upper) */
                else
                {
                    LUtemp->j[workspace->color_top[nc]] = nr;
                    LUtemp->val[workspace->color_top[nc]] = LU->val[pj];
                    ++workspace->color_top[nc];
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
 * control: container for control info
 * workspace: storage container for workspace structures
 * H: symmetric, lower triangular portion only, stored in CSR format;
 * H_full: symmetric, stored in CSR format;
 * H_p (output): permuted copy of H based on coloring, lower half stored, CSR format
 */
void setup_graph_coloring( const control_params * const control,
        const static_storage * const workspace, const sparse_matrix * const H,
        sparse_matrix ** H_full, sparse_matrix ** H_p )
{
    if ( *H_p == NULL )
    {
        Allocate_Matrix( H_p, H->n, H->m );
    }
    else if ( (*H_p)->m < H->m )
    {
        Deallocate_Matrix( *H_p );
        Allocate_Matrix( H_p, H->n, H->m );
    }

    compute_full_sparse_matrix( H, H_full );

    graph_coloring( control, (static_storage *) workspace, *H_full, LOWER );
    sort_rows_by_colors( workspace, (*H_full)->n, LOWER );

    memcpy( (*H_p)->start, H->start, sizeof(int) * (H->n + 1) );
    memcpy( (*H_p)->j, H->j, sizeof(int) * (H->start[H->n]) );
    memcpy( (*H_p)->val, H->val, sizeof(real) * (H->start[H->n]) );
    permute_matrix( workspace, (*H_p), LOWER );
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
 * the coefficient matrix in the triangular system
 *
 * workspace: storage container for workspace structures
 * R:
 * Dinv:
 * b:
 * x (output):
 * tri:
 * maxiter:
 * */
void jacobi_iter( const static_storage * const workspace,
        const sparse_matrix * const R, const real * const Dinv,
        const real * const b, real * const x, const TRIANGULARITY tri,
        const unsigned int maxiter )
{
    unsigned int i, k, si, ei, iter;
    real *p1, *p2, *p3;

    si = 0;
    ei = 0;
    iter = 0;
    p1 = workspace->rp;
    p2 = workspace->rp2;

    Vector_MakeZero( p1, R->n );

    /* precompute and cache, as invariant in loop below */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( i = 0; i < R->n; ++i )
    {
        workspace->Dinv_b[i] = Dinv[i] * b[i];
    }

    do
    {
        /* x_{k+1} = G*x_{k} + Dinv*b */
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

            p2[i] = 0.;

            for ( k = si; k < ei; ++k )
            {
                p2[i] += R->val[k] * p1[R->j[k]];
            }

            p2[i] *= -Dinv[i];
            p2[i] += workspace->Dinv_b[i];
        }

        p3 = p1;
        p1 = p2;
        p2 = p3;

        ++iter;
    }
    while ( iter < maxiter );

    Vector_Copy( x, p1, R->n );
}


/* Apply left-sided preconditioning while solver M^{-1}Ax = M^{-1}b
 *
 * workspace: data struct containing matrices, stored in CSR
 * control: data struct containing parameters
 * y: vector to which to apply preconditioning,
 *  specific to internals of iterative solver being used
 * x: preconditioned vector (output)
 * fresh_pre: parameter indicating if this is a newly computed (fresh) preconditioner
 * side: used in determining how to apply preconditioner if the preconditioner is
 *  factorized as M = M_{1}M_{2} (e.g., incomplete LU, A \approx LU)
 *
 * Assumptions:
 *   Matrices have non-zero diagonals
 *   Each row of a matrix has at least one non-zero (i.e., no rows with all zeros) */
static void apply_preconditioner( const static_storage * const workspace, const control_params * const control,
                                  const real * const y, real * const x, const int fresh_pre,
                                  const int side )
{
    int i, si;

    /* no preconditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        if ( x != y )
        {
            Vector_Copy( x, y, workspace->H->n );
        }
    }
    else
    {
        switch ( side )
        {
        case LEFT:
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
                    break;
                case SAI_PC:
                    Sparse_MatVec_full( workspace->H_app_inv, y, x );
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
                    tri_solve_level_sched( (static_storage *) workspace,
                            workspace->L, y, x, workspace->L->n, LOWER, fresh_pre );
                    break;
                case SAI_PC:
                    Sparse_MatVec_full( workspace->H_app_inv, y, x );
                    break;
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
                    #pragma omp for schedule(static)
#endif
                    for ( i = 0; i < workspace->H->n; ++i )
                    {
                        workspace->y_p[i] = y[i];
                    }

                    permute_vector( workspace, workspace->y_p, workspace->H->n, FALSE, LOWER );
                    tri_solve_level_sched( (static_storage *) workspace,
                            workspace->L, workspace->y_p, x, workspace->L->n, LOWER, fresh_pre );
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
                    /* construct D^{-1}_L */
                    if ( fresh_pre == TRUE )
                    {
#ifdef _OPENMP
                        #pragma omp for schedule(static)
#endif
                        for ( i = 0; i < workspace->L->n; ++i )
                        {
                            si = workspace->L->start[i + 1] - 1;
                            workspace->Dinv_L[i] = 1.0 / workspace->L->val[si];
                        }
                    }

                    jacobi_iter( workspace, workspace->L, workspace->Dinv_L,
                            y, x, LOWER, control->cm_solver_pre_app_jacobi_iters );
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
            break;

        case RIGHT:
            switch ( control->cm_solver_pre_app_type )
            {
            case TRI_SOLVE_PA:
                switch ( control->cm_solver_pre_comp_type )
                {
                case DIAG_PC:
                case SAI_PC:
                    if ( x != y )
                    {
                        Vector_Copy( x, y, workspace->H->n );
                    }
                    break;
                case ICHOLT_PC:
                case ILU_PAR_PC:
                case ILUT_PAR_PC:
                    tri_solve( workspace->U, y, x, workspace->U->n, UPPER );
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
                case SAI_PC:
                    if ( x != y )
                    {
                        Vector_Copy( x, y, workspace->H->n );
                    }
                    break;
                case ICHOLT_PC:
                case ILU_PAR_PC:
                case ILUT_PAR_PC:
                    tri_solve_level_sched( (static_storage *) workspace,
                            workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
                    break;
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
                    tri_solve_level_sched( (static_storage *) workspace,
                            workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
                    permute_vector( workspace, x, workspace->H->n, TRUE, UPPER );
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
                    /* construct D^{-1}_U */
                    if ( fresh_pre == TRUE )
                    {
#ifdef _OPENMP
                        #pragma omp for schedule(static)
#endif
                        for ( i = 0; i < workspace->U->n; ++i )
                        {
                            si = workspace->U->start[i];
                            workspace->Dinv_U[i] = 1.0 / workspace->U->val[si];
                        }
                    }

                    jacobi_iter( workspace, workspace->U, workspace->Dinv_U,
                            y, x, UPPER, control->cm_solver_pre_app_jacobi_iters );
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
            break;
        }
    }
}


/* Generalized minimual residual method with restarting and
 * left preconditioning for sparse linear systems */
int GMRES( const static_storage * const workspace, const control_params * const control,
           simulation_data * const data, const sparse_matrix * const H, const real * const b,
           const real tol, real * const x, const int fresh_pre )
{
    int i, j, k, itr, N, g_j, g_itr;
    real cc, tmp1, tmp2, temp, bnorm;
    real t_start, t_ortho, t_pa, t_spmv, t_ts, t_vops;

    N = H->n;
    g_j = 0;
    g_itr = 0;
    t_ortho = 0.0;
    t_pa = 0.0;
    t_spmv = 0.0;
    t_ts = 0.0;
    t_vops = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(none) \
    private(i, j, k, itr, bnorm, temp, t_start) \
    shared(N, cc, tmp1, tmp2, g_itr, g_j, stderr) \
    reduction(+: t_ortho, t_pa, t_spmv, t_ts, t_vops)
#endif
    {
        j = 0;
        itr = 0;
        t_ortho = 0.0;
        t_pa = 0.0;
        t_spmv = 0.0;
        t_ts = 0.0;
        t_vops = 0.0;

        t_start = Get_Time( );
        bnorm = Norm( b, N );
        t_vops += Get_Timing_Info( t_start );

        /* GMRES outer-loop */
        for ( itr = 0; itr < control->cm_solver_max_iters; ++itr )
        {
            /* calculate r0 */
            t_start = Get_Time( );
            Sparse_MatVec( workspace, H, x, workspace->b_prm );
            t_spmv += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            Vector_Sum( workspace->b_prc, 1.0, b, -1.0, workspace->b_prm, N );
            t_vops += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            apply_preconditioner( workspace, control, workspace->b_prc, workspace->b_prm,
                                  itr == 0 ? fresh_pre : FALSE, LEFT );
            apply_preconditioner( workspace, control, workspace->b_prm, workspace->v[0],
                                  itr == 0 ? fresh_pre : FALSE, RIGHT );
            t_pa += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            temp = Norm( workspace->v[0], N );

#ifdef _OPENMP
            #pragma omp single
#endif
            workspace->g[0] = temp;

            Vector_Scale( workspace->v[0], 1.0 / temp, workspace->v[0], N );
            t_vops += Get_Timing_Info( t_start );

            /* GMRES inner-loop */
            for ( j = 0; j < control->cm_solver_restart && FABS(workspace->g[j]) / bnorm > tol; j++ )
            {
                /* matvec */
                t_start = Get_Time( );
                Sparse_MatVec( workspace, H, workspace->v[j], workspace->b_prc );
                t_spmv += Get_Timing_Info( t_start );

                t_start = Get_Time( );
                apply_preconditioner( workspace, control, workspace->b_prc,
                        workspace->b_prm, FALSE, LEFT );
                apply_preconditioner( workspace, control, workspace->b_prm,
                        workspace->v[j + 1], FALSE, RIGHT );
                t_pa += Get_Timing_Info( t_start );

                /* apply modified Gram-Schmidt to orthogonalize the new residual */
                t_start = Get_Time( );
                for ( i = 0; i <= j; i++ )
                {
                    temp = Dot( workspace->v[i], workspace->v[j + 1], N );

#ifdef _OPENMP
                    #pragma omp single
#endif
                    workspace->h[i][j] = temp;

                    Vector_Add( workspace->v[j + 1], -1.0 * temp, workspace->v[i], N );

                }
                t_vops += Get_Timing_Info( t_start );

                t_start = Get_Time( );
                temp = Norm( workspace->v[j + 1], N );

#ifdef _OPENMP
                #pragma omp single
#endif
                workspace->h[j + 1][j] = temp;

                Vector_Scale( workspace->v[j + 1], 1.0 / temp, workspace->v[j + 1], N );
                t_vops += Get_Timing_Info( t_start );

                t_start = Get_Time( );
                /* Givens rotations on the upper-Hessenberg matrix to make it U */
#ifdef _OPENMP
                #pragma omp single
#endif
                {
                    for ( i = 0; i <= j; i++ )
                    {
                        if ( i == j )
                        {
                            cc = SQRT( SQR(workspace->h[j][j]) + SQR(workspace->h[j + 1][j]) );
                            workspace->hc[j] = workspace->h[j][j] / cc;
                            workspace->hs[j] = workspace->h[j + 1][j] / cc;
                        }

                        tmp1 = workspace->hc[i] * workspace->h[i][j] +
                            workspace->hs[i] * workspace->h[i + 1][j];
                        tmp2 = -1.0 * workspace->hs[i] * workspace->h[i][j] +
                            workspace->hc[i] * workspace->h[i + 1][j];

                        workspace->h[i][j] = tmp1;
                        workspace->h[i + 1][j] = tmp2;
                    }

                    /* apply Givens rotations to the rhs as well */
                    tmp1 = workspace->hc[j] * workspace->g[j];
                    tmp2 = -1.0 * workspace->hs[j] * workspace->g[j];
                    workspace->g[j] = tmp1;
                    workspace->g[j + 1] = tmp2;

                }
                t_ortho += Get_Timing_Info( t_start );
            }

            /* solve Hy = g: H is now upper-triangular, do back-substitution */
            t_start = Get_Time( );
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                for ( i = j - 1; i >= 0; i-- )
                {
                    temp = workspace->g[i];
                    for ( k = j - 1; k > i; k-- )
                    {
                        temp -= workspace->h[i][k] * workspace->y[k];
                    }

                    workspace->y[i] = temp / workspace->h[i][i];
                }
            }
            t_ts += Get_Timing_Info( t_start );

            /* update x = x_0 + Vy */
            t_start = Get_Time( );
            Vector_MakeZero( workspace->p, N );

            for ( i = 0; i < j; i++ )
            {
                Vector_Add( workspace->p, workspace->y[i], workspace->v[i], N );
            }

            Vector_Add( x, 1.0, workspace->p, N );
            t_vops += Get_Timing_Info( t_start );

            /* stopping condition */
            if ( FABS(workspace->g[j]) / bnorm <= tol )
            {
                break;
            }
        }

#ifdef _OPENMP
        #pragma omp single
#endif
        {
            g_j = j;
            g_itr = itr;
        }
    }

    data->timing.cm_solver_orthog += t_ortho / control->num_threads;
    data->timing.cm_solver_pre_app += t_pa / control->num_threads;
    data->timing.cm_solver_spmv += t_spmv / control->num_threads;
    data->timing.cm_solver_tri_solve += t_ts / control->num_threads;
    data->timing.cm_solver_vector_ops += t_vops / control->num_threads;

    if ( g_itr >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] GMRES convergence failed (%d outer iters)\n", g_itr );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", FABS(workspace->g[g_j]) / bnorm );
        return g_itr * (control->cm_solver_restart + 1) + g_j + 1;
    }

    return g_itr * (control->cm_solver_restart + 1) + g_j + 1;
}


int GMRES_HouseHolder( const static_storage * const workspace,
                       const control_params * const control, simulation_data * const data,
                       const sparse_matrix * const H, const real * const b, real tol,
                       real * const x, const int fresh_pre )
{
    int i, j, k, itr, N, g_j, g_itr;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[control->cm_solver_restart + 2][10000], w[control->cm_solver_restart + 2];
    real u[control->cm_solver_restart + 2][10000];
    real t_start, t_ortho, t_pa, t_spmv, t_ts, t_vops;

    j = 0;
    N = H->n;
    t_ortho = 0.0;
    t_pa = 0.0;
    t_spmv = 0.0;
    t_ts = 0.0;
    t_vops = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(none) \
    private(i, j, k, itr, bnorm, temp, t_start) \
    shared(v, z, w, u, tol, N, cc, tmp1, tmp2, g_itr, g_j, stderr) \
    reduction(+: t_ortho, t_pa, t_spmv, t_ts, t_vops)
#endif
    {
        j = 0;
        t_ortho = 0.0;
        t_pa = 0.0;
        t_spmv = 0.0;
        t_ts = 0.0;
        t_vops = 0.0;

        t_start = Get_Time( );
        bnorm = Norm( b, N );
        t_vops += Get_Timing_Info( t_start );

        // memset( x, 0, sizeof(real) * N );

        /* GMRES outer-loop */
        for ( itr = 0; itr < control->cm_solver_max_iters; ++itr )
        {
            /* compute z = r0 */
            t_start = Get_Time( );
            Sparse_MatVec( workspace, H, x, workspace->b_prm );
            t_spmv += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            Vector_Sum( workspace->b_prc, 1.,  workspace->b, -1., workspace->b_prm, N );
            t_vops += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            apply_preconditioner( workspace, control, workspace->b_prc,
                    workspace->b_prm, fresh_pre, LEFT );
            apply_preconditioner( workspace, control, workspace->b_prm,
                    z[0], fresh_pre, RIGHT );
            t_pa += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            Vector_MakeZero( w, control->cm_solver_restart + 1 );
            w[0] = Norm( z[0], N );

            Vector_Copy( u[0], z[0], N );
            u[0][0] += ( u[0][0] < 0.0 ? -1 : 1 ) * w[0];
            Vector_Scale( u[0], 1.0 / Norm( u[0], N ), u[0], N );

            w[0] *= ( u[0][0] < 0.0 ?  1 : -1 );
            t_vops += Get_Timing_Info( t_start );

            /* GMRES inner-loop */
            for ( j = 0; j < control->cm_solver_restart && FABS( w[j] ) / bnorm > tol; j++ )
            {
                /* compute v_j */
                t_start = Get_Time( );
                Vector_Scale( z[j], -2.0 * u[j][j], u[j], N );
                z[j][j] += 1.; /* due to e_j */

                for ( i = j - 1; i >= 0; --i )
                {
                    Vector_Add( z[j] + i, -2.0 * Dot( u[i] + i, z[j] + i, N - i ),
                            u[i] + i, N - i );
                }
                t_vops += Get_Timing_Info( t_start );

                /* matvec */
                t_start = Get_Time( );
                Sparse_MatVec( workspace, H, z[j], workspace->b_prc );
                t_spmv += Get_Timing_Info( t_start );

                t_start = Get_Time( );
                apply_preconditioner( workspace, control, workspace->b_prc,
                        workspace->b_prm, fresh_pre, LEFT );
                apply_preconditioner( workspace, control, workspace->b_prm,
                        v, fresh_pre, RIGHT );
                t_pa += Get_Timing_Info( t_start );

                t_start = Get_Time( );
                for ( i = 0; i <= j; ++i )
                {
                    Vector_Add( v + i, -2.0 * Dot( u[i] + i, v + i, N - i ),
                                u[i] + i, N - i );
                }

                if ( !Vector_isZero( v + (j + 1), N - (j + 1) ) )
                {
                    /* compute the HouseHolder unit vector u_j+1 */
                    Vector_MakeZero( u[j + 1], j + 1 );
                    Vector_Copy( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) );
                    temp = Norm( v + (j + 1), N - (j + 1) );
#ifdef _OPENMP
                    #pragma omp single
#endif
                    u[j + 1][j + 1] += ( v[j + 1] < 0.0 ? -1.0 : 1.0 ) * temp;

#ifdef _OPENMP
                    #pragma omp barrier
#endif

                    Vector_Scale( u[j + 1], 1.0 / Norm( u[j + 1], N ), u[j + 1], N );

                    /* overwrite v with P_m+1 * v */
                    temp = 2.0 * Dot( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) )
                           * u[j + 1][j + 1];
#ifdef _OPENMP
                    #pragma omp single
#endif
                    v[j + 1] -= temp;

#ifdef _OPENMP
                    #pragma omp barrier
#endif

                    Vector_MakeZero( v + (j + 2), N - (j + 2) );
//                    Vector_Add( v, -2.0 * Dot( u[j+1], v, N ), u[j+1], N );
                }
                t_vops += Get_Timing_Info( t_start );

                /* prev Givens rots on the upper-Hessenberg matrix to make it U */
                t_start = Get_Time( );
#ifdef _OPENMP
                #pragma omp single
#endif
                {
                    for ( i = 0; i < j; i++ )
                    {
                        tmp1 =  workspace->hc[i] * v[i] + workspace->hs[i] * v[i + 1];
                        tmp2 = -workspace->hs[i] * v[i] + workspace->hc[i] * v[i + 1];

                        v[i] = tmp1;
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
                }
                t_ortho += Get_Timing_Info( t_start );
            }


            /* solve Hy = w.
               H is now upper-triangular, do back-substitution */
            t_start = Get_Time( );
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                for ( i = j - 1; i >= 0; i-- )
                {
                    temp = w[i];
                    for ( k = j - 1; k > i; k-- )
                    {
                        temp -= workspace->h[i][k] * workspace->y[k];
                    }

                    workspace->y[i] = temp / workspace->h[i][i];
                }
            }
            t_ts += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            for ( i = j - 1; i >= 0; i-- )
            {
                Vector_Add( x, workspace->y[i], z[i], N );
            }
            t_vops += Get_Timing_Info( t_start );

            /* stopping condition */
            if ( FABS( w[j] ) / bnorm <= tol )
            {
                break;
            }
        }

#ifdef _OPENMP
        #pragma omp single
#endif
        {
            g_j = j;
            g_itr = itr;
        }
    }

    data->timing.cm_solver_orthog += t_ortho / control->num_threads;
    data->timing.cm_solver_pre_app += t_pa / control->num_threads;
    data->timing.cm_solver_spmv += t_spmv / control->num_threads;
    data->timing.cm_solver_tri_solve += t_ts / control->num_threads;
    data->timing.cm_solver_vector_ops += t_vops / control->num_threads;

    if ( g_itr >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] GMRES convergence failed (%d outer iters)\n", g_itr );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", FABS(w[g_j]) / bnorm );
        return g_itr * (control->cm_solver_restart + 1) + j + 1;
    }

    return g_itr * (control->cm_solver_restart + 1) + g_j + 1;
}


/* Conjugate Gradient */
int CG( const static_storage * const workspace, const control_params * const control,
        simulation_data * const data, const sparse_matrix * const H, const real * const b,
        const real tol, real * const x, const int fresh_pre )
{
    int i, g_itr, N;
    real tmp, alpha, beta, b_norm, r_norm;
    real *d, *r, *p, *z;
    real sig_old, sig_new;
    real t_start, t_pa, t_spmv, t_vops;

    N = H->n;
    d = workspace->d;
    r = workspace->r;
    p = workspace->q;
    z = workspace->p;
    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(none) \
    private(i, tmp, alpha, beta, b_norm, r_norm, sig_old, sig_new, t_start) \
    reduction(+: t_pa, t_spmv, t_vops) \
    shared(g_itr, N, d, r, p, z)
#endif
    {
        t_pa = 0.0;
        t_spmv = 0.0;
        t_vops = 0.0;

        t_start = Get_Time( );
        b_norm = Norm( b, N );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        Sparse_MatVec( workspace, H, x, d );
        t_spmv += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        Vector_Sum( r, 1.0,  b, -1.0, d, N );
        r_norm = Norm( r, N );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        apply_preconditioner( workspace, control, r, d, fresh_pre, LEFT );
        apply_preconditioner( workspace, control, d, z, fresh_pre, RIGHT );
        t_pa += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        Vector_Copy( p, z, N );
        sig_new = Dot( r, p, N );
        t_vops += Get_Timing_Info( t_start );

        for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
        {
            t_start = Get_Time( );
            Sparse_MatVec( workspace, H, p, d );
            t_spmv += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            tmp = Dot( d, p, N );
            alpha = sig_new / tmp;
            Vector_Add( x, alpha, p, N );
            Vector_Add( r, -1.0 * alpha, d, N );
            r_norm = Norm( r, N );
            t_vops += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            apply_preconditioner( workspace, control, r, d, FALSE, LEFT );
            apply_preconditioner( workspace, control, d, z, FALSE, RIGHT );
            t_pa += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            sig_old = sig_new;
            sig_new = Dot( r, z, N );
            beta = sig_new / sig_old;
            Vector_Sum( p, 1.0, z, beta, p, N );
            t_vops += Get_Timing_Info( t_start );
        }

#ifdef _OPENMP
        #pragma omp single
#endif
        g_itr = i;
    }

    data->timing.cm_solver_pre_app += t_pa / control->num_threads;
    data->timing.cm_solver_spmv += t_spmv / control->num_threads;
    data->timing.cm_solver_vector_ops += t_vops / control->num_threads;

    if ( g_itr >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] CG convergence failed (%d iters)\n", g_itr );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", r_norm / b_norm );
        return g_itr;
    }

    return g_itr;
}


/* Bi-conjugate gradient stabalized method with left preconditioning for
 * solving nonsymmetric linear systems
 *
 * workspace: struct containing storage for workspace for the linear solver
 * control: struct containing parameters governing the simulation and numeric methods
 * data: struct containing simulation data (e.g., atom info)
 * H: sparse, symmetric matrix, lower half stored in CSR format
 * b: right-hand side of the linear system
 * tol: tolerence compared against the relative residual for determining convergence
 * x: inital guess
 * fresh_pre: flag for determining if preconditioners should be recomputed
 * */
int BiCGStab( const static_storage * const workspace, const control_params * const control,
        simulation_data * const data, const sparse_matrix * const H, const real * const b,
        const real tol, real * const x, const int fresh_pre )
{
    int i, g_itr, N;
    real tmp, alpha, beta, omega, rho, rho_old, sigma, r_norm, b_norm;
    real t_start, t_pa, t_spmv, t_vops;

    N = H->n;
    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(none) \
    private(i, tmp, alpha, beta, omega, rho, rho_old, sigma, r_norm, b_norm, t_start) \
    reduction(+: t_pa, t_spmv, t_vops) \
    shared(g_itr, N)
#endif
    {
        t_pa = 0.0;
        t_spmv = 0.0;
        t_vops = 0.0;

        t_start = Get_Time( );
        Sparse_MatVec( workspace, H, x, workspace->d );
        t_spmv += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        b_norm = Norm( b, N );
        Vector_Sum( workspace->r, 1.0,  b, -1.0, workspace->d, N );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        Vector_Copy( workspace->r_hat, workspace->r, N );
        r_norm = Norm( workspace->r, N );
        Vector_Copy( workspace->p, workspace->r, N );
        rho_old = Dot( workspace->r, workspace->r_hat, N );
        t_vops += Get_Timing_Info( t_start );

        for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
        {
            t_start = Get_Time( );
            Sparse_MatVec( workspace, H, workspace->p, workspace->d );
            t_spmv += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            tmp = Dot( workspace->d, workspace->r_hat, N );
            alpha = rho_old / tmp;
            Vector_Sum( workspace->q, 1.0, workspace->r, -1.0 * alpha, workspace->d, N );
            t_vops += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            Sparse_MatVec( workspace, H, workspace->q, workspace->y );
            t_spmv += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            sigma = Dot( workspace->y, workspace->q, N );
            tmp = Dot( workspace->y, workspace->y, N );
            omega = sigma / tmp;
            Vector_Sum( workspace->z, alpha, workspace->p, omega, workspace->q, N );
            Vector_Add( x, 1.0, workspace->z, N );
            Vector_Sum( workspace->r, 1.0, workspace->q, -1.0 * omega, workspace->y, N );
            r_norm = Norm( workspace->r, N );
            t_vops += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            rho = Dot( workspace->r, workspace->r_hat, N );
            beta = (rho / rho_old) * (alpha / omega);
            Vector_Sum( workspace->z, 1.0, workspace->p, -1.0 * omega, workspace->d, N );
            Vector_Sum( workspace->p, 1.0, workspace->r, beta, workspace->z, N );
            rho_old = rho;
            t_vops += Get_Timing_Info( t_start );
        }

#ifdef _OPENMP
        #pragma omp single
#endif
        g_itr = i;
    }

    data->timing.cm_solver_pre_app += t_pa / control->num_threads;
    data->timing.cm_solver_spmv += t_spmv / control->num_threads;
    data->timing.cm_solver_vector_ops += t_vops / control->num_threads;

    if ( g_itr >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] BiCGStab convergence failed (%d iters)\n", g_itr );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", r_norm / b_norm );
        return g_itr;
    }

    return g_itr;
}


/* Steepest Descent */
int SDM( const static_storage * const workspace, const control_params * const control,
         simulation_data * const data, const sparse_matrix * const H, const real * const b,
         const real tol, real * const x, const int fresh_pre )
{
    int i, g_itr, N;
    real tmp, alpha, b_norm;
    real sig;
    real t_start, t_pa, t_spmv, t_vops;

    N = H->n;
    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(none) \
    private(i, tmp, alpha, b_norm, sig, t_start) \
    reduction(+: t_pa, t_spmv, t_vops) \
    shared(g_itr, N)
#endif
    {
        t_pa = 0.0;
        t_spmv = 0.0;
        t_vops = 0.0;

        t_start = Get_Time( );
        b_norm = Norm( b, N );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        Sparse_MatVec( workspace, H, x, workspace->q );
        t_spmv += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        Vector_Sum( workspace->r, 1.0,  b, -1.0, workspace->q, N );
        t_vops += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        apply_preconditioner( workspace, control, workspace->r, workspace->q, fresh_pre, LEFT );
        apply_preconditioner( workspace, control, workspace->q, workspace->d, fresh_pre, RIGHT );
        t_pa += Get_Timing_Info( t_start );

        t_start = Get_Time( );
        sig = Dot( workspace->r, workspace->d, N );
        t_vops += Get_Timing_Info( t_start );

        for ( i = 0; i < control->cm_solver_max_iters && SQRT(sig) / b_norm > tol; ++i )
        {
            t_start = Get_Time( );
            Sparse_MatVec( workspace, H, workspace->d, workspace->q );
            t_spmv += Get_Timing_Info( t_start );

            t_start = Get_Time( );
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
            t_vops += Get_Timing_Info( t_start );

            t_start = Get_Time( );
            apply_preconditioner( workspace, control, workspace->r, workspace->q, FALSE, LEFT );
            apply_preconditioner( workspace, control, workspace->q, workspace->d, FALSE, RIGHT );
            t_pa += Get_Timing_Info( t_start );
        }

#ifdef _OPENMP
        #pragma omp single
#endif
        g_itr = i;
    }

    data->timing.cm_solver_pre_app += t_pa / control->num_threads;
    data->timing.cm_solver_spmv += t_spmv / control->num_threads;
    data->timing.cm_solver_vector_ops += t_vops / control->num_threads;

    if ( g_itr >= control->cm_solver_max_iters  )
    {
        fprintf( stderr, "[WARNING] SDM convergence failed (%d iters)\n", g_itr );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", SQRT(sig) / b_norm );
        return g_itr;
    }

    return g_itr;
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

    e = (real*) smalloc( sizeof(real) * N, "condest::e" );

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
