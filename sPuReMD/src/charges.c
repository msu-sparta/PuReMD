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

#include "charges.h"

#include "allocate.h"
#include "list.h"
#include "lin_alg.h"
#include "print_utils.h"
#include "tool_box.h"
#include "vector.h"
#if defined(HAVE_SUPERLU_MT)
  #include "slu_mt_ddefs.h"
#endif


typedef struct
{
    unsigned int j;
    real val;
} sparse_matrix_entry;


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
static void Sort_Matrix_Rows( sparse_matrix * const A )
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


static void Calculate_Droptol( const sparse_matrix * const A,
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


static int Estimate_LU_Fill( const sparse_matrix * const A, const real * const droptol )
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
static real SuperLU_Factorize( const sparse_matrix * const A,
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

//#if defined(DEBUG)
    fprintf( stderr, "nprocs = %d\n", nprocs );
    fprintf( stderr, "Panel size = %d\n", panel_size );
    fprintf( stderr, "Relax = %d\n", relax );
//#endif

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

//#if defined(DEBUG)
    printf("\n** Result of sparse LU **\n");
    L_S_store = (SCPformat *) L_S.Store;
    U_S_store = (NCPformat *) U_S.Store;
    printf( "No of nonzeros in factor L = " IFMT "\n", L_S_store->nnz );
    printf( "No of nonzeros in factor U = " IFMT "\n", U_S_store->nnz );
    fflush( stdout );
//#endif

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
static real diag_pre_comp( const sparse_matrix * const H, real * const Hdia_inv )
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
static real ICHOLT( const sparse_matrix * const A, const real * const droptol,
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
static real ICHOL_PAR( const sparse_matrix * const A, const unsigned int sweeps,
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
static real ILU_PAR( const sparse_matrix * const A, const unsigned int sweeps,
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
static real ILUT_PAR( const sparse_matrix * const A, const real * droptol,
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


static void Extrapolate_Charges_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int i;
    real s_tmp, t_tmp;

    /* extrapolation for s & t */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        default(none) private(i, s_tmp, t_tmp)
#endif
    for ( i = 0; i < system->N_cm; ++i )
    {
        // no extrapolation
        //s_tmp = workspace->s[0][i];
        //t_tmp = workspace->t[0][i];

        // linear
        //s_tmp = 2 * workspace->s[0][i] - workspace->s[1][i];
        //t_tmp = 2 * workspace->t[0][i] - workspace->t[1][i];

        // quadratic
        //s_tmp = workspace->s[2][i] + 3 * (workspace->s[0][i]-workspace->s[1][i]);
        t_tmp = workspace->t[2][i] + 3 * (workspace->t[0][i] - workspace->t[1][i]);

        // cubic
        s_tmp = 4 * (workspace->s[0][i] + workspace->s[2][i]) -
                (6 * workspace->s[1][i] + workspace->s[3][i] );
        //t_tmp = 4 * (workspace->t[0][i] + workspace->t[2][i]) -
        //  (6 * workspace->t[1][i] + workspace->t[3][i] );

        // 4th order
        //s_tmp = 5 * (workspace->s[0][i] - workspace->s[3][i]) +
        //  10 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];
        //t_tmp = 5 * (workspace->t[0][i] - workspace->t[3][i]) +
        //  10 * (-workspace->t[1][i] + workspace->t[2][i] ) + workspace->t[4][i];

        workspace->s[4][i] = workspace->s[3][i];
        workspace->s[3][i] = workspace->s[2][i];
        workspace->s[2][i] = workspace->s[1][i];
        workspace->s[1][i] = workspace->s[0][i];
        workspace->s[0][i] = s_tmp;

        workspace->t[4][i] = workspace->t[3][i];
        workspace->t[3][i] = workspace->t[2][i];
        workspace->t[2][i] = workspace->t[1][i];
        workspace->t[1][i] = workspace->t[0][i];
        workspace->t[0][i] = t_tmp;
    }
}


static void Extrapolate_Charges_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int i;
    real s_tmp;

    /* extrapolation for s */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        default(none) private(i, s_tmp)
#endif
    for ( i = 0; i < system->N_cm; ++i )
    {
        // no extrapolation
        //s_tmp = workspace->s[0][i];

        // linear
        //s_tmp = 2 * workspace->s[0][i] - workspace->s[1][i];

        // quadratic
        //s_tmp = workspace->s[2][i] + 3 * (workspace->s[0][i]-workspace->s[1][i]);

        // cubic
        s_tmp = 4 * (workspace->s[0][i] + workspace->s[2][i]) -
                (6 * workspace->s[1][i] + workspace->s[3][i] );

        // 4th order
        //s_tmp = 5 * (workspace->s[0][i] - workspace->s[3][i]) +
        //  10 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];

        workspace->s[4][i] = workspace->s[3][i];
        workspace->s[3][i] = workspace->s[2][i];
        workspace->s[2][i] = workspace->s[1][i];
        workspace->s[1][i] = workspace->s[0][i];
        workspace->s[0][i] = s_tmp;
    }
}


/* Compute preconditioner for QEq
 */
static void Compute_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        if ( control->cm_domain_sparsify_enabled == TRUE )
        {
            Hptr = setup_graph_coloring( workspace->H_sp );
        }
        else
        {
            Hptr = setup_graph_coloring( workspace->H );
        }

        Sort_Matrix_Rows( Hptr );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            data->timing.cm_solver_pre_comp +=
                ICHOLT( Hptr, workspace->droptol, workspace->L, workspace->U );
            break;

        case ILU_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L, workspace->U );
            break;

        case ILUT_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        workspace->L, workspace->U );
            break;

        case ILU_SUPERLU_MT_PC:
#if defined(HAVE_SUPERLU_MT)
            data->timing.cm_solver_pre_comp +=
                SuperLU_Factorize( Hptr, workspace->L, workspace->U );
#else
            fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

#if defined(DEBUG)
    if ( control->cm_solver_pre_comp_type != NONE_PC && 
            control->cm_solver_pre_comp_type != DIAG_PC )
    {
        fprintf( stderr, "condest = %f\n", condest(workspace->L, workspace->U) );

#if defined(DEBUG_FOCUS)
        sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        sprintf( fname, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#endif
    }
#endif
}


/* Compute preconditioner for EE
 */
//static void Compute_Preconditioner_EE( const reax_system * const system,
//        const control_params * const control,
//        simulation_data * const data, static_storage * const workspace,
//        const reax_list * const far_nbrs )
//{
//    int i, top;
//    static real * ones = NULL, * x = NULL, * y = NULL;
//    sparse_matrix *Hptr;
//
//    Hptr = workspace->H_EE;
//
//#if defined(TEST_MAT)
//    Hptr = create_test_mat( );
//#endif
//
//    if ( ones == NULL )
//    {
//        if ( ( ones = (real*) malloc( system->N * sizeof(real)) ) == NULL ||
//            ( x = (real*) malloc( system->N * sizeof(real)) ) == NULL ||
//            ( y = (real*) malloc( system->N * sizeof(real)) ) == NULL )
//        {
//            fprintf( stderr, "Not enough space for preconditioner computation. Terminating...\n" );
//            exit( INSUFFICIENT_MEMORY );
//        }
//
//        for ( i = 0; i < system->N; ++i )
//        {
//            ones[i] = 1.0;
//        }
//    }
//
//    switch ( control->cm_solver_pre_comp_type )
//    {
//    case DIAG_PC:
//        data->timing.cm_solver_pre_comp +=
//            diag_pre_comp( Hptr, workspace->Hdia_inv );
//        break;
//
//    case ICHOLT_PC:
//        data->timing.cm_solver_pre_comp +=
//            ICHOLT( Hptr, workspace->droptol, workspace->L_EE, workspace->U_EE );
//        break;
//
//    case ILU_PAR_PC:
//        data->timing.cm_solver_pre_comp +=
//            ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L_EE, workspace->U_EE );
//        break;
//
//    case ILUT_PAR_PC:
//        data->timing.cm_solver_pre_comp +=
//            ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
//                    workspace->L_EE, workspace->U_EE );
//        break;
//
//    case ILU_SUPERLU_MT_PC:
//#if defined(HAVE_SUPERLU_MT)
//        data->timing.cm_solver_pre_comp +=
//            SuperLU_Factorize( Hptr, workspace->L_EE, workspace->U_EE );
//#else
//        fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
//        exit( INVALID_INPUT );
//#endif
//        break;
//
//    default:
//        fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
//        exit( INVALID_INPUT );
//        break;
//    }
//
//    if ( control->cm_solver_pre_comp_type != DIAG_PC )
//    {
//        switch ( control->cm_solver_pre_app_type )
//        {
//            case TRI_SOLVE_PA:
//                tri_solve( workspace->L_EE, ones, x, workspace->L_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//                tri_solve( workspace->U_EE, ones, y, workspace->U_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//
//                memcpy( workspace->L->start, workspace->L_EE->start, sizeof(unsigned int) * (system->N + 1) );
//                memcpy( workspace->L->j, workspace->L_EE->j, sizeof(unsigned int) * workspace->L_EE->start[workspace->L_EE->n] );
//                memcpy( workspace->L->val, workspace->L_EE->val, sizeof(real) * workspace->L_EE->start[workspace->L_EE->n] );
//
//                top = workspace->L->start[system->N];
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->L->j[top] = i;
//                    workspace->L->val[top] = x[i];
//                    ++top;
//                }
//
//                workspace->L->j[top] = system->N_cm - 1;
//                workspace->L->val[top] = 1.0;
//                ++top;
//
//                workspace->L->start[system->N_cm] = top;
//
//                top = 0;
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->U->start[i] = top;
//                    memcpy( workspace->U->j + top, workspace->U_EE->j + workspace->U_EE->start[i],
//                            sizeof(unsigned int) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    memcpy( workspace->U->val + top, workspace->U_EE->val + workspace->U_EE->start[i],
//                            sizeof(real) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    top += (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]);
//
//                    workspace->U->j[top] = system->N_cm - 1;
//                    workspace->U->val[top] = y[i];
//                    ++top;
//                }
//
//                workspace->U->start[system->N_cm - 1] = top;
//
//                workspace->U->j[top] = system->N_cm - 1;
//                workspace->U->val[top] = -Dot( x, y, system->N );
//                ++top;
//
//                workspace->U->start[system->N_cm] = top;
//                break;
//
//            case TRI_SOLVE_LEVEL_SCHED_PA:
//                tri_solve_level_sched( workspace->L_EE, ones, x, workspace->L_EE->n, LOWER, TRUE );
//                Transpose_I( workspace->U_EE );
//                tri_solve_level_sched( workspace->U_EE, ones, y, workspace->U_EE->n, LOWER, TRUE );
//                Transpose_I( workspace->U_EE );
//
//                memcpy( workspace->L->start, workspace->L_EE->start, sizeof(unsigned int) * (system->N + 1) );
//                memcpy( workspace->L->j, workspace->L_EE->j, sizeof(unsigned int) * workspace->L_EE->start[workspace->L_EE->n] );
//                memcpy( workspace->L->val, workspace->L_EE->val, sizeof(real) * workspace->L_EE->start[workspace->L_EE->n] );
//
//                top = workspace->L->start[system->N];
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->L->j[top] = i;
//                    workspace->L->val[top] = x[i];
//                    ++top;
//                }
//
//                workspace->L->j[top] = system->N_cm - 1;
//                workspace->L->val[top] = 1.0;
//                ++top;
//
//                workspace->L->start[system->N_cm] = top;
//
//                top = 0;
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->U->start[i] = top;
//                    memcpy( workspace->U->j + top, workspace->U_EE->j + workspace->U_EE->start[i],
//                            sizeof(unsigned int) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    memcpy( workspace->U->val + top, workspace->U_EE->val + workspace->U_EE->start[i],
//                            sizeof(real) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    top += (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]);
//
//                    workspace->U->j[top] = system->N_cm - 1;
//                    workspace->U->val[top] = y[i];
//                    ++top;
//                }
//
//                workspace->U->start[system->N_cm - 1] = top;
//
//                workspace->U->j[top] = system->N_cm - 1;
//                workspace->U->val[top] = -Dot( x, y, system->N );
//                ++top;
//
//                workspace->U->start[system->N_cm] = top;
//                break;
//
//            //TODO: add Jacobi iter, etc.?
//            default:
//                tri_solve( workspace->L_EE, ones, x, workspace->L_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//                tri_solve( workspace->U_EE, ones, y, workspace->U_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//
//                memcpy( workspace->L->start, workspace->L_EE->start, sizeof(unsigned int) * (system->N + 1) );
//                memcpy( workspace->L->j, workspace->L_EE->j, sizeof(unsigned int) * workspace->L_EE->start[workspace->L_EE->n] );
//                memcpy( workspace->L->val, workspace->L_EE->val, sizeof(real) * workspace->L_EE->start[workspace->L_EE->n] );
//
//                top = workspace->L->start[system->N];
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->L->j[top] = i;
//                    workspace->L->val[top] = x[i];
//                    ++top;
//                }
//
//                workspace->L->j[top] = system->N_cm - 1;
//                workspace->L->val[top] = 1.0;
//                ++top;
//
//                workspace->L->start[system->N_cm] = top;
//
//                top = 0;
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->U->start[i] = top;
//                    memcpy( workspace->U->j + top, workspace->U_EE->j + workspace->U_EE->start[i],
//                            sizeof(unsigned int) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    memcpy( workspace->U->val + top, workspace->U_EE->val + workspace->U_EE->start[i],
//                            sizeof(real) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    top += (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]);
//
//                    workspace->U->j[top] = system->N_cm - 1;
//                    workspace->U->val[top] = y[i];
//                    ++top;
//                }
//
//                workspace->U->start[system->N_cm - 1] = top;
//
//                workspace->U->j[top] = system->N_cm - 1;
//                workspace->U->val[top] = -Dot( x, y, system->N );
//                ++top;
//
//                workspace->U->start[system->N_cm] = top;
//                break;
//        }
//    }
//
//#if defined(DEBUG)
//    if ( control->cm_solver_pre_comp_type != DIAG_PC )
//    {
//        fprintf( stderr, "condest = %f\n", condest(workspace->L) );
//
//#if defined(DEBUG_FOCUS)
//        sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
//        Print_Sparse_Matrix2( workspace->L, fname, NULL );
//        sprintf( fname, "%s.U%d.out", control->sim_name, data->step );
//        Print_Sparse_Matrix2( workspace->U, fname, NULL );
//
//        fprintf( stderr, "icholt-" );
//        sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
//        Print_Sparse_Matrix2( workspace->L, fname, NULL );
//        Print_Sparse_Matrix( U );
//#endif
//    }
//#endif
//}


/* Compute preconditioner for EE
 */
static void Compute_Preconditioner_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        if ( control->cm_domain_sparsify_enabled == TRUE )
        {
            Hptr = setup_graph_coloring( workspace->H_sp );
        }
        else
        {
            Hptr = setup_graph_coloring( workspace->H );
        }

        Sort_Matrix_Rows( Hptr );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            data->timing.cm_solver_pre_comp +=
                ICHOLT( Hptr, workspace->droptol, workspace->L, workspace->U );
            break;

        case ILU_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L, workspace->U );
            break;

        case ILUT_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        workspace->L, workspace->U );
            break;

        case ILU_SUPERLU_MT_PC:
#if defined(HAVE_SUPERLU_MT)
            data->timing.cm_solver_pre_comp +=
                SuperLU_Factorize( Hptr, workspace->L, workspace->U );
#else
            fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;

#if defined(DEBUG)
    if ( control->cm_solver_pre_comp_type != NONE_PC && 
            control->cm_solver_pre_comp_type != DIAG_PC )
    {
        fprintf( stderr, "condest = %f\n", condest(workspace->L, workspace->U) );

#if defined(DEBUG_FOCUS)
        sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        sprintf( fname, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#endif
    }
#endif
}


/* Compute preconditioner for ACKS2
 */
static void Compute_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        if ( control->cm_domain_sparsify_enabled == TRUE )
        {
            Hptr = setup_graph_coloring( workspace->H_sp );
        }
        else
        {
            Hptr = setup_graph_coloring( workspace->H );
        }

        Sort_Matrix_Rows( Hptr );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;
    Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            data->timing.cm_solver_pre_comp +=
                ICHOLT( Hptr, workspace->droptol, workspace->L, workspace->U );
            break;

        case ILU_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L, workspace->U );
            break;

        case ILUT_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        workspace->L, workspace->U );
            break;

        case ILU_SUPERLU_MT_PC:
#if defined(HAVE_SUPERLU_MT)
            data->timing.cm_solver_pre_comp +=
                SuperLU_Factorize( Hptr, workspace->L, workspace->U );
#else
            fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;
    Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;

#if defined(DEBUG)
    if ( control->cm_solver_pre_comp_type != NONE_PC || 
            control->cm_solver_pre_comp_type != DIAG_PC )
    {
        fprintf( stderr, "condest = %f\n", condest(workspace->L, workspace->U) );

#if defined(DEBUG_FOCUS)
        sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        sprintf( fname, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#endif
    }
#endif
}


/* Setup routines before computing the preconditioner for QEq
 */
static void Setup_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    int fillin;
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = Get_Time( );
    Sort_Matrix_Rows( workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

#if defined(DEBUG)
    fprintf( stderr, "H matrix sorted\n" );
#endif

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                if ( ( workspace->Hdia_inv = (real *) calloc( Hptr->n, sizeof( real ) ) ) == NULL )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            fillin = Estimate_LU_Fill( Hptr, workspace->droptol );

#if defined(DEBUG)
            fprintf( stderr, "fillin = %d\n", fillin );
            fprintf( stderr, "allocated memory: L = U = %ldMB\n",
                     fillin * (sizeof(real) + sizeof(unsigned int)) / (1024 * 1024) );
#endif

            if ( workspace->L == NULL )
            {
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, fillin ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, fillin ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }

            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILU_PAR_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILUT_PAR_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            if ( workspace->L == NULL )
            {
                /* TODO: safest storage estimate is ILU(0) (same as lower triangular portion of H), could improve later */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILU_SUPERLU_MT_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }
}


/* Setup routines before computing the preconditioner for EE
 */
static void Setup_Preconditioner_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    int fillin;
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    /* sorted H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = Get_Time( );
    Sort_Matrix_Rows( workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;

#if defined(DEBUG)
    fprintf( stderr, "H matrix sorted\n" );
#endif

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                if ( ( workspace->Hdia_inv = (real *) calloc( system->N_cm, sizeof( real ) ) ) == NULL )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            fillin = Estimate_LU_Fill( Hptr, workspace->droptol );

#if defined(DEBUG)
            fprintf( stderr, "fillin = %d\n", fillin );
            fprintf( stderr, "allocated memory: L = U = %ldMB\n",
                     fillin * (sizeof(real) + sizeof(unsigned int)) / (1024 * 1024) );
#endif

            if ( workspace->L == NULL )
            {
                if ( Allocate_Matrix( &(workspace->L), system->N_cm, fillin + system->N_cm ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), system->N_cm, fillin + system->N_cm ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }

            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILU_PAR_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILUT_PAR_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            if ( workspace->L == NULL )
            {
                /* TODO: safest storage estimate is ILU(0) (same as lower triangular portion of H), could improve later */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILU_SUPERLU_MT_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;
}


/* Setup routines before computing the preconditioner for ACKS2
 */
static void Setup_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    int fillin;
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = Get_Time( );
    Sort_Matrix_Rows( workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;
    Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;

#if defined(DEBUG)
    fprintf( stderr, "H matrix sorted\n" );
#endif

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                if ( ( workspace->Hdia_inv = (real *) calloc( Hptr->n, sizeof( real ) ) ) == NULL )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            fillin = Estimate_LU_Fill( Hptr, workspace->droptol );

#if defined(DEBUG)
            fprintf( stderr, "fillin = %d\n", fillin );
            fprintf( stderr, "allocated memory: L = U = %ldMB\n",
                     fillin * (sizeof(real) + sizeof(unsigned int)) / (1024 * 1024) );
#endif

            if ( workspace->L == NULL )
            {
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, fillin ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, fillin ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILU_PAR_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILUT_PAR_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            if ( workspace->L == NULL )
            {
                /* TODO: safest storage estimate is ILU(0) (same as lower triangular portion of H), could improve later */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        case ILU_SUPERLU_MT_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            else
            {
                //TODO: reallocate
            }
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;
    Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;
}


/* Combine ficticious charges s and t to get atomic charge q for QEq method
 */
static void Calculate_Charges_QEq( const reax_system * const system,
        static_storage * const workspace )
{
    int i;
    real u, s_sum, t_sum;

    s_sum = t_sum = 0.;
    for ( i = 0; i < system->N_cm; ++i )
    {
        s_sum += workspace->s[0][i];
        t_sum += workspace->t[0][i];
    }

    u = s_sum / t_sum;
    for ( i = 0; i < system->N_cm; ++i )
    {
        system->atoms[i].q = workspace->s[0][i] - u * workspace->t[0][i];

#if defined(DEBUG_FOCUS)
        printf("atom %4d: %f\n", i, system->atoms[i].q);
        printf("  x[0]: %10.5f, x[1]: %10.5f, x[2]:  %10.5f\n",
                system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2]);
#endif
    }
}


/* Get atomic charge q for EE method
 */
static void Calculate_Charges_EE( const reax_system * const system,
        static_storage * const workspace )
{
    int i;

    for ( i = 0; i < system->N; ++i )
    {
        system->atoms[i].q = workspace->s[0][i];

#if defined(DEBUG_FOCUS)
        printf( "atom %4d: %f\n", i, system->atoms[i].q );
        printf( "  x[0]: %10.5f, x[1]: %10.5f, x[2]:  %10.5f\n",
               system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2] );
#endif
    }
}


/* Main driver method for QEq kernel
 *
 * Rough outline:
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 2 linear solves
 *  5) compute atomic charges based on output of (4)
 */
static void QEq( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
    int iters;

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
        
    {
        Setup_Preconditioner_QEq( system, control, data, workspace, far_nbrs );

        Compute_Preconditioner_QEq( system, control, data, workspace, far_nbrs );
    }

    Extrapolate_Charges_QEq( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        iters += GMRES( workspace, control, data, workspace->H,
                workspace->b_t, control->cm_solver_q_err, workspace->t[0], FALSE );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        iters += GMRES_HouseHolder( workspace, control, data, workspace->H,
                workspace->b_t, control->cm_solver_q_err, workspace->t[0], 0 );
        break;

    case CG_S:
        iters = CG( workspace, control, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        iters += CG( workspace, control, workspace->H, workspace->b_t, control->cm_solver_q_err,
                workspace->t[0], FALSE ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        iters += SDM( workspace,control,  workspace->H, workspace->b_t, control->cm_solver_q_err,
                      workspace->t[0], FALSE ) + 1;
        break;

    default:
        fprintf( stderr, "Unrecognized QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "linsolve-" );
#endif

    Calculate_Charges_QEq( system, workspace );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "%d %.9f %.9f %.9f %.9f %.9f %.9f\n", data->step,
       workspace->s[0][0], workspace->t[0][0],
       workspace->s[0][1], workspace->t[0][1],
       workspace->s[0][2], workspace->t[0][2] );
    if( data->step == control->nsteps )
    {
        Print_Charges( system, control, workspace, data->step );
    }
#endif
}


/* Main driver method for EE kernel
 *
 * Rough outline:
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 1 linear solve
 *  5) compute atomic charges based on output of (4)
 */
static void EE( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
    int iters;

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
    {
        Setup_Preconditioner_EE( system, control, data, workspace, far_nbrs );

        Compute_Preconditioner_EE( system, control, data, workspace, far_nbrs );
    }

    Extrapolate_Charges_EE( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data,workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0 );
        break;

    case CG_S:
        iters = CG( workspace, control, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    default:
        fprintf( stderr, "Unrecognized EE solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "linsolve-" );
#endif

    Calculate_Charges_EE( system, workspace );

    // if( data->step == control->nsteps )
    //Print_Charges( system, control, workspace, data->step );
}


/* Main driver method for ACKS2 kernel
 *
 * Rough outline:
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 1 linear solve
 *  5) compute atomic charges based on output of (4)
 */
static void ACKS2( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
    int iters;

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
    {
        Setup_Preconditioner_ACKS2( system, control, data, workspace, far_nbrs );

        Compute_Preconditioner_ACKS2( system, control, data, workspace, far_nbrs );
    }

//   Print_Linear_System( system, control, workspace, data->step );

    Extrapolate_Charges_EE( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data,workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0 );
        break;

    case CG_S:
        iters = CG( workspace, control, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    default:
        fprintf( stderr, "Unrecognized ACKS2 solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "linsolve-" );
#endif

    Calculate_Charges_EE( system, workspace );
}


void Compute_Charges( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
#if defined(DEBUG_FOCUS)
    char fname[200];
    FILE * fp;

    if ( data->step >= 100 )
    {
        sprintf( fname, "s_%d_%s.out", data->step, control->sim_name );
        fp = fopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->s[0], system->N_cm );
        fclose( fp );

        sprintf( fname, "t_%d_%s.out", data->step, control->sim_name );
        fp = fopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->t[0], system->N_cm );
        fclose( fp );
    }
#endif

    switch ( control->charge_method )
    {
    case QEQ_CM:
        QEq( system, control, data, workspace, far_nbrs, out_control );
        break;

    case EE_CM:
        EE( system, control, data, workspace, far_nbrs, out_control );
        break;

    case ACKS2_CM:
        ACKS2( system, control, data, workspace, far_nbrs, out_control );
        break;

    default:
        fprintf( stderr, "Invalid charge method. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

#if defined(DEBUG_FOCUS)
    if ( data->step >= 100 )
    {
        sprintf( fname, "H_%d_%s.out", data->step, control->sim_name );
        Print_Sparse_Matrix2( workspace->H, fname, NULL );
//        Print_Sparse_Matrix_Binary( workspace->H, fname );

        sprintf( fname, "b_s_%d_%s.out", data->step, control->sim_name );
        fp = fopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->b_s, system->N_cm );
        fclose( fp );

        sprintf( fname, "b_t_%d_%s.out", data->step, control->sim_name );
        fp = fopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->b_t, system->N_cm );
        fclose( fp );
    }
#endif
}
