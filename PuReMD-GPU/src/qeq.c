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

#include "qeq.h"

#include "allocate.h"
#include "index_utils.h"
#include "list.h"
#include "lin_alg.h"
#include "print_utils.h"
#include "tool_box.h"
#if defined(HAVE_SUPERLU_MT)
#include "slu_mt_ddefs.h"
#endif


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
    return *(unsigned int *)v1 - *(unsigned int *)v2;
}


/* Routine used for sorting nonzeros within a sparse matrix row;
 *  internally, a combination of qsort and manual sorting is utilized
 *  (parallel calls to qsort when multithreading, rows mapped to threads)
 *
 * A: sparse matrix for which to sort nonzeros within a row, stored in CSR format
 */
static void Sort_Matrix_Rows( sparse_matrix * const A )
{
    unsigned int i, j, k, si, ei, *temp_j;
    real *temp_val;

    #pragma omp parallel default(none) private(i, j, k, si, ei, temp_j, temp_val) shared(stderr)
    {
        if ( ( temp_j = (unsigned int*) malloc( A->n * sizeof(unsigned int)) ) == NULL
                || ( temp_val = (real*) malloc( A->n * sizeof(real)) ) == NULL )
        {
            fprintf( stderr, "Not enough space for matrix row sort. Terminating...\n" );
            exit( INSUFFICIENT_MEMORY );
        }

        /* sort each row of A using column indices */
        #pragma omp for schedule(guided)
        for ( i = 0; i < A->n; ++i )
        {
            si = A->start[i];
            ei = A->start[i + 1];
            memcpy( temp_j, A->j + si, sizeof(unsigned int) * (ei - si) );
            memcpy( temp_val, A->val + si, sizeof(real) * (ei - si) );

            //TODO: consider implementing single custom one-pass sort instead of using qsort + manual sort
            /* polymorphic sort in standard C library using column indices */
            qsort( temp_j, ei - si, sizeof(unsigned int), compare_matrix_entry );

            /* manually sort vals */
            for ( j = 0; j < (ei - si); ++j )
            {
                for ( k = 0; k < (ei - si); ++k )
                {
                    if ( A->j[si + j] == temp_j[k] )
                    {
                        A->val[si + k] = temp_val[j];
                        break;
                    }

                }
            }

            /* copy sorted column indices */
            memcpy( A->j + si, temp_j, sizeof(unsigned int) * (ei - si) );
        }

        free( temp_val );
        free( temp_j );
    }
}


static void Calculate_Droptol( const sparse_matrix * const A, real * const droptol,
        const real dtol )
{
    int i, j, k;
    real val;
#ifdef _OPENMP
    static real *droptol_local;
    unsigned int tid;
#endif

    #pragma omp parallel default(none) private(i, j, k, val, tid), shared(droptol_local, stderr)
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

        #pragma omp barrier

        /* calculate sqaure of the norm of each row */
        #pragma omp for schedule(static)
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

            val = A->val[k]; // diagonal entry
#ifdef _OPENMP
            droptol_local[tid * A->n + i] += val * val;
#else
            droptol[i] += val * val;
#endif
        }

        #pragma omp barrier

#ifdef _OPENMP
        #pragma omp for schedule(static)
        for ( i = 0; i < A->n; ++i )
        {
            droptol[i] = 0.0;
            for ( k = 0; k < omp_get_num_threads(); ++k )
            {
                droptol[i] += droptol_local[k * A->n + i];
            }
        }
#endif

        #pragma omp barrier

        /* calculate local droptol for each row */
        //fprintf( stderr, "droptol: " );
        #pragma omp for schedule(static)
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
    int i, j, pj;
    int fillin;
    real val;

    fillin = 0;

    #pragma omp parallel for schedule(static) \
    default(none) private(i, j, pj, val) reduction(+: fillin)
    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            j = A->j[pj];
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
    free( xa );
    free( asub );
    free( a );
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

    free( Utop );
    free( Ltop );

    //TODO: return iters
    return 0.;
}
#endif


/* Diagonal (Jacobi) preconditioner computation */
static real diag_pre_comp( const reax_system * const system, real * const Hdia_inv )
{
    unsigned int i;
    real start;

    start = Get_Time( );

    #pragma omp parallel for schedule(static) \
    default(none) private(i)
    for ( i = 0; i < system->N; ++i )
    {
        Hdia_inv[i] = 1.0 / system->reaxprm.sbp[system->atoms[i].type].eta;
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
    int *Utop;

    start = Get_Time( );

    if ( ( Utop = (int*) malloc((A->n + 1) * sizeof(int)) ) == NULL ||
            ( tmp_j = (int*) malloc(A->n * sizeof(int)) ) == NULL ||
            ( tmp_val = (real*) malloc(A->n * sizeof(real)) ) == NULL )
    {
        fprintf( stderr, "not enough memory for ICHOLT preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    // clear variables
    Ltop = 0;
    tmptop = 0;
    memset( L->start, 0, (A->n + 1) * sizeof(unsigned int) );
    memset( U->start, 0, (A->n + 1) * sizeof(unsigned int) );
    memset( Utop, 0, A->n * sizeof(unsigned int) );

    //fprintf( stderr, "n: %d\n", A->n );
    for ( i = 0; i < A->n; ++i )
    {
        L->start[i] = Ltop;
        tmptop = 0;

        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            j = A->j[pj];
            val = A->val[pj];
            //fprintf( stderr, "i: %d, j: %d", i, j );

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
            //fprintf( stderr, " -- done\n" );
        }

        // sanity check
        if ( A->j[pj] != i )
        {
            fprintf( stderr, "i=%d, badly built A matrix!\n", i );
            exit( NUMERIC_BREAKDOWN );
        }

        // compute the ith diagonal in L
        val = A->val[pj];
        for ( k1 = 0; k1 < tmptop; ++k1 )
        {
            val -= (tmp_val[k1] * tmp_val[k1]);
        }

        tmp_j[tmptop] = i;
        tmp_val[tmptop] = SQRT(val);

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

    free( tmp_val );
    free( tmp_j );
    free( Utop );

    return Get_Timing_Info( start );
}


/* Fine-grained (parallel) incomplete Cholesky factorization
 *
 * Reference:
 * Edmond Chow and Aftab Patel
 * Fine-Grained Parallel Incomplete LU Factorization
 * SIAM J. Sci. Comp. */
static real ICHOL_PAR( const sparse_matrix * const A, const unsigned int sweeps,
        sparse_matrix * const U_t, sparse_matrix * const U )
{
    unsigned int i, j, k, pj, x = 0, y = 0, ei_x, ei_y;
    real *D, *D_inv, sum, start;
    sparse_matrix *DAD;
    int *Utop;

    start = Get_Time( );

    if ( Allocate_Matrix( DAD, A->n, A->m ) == FAILURE ||
            ( D = (real*) malloc(A->n * sizeof(real)) ) == NULL ||
            ( D_inv = (real*) malloc(A->n * sizeof(real)) ) == NULL ||
            ( Utop = (int*) malloc((A->n + 1) * sizeof(int)) ) == NULL )
    {
        fprintf( stderr, "not enough memory for ICHOL_PAR preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    #pragma omp parallel for schedule(static) \
        default(none) shared(D_inv, D) private(i)
    for ( i = 0; i < A->n; ++i )
    {
        D_inv[i] = SQRT( A->val[A->start[i + 1] - 1] );
        D[i] = 1. / D_inv[i];
    }

    memset( U->start, 0, sizeof(unsigned int) * (A->n + 1) );
    memset( Utop, 0, sizeof(unsigned int) * (A->n + 1) );

    /* to get convergence, A must have unit diagonal, so apply
     * transformation DAD, where D = D(1./sqrt(D(A))) */
    memcpy( DAD->start, A->start, sizeof(int) * (A->n + 1) );
    #pragma omp parallel for schedule(guided) \
        default(none) shared(DAD, D_inv, D) private(i, pj)
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
        #pragma omp parallel for schedule(static) \
            default(none) shared(DAD, stderr) private(sum, ei_x, ei_y, k) firstprivate(x, y)
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
                    fprintf( stderr, "Numeric breakdown in ICHOL Terminating.\n");
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
    #pragma omp parallel for schedule(guided) \
        default(none) shared(D_inv) private(i, pj)
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
    free(D_inv);
    free(D);
    free(Utop);

    return Get_Timing_Info( start );
}


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

    if ( Allocate_Matrix( DAD, A->n, A->m ) == FAILURE ||
            ( D = (real*) malloc(A->n * sizeof(real)) ) == NULL ||
            ( D_inv = (real*) malloc(A->n * sizeof(real)) ) == NULL )
    {
        fprintf( stderr, "not enough memory for ILU_PAR preconditioning matrices. terminating.\n" );
        exit( INSUFFICIENT_MEMORY );
    }

    #pragma omp parallel for schedule(static) \
        default(none) shared(D, D_inv) private(i)
    for ( i = 0; i < A->n; ++i )
    {
        D_inv[i] = SQRT( A->val[A->start[i + 1] - 1] );
        D[i] = 1.0 / D_inv[i];
    }

    /* to get convergence, A must have unit diagonal, so apply
     * transformation DAD, where D = D(1./sqrt(D(A))) */
    memcpy( DAD->start, A->start, sizeof(int) * (A->n + 1) );
    #pragma omp parallel for schedule(static) \
        default(none) shared(DAD, D) private(i, pj)
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
    #pragma omp parallel for schedule(static) default(none) private(i)
    for ( i = 0; i < A->n; ++i )
    {
        L->val[L->start[i + 1] - 1] = 1.0;
    }

    for ( i = 0; i < sweeps; ++i )
    {
        /* for each nonzero in L */
        #pragma omp parallel for schedule(static) \
            default(none) shared(DAD) private(j, k, x, y, ei_x, ei_y, sum)
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

        #pragma omp parallel for schedule(static) \
            default(none) shared(DAD) private(j, k, x, y, ei_x, ei_y, sum)
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
    #pragma omp parallel for schedule(static) \
        default(none) shared(DAD, D_inv) private(i, pj)
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
    free( D_inv );
    free( D );

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

    if ( Allocate_Matrix( DAD, A->n, A->m ) == FAILURE ||
            Allocate_Matrix( L_temp, A->n, A->m ) == FAILURE ||
            Allocate_Matrix( U_temp, A->n, A->m ) == FAILURE )
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

    #pragma omp parallel for schedule(static) \
        default(none) shared(D, D_inv) private(i)
    for ( i = 0; i < A->n; ++i )
    {
        D_inv[i] = SQRT( A->val[A->start[i + 1] - 1] );
        D[i] = 1.0 / D_inv[i];
    }

    /* to get convergence, A must have unit diagonal, so apply
     * transformation DAD, where D = D(1./sqrt(D(A))) */
    memcpy( DAD->start, A->start, sizeof(int) * (A->n + 1) );
    #pragma omp parallel for schedule(static) \
        default(none) shared(DAD, D) private(i, pj)
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
    #pragma omp parallel for schedule(static) \
        default(none) private(i) shared(L_temp)
    for ( i = 0; i < A->n; ++i )
    {
        L_temp->val[L_temp->start[i + 1] - 1] = 1.0;
    }

    for ( i = 0; i < sweeps; ++i )
    {
        /* for each nonzero in L */
        #pragma omp parallel for schedule(static) \
        default(none) shared(DAD, L_temp, U_temp) private(j, k, x, y, ei_x, ei_y, sum)
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

        #pragma omp parallel for schedule(static) \
            default(none) shared(DAD, L_temp, U_temp) private(j, k, x, y, ei_x, ei_y, sum)
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
    #pragma omp parallel for schedule(static) \
    default(none) shared(DAD, L_temp, U_temp, D_inv) private(i, pj)
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
    free( D_inv );
    free( D );

    return Get_Timing_Info( start );
}


/* Setup routine which performs the following:
 *  1) init storage for QEq matrices and other dependent routines
 *  2) compute preconditioner (if sim. step matches refactor step)
 *  3) extrapolate ficticious charges s and t
 */
static void Init_MatVec( const reax_system * const system, const control_params * const control,
        simulation_data * const data, static_storage * const workspace, const list * const far_nbrs )
{
    int i, fillin;
    real s_tmp, t_tmp, time;
    sparse_matrix *Hptr;
//    char fname[100];

    if (control->qeq_domain_sparsify_enabled)
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    if (control->pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->pre_comp_refactor == 0 || workspace->L == NULL))
    {
        //Print_Linear_System( system, control, workspace, data->step );

        time = Get_Time( );
        if ( control->pre_comp_type != DIAG_PC )
        {
            Sort_Matrix_Rows( workspace->H );
            if ( control->qeq_domain_sparsify_enabled == TRUE )
            {
                Sort_Matrix_Rows( workspace->H_sp );
            }

            if ( control->pre_app_type == TRI_SOLVE_GC_PA )
            {
                if ( control->qeq_domain_sparsify_enabled == TRUE )
                {
                    Hptr = setup_graph_coloring( workspace->H_sp );
                }
                else
                {
                    Hptr = setup_graph_coloring( workspace->H );
                }

                Sort_Matrix_Rows( Hptr );
            }
        }
        data->timing.QEq_sort_mat_rows += Get_Timing_Info( time );

#if defined(DEBUG)
        fprintf( stderr, "H matrix sorted\n" );
#endif

        switch ( control->pre_comp_type )
        {
        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                if ( ( workspace->Hdia_inv = (real *) calloc( system->N, sizeof( real ) ) ) == NULL )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            data->timing.pre_comp += diag_pre_comp( system, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->pre_comp_droptol );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            if ( workspace->L == NULL )
            {
                fillin = Estimate_LU_Fill( Hptr, workspace->droptol );
                if ( Allocate_Matrix( workspace->L, far_nbrs->n, fillin ) == FAILURE ||
                        Allocate_Matrix( workspace->U, far_nbrs->n, fillin ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }

#if defined(DEBUG)
                fprintf( stderr, "fillin = %d\n", fillin );
                fprintf( stderr, "allocated memory: L = U = %ldMB\n",
                         fillin * sizeof(sparse_matrix_entry) / (1024 * 1024) );
#endif
            }

            data->timing.pre_comp += ICHOLT( Hptr, workspace->droptol, workspace->L, workspace->U );
            break;

        case ILU_PAR_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( workspace->L, Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( workspace->U, Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }

            data->timing.pre_comp += ILU_PAR( Hptr, control->pre_comp_sweeps, workspace->L, workspace->U );
            break;

        case ILUT_PAR_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->pre_comp_droptol );
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "drop tolerances calculated\n" );
#endif

            if ( workspace->L == NULL )
            {
                /* TODO: safest storage estimate is ILU(0) (same as lower triangular portion of H), could improve later */
                if ( Allocate_Matrix( workspace->L, Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( workspace->U, Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }

            data->timing.pre_comp += ILUT_PAR( Hptr, workspace->droptol, control->pre_comp_sweeps,
                    workspace->L, workspace->U );
            break;

        case ILU_SUPERLU_MT_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                if ( Allocate_Matrix( workspace->L, Hptr->n, Hptr->m ) == FAILURE ||
                        Allocate_Matrix( workspace->U, Hptr->n, Hptr->m ) == FAILURE )
                {
                    fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }

#if defined(HAVE_SUPERLU_MT)
            data->timing.pre_comp += SuperLU_Factorize( Hptr, workspace->L, workspace->U );
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
        fprintf( stderr, "condest = %f\n", condest(workspace->L, workspace->U) );
#endif

#if defined(DEBUG_FOCUS)
        sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname );
        sprintf( fname, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname );

        fprintf( stderr, "icholt-" );
        //sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        //Print_Sparse_Matrix2( workspace->L, fname );
        //Print_Sparse_Matrix( U );
#endif
    }

    /* extrapolation for s & t */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
    #pragma omp parallel for schedule(static) \
        default(none) private(i, s_tmp, t_tmp)
    for ( i = 0; i < system->N; ++i )
    {
        // no extrapolation
        //s_tmp = workspace->s[index_wkspace_sys(0,i,system->N)];
        //t_tmp = workspace->t[index_wkspace_sys(0,i,system->N)];

        // linear
        //s_tmp = 2 * workspace->s[index_wkspace_sys(0,i,system->N)] - workspace->s[index_wkspace_sys(1,i,system->N)];
        //t_tmp = 2 * workspace->t[index_wkspace_sys(0,i,system->N)] - workspace->t[index_wkspace_sys(1,i,system->N)];

        // quadratic
//        s_tmp = workspace->s[index_wkspace_sys(2,i,system->N)] +
//            3 * (workspace->s[index_wkspace_sys(0,i,system->N)]-workspace->s[index_wkspace_sys(1,i,system->N)]);
        t_tmp = workspace->t[index_wkspace_sys(2,i,system->N)] +
            3 * (workspace->t[index_wkspace_sys(0,i,system->N)] -workspace->t[index_wkspace_sys(1,i,system->N)]);

        // cubic
        s_tmp = 4 * (workspace->s[index_wkspace_sys(0,i,system->N)] + workspace->s[index_wkspace_sys(2,i,system->N)]) -
            (6 * workspace->s[index_wkspace_sys(1,i,system->N)] + workspace->s[index_wkspace_sys(3,i,system->N)]);
        //t_tmp = 4 * (workspace->t[index_wkspace_sys(0,i,system->N)] + workspace->t[index_wkspace_sys(2,i,system->N)]) -
        //  (6 * workspace->t[index_wkspace_sys(1,i,system->N)] + workspace->t[index_wkspace_sys(3,i,system->N)] );

        // 4th order
//        s_tmp = 5 * (workspace->s[index_wkspace_sys(0,i,system->N)] -
//                workspace->s[index_wkspace_sys(3,i,system->N)]) + 10 *
//            (-workspace->s[index_wkspace_sys(1,i,system->N)] +
//             workspace->s[index_wkspace_sys(2,i,system->N)] ) +
//            workspace->s[index_wkspace_sys(4,i,system->N)];
//        t_tmp = 5 * (workspace->t[index_wkspace_sys(0,i,system->N)] -
//                workspace->t[index_wkspace_sys(3,i,system->N)]) + 10 *
//            (-workspace->t[index_wkspace_sys(1,i,system->N)] +
//             workspace->t[index_wkspace_sys(2,i,system->N)] ) +
//            workspace->t[index_wkspace_sys(4,i,system->N)];

        workspace->s[index_wkspace_sys(4,i,system->N)] = workspace->s[index_wkspace_sys(3,i,system->N)];
        workspace->s[index_wkspace_sys(3,i,system->N)] = workspace->s[index_wkspace_sys(2,i,system->N)]; 
        workspace->s[index_wkspace_sys(2,i,system->N)] = workspace->s[index_wkspace_sys(1,i,system->N)];
        workspace->s[index_wkspace_sys(1,i,system->N)] = workspace->s[index_wkspace_sys(0,i,system->N)];
        workspace->s[index_wkspace_sys(0,i,system->N)] = s_tmp;

        workspace->t[index_wkspace_sys(4,i,system->N)] = workspace->t[index_wkspace_sys(3,i,system->N)];
        workspace->t[index_wkspace_sys(3,i,system->N)] = workspace->t[index_wkspace_sys(2,i,system->N)]; 
        workspace->t[index_wkspace_sys(2,i,system->N)] = workspace->t[index_wkspace_sys(1,i,system->N)];
        workspace->t[index_wkspace_sys(1,i,system->N)] = workspace->t[index_wkspace_sys(0,i,system->N)];
        workspace->t[index_wkspace_sys(0,i,system->N)] = t_tmp;
    }
}


/* Combine ficticious charges s and t to get atomic charge q
 */
static void Calculate_Charges( const reax_system * const system, static_storage * const workspace )
{
    int i;
    real u, s_sum, t_sum;

    s_sum = t_sum = 0.;
    for ( i = 0; i < system->N; ++i )
    {
        s_sum += workspace->s[index_wkspace_sys(0,i,system->N)];
        t_sum += workspace->t[index_wkspace_sys(0,i,system->N)];
    }

    u = s_sum / t_sum;
    for ( i = 0; i < system->N; ++i )
    {
        system->atoms[i].q = workspace->s[index_wkspace_sys(0,i,system->N)]
            - u * workspace->t[index_wkspace_sys(0,i,system->N)];
    }
}


/* Main driver method for QEq kernel
 *
 * Rough outline:
 *  1) init / setup routines
 *  2) perform 2 linear solves
 *  3) compute atomic charges based on output of 2)
 */
void QEq( reax_system * const system, control_params * const control, simulation_data * const data,
          static_storage * const workspace, const list * const far_nbrs,
          const output_controls * const out_control )
{
    int iters;

    Init_MatVec( system, control, data, workspace, far_nbrs );

    switch ( control->qeq_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, workspace->H, workspace->b_s, control->qeq_solver_q_err,
                &workspace->s[index_wkspace_sys(0,0,system->N)], out_control->log,
                ((data->step - data->prev_steps) % control->pre_comp_refactor == 0) ? TRUE : FALSE );
        iters += GMRES( workspace, control, data, workspace->H, workspace->b_t, control->qeq_solver_q_err,
                &workspace->t[index_wkspace_sys(0,0,system->N)], out_control->log, FALSE );
        break;
    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data, workspace->H, workspace->b_s, control->qeq_solver_q_err,
                &workspace->s[index_wkspace_sys(0,0,system->N)], out_control->log, (data->step - data->prev_steps) % control->pre_comp_refactor == 0 );
        iters += GMRES_HouseHolder( workspace, control, data, workspace->H, workspace->b_t, control->qeq_solver_q_err,
                &workspace->t[index_wkspace_sys(0,0,system->N)], out_control->log, 0 );
        break;
    case CG_S:
        iters = CG( workspace, workspace->H, workspace->b_s, control->qeq_solver_q_err,
                &workspace->s[index_wkspace_sys(0,0,system->N)], out_control->log ) + 1;
        iters += CG( workspace, workspace->H, workspace->b_t, control->qeq_solver_q_err,
                &workspace->t[index_wkspace_sys(0,0,system->N)], out_control->log ) + 1;
        break;
    case SDM_S:
        iters = SDM( workspace, workspace->H, workspace->b_s, control->qeq_solver_q_err,
                &workspace->s[index_wkspace_sys(0,0,system->N)], out_control->log ) + 1;
        iters += SDM( workspace, workspace->H, workspace->b_t, control->qeq_solver_q_err,
                &workspace->t[index_wkspace_sys(0,0,system->N)], out_control->log ) + 1;
        break;
    default:
        fprintf( stderr, "Unrecognized QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.solver_iters += iters;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "linsolve-" );
#endif

    Calculate_Charges( system, workspace );
}
