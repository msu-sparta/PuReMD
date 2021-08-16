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

#include "lin_alg.h"

#include "allocate.h"
#include "basic_comm.h"
#include "comm_tools.h"
#include "io_tools.h"
#include "tool_box.h"
#include "vector.h"

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


static int compare_matrix_entry( const void * const v1, const void * const v2 )
{
    return ((sparse_matrix_entry *)v1)->j - ((sparse_matrix_entry *)v2)->j;
}


/* Routine used for sorting nonzeros within a sparse matrix row;
 *  internally, a combination of qsort and manual sorting is utilized
 *
 * A: sparse matrix for which to sort nonzeros within a row, stored in CSR format
 */
void Sort_Matrix_Rows( sparse_matrix * const A )
{
    unsigned int i, pj, si, ei, temp_size;
    sparse_matrix_entry *temp;

    temp = NULL;
    temp_size = 0;

    /* sort each row of A using column indices */
    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];
        ei = A->end[i];

        if ( temp == NULL )
        {
            temp = smalloc( sizeof(sparse_matrix_entry) * (ei - si), "Sort_Matrix_Rows::temp" );
            temp_size = ei - si;
        }
        else if ( temp_size < ei - si )
        {
            sfree( temp, "Sort_Matrix_Rows::temp" );
            temp = smalloc( sizeof(sparse_matrix_entry) * (ei - si), "Sort_Matrix_Rows::temp" );
            temp_size = ei - si;
        }

        for ( pj = 0; pj < (ei - si); ++pj )
        {
            temp[pj].j = A->j[si + pj];
            temp[pj].val = A->val[si + pj];
        }

        /* polymorphic sort in standard C library using column indices */
        qsort( temp, ei - si, sizeof(sparse_matrix_entry), compare_matrix_entry );

        for ( pj = 0; pj < (ei - si); ++pj )
        {
            A->j[si + pj] = temp[pj].j;
            A->val[si + pj] = temp[pj].val;
        }
    }

    sfree( temp, "Sort_Matrix_Rows::temp" );
}


static int compare_dbls( const void* arg1, const void* arg2 )
{   
    int ret;
    double a1, a2;

    a1 = *(double *) arg1;
    a2 = *(double *) arg2;

    if ( a1 < a2 )
    {   
        ret = -1;
    }
    else if (a1 == a2)
    {   
        ret = 0;
    }
    else
    {   
        ret = 1;
    }

    return ret;
}


static void qsort_dbls( double *array, int array_len )
{
    qsort( array, (size_t) array_len, sizeof(double),
            compare_dbls );
}


static int find_bucket( double *list, int len, double a )
{
    int s, e, m;

    if ( len == 0 )
    {
        return 0;
    }

    if ( a > list[len - 1] )
    {
        return len;
    }

    s = 0;
    e = len - 1;

    while ( s < e )
    {
        m = (s + e) / 2;

        if ( list[m] < a )
        {
            s = m + 1;
        }
        else
        {
            e = m;
        }
    }

    return s;
}


/* Compute diagonal inverese (Jacobi) preconditioner
 *
 * H: matrix used to compute preconditioner, in CSR format
 * Hdia_inv: computed diagonal inverse preconditioner
 */
void jacobi( sparse_matrix const * const H, real * const Hdia_inv )
{
    unsigned int i, pj;

    if ( H->format == SYM_HALF_MATRIX )
    {
        for ( i = 0; i < H->n; ++i )
        {
            if ( FABS( H->val[H->start[i]] ) > 1.0e-15 )
            {
                Hdia_inv[i] = 1.0 / H->val[H->start[i]];
            }
            else
            {
                Hdia_inv[i] = 1.0;
            }
        }
    }
    else if ( H->format == SYM_FULL_MATRIX || H->format == FULL_MATRIX )
    {
        for ( i = 0; i < H->n; ++i )
        {
            for ( pj = H->start[i]; pj < H->start[i + 1]; ++pj )
            {
                if ( H->j[pj] == i )
                {
                    if ( FABS( H->val[H->start[i]] ) > 1.0e-15 )
                    {
                        Hdia_inv[i] = 1.0 / H->val[pj];
                    }
                    else
                    {
                        Hdia_inv[i] = 1.0;
                    }

                    break;
                }
            }
        }
    }
}


/* Apply diagonal inverse (Jacobi) preconditioner to system residual
 *
 * Hdia_inv: diagonal inverse preconditioner (constructed using H)
 * y: current residuals
 * x: preconditioned residuals
 * N: dimensions of preconditioner and vectors (# rows in H)
 */
static void dual_jacobi_app( const real * const Hdia_inv, const rvec2 * const y,
        rvec2 * const x, const int N )
{
    unsigned int i;

    for ( i = 0; i < N; ++i )
    {
        x[i][0] = y[i][0] * Hdia_inv[i];
        x[i][1] = y[i][1] * Hdia_inv[i];
    }
}


/* Apply diagonal inverse (Jacobi) preconditioner to system residual
 *
 * Hdia_inv: diagonal inverse preconditioner (constructed using H)
 * y: current residual
 * x: preconditioned residual
 * N: dimensions of preconditioner and vectors (# rows in H)
 */
static void jacobi_app( const real * const Hdia_inv, const real * const y,
        real * const x, const int N )
{
    unsigned int i;

    for ( i = 0; i < N; ++i )
    {
        x[i] = y[i] * Hdia_inv[i];
    }
}


/* Local arithmetic portion of dual sparse matrix-dense vector multiplication Ax = b
 *
 * A: sparse matrix, 1D partitioned row-wise
 * x: two dense vectors
 * b (output): two dense vectors
 * N: number of entries in both vectors in b (must be equal)
 */
static void Dual_Sparse_MatVec_local( sparse_matrix const * const A,
        rvec2 const * const x, rvec2 * const b, int N )
{
    int i, j, k, si, num_rows;
    real val;

    for ( i = 0; i < N; ++i )
    {
        b[i][0] = 0.0;
        b[i][1] = 0.0;
    }

#if defined(NEUTRAL_TERRITORY)
    num_rows = A->NT;

    if ( A->format == SYM_HALF_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            /* diagonal only contributes once */
            if ( i < A->n )
            {
                b[i][0] += A->val[si] * x[i][0];
                b[i][1] += A->val[si] * x[i][1];
                k = si + 1;
            }
            /* zeros on the diagonal for i >= A->n,
             * so skip the diagonal multplication step as zeros
             * are not stored (idea: keep the NNZ's the same
             * for full shell and neutral territory half-stored
             * charge matrices to make debugging easier) */
            else
            {
                k = si;
            }

            for ( ; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i][0] += val * x[j][0];
                b[i][1] += val * x[j][1];
                
                b[j][0] += val * x[i][0];
                b[j][1] += val * x[i][1];
            }
        }
    }
    else if ( A->format == SYM_FULL_MATRIX || A->format == FULL_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            for ( k = si; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i][0] += val * x[j][0];
                b[i][1] += val * x[j][1];
            }
        }
    }
#else
    num_rows = A->n;

    if ( A->format == SYM_HALF_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            /* diagonal only contributes once */
            b[i][0] += A->val[si] * x[i][0];
            b[i][1] += A->val[si] * x[i][1];

            for ( k = si + 1; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i][0] += val * x[j][0];
                b[i][1] += val * x[j][1];
                
                b[j][0] += val * x[i][0];
                b[j][1] += val * x[i][1];
            }
        }
    }
    else if ( A->format == SYM_FULL_MATRIX || A->format == FULL_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            for ( k = si; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i][0] += val * x[j][0];
                b[i][1] += val * x[j][1];
            }
        }
    }
#endif
}


/* Communications for sparse matrix-dense vector multiplication Ax = b
 *
 * system:
 * control: 
 * mpi_data:
 * x: dense vector
 * buf_type: data structure type for x
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static void Sparse_MatVec_Comm_Part1( const reax_system * const system,
        const control_params * const control, mpi_datatypes * const mpi_data,
        void const * const x, int buf_type, MPI_Datatype mpi_type )
{
    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, x, buf_type, mpi_type );
}


/* Local arithmetic portion of sparse matrix-dense vector multiplication Ax = b
 *
 * A: sparse matrix, 1D partitioned row-wise
 * x: dense vector
 * b (output): dense vector
 * N: number of entries in b
 */
static void Sparse_MatVec_local( sparse_matrix const * const A,
        real const * const x, real * const b, int N )
{
    int i, j, k, si, num_rows;
    real val;

    for ( i = 0; i < N; ++i )
    {
        b[i] = 0.0;
    }

#if defined(NEUTRAL_TERRITORY)
    num_rows = A->NT;

    if ( A->format == SYM_HALF_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            /* diagonal only contributes once */
            if ( i < A->n )
            {
                b[i] += A->val[si] * x[i];
                k = si + 1;
            }
            /* zeros on the diagonal for i >= A->n,
             * so skip the diagonal multplication step as zeros
             * are not stored (idea: keep the NNZ's the same
             * for full shell and neutral territory half-stored
             * charge matrices to make debugging easier) */
            else
            {
                k = si;
            }

            for ( ; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i] += val * x[j];
                b[j] += val * x[i];
            }
        }
    }
    else if ( A->format == SYM_FULL_MATRIX || A->format == FULL_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            for ( k = si; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i] += val * x[j];
            }
        }
    }
#else
    num_rows = A->n;

    if ( A->format == SYM_HALF_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            /* A symmetric, upper triangular portion stored
             * => diagonal only contributes once */
            b[i] += A->val[si] * x[i];

            for ( k = si + 1; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i] += val * x[j];
                b[j] += val * x[i];
            }
        }
    }
    else if ( A->format == SYM_FULL_MATRIX || A->format == FULL_MATRIX )
    {
        for ( i = 0; i < num_rows; ++i )
        {
            si = A->start[i];

            for ( k = si; k < A->end[i]; ++k )
            {
                j = A->j[k];
                val = A->val[k];

                b[i] += val * x[j];
            }
        }
    }
#endif
}


/* Communications for sparse matrix-dense vector multiplication Ax = b
 *
 * system:
 * control:
 * mpi_data:
 * mat_format: storage type of sparse matrix A
 * b: dense vector
 * buf_type: data structure type for b
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static void Sparse_MatVec_Comm_Part2( const reax_system * const system,
        const control_params * const control, mpi_datatypes * const mpi_data,
        int mat_format, void * const b, int buf_type, MPI_Datatype mpi_type )
{
    if ( mat_format == SYM_HALF_MATRIX )
    {
        Coll( system, mpi_data, b, buf_type, mpi_type );
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        Coll( system, mpi_data, b, buf_type, mpi_type );
    }
#endif
}


/* sparse matrix, dense vector multiplication AX = B
 *
 * system:
 * control:
 * data:
 * A: symmetric matrix, stored in CSR format
 * X: dense vector
 * n: number of entries in x
 * B (output): dense vector */
static void Dual_Sparse_MatVec( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        mpi_datatypes * const mpi_data, sparse_matrix const * const A,
        rvec2 const * const x, int n, rvec2 * const b )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Sparse_MatVec_Comm_Part1( system, control, mpi_data, x,
            RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif

    Dual_Sparse_MatVec_local( A, x, b, n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_spmv );
#endif

    Sparse_MatVec_Comm_Part2( system, control, mpi_data, A->format, b,
            RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif
}


/* sparse matrix, dense vector multiplication Ax = b
 *
 * system:
 * control:
 * data:
 * A: symmetric matrix, stored in CSR format
 * x: dense vector
 * n: number of entries in x
 * b (output): dense vector */
static void Sparse_MatVec( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        mpi_datatypes * const mpi_data, sparse_matrix const * const A,
        real const * const x, int n, real * const b )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Sparse_MatVec_Comm_Part1( system, control, mpi_data, x,
            REAL_PTR_TYPE, MPI_DOUBLE );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif

    Sparse_MatVec_local( A, x, b, n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_spmv );
#endif

    Sparse_MatVec_Comm_Part2( system, control, mpi_data, A->format, b,
            REAL_PTR_TYPE, MPI_DOUBLE );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif
}


void setup_sparse_approx_inverse( reax_system const * const system,
        simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data,
        sparse_matrix * const A, sparse_matrix *A_spar_patt,
        int nprocs, real filter )
{
    int i, ret, bin, total, pos;
    int n, n_gather, s_local, s, n_local;
    int target_proc;
    int k, pj, size;
    int left, right, p, turn;
    int num_rows;
    int *srecv, *sdispls;
    int *scounts_local, *scounts;
    int *dspls_local, *dspls;
    int *bin_elements;
    real threshold, pivot, tmp;
    real *input_array;
    real *samplelist_local, *samplelist;
    real *pivotlist;
    real *bucketlist_local, *bucketlist;
    real t_start, t_comm;
    real total_comm;

    srecv = NULL;
    sdispls = NULL;
    samplelist_local = NULL;
    samplelist = NULL;
    pivotlist = NULL;
    input_array = NULL;
    bucketlist_local = NULL;
    bucketlist = NULL;
    scounts_local = NULL;
    scounts = NULL;
    dspls_local = NULL;
    dspls = NULL;
    bin_elements = NULL;
    t_comm = 0.0;

#if defined(NEUTRAL_TERRITORY)
    num_rows = A->NT;
#else
    num_rows = A->n;
#endif

    if ( A_spar_patt->allocated == FALSE )
    {
        Allocate_Matrix( A_spar_patt, A->n,
#if defined(NEUTRAL_TERRITORY)
                A->NT,
#else
                A->n_max,
#endif
                A->m, A->format );
    }
    else if ( A_spar_patt->m < A->m 
            || A_spar_patt->n_max < A->n_max )
    {
        Deallocate_Matrix( A_spar_patt );

        Allocate_Matrix( A_spar_patt, A->n,
#if defined(NEUTRAL_TERRITORY)
                A->NT,
#else
                A->n_max,
#endif
                A->m, A->format );
    }

    A_spar_patt->n = A->n;

    n_local = 0;
    for ( i = 0; i < num_rows; ++i )
    {
        n_local += (A->end[i] - A->start[i] + 9) / 10;
    }
    s_local = (int) (12.0 * (LOG2(n_local) + LOG2(nprocs)));
    
    t_start = Get_Time( );
    ret = MPI_Allreduce( &n_local, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Reduce( &s_local, &s, 1, MPI_INT, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

    /* count num. bin elements for each processor, uniform bin sizes */
    input_array = smalloc( sizeof(real) * n_local,
           "setup_sparse_approx_inverse::input_array" );
    scounts_local = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::scounts_local" );
    scounts = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::scounts" );
    bin_elements = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::bin_elements" );
    dspls_local = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::displs_local" );
    bucketlist_local = smalloc( sizeof(real) * n_local,
          "setup_sparse_approx_inverse::bucketlist_local" );
    dspls = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::dspls" );
    if ( nprocs > 1 )
    {
        pivotlist = smalloc( sizeof(real) *  (nprocs - 1),
                "setup_sparse_approx_inverse::pivotlist" );
    }
    samplelist_local = smalloc( sizeof(real) * s_local,
           "setup_sparse_approx_inverse::samplelist_local" );
    if ( system->my_rank == MASTER_NODE )
    {
        samplelist = smalloc( sizeof(real) * s,
               "setup_sparse_approx_inverse::samplelist" );
        srecv = smalloc( sizeof(int) * nprocs,
               "setup_sparse_approx_inverse::srecv" );
        sdispls = smalloc( sizeof(int) * nprocs,
               "setup_sparse_approx_inverse::sdispls" );
    }

    n_local = 0;
    for ( i = 0; i < num_rows; ++i )
    {
        for ( pj = A->start[i]; pj < A->end[i]; pj += 10 )
        {
            input_array[n_local++] = A->val[pj];
        }
    }

    for ( i = 0; i < s_local; i++)
    {
        /* samplelist_local[i] = input_array[rand( ) % n_local]; */
        samplelist_local[i] = input_array[ i ];
    }

    /* gather samples at the root process */
    t_start = Get_Time( );
    ret = MPI_Gather( &s_local, 1, MPI_INT, srecv, 1, MPI_INT, MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

    if( system->my_rank == MASTER_NODE )
    {
        sdispls[0] = 0;
        for ( i = 0; i < nprocs - 1; ++i )
        {
            sdispls[i + 1] = sdispls[i] + srecv[i];
        }
    }

    t_start = Get_Time( );
    ret = MPI_Gatherv( samplelist_local, s_local, MPI_DOUBLE,
            samplelist, srecv, sdispls, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD);
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

    /* sort samples at the root process and select pivots */
    if ( system->my_rank == MASTER_NODE )
    {
        qsort_dbls( samplelist, s );

        for ( i = 1; i < nprocs; ++i )
        {
            pivotlist[i - 1] = samplelist[(i * s) / nprocs];
        }
    }

    /* broadcast pivots */
    t_start = Get_Time( );
    ret = MPI_Bcast( pivotlist, nprocs - 1, MPI_DOUBLE, MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

    for ( i = 0; i < nprocs; ++i )
    {
        scounts_local[i] = 0;
    }

    for ( i = 0; i < n_local; ++i )
    {
        pos = find_bucket( pivotlist, nprocs - 1, input_array[i] );
        scounts_local[pos]++;
    }

    for ( i = 0; i < nprocs; ++i )
    {
        bin_elements[i] = scounts_local[i];
        scounts[i] = scounts_local[i];
    }

    /* compute displacements for MPI comm */
    dspls_local[0] = 0;
    for ( i = 0; i < nprocs - 1; ++i )
    {
        dspls_local[i + 1] = dspls_local[i] + scounts_local[i];
    }

    /* bin elements */
    for ( i = 0; i < n_local; ++i )
    {
        bin = find_bucket( pivotlist, nprocs - 1, input_array[i] );
        pos = dspls_local[bin] + scounts_local[bin] - bin_elements[bin];
        bucketlist_local[pos] = input_array[i];
        bin_elements[bin]--;
    }

    /* determine counts for elements per process */
    t_start = Get_Time( );
    ret = MPI_Allreduce( MPI_IN_PLACE, scounts, nprocs, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

    /* find the target process */
    target_proc = 0;
    total = 0;
    k = n * filter;
    for ( i = nprocs - 1; i >= 0; --i )
    {
        if ( total + scounts[i] >= k )
        {
            /* global k becomes local k*/
            k -= total;
            target_proc = i;
            break;
        }
        total += scounts[i];
    }

    n_gather = scounts[target_proc];
    if ( system->my_rank == target_proc )
    {
        bucketlist = smalloc( sizeof( real ) * n_gather,
               "setup_sparse_approx_inverse::bucketlist" );
    }

    /* send local buckets to target processor for quickselect */
    t_start = Get_Time( );
    ret = MPI_Gather( scounts_local + target_proc, 1, MPI_INT, scounts,
            1, MPI_INT, target_proc, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

    if ( system->my_rank == target_proc )
    {
        dspls[0] = 0;
        for ( i = 0; i < nprocs - 1; ++i )
        {
            dspls[i + 1] = dspls[i] + scounts[i];
        }
    }

    t_start = Get_Time( );
    ret = MPI_Gatherv( bucketlist_local + dspls_local[target_proc], scounts_local[target_proc], MPI_DOUBLE,
            bucketlist, scounts, dspls, MPI_DOUBLE, target_proc, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

    /* apply quick select algorithm at the target process */
    if ( system->my_rank == target_proc )
    {
        left = 0;
        right = n_gather-1;

        turn = 0;
        while ( k )
        {
            p  = left;
            turn = 1 - turn;

            /* alternating pivots in order to handle corner cases */
            if ( turn == 1 )
            {
                pivot = bucketlist[right];
            }
            else
            {
                pivot = bucketlist[left];
            }
            for ( i = left + 1 - turn; i <= right-turn; ++i )
            {
                if ( bucketlist[i] > pivot )
                {
                    tmp = bucketlist[i];
                    bucketlist[i] = bucketlist[p];
                    bucketlist[p] = tmp;
                    p++;
                }
            }
            if ( turn == 1 )
            {
                tmp = bucketlist[p];
                bucketlist[p] = bucketlist[right];
                bucketlist[right] = tmp;
            }
            else
            {
                tmp = bucketlist[p];
                bucketlist[p] = bucketlist[left];
                bucketlist[left] = tmp;
            }

            if ( p == k - 1)
            {
                threshold = bucketlist[p];
                break;
            }
            else if ( p > k - 1 )
            {
                right = p - 1;
            }
            else
            {
                left = p + 1;
            }
        }
        /* comment out if ACKS2 and/or EE is not an option */
//        if ( threshold < 1.000000 )
//        {
//            threshold = 1.000001;
//        }
    }

    /* broadcast the filtering value */
    t_start = Get_Time( );
    ret = MPI_Bcast( &threshold, 1, MPI_DOUBLE, target_proc, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    t_comm += Get_Time( ) - t_start;

#if defined(DEBUG_FOCUS)
    int nnz = 0;
#endif

    /* build entries of that pattern*/
    for ( i = 0; i < num_rows; ++i )
    {
        A_spar_patt->start[i] = A->start[i];
        size = A->start[i];

        for ( pj = A->start[i]; pj < A->end[i]; ++pj )
        {
            if ( ( A->val[pj] >= threshold )  || ( A->j[pj] == i ) )
            {
                A_spar_patt->val[size] = A->val[pj];
                A_spar_patt->j[size] = A->j[pj];
                size++;

#if defined(DEBUG_FOCUS)
                nnz++;
#endif
            }
        }
        A_spar_patt->end[i] = size;
    }

#if defined(DEBUG_FOCUS)
    ret = MPI_Allreduce( MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stdout, "    [INFO] \ntotal nnz in all charge matrices = %d\ntotal nnz in all sparsity patterns = %d\nthreshold = %.15lf\n",
                n, nnz, threshold );
        fflush( stdout );
    }
#endif
 
    ret = MPI_Reduce( &t_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM,
            MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_comm += total_comm / nprocs;
    }

    sfree( input_array, "setup_sparse_approx_inverse::input_array" );
    sfree( scounts_local, "setup_sparse_approx_inverse::scounts_local" );
    sfree( scounts, "setup_sparse_approx_inverse::scounts" );
    sfree( bin_elements, "setup_sparse_approx_inverse::bin_elements" );
    sfree( dspls_local, "setup_sparse_approx_inverse::displs_local" );
    sfree( bucketlist_local, "setup_sparse_approx_inverse::bucketlist_local" );
    sfree( dspls, "setup_sparse_approx_inverse::dspls" );
    if ( nprocs > 1 )
    {
        sfree( pivotlist, "setup_sparse_approx_inverse::pivotlist" );
    }
    sfree( samplelist_local, "setup_sparse_approx_inverse::samplelist_local" );
    if ( system->my_rank == MASTER_NODE )
    {
        sfree( samplelist, "setup_sparse_approx_inverse::samplelist" );
        sfree( srecv, "setup_sparse_approx_inverse::srecv" );
        sfree( sdispls, "setup_sparse_approx_inverse::sdispls" );
    }
    if ( system->my_rank == target_proc )
    {
        sfree( bucketlist, "setup_sparse_approx_inverse::bucketlist" );
    }
}


#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
#if defined(NEUTRAL_TERRITORY)
real sparse_approx_inverse( reax_system const * const system,
        simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data, 
        sparse_matrix * const A, sparse_matrix * const A_spar_patt,
        sparse_matrix * const A_app_inv, int nprocs )
{
    int i, k, pj, j_temp, ret;
    int N, M, d_i, d_j;
    int local_pos, atom_pos, identity_pos;
    int *pos_x, *X;
    int cnt;
    int *row_nnz;
    int **j_list;
    int d, count, index;
    int *j_send, *j_recv[6];
    real *e_j, *dense_matrix;
    real *val_send, *val_recv[6];
    reax_atom *atom;
    real **val_list;
    real start, t_start, t_comm, total_comm;
    mpi_out_data *out_bufs;
    neighbor_proc *nbr;
    MPI_Request req[12];
    MPI_Status stat[12];
    lapack_int m, n, nrhs, lda, ldb, info;

    start = Get_Time( );
    t_comm = 0.0;

    if ( A_app_inv->allocated == FALSE )
    {
        //TODO: FULL_MATRIX?
        Allocate_Matrix( A_app_inv, A_spar_patt->n, A->NT, A_spar_patt->m, SYM_FULL_MATRIX );
    }
    
    else /* if ( A_app_inv->m < A_spar_patt->m ) */
    {
        Deallocate_Matrix( A_app_inv );
        Allocate_Matrix( A_app_inv, A_spar_patt->n, A->NT, A_spar_patt->m, SYM_FULL_MATRIX );
    }

    pos_x = NULL;
    X = NULL;

    row_nnz = NULL;
    j_list = NULL;
    val_list = NULL;

    j_send = NULL;
    val_send = NULL;
    for ( d = 0; d < 6; ++d )
    {
        j_recv[d] = NULL;
        val_recv[d] = NULL;
    }
    ////////////////////
    row_nnz = (int *) malloc( sizeof(int) * A->NT );

    //TODO: allocation size
    j_list = (int **) malloc( sizeof(int *) * system->N );
    val_list = (real **) malloc( sizeof(real *) * system->N );

    for ( i = 0; i < A->NT; ++i )
    {
        row_nnz[i] = 0;
    }

    /* mark the atoms that already have their row stored in the local matrix */
    for ( i = 0; i < A->n; ++i )
    {
        row_nnz[i] = A->end[i] - A->start[i];
    }

    /* Announce the nnz's in each row that will be communicated later */
    t_start = Get_Time( );
    Dist( system, mpi_data, row_nnz, REAL_PTR_TYPE, MPI_INT );
    t_comm += Get_Time( ) - t_start;
    fprintf( stdout,"SAI after Dist call\n" );
    fflush( stdout );

    out_bufs = mpi_data->out_nt_buffers;
    count = 0;

    /*  use a Dist-like approach to send the row information */
    for ( d = 0; d < 6; ++d)
    {
        /* initiate recvs */
        nbr = &system->my_nt_nbrs[d];
        if ( nbr->atoms_cnt )
        {
            /* calculate the total data that will be received */
            cnt = 0;
            for ( i = nbr->atoms_str; i < (nbr->atoms_str + nbr->atoms_cnt); ++i )
            {
                cnt += row_nnz[i];
            }

            /* initiate Irecv */
            if ( cnt )
            {
                count += 2;

                j_recv[d] = (int *) malloc( sizeof(int) * cnt );
                val_recv[d] = (real *) malloc( sizeof(real) * cnt );

                fprintf( stdout, "Dist communication receive phase direction %d will receive %d\n", d, cnt );
                fflush( stdout );
                t_start = Get_Time( );
                ret = MPI_Irecv( j_recv + d, cnt, MPI_INT, nbr->receive_rank,
                        d, mpi_data->comm_mesh3D, &req[2 * d] );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                ret = MPI_Irecv( val_recv + d, cnt, MPI_DOUBLE, nbr->receive_rank,
                        d, mpi_data->comm_mesh3D, &req[2 * d + 1] );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                t_comm += Get_Time( ) - t_start;
            }
        }
    }
    /////////////////////
    for( d = 0; d < 6; ++d)
    {
        nbr = &system->my_nt_nbrs[d];
        /* send both messages in dimension d */
        if ( out_bufs[d].cnt )
        {
            cnt = 0;
            for ( i = 0; i < out_bufs[d].cnt; ++i )
            {
                cnt += A->end[ out_bufs[d].index[i] ] - A->start[ out_bufs[d].index[i] ];
                if ( out_bufs[d].index[i] < 0 || out_bufs[d].index[i] >= A->n )
                {
                    fprintf( stdout, "INDEXING ERROR %d > %d\n", out_bufs[d].index[i], A->n );
                    fflush( stdout );
                }
               //     row_nnz[ out_bufs[d].index[i] ];
            }
            fprintf( stdout,"Dist communication    send phase direction %d should  send %d\n", d, cnt );
            fflush( stdout );

            if ( cnt )
            {
                j_send = (int *) malloc( sizeof(int) * cnt );
                val_send = (real *) malloc( sizeof(real) * cnt );

                cnt = 0;
                for ( i = 0; i < out_bufs[d].cnt; ++i )
                {
                    for ( pj = A->start[ out_bufs[d].index[i] ]; pj < A->end[ out_bufs[d].index[i] ]; ++pj )
                    {
                        atom = &system->my_atoms[ A->j[pj] ];
                        j_send[cnt] = atom->orig_id;
                        val_send[cnt] = A->val[pj];
                        cnt++;
                    }
                }

                fprintf( stdout,"Dist communication    send phase direction %d will    send %d\n", d, cnt );
                fflush( stdout );

                t_start = Get_Time( );
                ret = MPI_Send( j_send, cnt, MPI_INT, nbr->rank,
                        d, mpi_data->comm_mesh3D );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                fprintf( stdout,"Dist communication send phase direction %d cnt = %d\n", d, cnt );
                fflush( stdout );
                ret = MPI_Send( val_send, cnt, MPI_DOUBLE, nbr->rank,
                        d, mpi_data->comm_mesh3D );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                fprintf( stdout,"Dist communication send phase direction %d cnt = %d\n", d, cnt );
                fflush( stdout );
                t_comm += Get_Time( ) - t_start;
            }
        }
    }
    fprintf( stdout," Dist communication for sending row info before waitany\n" );
    fflush( stdout );
    ///////////////////////
    for ( d = 0; d < count; ++d )
    {
        t_start = Get_Time( );
        ret = MPI_Waitany( MAX_NT_NBRS, req, &index, stat );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        t_comm += Get_Time( ) - t_start;

        nbr = &system->my_nt_nbrs[index / 2];
        cnt = 0;
        for ( i = nbr->atoms_str; i < (nbr->atoms_str + nbr->atoms_cnt); ++i )
        {
            if ( index % 2 == 0 )
            {
                j_list[i] = (int *) malloc( sizeof(int) *  row_nnz[i] );
                for ( pj = 0; pj < row_nnz[i]; ++pj )
                {
                    j_list[i][pj] = j_recv[index / 2][cnt];
                    cnt++;
                }
            }
            else
            {
                val_list[i] = (real *) malloc( sizeof(real) * row_nnz[i] );
                for ( pj = 0; pj < row_nnz[i]; ++pj )
                {
                    val_list[i][pj] = val_recv[index / 2][cnt];
                    cnt++;
                }
            }

        }
    }
    //////////////////////
    fprintf( stdout, "Dist communication for sending row info worked\n" );
    fflush( stdout );
    //TODO: size?
    X = (int *) malloc( sizeof(int) * (system->bigN + 1) );
    pos_x = (int *) malloc( sizeof(int) * (system->bigN + 1) );

    for ( i = 0; i < A_spar_patt->NT; ++i )
    {
        N = 0;
        M = 0;
        for ( k = 0; k <= system->bigN; ++k )
        {
            X[k] = 0;
            pos_x[k] = 0;
        }

        /* find column indices of nonzeros (which will be the columns indices of the dense matrix) */
        for ( pj = A_spar_patt->start[i]; pj < A_spar_patt->end[i]; ++pj )
        {
            j_temp = A_spar_patt->j[pj];
            atom = &system->my_atoms[j_temp];
            ++N;

            /* for each of those indices
             * search through the row of full A of that index */

            /* the case where the local matrix has that index's row */
            if ( j_temp < A->NT )
            {
                for ( k = A->start[ j_temp ]; k < A->end[ j_temp ]; ++k )
                {
                    /* and accumulate the nonzero column indices to serve as the row indices of the dense matrix */
                    atom = &system->my_atoms[ A->j[k] ];
                    X[atom->orig_id] = 1;
                }
            }

            /* the case where we communicated that index's row */
            else
            {
                for ( k = 0; k < row_nnz[j_temp]; ++k )
                {
                    /* and accumulate the nonzero column indices to serve as the row indices of the dense matrix */
                    X[ j_list[j_temp][k] ] = 1;
                }
            }
        }

        /* enumerate the row indices from 0 to (# of nonzero rows - 1) for the dense matrix */
        identity_pos = M;
        atom = &system->my_atoms[ i ];
        atom_pos = atom->orig_id;

        for ( k = 0; k <= system->bigN; k++)
        {
            if ( X[k] != 0 )
            {
                pos_x[k] = M;
                if ( k == atom_pos )
                {
                    identity_pos = M;
                }
                ++M;
            }
        }

        /* allocate memory for NxM dense matrix */
        dense_matrix = (real *) malloc( sizeof(real) * N * M );

        /* fill in the entries of dense matrix */
        for ( d_j = 0; d_j < N; ++d_j)
        {
            /* all rows are initialized to zero */
            for ( d_i = 0; d_i < M; ++d_i )
            {
                dense_matrix[d_i * N + d_j] = 0.0;
            }
            /* change the value if any of the column indices is seen */

            /* it is in the original list */
            local_pos = A_spar_patt->j[ A_spar_patt->start[i] + d_j ];
            if ( local_pos < 0 || local_pos >= system->N )
            {
                fprintf( stderr, "THE LOCAL POSITION OF THE ATOM IS NOT VALID, STOP THE EXECUTION\n" );
                fflush( stderr );

            }
            /////////////////////////////
            if ( local_pos < A->NT )
            {
                for ( d_i = A->start[local_pos]; d_i < A->end[local_pos]; ++d_i )
                {
                    atom = &system->my_atoms[ A->j[d_i] ];
                    if ( pos_x[ atom->orig_id ] >= M || d_j >=  N )
                    {
                        fprintf( stderr, "CANNOT MAP IT TO THE DENSE MATRIX, STOP THE EXECUTION, orig_id = %d, i =  %d, j = %d, M = %d N = %d\n", atom->orig_id, pos_x[ atom->orig_id ], d_j, M, N );
                        fflush( stderr );
                    }
                    if ( X[ atom->orig_id ] == 1 )
                    {
                        dense_matrix[ pos_x[ atom->orig_id ] * N + d_j ] = A->val[d_i];
                    }
                }
            }
            else
            {
                for ( d_i = 0; d_i < row_nnz[ local_pos ]; ++d_i )
                {
                    if ( pos_x[ j_list[local_pos][d_i] ] >= M || d_j  >= N )
                    {
                        fprintf( stderr, "CANNOT MAP IT TO THE DENSE MATRIX, STOP THE EXECUTION, %d %d\n", pos_x[ j_list[local_pos][d_i] ], d_j);
                        fflush( stderr );
                    }
                    if ( X[ j_list[local_pos][d_i] ] == 1 )
                    {
                        dense_matrix[ pos_x[ j_list[local_pos][d_i] ] * N + d_j ] = val_list[local_pos][d_i];
                    }
                }
            }
        }

        /* create the right hand side of the linear equation
         * that is the full column of the identity matrix */
        e_j = (real *) malloc( sizeof(real) * M );
        //////////////////////
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

        /* accumulate the resulting vector to build A_app_inv */
        A_app_inv->start[i] = A_spar_patt->start[i];
        A_app_inv->end[i] = A_spar_patt->end[i];
        for ( k = A_app_inv->start[i]; k < A_app_inv->end[i]; ++k)
        {
            A_app_inv->j[k] = A_spar_patt->j[k];
            A_app_inv->val[k] = e_j[k - A_spar_patt->start[i]];
        }
        free( dense_matrix );
        free( e_j );
    }

    free( pos_x);
    free( X );
    /////////////////////
    ret = MPI_Reduce( &t_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM,
            MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_comm += total_comm / nprocs;
    }

    return Get_Time( ) - start;
}


#else
real sparse_approx_inverse( reax_system const * const system,
        simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data, 
        sparse_matrix * const A, sparse_matrix * const A_spar_patt,
        sparse_matrix * const A_app_inv, int nprocs )
{
    int i, k, pj, j_temp, push, ret;
    int N, M, d_i, d_j, mark;
    int local_pos, atom_pos, identity_pos;
    int *X, *q;
    int size_e, size_dense;
    int cnt;
    int *row_nnz;
    int **j_list;
    int d;
    int flag1, flag2;
    int *j_send, *j_recv1, *j_recv2;
    int size_send, size_recv1, size_recv2;
    real *e_j, *dense_matrix;
    real **val_list;
    real *val_send, *val_recv1, *val_recv2;
    real start, t_start, t_comm, total_comm;
    reax_atom *atom;
    mpi_out_data *out_bufs;
    const neighbor_proc *nbr1, *nbr2;
    MPI_Request req1, req2, req3, req4;
    MPI_Status stat1, stat2, stat3, stat4;
    lapack_int m, n, nrhs, lda, ldb, info;

    start = Get_Time( );
    t_comm = 0.0;

    if ( A_app_inv->allocated == FALSE )
    {
        Allocate_Matrix( A_app_inv, A_spar_patt->n, A_spar_patt->n_max, A_spar_patt->m,
                SYM_FULL_MATRIX );
    }
    else if ( A_app_inv->m < A_spar_patt->m
            || A_app_inv->n_max < A_spar_patt->n_max )
    {
        Deallocate_Matrix( A_app_inv );

        Allocate_Matrix( A_app_inv, A_spar_patt->n, A_spar_patt->n_max, A_spar_patt->m,
                SYM_FULL_MATRIX );
    }

    A_app_inv->n = A_spar_patt->n;

    X = NULL;
    j_send = NULL;
    val_send = NULL;
    j_recv1 = NULL;
    j_recv2 = NULL;
    val_recv1 = NULL;
    val_recv2 = NULL;
    size_send = 0;
    size_recv1 = 0;
    size_recv2 = 0;

    e_j = NULL;
    dense_matrix = NULL;
    size_e = 0;
    size_dense = 0;

    row_nnz = smalloc( sizeof(int) * system->total_cap,
           "sparse_approx_inverse::row_nnz" );
    j_list = smalloc( sizeof(int *) * system->N,
           "sparse_approx_inverse::j_list" );
    val_list = smalloc( sizeof(real *) * system->N,
           "sparse_approx_inverse::val_list" );
    for ( i = 0; i < system->N; ++i )
    {
        j_list[i] = NULL;
    }
    for ( i = 0; i < system->N; ++i )
    {
        val_list[i] = NULL;
    }

    for ( i = 0; i < system->total_cap; ++i )
    {
        row_nnz[i] = 0;
    }

    /* mark the atoms that already have their row stored in the local matrix */
    for ( i = 0; i < system->n; ++i )
    {
        row_nnz[i] = A->end[i] - A->start[i];
    }

    /* Announce the nnz's in each row that will be communicated later */
    t_start = Get_Time( );
    Dist( system, mpi_data, row_nnz, INT_PTR_TYPE, MPI_INT );
    t_comm += Get_Time( ) - t_start;

    out_bufs = mpi_data->out_buffers;

    /* use a Dist-like approach to send the row information */
    for ( d = 0; d < 3; ++d )
    {
        flag1 = 0;
        flag2 = 0;
        cnt = 0;

        /* initiate recvs */
        nbr1 = &system->my_nbrs[2 * d];
        if ( nbr1->atoms_cnt )
        {
            cnt = 0;

            /* calculate the total data that will be received */
            for ( i = nbr1->atoms_str; i < (nbr1->atoms_str + nbr1->atoms_cnt); ++i )
            {
                cnt += row_nnz[i];
            }

            /* initiate Irecv */
            if ( cnt > 0 )
            {
                flag1 = 1;
                
                if ( size_recv1 < cnt )
                {
                    if ( size_recv1 )
                    {
                        sfree( j_recv1, "sparse_approx_inverse::j_recv1" );
                        sfree( val_recv1, "sparse_approx_inverse::val_recv1" );
                    }

                    size_recv1 = cnt * SAFE_ZONE;

                    j_recv1 = smalloc( sizeof(int) * size_recv1,
                            "sparse_approx_inverse::j_recv1" );
                    val_recv1 = smalloc( sizeof(real) * size_recv1,
                            "sparse_approx_inverse::val_recv1" );
                }

                t_start = Get_Time( );
                ret = MPI_Irecv( j_recv1, cnt, MPI_INT, nbr1->rank,
                        2 * d + 1, mpi_data->comm_mesh3D, &req1 );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                ret = MPI_Irecv( val_recv1, cnt, MPI_DOUBLE, nbr1->rank,
                        2 * d + 1, mpi_data->comm_mesh3D, &req2 );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                t_comm += Get_Time( ) - t_start;
            }
        }

        nbr2 = &system->my_nbrs[2 * d + 1];
        if ( nbr2->atoms_cnt )
        {
            /* calculate the total data that will be received */
            cnt = 0;
            for ( i = nbr2->atoms_str; i < (nbr2->atoms_str + nbr2->atoms_cnt); ++i )
            {
                cnt += row_nnz[i];
            }

            /* initiate Irecv */
            if ( cnt > 0 )
            {
                flag2 = 1;

                if ( size_recv2 < cnt )
                {
                    if ( size_recv2 )
                    {
                        sfree( j_recv2, "sparse_approx_inverse::j_recv2" );
                        sfree( val_recv2, "sparse_approx_inverse::val_recv2" );
                    }

                    size_recv2 = cnt * SAFE_ZONE;

                    j_recv2 = smalloc( sizeof(int) * size_recv2,
                            "sparse_approx_inverse::j_recv2" );
                    val_recv2 = smalloc( sizeof(real) * size_recv2,
                            "sparse_approx_inverse::val_recv2" );
                }

                t_start = Get_Time( );
                ret = MPI_Irecv( j_recv2, cnt, MPI_INT, nbr2->rank,
                        2 * d, mpi_data->comm_mesh3D, &req3 );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                ret = MPI_Irecv( val_recv2, cnt, MPI_DOUBLE, nbr2->rank,
                        2 * d, mpi_data->comm_mesh3D, &req4 );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                t_comm += Get_Time( ) - t_start;
            }
        }

        /* send both messages in dimension d */
        if ( out_bufs[2 * d].cnt )
        {
            cnt = 0;
            for ( i = 0; i < out_bufs[2 * d].cnt; ++i )
            {
                cnt += row_nnz[ out_bufs[2 * d].index[i] ];
            }

            if ( cnt > 0 )
            {
                if ( size_send < cnt )
                {
                    if ( size_send )
                    {
                        sfree( j_send, "sparse_approx_inverse::j_send" );
                        sfree( val_send, "sparse_approx_inverse::val_send" );
                    }

                    size_send = cnt * SAFE_ZONE;

                    j_send = smalloc( sizeof(int) * size_send,
                            "sparse_approx_inverse::j_send" );
                    val_send = smalloc( sizeof(real) * size_send,
                            "sparse_approx_inverse::j_send" );
                }

                cnt = 0;
                for ( i = 0; i < out_bufs[2 * d].cnt; ++i )
                {
                    if ( out_bufs[2 * d].index[i] < A->n )
                    {
                        for ( pj = A->start[ out_bufs[2 * d].index[i] ]; pj < A->end[ out_bufs[2 * d].index[i] ]; ++pj )
                        {
                            atom = &system->my_atoms[ A->j[pj] ];
                            j_send[cnt] = atom->orig_id;
                            val_send[cnt] = A->val[pj];
                            cnt++;
                        }
                    }
                    else
                    {
                        for ( pj = 0; pj < row_nnz[ out_bufs[2 * d].index[i] ]; ++pj )
                        {
                            j_send[cnt] = j_list[ out_bufs[2 * d].index[i] ][pj];
                            val_send[cnt] = val_list[ out_bufs[2 * d].index[i] ][pj];
                            cnt++;
                        }
                    }
                }

                t_start = Get_Time( );
                ret = MPI_Send( j_send, cnt, MPI_INT, nbr1->rank,
                        2 * d, mpi_data->comm_mesh3D );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                ret = MPI_Send( val_send, cnt, MPI_DOUBLE, nbr1->rank,
                        2 * d, mpi_data->comm_mesh3D );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                t_comm += Get_Time( ) - t_start;
            }
        }

        if ( out_bufs[2 * d + 1].cnt )
        {
            cnt = 0;
            for ( i = 0; i < out_bufs[2 * d + 1].cnt; ++i )
            {
                cnt += row_nnz[ out_bufs[2 * d + 1].index[i] ];
            }

            if ( cnt > 0 )
            {

                if ( size_send < cnt )
                {
                    if ( size_send )
                    {
                        sfree( j_send, "sparse_approx_inverse::j_send" );
                        sfree( val_send, "sparse_approx_inverse::j_send" );
                    }

                    size_send = cnt * SAFE_ZONE;

                    j_send = smalloc( sizeof(int) * size_send,
                            "sparse_approx_inverse::j_send" );
                    val_send = smalloc( sizeof(real) * size_send,
                            "sparse_approx_inverse::val_send" );
                }

                cnt = 0;
                for ( i = 0; i < out_bufs[2 * d + 1].cnt; ++i )
                {
                    if ( out_bufs[2 * d + 1].index[i] < A->n )
                    {
                        for ( pj = A->start[ out_bufs[2 * d + 1].index[i] ]; pj < A->end[ out_bufs[2 * d + 1].index[i] ]; ++pj )
                        {
                            atom = &system->my_atoms[ A->j[pj] ];
                            j_send[cnt] = atom->orig_id;
                            val_send[cnt] = A->val[pj];
                            cnt++;
                        }
                    }
                    else
                    {
                        for ( pj = 0; pj < row_nnz[ out_bufs[2 * d + 1].index[i] ]; ++pj )
                        {
                            j_send[cnt] = j_list[ out_bufs[2 * d + 1].index[i] ][pj];
                            val_send[cnt] = val_list[ out_bufs[2 * d + 1].index[i] ][pj];
                            cnt++;
                        }
                    }
                }

                t_start = Get_Time( );
                ret = MPI_Send( j_send, cnt, MPI_INT, nbr2->rank,
                        2 * d + 1, mpi_data->comm_mesh3D );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                ret = MPI_Send( val_send, cnt, MPI_DOUBLE, nbr2->rank,
                        2 * d + 1, mpi_data->comm_mesh3D );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                t_comm += Get_Time( ) - t_start;
            }

        }

        if ( flag1 )
        {
            t_start = Get_Time( );
            ret = MPI_Wait( &req1, &stat1 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            ret = MPI_Wait( &req2, &stat2 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            t_comm += Get_Time( ) - t_start;

            cnt = 0;
            for ( i = nbr1->atoms_str; i < (nbr1->atoms_str + nbr1->atoms_cnt); ++i )
            {
                j_list[i] = smalloc( sizeof(int) * row_nnz[i],
                       "sparse_approx_inverse::j_list[i]" );
                val_list[i] = smalloc( sizeof(real) * row_nnz[i],
                       "sparse_approx_inverse::val_list[i]" );

                for ( pj = 0; pj < row_nnz[i]; ++pj )
                {
                    j_list[i][pj] = j_recv1[cnt];
                    val_list[i][pj] = val_recv1[cnt];
                    cnt++;
                }
            }
        }

        if ( flag2 )
        {
            t_start = Get_Time( );
            ret = MPI_Wait( &req3, &stat3 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            ret = MPI_Wait( &req4, &stat4 );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
            t_comm += Get_Time( ) - t_start;

            cnt = 0;
            for ( i = nbr2->atoms_str; i < (nbr2->atoms_str + nbr2->atoms_cnt); ++i )
            {
                j_list[i] = smalloc( sizeof(int) *  row_nnz[i],
                       "sparse_approx_inverse::j_list[i]" );
                val_list[i] = smalloc( sizeof(real) * row_nnz[i],
                       "sparse_approx_inverse::val_list[i]" );

                for ( pj = 0; pj < row_nnz[i]; ++pj )
                {
                    j_list[i][pj] = j_recv2[cnt];
                    val_list[i][pj] = val_recv2[cnt];
                    cnt++;
                }
            }
        }
    }

    sfree( j_send, "sparse_approx_inverse::j_send" );
    sfree( val_send, "sparse_approx_inverse::val_send" );
    sfree( j_recv1, "sparse_approx_inverse::j_recv1" );
    sfree( j_recv2, "sparse_approx_inverse::j_recv2" );
    sfree( val_recv1, "sparse_approx_inverse::val_recv1" );
    sfree( val_recv2, "sparse_approx_inverse::val_recv2" );

    X = smalloc( sizeof(int) * (system->bigN + 1),
            "sparse_approx_inverse::X" );
    //size of q should be equal to the maximum possible cardinalty 
    //of the set formed by neighbors of neighbors of an atom
    //i.e, maximum number of rows of dense matrix
    //for water systems, this number is 34000
    //for silica systems, it is 12000
    q = smalloc( sizeof(int) * 50000,
            "sparse_approx_inverse::q" );

    for ( i = 0; i <= system->bigN; ++i )
    {
        X[i] = -1;
    }

    for ( i = 0; i < A_spar_patt->n; ++i )
    {
        N = 0;
        M = 0;
        push = 0;
        mark = i + system->bigN;
        
        /* find column indices of nonzeros (which will be the columns indices of the dense matrix) */
        for ( pj = A_spar_patt->start[i]; pj < A_spar_patt->end[i]; ++pj )
        {
            j_temp = A_spar_patt->j[pj];
            atom = &system->my_atoms[j_temp];
            ++N;

            /* for each of those indices
             * search through the row of full A of that index */

            /* the case where the local matrix has that index's row */
            if ( j_temp < A->n )
            {
                for ( k = A->start[ j_temp ]; k < A->end[ j_temp ]; ++k )
                {
                    /* and accumulate the nonzero column indices to serve as the row indices of the dense matrix */
                    atom = &system->my_atoms[ A->j[k] ];
                    X[atom->orig_id] = mark;
                    q[push++] = atom->orig_id;
                }
            }

            /* the case where we communicated that index's row */
            else
            {
                for ( k = 0; k < row_nnz[j_temp]; ++k )
                {
                    /* and accumulate the nonzero column indices to serve as the row indices of the dense matrix */
                    X[ j_list[j_temp][k] ] = mark;
                    q[push++] = j_list[j_temp][k];
                }
            }
        }

        /* enumerate the row indices from 0 to (# of nonzero rows - 1) for the dense matrix */
        identity_pos = M;
        atom = &system->my_atoms[ i ];
        atom_pos = atom->orig_id;

        for ( k = 0; k < push; k++)
        {
            if ( X[ q[k] ] == mark )
            {
                X[ q[k] ] = M;
                ++M;
            }
        }
        identity_pos = X[atom_pos];

        /* allocate memory for NxM dense matrix */
        if ( size_dense < N * M )
        {
            if ( size_dense )
            {
                sfree( dense_matrix, "sparse_approx_inverse::dense_matrix" );
            }
            
            size_dense = N * M * SAFE_ZONE;

            dense_matrix = smalloc( sizeof(real) * size_dense,
                "sparse_approx_inverse::dense_matrix" );
        }

        /* fill in the entries of dense matrix */
        for ( d_j = 0; d_j < N; ++d_j)
        {
            /* all rows are initialized to zero */
            for ( d_i = 0; d_i < M; ++d_i )
            {
                dense_matrix[d_i * N + d_j] = 0.0;
            }
            /* change the value if any of the column indices is seen */

            /* it is in the original list */
            local_pos = A_spar_patt->j[ A_spar_patt->start[i] + d_j ];

            if ( local_pos < A->n )
            {
                for ( d_i = A->start[local_pos]; d_i < A->end[local_pos]; ++d_i )
                {
                    atom = &system->my_atoms[ A->j[d_i] ];
                    dense_matrix[ X[ atom->orig_id ] * N + d_j ] = A->val[d_i];
                }
            }
            else
            {
                for ( d_i = 0; d_i < row_nnz[ local_pos ]; ++d_i )
                {
                    dense_matrix[ X[ j_list[local_pos][d_i] ] * N + d_j ] = val_list[local_pos][d_i];
                }
            }
        }

        /* create the right hand side of the linear equation
         * that is the full column of the identity matrix */
        if ( size_e < M )
        {
            if ( size_e )
            {
                sfree( e_j, "sparse_approx_inverse::e_j" );
            }

            size_e = M * SAFE_ZONE;

            e_j = smalloc( sizeof(real) * size_e, "sparse_approx_inverse::e_j" );
        }

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
            fprintf( stderr, "[ERROR] The diagonal element %i of the triangular factor ", info );
            fprintf( stderr, "of A is zero, so that A does not have full rank;\n" );
            fprintf( stderr, "the least squares solution could not be computed.\n" );
            MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
        }

        /* accumulate the resulting vector to build A_app_inv */
        A_app_inv->start[i] = A_spar_patt->start[i];
        A_app_inv->end[i] = A_spar_patt->end[i];
        for ( k = A_app_inv->start[i]; k < A_app_inv->end[i]; ++k )
        {
            A_app_inv->j[k] = A_spar_patt->j[k];
            A_app_inv->val[k] = e_j[k - A_spar_patt->start[i]];
        }
    }

    sfree( dense_matrix, "sparse_approx_inverse::dense_matrix" );
    sfree( e_j, "sparse_approx_inverse::e_j" );
    sfree( X, "sparse_approx_inverse::X" );
    for ( i = 0; i < system->N; ++i )
    {
        if ( j_list[i] != NULL )
        {
            sfree( j_list[i], "sparse_approx_inverse::j_list" );
        }
    }
    sfree( j_list, "sparse_approx_inverse::j_list" );
    for ( i = 0; i < system->N; ++i )
    {
        if ( val_list[i] != NULL )
        {
            sfree( val_list[i], "sparse_approx_inverse::val_list" );
        }
    }
    sfree( val_list, "sparse_approx_inverse::val_list" );
    sfree( row_nnz, "sparse_approx_inverse::row_nnz" );

    ret = MPI_Reduce( &t_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM,
            MASTER_NODE, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_comm += total_comm / nprocs;
    }

    return Get_Time( ) - start;
}
#endif
#endif


/* Apply left-sided preconditioning while solving M^{-1}AX = M^{-1}B
 *
 * system:
 * workspace: data struct containing matrices and vectors, stored in CSR
 * control: data struct containing parameters
 * data: struct containing timing simulation data (including performance data)
 * y: vector to which to apply preconditioning,
 *  specific to internals of iterative solver being used
 * x (output): preconditioned vector
 * fresh_pre: parameter indicating if this is a newly computed (fresh) preconditioner
 * side: used in determining how to apply preconditioner if the preconditioner is
 *  factorized as M = M_{1}M_{2} (e.g., incomplete LU, A \approx LU)
 *
 * Assumptions:
 *   Matrices have non-zero diagonals
 *   Each row of a matrix has at least one non-zero (i.e., no rows with all zeros) */
static void dual_apply_preconditioner( reax_system const * const system,
        storage const * const workspace, control_params const * const control,
        simulation_data * const data, mpi_datatypes * const  mpi_data,
        rvec2 const * const y, rvec2 * const x, int fresh_pre, int side )
{
//    int i, si;

    /* no preconditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        if ( x != y )
        {
            Vector_Copy_rvec2( x, y, system->n );
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
                            case JACOBI_PC:
                                dual_jacobi_app( workspace->Hdia_inv, y, x, system->n );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->L, y, x, workspace->L->n, LOWER );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Dual_Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Dual_Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, system->n, x );
#endif
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
                            case JACOBI_PC:
                                dual_jacobi_app( workspace->Hdia_inv, y, x, system->n );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->L, y, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Dual_Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Dual_Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, system->n, x );
#endif
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
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  for ( i = 0; i < workspace->H->n; ++i )
//                                  {
//                                      workspace->y_p[i] = y[i];
//                                  }
//
//                                  permute_vector( workspace, workspace->y_p, workspace->H->n, FALSE, LOWER );
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->L, workspace->y_p, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                // construct D^{-1}_L
//                                if ( fresh_pre == TRUE )
//                                {
//                                    for ( i = 0; i < workspace->L->n; ++i )
//                                    {
//                                        si = workspace->L->start[i + 1] - 1;
//                                        workspace->Dinv_L[i] = 1.0 / workspace->L->val[si];
//                                    }
//                                }
//
//                                jacobi_iter( workspace, workspace->L, workspace->Dinv_L,
//                                        y, x, LOWER, control->cm_solver_pre_app_jacobi_iters );
//                                break;
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
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy_rvec2( x, y, system->n );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->U, y, x, workspace->U->n, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_LEVEL_SCHED_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy_rvec2( x, y, system->n );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_GC_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  permute_vector( workspace, x, workspace->H->n, TRUE, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  if ( fresh_pre == TRUE )
//                                  {
//                                      for ( i = 0; i < workspace->U->n; ++i )
//                                      {
//                                          si = workspace->U->start[i];
//                                          workspace->Dinv_U[i] = 1.0 / workspace->U->val[si];
//                                      }
//                                  }
//
//                                  jacobi_iter( workspace, workspace->U, workspace->Dinv_U,
//                                          y, x, UPPER, control->cm_solver_pre_app_jacobi_iters );
//                                  break;
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


/* Apply left-sided preconditioning while solving M^{-1}Ax = M^{-1}b
 *
 * system:
 * workspace: data struct containing matrices and vectors, stored in CSR
 * control: data struct containing parameters
 * data: struct containing timing simulation data (including performance data)
 * y: vector to which to apply preconditioning,
 *  specific to internals of iterative solver being used
 * x (output): preconditioned vector
 * fresh_pre: parameter indicating if this is a newly computed (fresh) preconditioner
 * side: used in determining how to apply preconditioner if the preconditioner is
 *  factorized as M = M_{1}M_{2} (e.g., incomplete LU, A \approx LU)
 *
 * Assumptions:
 *   Matrices have non-zero diagonals
 *   Each row of a matrix has at least one non-zero (i.e., no rows with all zeros) */
static void apply_preconditioner( reax_system const * const system,
        storage const * const workspace, control_params const * const control,
        simulation_data * const data, mpi_datatypes * const  mpi_data,
        real const * const y, real * const x, int fresh_pre, int side )
{
//    int i, si;

    /* no preconditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        if ( x != y )
        {
            Vector_Copy( x, y, system->n );
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
                            case JACOBI_PC:
                                jacobi_app( workspace->Hdia_inv, y, x, system->n );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->L, y, x, workspace->L->n, LOWER );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, system->n, x );
#endif
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
                            case JACOBI_PC:
                                jacobi_app( workspace->Hdia_inv, y, x, system->n );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->L, y, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Sparse_MatVec( system, control, data, mpi_data, &workspace->H_app_inv,
                                        y, system->n, x );
#endif
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
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  for ( i = 0; i < workspace->H->n; ++i )
//                                  {
//                                      workspace->y_p[i] = y[i];
//                                  }
//
//                                  permute_vector( workspace, workspace->y_p, workspace->H->n, FALSE, LOWER );
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->L, workspace->y_p, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                // construct D^{-1}_L
//                                if ( fresh_pre == TRUE )
//                                {
//                                    for ( i = 0; i < workspace->L->n; ++i )
//                                    {
//                                        si = workspace->L->start[i + 1] - 1;
//                                        workspace->Dinv_L[i] = 1.0 / workspace->L->val[si];
//                                    }
//                                }
//
//                                jacobi_iter( workspace, workspace->L, workspace->Dinv_L,
//                                        y, x, LOWER, control->cm_solver_pre_app_jacobi_iters );
//                                break;
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
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy( x, y, system->n );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->U, y, x, workspace->U->n, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_LEVEL_SCHED_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy( x, y, system->n );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_GC_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  permute_vector( workspace, x, workspace->H->n, TRUE, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  if ( fresh_pre == TRUE )
//                                  {
//                                      for ( i = 0; i < workspace->U->n; ++i )
//                                      {
//                                          si = workspace->U->start[i];
//                                          workspace->Dinv_U[i] = 1.0 / workspace->U->val[si];
//                                      }
//                                  }
//
//                                  jacobi_iter( workspace, workspace->U, workspace->Dinv_U,
//                                          y, x, UPPER, control->cm_solver_pre_app_jacobi_iters );
//                                  break;
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


/* Steepest Descent 
 * This function performs dual iteration for QEq (2 simultaneous solves)
 * */
int dual_SDM( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix * const H, rvec2 * const b, real tol,
        rvec2 * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    rvec2 tmp, alpha, bnorm, sig;
    real redux[4];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->q2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->q2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->q2, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r2,
            workspace->q2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q2,
            workspace->d2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Dot_local_rvec2( b, b, system->n, &redux[0], &redux[1] );
    Dot_local_rvec2( workspace->r2, workspace->d2, system->n, &redux[2], &redux[3] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    bnorm[0] = SQRT( redux[0] );
    bnorm[1] = SQRT( redux[1] );
    sig[0] = redux[2];
    sig[1] = redux[3];

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( SQRT(sig[0]) / bnorm[0] <= tol || SQRT(sig[1]) / bnorm[1] <= tol )
        {
            break;
        }

#if defined(NEUTRAL_TERRITORY)
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->d2,
                H->NT, workspace->q2 );
#else
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->d2,
                system->N, workspace->q2 );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace->r2, workspace->d2, system->n, &redux[0], &redux[1] );
        Dot_local_rvec2( workspace->d2, workspace->q2, system->n, &redux[2], &redux[3] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig[0] = redux[0];
        sig[1] = redux[1];
        tmp[0] = redux[2];
        tmp[1] = redux[3];
        alpha[0] = sig[0] / tmp[0];
        alpha[1] = sig[1] / tmp[1];
        Vector_Add_rvec2( x, alpha[0], alpha[1], workspace->d2, system->n );
        Vector_Add_rvec2( workspace->r2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->q2, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r2,
                workspace->q2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q2,
                workspace->d2, FALSE, RIGHT );
    }

    /* continue to solve the system that has not converged yet */
    if ( sig[0] / bnorm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->s, workspace->x, 0, system->n );

        i += SDM( system, control, data, workspace,
                H, workspace->b_s, tol, workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->s, 0, system->n );
    }
    else if ( sig[1] / bnorm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->t, workspace->x, 1, system->n );

        i += SDM( system, control, data, workspace,
                H, workspace->b_t, tol, workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->t, 1, system->n );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] SDM convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual error (s solve): %f\n", SQRT(sig[0]) / bnorm[0] );
        fprintf( stderr, "  [INFO] Rel. residual error (t solve): %f\n", SQRT(sig[1]) / bnorm[1] );
        return i;
    }

    return i;
}


/* Steepest Descent */
int SDM( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix * const H, real * const b, real tol,
        real * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    real tmp, alpha, bnorm, sig;
    real redux[2];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->q );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->q );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->r, 1.0,  b, -1.0, workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r,
            workspace->q, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q,
            workspace->d, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    redux[0] = Dot_local( b, b, system->n );
    redux[1] = Dot_local( workspace->r, workspace->d, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    bnorm = SQRT( redux[0] );
    sig = redux[1];

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && SQRT(sig) / bnorm > tol; ++i )
    {
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->d,
                H->NT, workspace->q );
#else
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->d,
                system->N, workspace->q );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace->r, workspace->d, system->n );
        redux[1] = Dot_local( workspace->d, workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig = redux[0];
        tmp = redux[1];
        alpha = sig / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -1.0 * alpha, workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r,
                workspace->q, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q,
                workspace->d, FALSE, RIGHT );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] SDM convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", SQRT(sig) / bnorm );
        return i;
    }

    return i;
}


/* Dual iteration of the Preconditioned Conjugate Gradient Method
 * for QEq (2 simultaneous solves) */
int dual_CG( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, rvec2 * const b,
        real tol, rvec2 * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    rvec2 tmp, alpha, beta, r_norm, b_norm, sig_old, sig_new;
    real redux[6];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x,
            H->NT, workspace->q2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x,
            system->N, workspace->q2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, b, -1.0, -1.0, workspace->q2, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r2,
            workspace->q2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q2,
            workspace->d2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    for ( j = 0; j < 6; ++j )
    {
        redux[j] = 0.0;
    }

    Dot_local_rvec2( workspace->r2, workspace->d2, system->n, &redux[0], &redux[1] );
    Dot_local_rvec2( workspace->d2, workspace->d2, system->n, &redux[2], &redux[3] );
    Dot_local_rvec2( b, b, system->n, &redux[4], &redux[5] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    sig_new[0] = redux[0];
    sig_new[1] = redux[1];
    r_norm[0] = SQRT( redux[2] );
    r_norm[1] = SQRT( redux[3] );
    b_norm[0] = SQRT( redux[4] );
    b_norm[1] = SQRT( redux[5] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

#if defined(NEUTRAL_TERRITORY)
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->d2,
                H->NT, workspace->q2 );
#else
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->d2,
                system->N, workspace->q2 );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = 0.0;
        redux[1] = 0.0;
        Dot_local_rvec2( workspace->d2, workspace->q2, system->n, &redux[0], &redux[1] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( &redux, &tmp, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        Vector_Add_rvec2( x, alpha[0], alpha[1], workspace->d2, system->n );
        Vector_Add_rvec2( workspace->r2, -1.0 * alpha[0], -1.0 * alpha[1],
                workspace->q2, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r2,
                workspace->q2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q2,
                workspace->p2, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = 0.0;
        redux[1] = 0.0;
        redux[2] = 0.0;
        redux[3] = 0.0;
        Dot_local_rvec2( workspace->r2, workspace->p2, system->n, &redux[0], &redux[1] );
        Dot_local_rvec2( workspace->p2, workspace->p2, system->n, &redux[2], &redux[3] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif
        
        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        sig_new[0] = redux[0];
        sig_new[1] = redux[1];
        r_norm[0] = SQRT( redux[2] );
        r_norm[1] = SQRT( redux[3] );
        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        Vector_Sum_rvec2( workspace->d2, 1.0, 1.0, workspace->p2, beta[0], beta[1],
                workspace->d2, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    /* continue to solve the system that has not converged yet */
    if ( r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->s, workspace->x, 0, system->n );

        i += CG( system, control, data, workspace,
                H, workspace->b_s, tol, workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->s, 0, system->n );
    }
    else if ( r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->t, workspace->x, 1, system->n );

        i += CG( system, control, data, workspace,
                H, workspace->b_t, tol, workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->t, 1, system->n );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] CG convergence failed!\n" );
        return i;
    }

    return i;

}


/* Preconditioned Conjugate Gradient Method */
int CG( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, real * const b,
        real tol, real * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    real tmp, alpha, beta, r_norm, b_norm;
    real sig_old, sig_new;
    real redux[3];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->q );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->q );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->r, 1.0, b, -1.0, workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r,
            workspace->q, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q,
            workspace->d, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    redux[0] = Dot_local( workspace->r, workspace->d, system->n );
    redux[1] = Dot_local( workspace->d, workspace->d, system->n );
    redux[2] = Dot_local( b, b, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    sig_new = redux[0];
    r_norm = SQRT( redux[1] );
    b_norm = SQRT( redux[2] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->d, 
                H->NT, workspace->q );
#else
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->d, 
                system->N, workspace->q );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        tmp = Parallel_Dot( workspace->d, workspace->q, system->n, MPI_COMM_WORLD );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -1.0 * alpha, workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r,
                workspace->q, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q,
                workspace->p, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace->r, workspace->p, system->n );
        redux[1] = Dot_local( workspace->p, workspace->p, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig_old = sig_new;
        sig_new = redux[0];
        r_norm = SQRT( redux[1] );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1.0, workspace->p, beta, workspace->d, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] CG convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual error: %e\n", r_norm / b_norm );
        return i;
    }

    return i;
}


/* Bi-conjugate gradient stabalized method with left preconditioning for
 * solving nonsymmetric linear systems.
 * This function performs dual iteration for QEq (2 simultaneous solves)
 *
 * system: 
 * workspace: struct containing storage for workspace for the linear solver
 * control: struct containing parameters governing the simulation and numeric methods
 * data: struct containing simulation data (e.g., atom info)
 * H: sparse, symmetric matrix, lower half stored in CSR format
 * b: right-hand side of the linear system
 * tol: tolerence compared against the relative residual for determining convergence
 * x: inital guess
 * mpi_data: 
 *
 * Reference: Netlib (in MATLAB)
 *  http://www.netlib.org/templates/matlab/bicgstab.m
 * */
int dual_BiCGStab( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, rvec2 * const b,
        real tol, rvec2 * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    rvec2 tmp, alpha, beta, omega, sigma, rho, rho_old, r_norm, b_norm;
    real time, redux[4];

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->d2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->d2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, b, -1.0, -1.0, workspace->d2, system->n );
    Dot_local_rvec2( b, b, system->n, &redux[0], &redux[1] );
    Dot_local_rvec2( workspace->r2, workspace->r2, system->n, &redux[2], &redux[3] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    b_norm[0] = SQRT( redux[0] );
    b_norm[1] = SQRT( redux[1] );
    r_norm[0] = SQRT( redux[2] );
    r_norm[1] = SQRT( redux[3] );
    if ( b_norm[0] == 0.0 )
    {
        b_norm[0] = 1.0;
    }
    if ( b_norm[1] == 0.0 )
    {
        b_norm[1] = 1.0;
    }
    Vector_Copy_rvec2( workspace->r_hat2, workspace->r2, system->n );
    omega[0] = 1.0;
    omega[1] = 1.0;
    rho[0] = 1.0;
    rho[1] = 1.0;

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        Dot_local_rvec2( workspace->r_hat2, workspace->r2, system->n,
                &redux[0], &redux[1] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        rho[0] = redux[0];
        rho[1] = redux[1];
        if ( rho[0] == 0.0 || rho[1] == 0.0 )
        {
            break;
        }
        if ( i > 0 )
        {
            beta[0] = (rho[0] / rho_old[0]) * (alpha[0] / omega[0]);
            beta[1] = (rho[1] / rho_old[1]) * (alpha[1] / omega[1]);
            Vector_Sum_rvec2( workspace->q2, 1.0, 1.0, workspace->p2,
                    -1.0 * omega[0], -1.0 * omega[1], workspace->z2, system->n );
            Vector_Sum_rvec2( workspace->p2, 1.0, 1.0, workspace->r2,
                    beta[0], beta[1], workspace->q2, system->n );
        }
        else
        {
            Vector_Copy_rvec2( workspace->p2, workspace->r2, system->n );
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->p2,
                workspace->y2, i == 0 ? fresh_pre : FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->y2,
                workspace->d2, i == 0 ? fresh_pre : FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->d2,
                H->NT, workspace->z2 );
#else
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->d2,
                system->N, workspace->z2 );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace->r_hat2, workspace->z2, system->n,
                &redux[0], &redux[1] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp[0] = redux[0];
        tmp[1] = redux[1];
        alpha[0] = rho[0] / tmp[0];
        alpha[1] = rho[1] / tmp[1];
        Vector_Sum_rvec2( workspace->q2, 1.0, 1.0, workspace->r2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->z2, system->n );
        Dot_local_rvec2( workspace->q2, workspace->q2, system->n,
                &redux[0], &redux[1] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp[0] = redux[0];
        tmp[1] = redux[1];
        /* early convergence check */
        if ( tmp[0] < tol || tmp[1] < tol )
        {
            Vector_Add_rvec2( x, alpha[0], alpha[1], workspace->d2, system->n );
            break;
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q2,
                workspace->y2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->y2,
                workspace->q_hat2, FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->q_hat2,
                H->NT, workspace->y2 );
#else
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->q_hat2,
                system->N, workspace->y2 );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace->y2, workspace->q2, system->n, &redux[0], &redux[1] );
        Dot_local_rvec2( workspace->y2, workspace->y2, system->n, &redux[2], &redux[3] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sigma[0] = redux[0];
        sigma[1] = redux[1];
        tmp[0] = redux[2];
        tmp[1] = redux[3];
        omega[0] = sigma[0] / tmp[0];
        omega[1] = sigma[1] / tmp[1];
        Vector_Sum_rvec2( workspace->g2, alpha[0], alpha[1], workspace->d2,
                omega[0], omega[1], workspace->q_hat2, system->n );
        Vector_Add_rvec2( x, 1.0, 1.0, workspace->g2, system->n );
        Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, workspace->q2,
                -1.0 * omega[0], -1.0 * omega[1], workspace->y2, system->n );
        Dot_local_rvec2( workspace->r2, workspace->r2, system->n, &redux[0], &redux[1] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        r_norm[0] = SQRT( redux[0] );
        r_norm[1] = SQRT( redux[1] );
        if ( omega[0] == 0.0 || omega[1] == 0.0 )
        {
            break;
        }
        rho_old[0] = rho[0];
        rho_old[1] = rho[1];

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( (omega[0] == 0.0 || omega[1] == 0.0) && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab numeric breakdown (%d iters)\n", i );
        fprintf( stderr, "  [INFO] omega = %e\n", omega );
    }
    else if ( (rho[0] == 0.0 || rho[1] == 0.0) && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab numeric breakdown (%d iters)\n", i );
        fprintf( stderr, "  [INFO] rho = %e\n", rho );
    }

    /* continue to solve the system that has not converged yet */
    if ( r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->s, workspace->x, 0, system->n );

        i += BiCGStab( system, control, data, workspace,
                H, workspace->b_s, tol, workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->s, 0, system->n );
    }
    else if ( r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->t, workspace->x, 1, system->n );

        i += BiCGStab( system, control, data, workspace,
                H, workspace->b_t, tol, workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->t, 1, system->n );
    }


    if ( i >= control->cm_solver_max_iters
            && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual error (s solve): %e\n", r_norm[0] / b_norm[0] );
        fprintf( stderr, "  [INFO] Rel. residual error (t solve): %e\n", r_norm[1] / b_norm[1] );
    }

    return i;
}


/* Bi-conjugate gradient stabalized method with left preconditioning for
 * solving nonsymmetric linear systems
 *
 * system: 
 * workspace: struct containing storage for workspace for the linear solver
 * control: struct containing parameters governing the simulation and numeric methods
 * data: struct containing simulation data (e.g., atom info)
 * H: sparse, symmetric matrix, lower half stored in CSR format
 * b: right-hand side of the linear system
 * tol: tolerence compared against the relative residual for determining convergence
 * x: inital guess
 * mpi_data: 
 *
 * Reference: Netlib (in MATLAB)
 *  http://www.netlib.org/templates/matlab/bicgstab.m
 * */
int BiCGStab( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, real * const b,
        real tol, real * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    real tmp, alpha, beta, omega, sigma, rho, rho_old, r_norm, b_norm;
    real time, redux[2];

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->d );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->d );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->r, 1.0,  b, -1.0, workspace->d, system->n );
    redux[0] = Dot_local( b, b, system->n );
    redux[1] = Dot_local( workspace->r, workspace->r, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    b_norm = SQRT( redux[0] );
    r_norm = SQRT( redux[1] );
    if ( b_norm == 0.0 )
    {
        b_norm = 1.0;
    }
    Vector_Copy( workspace->r_hat, workspace->r, system->n );
    omega = 1.0;
    rho = 1.0;

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        redux[0] = Dot_local( workspace->r_hat, workspace->r, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        rho = redux[0];
        if ( rho == 0.0 )
        {
            break;
        }
        if ( i > 0 )
        {
            beta = (rho / rho_old) * (alpha / omega);
            Vector_Sum( workspace->q, 1.0, workspace->p, -1.0 * omega, workspace->z, system->n );
            Vector_Sum( workspace->p, 1.0, workspace->r, beta, workspace->q, system->n );
        }
        else
        {
            Vector_Copy( workspace->p, workspace->r, system->n );
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->p,
                workspace->y, i == 0 ? fresh_pre : FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->y,
                workspace->d, i == 0 ? fresh_pre : FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->d,
                H->NT, workspace->z );
#else
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->d,
                system->N, workspace->z );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace->r_hat, workspace->z, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp = redux[0];
        alpha = rho / tmp;
        Vector_Sum( workspace->q, 1.0, workspace->r, -1.0 * alpha, workspace->z, system->n );
        redux[0] = Dot_local( workspace->q, workspace->q, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp = redux[0];
        /* early convergence check */
        if ( tmp < tol )
        {
            Vector_Add( x, alpha, workspace->d, system->n );
            break;
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->q,
                workspace->y, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->y,
                workspace->q_hat, FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->q_hat,
                H->NT, workspace->y );
#else
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->q_hat,
                system->N, workspace->y );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace->y, workspace->q, system->n );
        redux[1] = Dot_local( workspace->y, workspace->y, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sigma = redux[0];
        tmp = redux[1];
        omega = sigma / tmp;
        Vector_Sum( workspace->g, alpha, workspace->d, omega, workspace->q_hat, system->n );
        Vector_Add( x, 1.0, workspace->g, system->n );
        Vector_Sum( workspace->r, 1.0, workspace->q, -1.0 * omega, workspace->y, system->n );
        redux[0] = Dot_local( workspace->r, workspace->r, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        r_norm = SQRT( redux[0] );
        if ( omega == 0.0 )
        {
            break;
        }
        rho_old = rho;

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( omega == 0.0 && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab numeric breakdown (%d iters)\n", i );
        fprintf( stderr, "  [INFO] omega = %e\n", omega );
    }
    else if ( rho == 0.0 && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab numeric breakdown (%d iters)\n", i );
        fprintf( stderr, "  [INFO] rho = %e\n", rho );
    }
    else if ( i >= control->cm_solver_max_iters
            && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual error: %e\n", r_norm / b_norm );
    }

    return i;
}


/* Dual iteration for the Pipelined Preconditioned Conjugate Gradient Method
 * for QEq (2 simaltaneous solves)
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 * 2) Scalable Non-blocking Preconditioned Conjugate Gradient Methods,
 *  Paul R. Eller and William Gropp, SC '16 Proceedings of the International Conference
 *  for High Performance Computing, Networking, Storage and Analysis, 2016.
 *  */
int dual_PIPECG( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, rvec2 * const b,
        real tol, rvec2 * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    rvec2 alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[8];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x,
            H->NT, workspace->u2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x,
            system->N, workspace->u2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, b, -1.0, -1.0, workspace->u2, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r2,
            workspace->m2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->m2,
            workspace->u2, fresh_pre, RIGHT );

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->u2,
            H->NT, workspace->w2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->u2,
            system->N, workspace->w2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    for ( j = 0; j < 8; ++j )
    {
        redux[j] = 0.0;
    }
    Dot_local_rvec2( workspace->w2, workspace->u2, system->n, &redux[0], &redux[1] );
    Dot_local_rvec2( workspace->r2, workspace->u2, system->n, &redux[2], &redux[3] );
    Dot_local_rvec2( workspace->u2, workspace->u2, system->n, &redux[4], &redux[5] );
    Dot_local_rvec2( b, b, system->n, &redux[6], &redux[7] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 8, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->w2,
            workspace->n2, FALSE, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n2,
            workspace->m2, FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->m2,
            H->NT, workspace->n2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->m2,
            system->N, workspace->n2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    delta[0] = redux[0];
    delta[1] = redux[1];
    gamma_new[0] = redux[2];
    gamma_new[1] = redux[3];
    r_norm[0] = SQRT( redux[4] );
    r_norm[1] = SQRT( redux[5] );
    b_norm[0] = SQRT( redux[6] );
    b_norm[1] = SQRT( redux[7] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }
        if ( i > 0 )
        {
            beta[0] = gamma_new[0] / gamma_old[0];
            beta[1] = gamma_new[1] / gamma_old[1];
            alpha[0] = gamma_new[0] / (delta[0] - beta[0] / alpha[0] * gamma_new[0]);
            alpha[1] = gamma_new[1] / (delta[1] - beta[1] / alpha[1] * gamma_new[1]);
        }
        else
        {
            beta[0] = 0.0;
            beta[1] = 0.0;
            alpha[0] = gamma_new[0] / delta[0];
            alpha[1] = gamma_new[1] / delta[1];
        }

        Vector_Sum_rvec2( workspace->z2, 1.0, 1.0, workspace->n2,
                beta[0], beta[1], workspace->z2, system->n );
        Vector_Sum_rvec2( workspace->q2, 1.0, 1.0, workspace->m2,
                beta[0], beta[1], workspace->q2, system->n );
        Vector_Sum_rvec2( workspace->p2, 1.0, 1.0, workspace->u2,
                beta[0], beta[1], workspace->p2, system->n );
        Vector_Sum_rvec2( workspace->d2, 1.0, 1.0, workspace->w2,
                beta[0], beta[1], workspace->d2, system->n );
        Vector_Sum_rvec2( x, 1.0, 1.0, x,
                alpha[0], alpha[1], workspace->p2, system->n );
        Vector_Sum_rvec2( workspace->u2, 1.0, 1.0, workspace->u2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->q2, system->n );
        Vector_Sum_rvec2( workspace->w2, 1.0, 1.0, workspace->w2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->z2, system->n );
        Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, workspace->r2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d2, system->n );

        for ( j = 0; j < 6; ++j )
        {
            redux[j] = 0.0;
        }
        Dot_local_rvec2( workspace->w2, workspace->u2, system->n, &redux[0], &redux[1] );
        Dot_local_rvec2( workspace->r2, workspace->u2, system->n, &redux[2], &redux[3] );
        Dot_local_rvec2( workspace->u2, workspace->u2, system->n, &redux[4], &redux[5] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->w2,
                workspace->n2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n2,
                workspace->m2, FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->m2,
                H->NT, workspace->n2 );
#else
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->m2,
                system->N, workspace->n2 );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        gamma_old[0] = gamma_new[0];
        gamma_old[1] = gamma_new[1];

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        delta[0] = redux[0];
        delta[1] = redux[1];
        gamma_new[0] = redux[2];
        gamma_new[1] = redux[3];
        r_norm[0] = SQRT( redux[4] );
        r_norm[1] = SQRT( redux[5] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif
    }

    /* continue to solve the system that has not converged yet */
    if ( r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->s, workspace->x, 0, system->n );

        i += PIPECG( system, control, data, workspace,
                H, workspace->b_s, tol, workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->s, 0, system->n );
    }
    else if ( r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->t, workspace->x, 1, system->n );

        i += PIPECG( system, control, data, workspace,
                H, workspace->b_t, tol, workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->t, 1, system->n );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] PIPECG convergence failed!\n" );
        return i;
    }

    return i;
}


/* Pipelined Preconditioned Conjugate Gradient Method
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 * 2) Scalable Non-blocking Preconditioned Conjugate Gradient Methods,
 *  Paul R. Eller and William Gropp, SC '16 Proceedings of the International Conference
 *  for High Performance Computing, Networking, Storage and Analysis, 2016.
 *  */
int PIPECG( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, real * const b,
        real tol, real * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    real alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[4];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->u );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->u );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->r, 1.0, b, -1.0, workspace->u, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r,
            workspace->m, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->m,
            workspace->u, fresh_pre, RIGHT );

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, workspace->u,
            H->NT, workspace->w );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, workspace->u,
            system->N, workspace->w );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    redux[0] = Dot_local( workspace->w, workspace->u, system->n );
    redux[1] = Dot_local( workspace->r, workspace->u, system->n );
    redux[2] = Dot_local( workspace->u, workspace->u, system->n );
    redux[3] = Dot_local( b, b, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->w,
            workspace->n, FALSE, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n,
            workspace->m, FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, workspace->m,
            H->NT, workspace->n );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, workspace->m,
            system->N, workspace->n );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    delta = redux[0];
    gamma_new = redux[1];
    r_norm = SQRT( redux[2] );
    b_norm = SQRT( redux[3] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        if ( i > 0 )
        {
            beta = gamma_new / gamma_old;
            alpha = gamma_new / (delta - beta / alpha * gamma_new);
        }
        else
        {
            beta = 0.0;
            alpha = gamma_new / delta;
        }

        Vector_Sum( workspace->z, 1.0, workspace->n, beta, workspace->z, system->n );
        Vector_Sum( workspace->q, 1.0, workspace->m, beta, workspace->q, system->n );
        Vector_Sum( workspace->p, 1.0, workspace->u, beta, workspace->p, system->n );
        Vector_Sum( workspace->d, 1.0, workspace->w, beta, workspace->d, system->n );
        Vector_Sum( x, 1.0, x, alpha, workspace->p, system->n );
        Vector_Sum( workspace->u, 1.0, workspace->u, -1.0 * alpha, workspace->q, system->n );
        Vector_Sum( workspace->w, 1.0, workspace->w, -1.0 * alpha, workspace->z, system->n );
        Vector_Sum( workspace->r, 1.0, workspace->r, -1.0 * alpha, workspace->d, system->n );
        redux[0] = Dot_local( workspace->w, workspace->u, system->n );
        redux[1] = Dot_local( workspace->r, workspace->u, system->n );
        redux[2] = Dot_local( workspace->u, workspace->u, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->w,
                workspace->n, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n,
                workspace->m, FALSE, RIGHT );

#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->m,
                H->NT, workspace->n );
#else
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->m,
                system->N, workspace->n );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        gamma_old = gamma_new;

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        delta = redux[0];
        gamma_new = redux[1];
        r_norm = SQRT( redux[2] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] PIPECG convergence failed!\n" );
        return i;
    }

    return i;
}


/* Pipelined Preconditioned Conjugate Residual Method.
 * This function performs dual iteration for QEq (2 simultaneous solves)
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 *  */
int dual_PIPECR( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, rvec2 * const b,
        real tol, rvec2 * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    rvec2 alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[6];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->u2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->u2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, b, -1.0, -1.0, workspace->u2, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r2,
            workspace->n2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n2,
            workspace->u2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Dot_local_rvec2( b, b, system->n, &redux[0], &redux[1] );
    Dot_local_rvec2( workspace->u2, workspace->u2, system->n, &redux[2], &redux[3] );

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

#if defined(NEUTRAL_TERRITORY)
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->u2,
            H->NT, workspace->w2 );
#else
    Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->u2,
            system->N, workspace->w2 );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    b_norm[0] = SQRT( redux[0] );
    b_norm[1] = SQRT( redux[1] );
    r_norm[0] = SQRT( redux[2] );
    r_norm[1] = SQRT( redux[3] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->w2,
                workspace->n2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n2,
                workspace->m2, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace->w2, workspace->u2, system->n, &redux[0], &redux[1] );
        Dot_local_rvec2( workspace->m2, workspace->w2, system->n, &redux[2], &redux[3] );
        Dot_local_rvec2( workspace->u2, workspace->u2, system->n, &redux[4], &redux[5] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(NEUTRAL_TERRITORY)
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->m2,
                H->NT, workspace->n2 );
#else
        Dual_Sparse_MatVec( system, control, data, mpi_data, H, workspace->m2,
                system->N, workspace->n2 );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        gamma_new[0] = redux[0];
        gamma_new[1] = redux[1];
        delta[0] = redux[2];
        delta[1] = redux[3];
        r_norm[0] = SQRT( redux[4] );
        r_norm[1] = SQRT( redux[5] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        if ( i > 0 )
        {
            beta[0] = gamma_new[0] / gamma_old[0];
            beta[1] = gamma_new[1] / gamma_old[1];
            alpha[0] = gamma_new[0] / (delta[0] - beta[0] / alpha[0] * gamma_new[0]);
            alpha[1] = gamma_new[1] / (delta[1] - beta[1] / alpha[1] * gamma_new[1]);
        }
        else
        {
            beta[0] = 0.0;
            beta[1] = 0.0;
            alpha[0] = gamma_new[0] / delta[0];
            alpha[1] = gamma_new[1] / delta[1];
        }

        Vector_Sum_rvec2( workspace->z2, 1.0, 1.0, workspace->n2,
                beta[0], beta[1], workspace->z2, system->n );
        Vector_Sum_rvec2( workspace->q2, 1.0, 1.0, workspace->m2,
                beta[0], beta[1], workspace->q2, system->n );
        Vector_Sum_rvec2( workspace->p2, 1.0, 1.0, workspace->u2,
                beta[0], beta[1], workspace->p2, system->n );
        Vector_Sum_rvec2( workspace->d2, 1.0, 1.0, workspace->w2,
                beta[0], beta[1], workspace->d2, system->n );
        Vector_Sum_rvec2( x, 1.0, 1.0, x, alpha[0], alpha[1], workspace->p2, system->n );
        Vector_Sum_rvec2( workspace->u2, 1.0, 1.0, workspace->u2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->q2, system->n );
        Vector_Sum_rvec2( workspace->w2, 1.0, 1.0, workspace->w2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->z2, system->n );
        Vector_Sum_rvec2( workspace->r2, 1.0, 1.0, workspace->r2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d2, system->n );

        gamma_old[0] = gamma_new[0];
        gamma_old[1] = gamma_new[1];

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    /* continue to solve the system that has not converged yet */
    if ( r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->s, workspace->x, 0, system->n );

        i += PIPECR( system, control, data, workspace,
                H, workspace->b_s, tol, workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->s, 0, system->n );
    }
    else if ( r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->t, workspace->x, 1, system->n );

        i += PIPECR( system, control, data, workspace,
                H, workspace->b_t, tol, workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->x, workspace->t, 1, system->n );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] PIPECR convergence failed!\n" );
        return i;
    }

    return i;
}


/* Pipelined Preconditioned Conjugate Residual Method
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 *  */
int PIPECR( reax_system const * const system, control_params const * const control,
        simulation_data * const data,
        storage * const workspace, sparse_matrix * const H, real * const b,
        real tol, real * const x, mpi_datatypes * const  mpi_data, int fresh_pre )
{
    int i, j, ret;
    real alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[3];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            H->NT, workspace->u );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, x, 
            system->N, workspace->u );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->r, 1.0, b, -1.0, workspace->u, system->n );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->r,
            workspace->n, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n,
            workspace->u, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    redux[0] = Dot_local( b, b, system->n );
    redux[1] = Dot_local( workspace->u, workspace->u, system->n );

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( system, control, data, mpi_data, H, workspace->u, 
            H->NT, workspace->w );
#else
    Sparse_MatVec( system, control, data, mpi_data, H, workspace->u, 
            system->N, workspace->w );
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    b_norm = SQRT( redux[0] );
    r_norm = SQRT( redux[1] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->w,
                workspace->n, fresh_pre, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data, workspace->n,
                workspace->m, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace->w, workspace->u, system->n );
        redux[1] = Dot_local( workspace->m, workspace->w, system->n );
        redux[2] = Dot_local( workspace->u, workspace->u, system->n );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->m, 
                H->NT, workspace->n );
#else
        Sparse_MatVec( system, control, data, mpi_data, H, workspace->m, 
                system->N, workspace->n );
#endif

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        gamma_new = redux[0];
        delta = redux[1];
        r_norm = SQRT( redux[2] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        if ( i > 0 )
        {
            beta = gamma_new / gamma_old;
            alpha = gamma_new / (delta - beta / alpha * gamma_new);
        }
        else
        {
            beta = 0.0;
            alpha = gamma_new / delta;
        }

        Vector_Sum( workspace->z, 1.0, workspace->n, beta, workspace->z, system->n );
        Vector_Sum( workspace->q, 1.0, workspace->m, beta, workspace->q, system->n );
        Vector_Sum( workspace->p, 1.0, workspace->u, beta, workspace->p, system->n );
        Vector_Sum( workspace->d, 1.0, workspace->w, beta, workspace->d, system->n );
        Vector_Sum( x, 1.0, x, alpha, workspace->p, system->n );
        Vector_Sum( workspace->u, 1.0, workspace->u, -1.0 * alpha, workspace->q, system->n );
        Vector_Sum( workspace->w, 1.0, workspace->w, -1.0 * alpha, workspace->z, system->n );
        Vector_Sum( workspace->r, 1.0, workspace->r, -1.0 * alpha, workspace->d, system->n );

        gamma_old = gamma_new;

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] PIPECR convergence failed!\n" );
        return i;
    }

    return i;
}
