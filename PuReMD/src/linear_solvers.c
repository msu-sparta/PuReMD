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

#include "linear_solvers.h"
#include "basic_comm.h"
#include "io_tools.h"
#include "tool_box.h"
#include "vector.h"
#include "allocate.h"

/* Intel MKL */
#if defined(HAVE_LAPACKE_MKL)
#include "mkl.h"
/* reference LAPACK */
#elif defined(HAVE_LAPACKE)
#include "lapacke.h"
#endif

/*#if defined(CG_PERFORMANCE)
real t_start, t_elapsed, matvec_time, dot_time;
#endif*/


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


static void dual_Sparse_MatVec( sparse_matrix *A, rvec2 *x, rvec2 *b, int N )
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
            if( i < A->n )
            {
                b[i][0] += A->entries[si].val * x[i][0];
                b[i][1] += A->entries[si].val * x[i][1];
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
                j = A->entries[k].j;
                val = A->entries[k].val;

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
                j = A->entries[k].j;
                val = A->entries[k].val;

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
            b[i][0] += A->entries[si].val * x[i][0];
            b[i][1] += A->entries[si].val * x[i][1];

            for ( k = si + 1; k < A->end[i]; ++k )
            {
                j = A->entries[k].j;
                val = A->entries[k].val;

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
                j = A->entries[k].j;
                val = A->entries[k].val;

                b[i][0] += val * x[j][0];
                b[i][1] += val * x[j][1];
            }
        }
    }
#endif
}


static void Sparse_MatVec( sparse_matrix *A, real *x, real *b, int N )
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
            if( i < A->n )
            {
                b[i] += A->entries[si].val * x[i];
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
                j = A->entries[k].j;
                val = A->entries[k].val;

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
                j = A->entries[k].j;
                val = A->entries[k].val;

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

            /* diagonal only contributes once */
            b[i] += A->entries[si].val * x[i];

            for ( k = si + 1; k < A->end[i]; ++k )
            {
                j = A->entries[k].j;
                val = A->entries[k].val;

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
                j = A->entries[k].j;
                val = A->entries[k].val;

                b[i] += val * x[j];
            }
        }
    }
#endif
}


real setup_sparse_approx_inverse( reax_system *system, simulation_data *data, storage *workspace,
        mpi_datatypes *mpi_data, sparse_matrix *A, sparse_matrix **A_spar_patt,
        int nprocs, real filter )
{
    int i, bin, total, pos;
    int n, n_gather, s_local, s, n_local;
    int target_proc;
    int k;
    int pj, size;
    int left, right, p, turn;
    int num_rows;

    real threshold, pivot, tmp;
    real *input_array;
    real *samplelist_local, *samplelist;
    real *pivotlist;
    real *bucketlist_local, *bucketlist;

    int *srecv, *sdispls;
    int *scounts_local, *scounts;
    int *dspls_local, *dspls;
    int *bin_elements;

    MPI_Comm comm;

    real start, t_start, t_comm;
    real total_comm;

    start = MPI_Wtime();
    t_comm = 0.0;

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

    comm = mpi_data->world;
#if defined(NEUTRAL_TERRITORY)
    num_rows = A->NT;
    fprintf( stdout,"%d %d %d\n", A->n, A->NT, A->m );
    fflush( stdout );
#else
    num_rows = A->n;
#endif

    if ( *A_spar_patt == NULL )
    {
#if defined(NEUTRAL_TERRITORY)
        Allocate_Matrix2( A_spar_patt, A->n, A->NT, A->m,
                A->format, comm );
#else
        Allocate_Matrix2( A_spar_patt, A->n, system->local_cap, A->m,
                A->format, comm );
#endif
    }

    else /*if ( (*A_spar_patt)->m < A->m )*/
    {
        Deallocate_Matrix( *A_spar_patt );
#if defined(NEUTRAL_TERRITORY)
        Allocate_Matrix2( A_spar_patt, A->n, A->NT, A->m,
                A->format, comm );
#else
        Allocate_Matrix2( A_spar_patt, A->n, system->local_cap, A->m,
                A->format, comm );
#endif
    }

    n_local = 0;
    for( i = 0; i < num_rows; ++i )
    {
        n_local += (A->end[i] - A->start[i] + 9)/10;
    }
    s_local = (int) (12.0 * (log2(n_local) + log2(nprocs)));
    
    t_start = MPI_Wtime();
    MPI_Allreduce( &n_local, &n, 1, MPI_INT, MPI_SUM, comm );
    MPI_Reduce( &s_local, &s, 1, MPI_INT, MPI_SUM, MASTER_NODE, comm );
    t_comm += MPI_Wtime() - t_start;

    /* count num. bin elements for each processor, uniform bin sizes */
    input_array = smalloc( sizeof(real) * n_local,
           "setup_sparse_approx_inverse::input_array", MPI_COMM_WORLD );
    scounts_local = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::scounts_local", MPI_COMM_WORLD );
    scounts = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::scounts", MPI_COMM_WORLD );
    bin_elements = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::bin_elements", MPI_COMM_WORLD );
    dspls_local = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::displs_local", MPI_COMM_WORLD );
    bucketlist_local = smalloc( sizeof(real) * n_local,
          "setup_sparse_approx_inverse::bucketlist_local", MPI_COMM_WORLD );
    dspls = smalloc( sizeof(int) * nprocs,
           "setup_sparse_approx_inverse::dspls", MPI_COMM_WORLD );
    if ( nprocs > 1 )
    {
        pivotlist = smalloc( sizeof(real) *  (nprocs - 1),
                "setup_sparse_approx_inverse::pivotlist", MPI_COMM_WORLD );
    }
    samplelist_local = smalloc( sizeof(real) * s_local,
           "setup_sparse_approx_inverse::samplelist_local", MPI_COMM_WORLD );
    if ( system->my_rank == MASTER_NODE )
    {
        samplelist = smalloc( sizeof(real) * s,
               "setup_sparse_approx_inverse::samplelist", MPI_COMM_WORLD );
        srecv = smalloc( sizeof(int) * nprocs,
               "setup_sparse_approx_inverse::srecv", MPI_COMM_WORLD );
        sdispls = smalloc( sizeof(int) * nprocs,
               "setup_sparse_approx_inverse::sdispls", MPI_COMM_WORLD );
    }

    n_local = 0;
    for ( i = 0; i < num_rows; ++i )
    {
        for ( pj = A->start[i]; pj < A->end[i]; pj += 10 )
        {
            input_array[n_local++] = A->entries[pj].val;
        }
    }

    for ( i = 0; i < s_local; i++)
    {
        /* samplelist_local[i] = input_array[rand( ) % n_local]; */
        samplelist_local[i] = input_array[ i ];
    }

    /* gather samples at the root process */
    t_start = MPI_Wtime();
    MPI_Gather( &s_local, 1, MPI_INT, srecv, 1, MPI_INT, MASTER_NODE, comm );
    t_comm += MPI_Wtime() - t_start;

    if( system->my_rank == MASTER_NODE )
    {
        sdispls[0] = 0;
        for ( i = 0; i < nprocs - 1; ++i )
        {
            sdispls[i + 1] = sdispls[i] + srecv[i];
        }
    }

    t_start = MPI_Wtime();
    MPI_Gatherv( samplelist_local, s_local, MPI_DOUBLE,
            samplelist, srecv, sdispls, MPI_DOUBLE, MASTER_NODE, comm);
    t_comm += MPI_Wtime() - t_start;

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
    t_start = MPI_Wtime();
    MPI_Bcast( pivotlist, nprocs - 1, MPI_DOUBLE, MASTER_NODE, comm );
    t_comm += MPI_Wtime() - t_start;

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
    t_start = MPI_Wtime();
    MPI_Allreduce( MPI_IN_PLACE, scounts, nprocs, MPI_INT, MPI_SUM, comm );
    t_comm += MPI_Wtime() - t_start;

    /* find the target process */
    target_proc = 0;
    total = 0;
    k = n * filter;
    for (i = nprocs - 1; i >= 0; --i )
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
               "setup_sparse_approx_inverse::bucketlist", MPI_COMM_WORLD );
    }

    /* send local buckets to target processor for quickselect */
    t_start = MPI_Wtime();
    MPI_Gather( scounts_local + target_proc, 1, MPI_INT, scounts,
            1, MPI_INT, target_proc, comm );
    t_comm += MPI_Wtime() - t_start;

    if ( system->my_rank == target_proc )
    {
        dspls[0] = 0;
        for ( i = 0; i < nprocs - 1; ++i )
        {
            dspls[i + 1] = dspls[i] + scounts[i];
        }
    }

    t_start = MPI_Wtime();
    MPI_Gatherv( bucketlist_local + dspls_local[target_proc], scounts_local[target_proc], MPI_DOUBLE,
            bucketlist, scounts, dspls, MPI_DOUBLE, target_proc, comm);
    t_comm += MPI_Wtime() - t_start;

    /* apply quick select algorithm at the target process */
    if ( system->my_rank == target_proc )
    {
        left = 0;
        right = n_gather-1;

        turn = 0;
        while( k )
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

            if( p == k - 1)
            {
                threshold = bucketlist[p];
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
        /* comment out if ACKS2 and/or EE is not an option
           if(threshold < 1.000000)
           {
           threshold = 1.000001;
           } */
    }

    /* broadcast the filtering value */
    t_start = MPI_Wtime();
    MPI_Bcast( &threshold, 1, MPI_DOUBLE, target_proc, comm );
    t_comm += MPI_Wtime() - t_start;

#if defined(DEBUG_FOCUS)
    int nnz = 0;
#endif

    /* build entries of that pattern*/
    for ( i = 0; i < num_rows; ++i )
    {
        (*A_spar_patt)->start[i] = A->start[i];
        size = A->start[i];

        for ( pj = A->start[i]; pj < A->end[i]; ++pj )
        {
            if ( ( A->entries[pj].val >= threshold )  || ( A->entries[pj].j == i ) )
            {
                (*A_spar_patt)->entries[size].val = A->entries[pj].val;
                (*A_spar_patt)->entries[size].j = A->entries[pj].j;
                size++;

#if defined(DEBUG_FOCUS)
                nnz++;
#endif
            }
        }
        (*A_spar_patt)->end[i] = size;
    }

#if defined(DEBUG_FOCUS)
    MPI_Allreduce( MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, comm );
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stdout, "    [INFO] \ntotal nnz in all charge matrices = %d\ntotal nnz in all sparsity patterns = %d\nthreshold = %.15lf\n",
                n, nnz, threshold );
        fprintf( stdout, "SAI SETUP takes %.2f seconds\n", MPI_Wtime() - start );
        fflush( stdout );
    }
#endif
 
    MPI_Reduce( &t_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE,
            mpi_data->world );

    if( system->my_rank == MASTER_NODE )
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
    if ( nprocs > 1)
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

    return MPI_Wtime() - start;
}


#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
#if defined(NEUTRAL_TERRITORY)
real sparse_approx_inverse( reax_system *system, simulation_data *data,
        storage *workspace, mpi_datatypes *mpi_data, 
        sparse_matrix *A, sparse_matrix *A_spar_patt,
        sparse_matrix **A_app_inv, int nprocs )
{
    ///////////////
    int N, M, d_i, d_j;
    int i, k, pj, j_temp;
    int local_pos, atom_pos, identity_pos;
    lapack_int m, n, nrhs, lda, ldb, info;
    int *pos_x, *X;
    real *e_j, *dense_matrix;
    int cnt;
    
    reax_atom *atom;
    int *row_nnz;
    int **j_list;
    real **val_list;

    int d, count, index;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req[12];
    MPI_Status stat[12];
    neighbor_proc *nbr;
    int *j_send, *j_recv[6];
    real *val_send, *val_recv[6];
    
    real start, t_start, t_comm;
    real total_comm;
    ///////////////////
    start = MPI_Wtime();
    t_comm = 0.0;

    comm = mpi_data->world;

    if ( *A_app_inv == NULL)
    {
        //TODO: FULL_MATRIX?
        Allocate_Matrix2( A_app_inv, A_spar_patt->n, A->NT, A_spar_patt->m,
                SYM_FULL_MATRIX, comm );
    }
    
    else /* if ( (*A_app_inv)->m < A_spar_patt->m ) */
    {
        Deallocate_Matrix( *A_app_inv );
        Allocate_Matrix2( A_app_inv, A_spar_patt->n, A->NT, A_spar_patt->m,
                SYM_FULL_MATRIX, comm );
    }

    pos_x = NULL;
    X = NULL;

    row_nnz = NULL;
    j_list = NULL;
    val_list = NULL;

    j_send = NULL;
    val_send = NULL;
    for( d = 0; d < 6; ++d )
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
    t_start = MPI_Wtime();
    Dist( system, mpi_data, row_nnz, REAL_PTR_TYPE, MPI_INT );
    t_comm += MPI_Wtime() - t_start;
    fprintf( stdout,"SAI after Dist call\n");
    fflush( stdout );

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_nt_buffers;
    count = 0;

    /*  use a Dist-like approach to send the row information */
    for ( d = 0; d < 6; ++d)
    {
        /* initiate recvs */
        nbr = &(system->my_nt_nbrs[d]);
        if ( nbr->atoms_cnt )
        {
            /* calculate the total data that will be received */
            cnt = 0;
            for( i = nbr->atoms_str; i < (nbr->atoms_str + nbr->atoms_cnt); ++i )
            {
                cnt += row_nnz[i];
            }

            /* initiate Irecv */
            if( cnt )
            {
                count += 2;

                j_recv[d] = (int *) malloc( sizeof(int) * cnt );
                val_recv[d] = (real *) malloc( sizeof(real) * cnt );

                fprintf( stdout,"Dist communication receive phase direction %d will receive %d\n", d, cnt);
                fflush( stdout );
                t_start = MPI_Wtime();
                MPI_Irecv( j_recv + d, cnt, MPI_INT, nbr->receive_rank, d, comm, &req[2 * d] );
                MPI_Irecv( val_recv + d, cnt, MPI_DOUBLE, nbr->receive_rank, d, comm, &req[2 * d + 1] );
                t_comm += MPI_Wtime() - t_start;
            }
        }
    }
    /////////////////////
    for( d = 0; d < 6; ++d)
    {
        nbr = &(system->my_nt_nbrs[d]);
        /* send both messages in dimension d */
        if( out_bufs[d].cnt )
        {
            cnt = 0;
            for( i = 0; i < out_bufs[d].cnt; ++i )
            {
                cnt += A->end[ out_bufs[d].index[i] ] - A->start[ out_bufs[d].index[i] ];
                if(out_bufs[d].index[i] < 0 || out_bufs[d].index[i] >= A->n)
                {
                    fprintf( stdout, "INDEXING ERROR %d > %d\n", out_bufs[d].index[i], A->n );
                    fflush( stdout );
                }
               //     row_nnz[ out_bufs[d].index[i] ];
            }
            fprintf( stdout,"Dist communication    send phase direction %d should  send %d\n", d, cnt);
            fflush( stdout );

            if( cnt )
            {
                j_send = (int *) malloc( sizeof(int) * cnt );
                val_send = (real *) malloc( sizeof(real) * cnt );

                cnt = 0;
                for( i = 0; i < out_bufs[d].cnt; ++i )
                {
                    for( pj = A->start[ out_bufs[d].index[i] ]; pj < A->end[ out_bufs[d].index[i] ]; ++pj )
                    {
                        atom = &system->my_atoms[ A->entries[pj].j ];
                        j_send[cnt] = atom->orig_id;
                        val_send[cnt] = A->entries[pj].val;
                        cnt++;
                    }
                }

                fprintf( stdout,"Dist communication    send phase direction %d will    send %d\n", d, cnt );
                fflush( stdout );

                t_start = MPI_Wtime();
                MPI_Send( j_send, cnt, MPI_INT, nbr->rank, d, comm );
                fprintf( stdout,"Dist communication send phase direction %d cnt = %d\n", d, cnt);
                fflush( stdout );
                MPI_Send( val_send, cnt, MPI_DOUBLE, nbr->rank, d, comm );
                fprintf( stdout,"Dist communication send phase direction %d cnt = %d\n", d, cnt);
                fflush( stdout );
                t_comm += MPI_Wtime() - t_start;
            }
        }
    }
    fprintf( stdout," Dist communication for sending row info before waitany\n");
    fflush( stdout );
    ///////////////////////
    for ( d = 0; d < count; ++d )
    {
        t_start = MPI_Wtime();
        MPI_Waitany( MAX_NT_NBRS, req, &index, stat);
        t_comm += MPI_Wtime() - t_start;

        nbr = &(system->my_nt_nbrs[index/2]);
        cnt = 0;
        for( i = nbr->atoms_str; i < (nbr->atoms_str + nbr->atoms_cnt); ++i )
        {
            if( (index%2) == 0 )
            {
                j_list[i] = (int *) malloc( sizeof(int) *  row_nnz[i] );
                for( pj = 0; pj < row_nnz[i]; ++pj )
                {
                    j_list[i][pj] = j_recv[index/2][cnt];
                    cnt++;
                }
            }
            else
            {
                val_list[i] = (real *) malloc( sizeof(real) * row_nnz[i] );
                for( pj = 0; pj < row_nnz[i]; ++pj )
                {
                    val_list[i][pj] = val_recv[index/2][cnt];
                    cnt++;
                }
            }

        }
    }
    //////////////////////
    fprintf( stdout," wow wow wow, Dist communication for sending row info worked\n");
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
            j_temp = A_spar_patt->entries[pj].j;
            atom = &system->my_atoms[j_temp];
            ++N;

            /* for each of those indices
             * search through the row of full A of that index */

            /* the case where the local matrix has that index's row */
            if( j_temp < A->NT )
            {
                for ( k = A->start[ j_temp ]; k < A->end[ j_temp ]; ++k )
                {
                    /* and accumulate the nonzero column indices to serve as the row indices of the dense matrix */
                    atom = &system->my_atoms[ A->entries[k].j ];
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
            local_pos = A_spar_patt->entries[ A_spar_patt->start[i] + d_j ].j;
            if( local_pos < 0 || local_pos >= system->N )
            {
                fprintf( stderr, "THE LOCAL POSITION OF THE ATOM IS NOT VALID, STOP THE EXECUTION\n");
                fflush( stderr );

            }
            /////////////////////////////
            if( local_pos < A->NT )
            {
                for ( d_i = A->start[local_pos]; d_i < A->end[local_pos]; ++d_i )
                {
                    atom = &system->my_atoms[ A->entries[d_i].j ];
                    if (pos_x[ atom->orig_id ] >= M || d_j >=  N )
                    {
                        fprintf( stderr, "CANNOT MAP IT TO THE DENSE MATRIX, STOP THE EXECUTION, orig_id = %d, i =  %d, j = %d, M = %d N = %d\n", atom->orig_id, pos_x[ atom->orig_id ], d_j, M, N );
                        fflush( stderr );
                    }
                    if ( X[ atom->orig_id ] == 1 )
                    {
                        dense_matrix[ pos_x[ atom->orig_id ] * N + d_j ] = A->entries[d_i].val;
                    }
                }
            }
            else
            {
                for ( d_i = 0; d_i < row_nnz[ local_pos ]; ++d_i )
                {
                    if (pos_x[ j_list[local_pos][d_i] ] >= M || d_j  >= N )
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
        (*A_app_inv)->start[i] = A_spar_patt->start[i];
        (*A_app_inv)->end[i] = A_spar_patt->end[i];
        for ( k = (*A_app_inv)->start[i]; k < (*A_app_inv)->end[i]; ++k)
        {
            (*A_app_inv)->entries[k].j = A_spar_patt->entries[k].j;
            (*A_app_inv)->entries[k].val = e_j[k - A_spar_patt->start[i]];
        }
        free( dense_matrix );
        free( e_j );
    }

    free( pos_x);
    free( X );
    /////////////////////
    MPI_Reduce(&t_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);

    if( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_comm += total_comm / nprocs;
    }

    return MPI_Wtime() - start;
}


#else
real sparse_approx_inverse( reax_system *system, simulation_data *data,
        storage *workspace, mpi_datatypes *mpi_data, 
        sparse_matrix *A, sparse_matrix *A_spar_patt,
        sparse_matrix **A_app_inv, int nprocs )
{
    int N, M, d_i, d_j, mark;
    int i, k, pj, j_temp, push;
    int local_pos, atom_pos, identity_pos;
    lapack_int m, n, nrhs, lda, ldb, info;
    int *X, *q;
    real *e_j, *dense_matrix;
    int size_e, size_dense;
    int cnt;
    reax_atom *atom;
    int *row_nnz;
    int **j_list;
    real **val_list;
    int d;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2, req3, req4;
    int flag1, flag2;
    MPI_Status stat1, stat2, stat3, stat4;
    const neighbor_proc *nbr1, *nbr2;
    int *j_send, *j_recv1, *j_recv2;
    int size_send, size_recv1, size_recv2;
    real *val_send, *val_recv1, *val_recv2;
    real start, t_start, t_comm;
    real total_comm;

    start = MPI_Wtime();
    t_comm = 0.0;

    comm = mpi_data->world;

    if ( *A_app_inv == NULL)
    {
        Allocate_Matrix2( A_app_inv, A_spar_patt->n, system->local_cap, A_spar_patt->m,
                SYM_FULL_MATRIX, comm );
    }
    else /* if ( (*A_app_inv)->m < A_spar_patt->m ) */
    {
        Deallocate_Matrix( *A_app_inv );
        Allocate_Matrix2( A_app_inv, A_spar_patt->n, system->local_cap, A_spar_patt->m,
                SYM_FULL_MATRIX, comm );
    }

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
           "sparse_approx_inverse::row_nnz", MPI_COMM_WORLD );
    j_list = smalloc( sizeof(int *) * system->N,
           "sparse_approx_inverse::j_list", MPI_COMM_WORLD );
    val_list = smalloc( sizeof(real *) * system->N,
           "sparse_approx_inverse::val_list", MPI_COMM_WORLD );

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
    t_start = MPI_Wtime();
    Dist( system, mpi_data, row_nnz, INT_PTR_TYPE, MPI_INT );
    t_comm += MPI_Wtime() - t_start;

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;

    /* use a Dist-like approach to send the row information */
    for ( d = 0; d < 3; ++d)
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
            for( i = nbr1->atoms_str; i < (nbr1->atoms_str + nbr1->atoms_cnt); ++i )
            {
                cnt += row_nnz[i];
            }

            /* initiate Irecv */
            if( cnt )
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
                            "sparse_approx_inverse::j_recv1", MPI_COMM_WORLD );
                    val_recv1 = smalloc( sizeof(real) * size_recv1,
                            "sparse_approx_inverse::val_recv1", MPI_COMM_WORLD );
                }

                t_start = MPI_Wtime();
                MPI_Irecv( j_recv1, cnt, MPI_INT, nbr1->rank, 2 * d + 1, comm, &req1 );
                MPI_Irecv( val_recv1, cnt, MPI_DOUBLE, nbr1->rank, 2 * d + 1, comm, &req2 );
                t_comm += MPI_Wtime() - t_start;
            }
        }

        nbr2 = &system->my_nbrs[2 * d + 1];
        if ( nbr2->atoms_cnt )
        {
            /* calculate the total data that will be received */
            cnt = 0;
            for( i = nbr2->atoms_str; i < (nbr2->atoms_str + nbr2->atoms_cnt); ++i )
            {
                cnt += row_nnz[i];
            }

            /* initiate Irecv */
            if( cnt )
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
                            "sparse_approx_inverse::j_recv2", MPI_COMM_WORLD );
                    val_recv2 = smalloc( sizeof(real) * size_recv2,
                            "sparse_approx_inverse::val_recv2", MPI_COMM_WORLD );
                }

                t_start = MPI_Wtime();
                MPI_Irecv( j_recv2, cnt, MPI_INT, nbr2->rank, 2 * d, comm, &req3 );
                MPI_Irecv( val_recv2, cnt, MPI_DOUBLE, nbr2->rank, 2 * d, comm, &req4 );
                t_comm += MPI_Wtime() - t_start;
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
                            "sparse_approx_inverse::j_send", MPI_COMM_WORLD );
                    val_send = smalloc( sizeof(real) * size_send,
                            "sparse_approx_inverse::j_send", MPI_COMM_WORLD );
                }

                cnt = 0;
                for ( i = 0; i < out_bufs[2 * d].cnt; ++i )
                {
                    if ( out_bufs[2 * d].index[i] < A->n )
                    {
                        for ( pj = A->start[ out_bufs[2 * d].index[i] ]; pj < A->end[ out_bufs[2 * d].index[i] ]; ++pj )
                        {
                            atom = &system->my_atoms[ A->entries[pj].j ];
                            j_send[cnt] = atom->orig_id;
                            val_send[cnt] = A->entries[pj].val;
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

                t_start = MPI_Wtime();
                MPI_Send( j_send, cnt, MPI_INT, nbr1->rank, 2 * d, comm );
                MPI_Send( val_send, cnt, MPI_DOUBLE, nbr1->rank, 2 * d, comm );
                t_comm += MPI_Wtime() - t_start;
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
                            "sparse_approx_inverse::j_send", MPI_COMM_WORLD );
                    val_send = smalloc( sizeof(real) * size_send,
                            "sparse_approx_inverse::val_send", MPI_COMM_WORLD );
                }

                cnt = 0;
                for ( i = 0; i < out_bufs[2 * d + 1].cnt; ++i )
                {
                    if ( out_bufs[2 * d + 1].index[i] < A->n )
                    {
                        for ( pj = A->start[ out_bufs[2 * d + 1].index[i] ]; pj < A->end[ out_bufs[2 * d + 1].index[i] ]; ++pj )
                        {
                            atom = &system->my_atoms[ A->entries[pj].j ];
                            j_send[cnt] = atom->orig_id;
                            val_send[cnt] = A->entries[pj].val;
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

                t_start = MPI_Wtime();
                MPI_Send( j_send, cnt, MPI_INT, nbr2->rank, 2 * d + 1, comm );
                MPI_Send( val_send, cnt, MPI_DOUBLE, nbr2->rank, 2 * d + 1, comm );
                t_comm += MPI_Wtime() - t_start;
            }

        }

        if ( flag1 )
        {
            t_start = MPI_Wtime();
            MPI_Wait( &req1, &stat1 );
            MPI_Wait( &req2, &stat2 );
            t_comm += MPI_Wtime() - t_start;

            cnt = 0;
            for ( i = nbr1->atoms_str; i < (nbr1->atoms_str + nbr1->atoms_cnt); ++i )
            {
                j_list[i] = smalloc( sizeof(int) *  row_nnz[i],
                       "sparse_approx_inverse::j_list[i]", MPI_COMM_WORLD );
                val_list[i] = smalloc( sizeof(real) * row_nnz[i],
                       "sparse_approx_inverse::val_list[i]", MPI_COMM_WORLD );

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
            t_start = MPI_Wtime();
            MPI_Wait( &req3, &stat3 );
            MPI_Wait( &req4, &stat4 );
            t_comm += MPI_Wtime() - t_start;

            cnt = 0;
            for ( i = nbr2->atoms_str; i < (nbr2->atoms_str + nbr2->atoms_cnt); ++i )
            {
                j_list[i] = smalloc( sizeof(int) *  row_nnz[i],
                       "sparse_approx_inverse::j_list[i]", MPI_COMM_WORLD );
                val_list[i] = smalloc( sizeof(real) * row_nnz[i],
                       "sparse_approx_inverse::val_list[i]", MPI_COMM_WORLD );

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
            "sparse_approx_inverse::X", MPI_COMM_WORLD );
    //size of q should be equal to the maximum possible cardinalty 
    //of the set formed by neighbors of neighbors of an atom
    //i.e, maximum number of rows of dense matrix
    //for water systems, this number is 34000
    //for silica systems, it is 12000
    q = smalloc( sizeof(int) * 50000,
            "sparse_approx_inverse::q", MPI_COMM_WORLD );

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
            j_temp = A_spar_patt->entries[pj].j;
            atom = &system->my_atoms[j_temp];
            ++N;

            /* for each of those indices
             * search through the row of full A of that index */

            /* the case where the local matrix has that index's row */
            if( j_temp < A->n )
            {
                for ( k = A->start[ j_temp ]; k < A->end[ j_temp ]; ++k )
                {
                    /* and accumulate the nonzero column indices to serve as the row indices of the dense matrix */
                    atom = &system->my_atoms[ A->entries[k].j ];
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
                "sparse_approx_inverse::dense_matrix", MPI_COMM_WORLD );
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
            local_pos = A_spar_patt->entries[ A_spar_patt->start[i] + d_j ].j;

            if ( local_pos < A->n )
            {
                for ( d_i = A->start[local_pos]; d_i < A->end[local_pos]; ++d_i )
                {
                    atom = &system->my_atoms[ A->entries[d_i].j ];
                    dense_matrix[ X[ atom->orig_id ] * N + d_j ] = A->entries[d_i].val;
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

            e_j = smalloc( sizeof(real) * size_e, "sparse_approx_inverse::e_j", MPI_COMM_WORLD );
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
        (*A_app_inv)->start[i] = A_spar_patt->start[i];
        (*A_app_inv)->end[i] = A_spar_patt->end[i];
        for ( k = (*A_app_inv)->start[i]; k < (*A_app_inv)->end[i]; ++k)
        {
            (*A_app_inv)->entries[k].j = A_spar_patt->entries[k].j;
            (*A_app_inv)->entries[k].val = e_j[k - A_spar_patt->start[i]];
        }
    }

    sfree( dense_matrix, "sparse_approx_inverse::dense_matrix" );
    sfree( e_j, "sparse_approx_inverse::e_j" );
    sfree( X, "sparse_approx_inverse::X" );
    /*for ( i = 0; i < system->N; ++i )
    {
        sfree( j_list[i], "sparse_approx_inverse::j_list" );
        sfree( val_list[i], "sparse_approx_inverse::val_list" );
    }
    sfree( j_list, "sparse_approx_inverse::j_list" );
    sfree( val_list, "sparse_approx_inverse::val_list" );*/
    sfree( row_nnz, "sparse_approx_inverse::row_nnz" );

    MPI_Reduce( &t_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE,
            mpi_data->world );

    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_comm += total_comm / nprocs;
    }

    return MPI_Wtime() - start;
}
#endif
#endif


/* Steepest Descent */
int SDM( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, real *b,
        real tol, real *x, mpi_datatypes* mpi_data )
{
    int i, j;
    real tmp, alpha, bnorm, sig;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[3];

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, x, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, x, workspace->q, H->NT );
#else
    Sparse_MatVec( H, x, workspace->q, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    Vector_Sum( workspace->r, 1.0,  b, -1.0, workspace->q, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    /* pre-conditioning */
    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->r, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, H->NT );
#else
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, system->n );
#endif
        t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC)
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    redux[0] = Dot_local( b, b, system->n );
    redux[1] = Dot_local( workspace->r, workspace->d, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
    MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
    t_allreduce += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
    bnorm = sqrt( redux[0] );
    sig = redux[1];
    t_vops += MPI_Wtime( ) - t_start;

    for ( i = 0; i < control->cm_solver_max_iters && SQRT(sig) / bnorm > tol; ++i )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->d, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( H, workspace->d, workspace->q, H->NT );
#else
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif

        t_start = MPI_Wtime( );
        redux[0] = Dot_local( workspace->r, workspace->d, system->n );
        redux[1] = Dot_local( workspace->d, workspace->q, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        sig = redux[0];
        tmp = redux[1];
        alpha = sig / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        t_vops += Get_Timing_Info( t_start );

        /* pre-conditioning */
        if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime( );
            Dist( system, mpi_data, workspace->r, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
            
            t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
            Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, H->NT );
#else
            Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, system->n );
#endif
            t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC)
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] SDM convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", SQRT(sig) / bnorm );
        return i;
    }

    return i;
}


int dual_CG( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, rvec2 *b,
        real tol, rvec2 *x, mpi_datatypes* mpi_data )
{
    int  i, j;
    rvec2 tmp, alpha, beta;
    rvec2 norm, b_norm;
    rvec2 sig_old, sig_new;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[6];

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, x, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    dual_Sparse_MatVec( H, x, workspace->q2, H->NT );
#else
    dual_Sparse_MatVec( H, x, workspace->q2, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;


    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    for ( j = 0; j < system->n; ++j )
    {
        // residual
        workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
        workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];
    }
    t_vops += MPI_Wtime( ) - t_start;

    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
         t_start = MPI_Wtime( );
         Dist( system, mpi_data, workspace->r2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
         t_comm += MPI_Wtime( ) - t_start;

         t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
         dual_Sparse_MatVec( workspace->H_app_inv, workspace->r2, workspace->d2, H->NT );
#else
         dual_Sparse_MatVec( workspace->H_app_inv, workspace->r2, workspace->d2, system->n );
#endif
         t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
            workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    for ( j = 0; j < 6; ++j )
    {
        redux[j] = 0;
    }
    for ( j = 0; j < system->n; ++j )
    {
        redux[0] += workspace->r2[j][0] * workspace->d2[j][0];
        redux[1] += workspace->r2[j][1] * workspace->d2[j][1];
        
        redux[2] += workspace->d2[j][0] * workspace->d2[j][0];
        redux[3] += workspace->d2[j][1] * workspace->d2[j][1];

        redux[4] += b[j][0] * b[j][0];
        redux[5] += b[j][1] * b[j][1];
    }
    t_vops += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
    MPI_Allreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM, mpi_data->world );
    t_allreduce += MPI_Wtime( ) - t_start;

    sig_new[0] = redux[0];
    sig_new[1] = redux[1];
    norm[0] = sqrt( redux[2] );
    norm[1] = sqrt( redux[3] );
    b_norm[0] = sqrt( redux[4] );
    b_norm[1] = sqrt( redux[5] );

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( norm[0] / b_norm[0] <= tol || norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->d2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        dual_Sparse_MatVec( H, workspace->d2, workspace->q2, H->NT );
#else
        dual_Sparse_MatVec( H, workspace->d2, workspace->q2, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif

        // dot product: d.q
        t_start =  MPI_Wtime( );
        redux[0] = redux[1] = 0;
        for ( j = 0; j < system->n; ++j )
        {
            redux[0] += workspace->d2[j][0] * workspace->q2[j][0];
            redux[1] += workspace->d2[j][1] * workspace->q2[j][1];
        }
        t_vops += MPI_Wtime( ) - t_start;

        t_start =  MPI_Wtime( );
        MPI_Allreduce( &redux, &tmp, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        for ( j = 0; j < system->n; ++j )
        {
            // update x
            x[j][0] += alpha[0] * workspace->d2[j][0];
            x[j][1] += alpha[1] * workspace->d2[j][1];
            // update residual
            workspace->r2[j][0] -= alpha[0] * workspace->q2[j][0];
            workspace->r2[j][1] -= alpha[1] * workspace->q2[j][1];
        }
        t_vops += MPI_Wtime( ) - t_start;

        if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
             t_start = MPI_Wtime( );
             Dist( system, mpi_data, workspace->r2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
             t_comm += MPI_Wtime( ) - t_start;

             t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
             dual_Sparse_MatVec( workspace->H_app_inv, workspace->r2, workspace->p2, H->NT );
#else
             dual_Sparse_MatVec( workspace->H_app_inv, workspace->r2, workspace->p2, system->n );
#endif
             t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC)
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->p2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
                workspace->p2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }

        t_start = MPI_Wtime( );
        redux[0] = 0.0;
        redux[1] = 0.0;
        redux[2] = 0.0;
        redux[3] = 0.0;
        for ( j = 0; j < system->n; ++j )
        {
            // dot product: r.p
            redux[0] += workspace->r2[j][0] * workspace->p2[j][0];
            redux[1] += workspace->r2[j][1] * workspace->p2[j][1];

            // dot product: p.p
            redux[2] += workspace->p2[j][0] * workspace->p2[j][0];
            redux[3] += workspace->p2[j][1] * workspace->p2[j][1];
        }
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        sig_new[0] = redux[0];
        sig_new[1] = redux[1];
        norm[0] = sqrt( redux[2] );
        norm[1] = sqrt( redux[3] );
        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        for ( j = 0; j < system->n; ++j )
        {
            // d = p + beta * d
            workspace->d2[j][0] = workspace->p2[j][0] + beta[0] * workspace->d2[j][0];
            workspace->d2[j][1] = workspace->p2[j][1] + beta[1] * workspace->d2[j][1];
        }
        t_vops += MPI_Wtime( ) - t_start;
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;
    
    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
 
        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    // continue to solve the system that has not converged yet
    if ( norm[0] / b_norm[0] > tol )
    {
        for ( j = 0; j < system->n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        i += CG( system, control, data, workspace,
                H, workspace->b_s, tol, workspace->s, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }
    }
    else if ( norm[1] / b_norm[1] > tol )
    {
        for ( j = 0; j < system->n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        i += CG( system, control, data, workspace,
                H, workspace->b_t, tol, workspace->t, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] CG convergence failed!\n" );
        return i;
    }

    return i;

}


/* Preconditioned Conjugate Gradient Method */
int CG( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, real *b,
        real tol, real *x, mpi_datatypes* mpi_data )
{
    int i, j;
    real tmp, alpha, beta, norm, b_norm;
    real sig_old, sig_new;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[3];

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, x, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, x, workspace->q, H->NT );
#else
    Sparse_MatVec( H, x, workspace->q, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    /* pre-conditioning */
    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->r, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, H->NT );
#else
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, system->n );
#endif
        t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC)
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    redux[0] = Dot_local( workspace->r, workspace->d, system->n );
    redux[1] = Dot_local( workspace->d, workspace->d, system->n );
    redux[2] = Dot_local( b, b, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
    MPI_Allreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM, mpi_data->world );
    t_allreduce += MPI_Wtime( ) - t_start;
    sig_new = redux[0];
    norm = sqrt( redux[1] );
    b_norm = sqrt( redux[2] );

    for ( i = 0; i < control->cm_solver_max_iters && norm / b_norm > tol; ++i )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->d, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( H, workspace->d, workspace->q, H->NT );
#else
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif

        t_start =  MPI_Wtime( );
        tmp = Parallel_Dot( workspace->d, workspace->q, system->n, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        /* pre-conditioning */
        if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime( );
            Dist( system, mpi_data, workspace->r, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;

            t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
            Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->p, H->NT );
#else
            Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->p, system->n );
#endif
            t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }

        t_start = MPI_Wtime( );
        redux[0] = Dot_local( workspace->r, workspace->p, system->n );
        redux[1] = Dot_local( workspace->p, workspace->p, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        sig_old = sig_new;
        sig_new = redux[0];
        norm = sqrt( redux[1] );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );
        t_vops += MPI_Wtime( ) - t_start;
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] CG convergence failed!\n" );
        return i;
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
int BiCGStab( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, real *b,
        real tol, real *x, mpi_datatypes* mpi_data )
{
    int i, j;
    real tmp, alpha, beta, omega, sigma, rho, rho_old, rnorm, bnorm;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[2];

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, x, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, x, workspace->d, H->NT );
#else
    Sparse_MatVec( H, x, workspace->d, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->d, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->d, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    Vector_Sum( workspace->r, 1.0,  b, -1.0, workspace->d, system->n );
    redux[0] = Dot_local( b, b, system->n );
    redux[1] = Dot_local( workspace->r, workspace->r, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
    MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
    t_allreduce += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
    bnorm = sqrt( redux[0] );
    rnorm = sqrt( redux[1] );
    if ( bnorm == 0.0 )
    {
        bnorm = 1.0;
    }
    Vector_Copy( workspace->r_hat, workspace->r, system->n );
    omega = 1.0;
    rho = 1.0;
    t_vops += MPI_Wtime( ) - t_start;

    for ( i = 0; i < control->cm_solver_max_iters && rnorm / bnorm > tol; ++i )
    {
        t_start = MPI_Wtime( );
        redux[0] = Dot_local( workspace->r_hat, workspace->r, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
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
        t_vops += MPI_Wtime( ) - t_start;

        /* pre-conditioning */
        if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime( );
            Dist( system, mpi_data, workspace->p, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;

            t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
            Sparse_MatVec( workspace->H_app_inv, workspace->p, workspace->d, H->NT );
#else
            Sparse_MatVec( workspace->H_app_inv, workspace->p, workspace->d, system->n );
#endif
            t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->d[j] = workspace->p[j] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }

        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->d, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( H, workspace->d, workspace->z, H->NT );
#else
        Sparse_MatVec( H, workspace->d, workspace->z, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->z, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->z, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif

        t_start = MPI_Wtime( );
        redux[0] = Dot_local( workspace->r_hat, workspace->z, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        tmp = redux[0];
        alpha = rho / tmp;
        Vector_Sum( workspace->q, 1.0, workspace->r, -1.0 * alpha, workspace->z, system->n );
        redux[0] = Dot_local( workspace->q, workspace->q, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        tmp = redux[0];
        /* early convergence check */
        if ( tmp < tol )
        {
            Vector_Add( x, alpha, workspace->d, system->n );
            break;
        }
        t_vops += MPI_Wtime( ) - t_start;

        /* pre-conditioning */
        if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime( );
            Dist( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;

            t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
            Sparse_MatVec( workspace->H_app_inv, workspace->q, workspace->q_hat, H->NT );
#else
            Sparse_MatVec( workspace->H_app_inv, workspace->q, workspace->q_hat, system->n );
#endif
            t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->q_hat[j] = workspace->q[j] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }

        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->q_hat, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( H, workspace->q_hat, workspace->y, H->NT );
#else
        Sparse_MatVec( H, workspace->q_hat, workspace->y, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->y, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->y, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif

        t_start = MPI_Wtime( );
        redux[0] = Dot_local( workspace->y, workspace->q, system->n );
        redux[1] = Dot_local( workspace->y, workspace->y, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        sigma = redux[0];
        tmp = redux[1];
        omega = sigma / tmp;
        Vector_Sum( workspace->g, alpha, workspace->d, omega, workspace->q_hat, system->n );
        Vector_Add( x, 1.0, workspace->g, system->n );
        Vector_Sum( workspace->r, 1.0, workspace->q, -1.0 * omega, workspace->y, system->n );
        redux[0] = Dot_local( workspace->r, workspace->r, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE, MPI_SUM, mpi_data->world );
        t_allreduce += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
        rnorm = sqrt( redux[0] );
        if ( omega == 0.0 )
        {
            break;
        }
        rho_old = rho;
        t_vops += MPI_Wtime( ) - t_start;
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    if ( omega == 0.0 && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab numeric breakdown (%d iters)\n", i );
        fprintf( stderr, "  [INFO] omega = %f\n", omega );
    }
    else if ( rho == 0.0 && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab numeric breakdown (%d iters)\n", i );
        fprintf( stderr, "  [INFO] rho = %f\n", rho );
    }
    else if ( i >= control->cm_solver_max_iters
            && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] BiCGStab convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual error: %f\n", rnorm / bnorm );
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
int dual_PIPECG( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, rvec2 *b,
        real tol, rvec2 *x, mpi_datatypes* mpi_data )
{
    int i, j;
    rvec2 alpha, beta, delta, gamma_old, gamma_new, norm, b_norm;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[8];
    MPI_Request req;

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, x, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    dual_Sparse_MatVec( H, x, workspace->u2, H->NT );
#else
    dual_Sparse_MatVec( H, x, workspace->u2, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->u2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->u2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    //Vector_Sum( workspace->r , 1.0,  b, -1.0, workspace->u, system->n );
    for ( j = 0; j < system->n; ++j )
    {
        workspace->r2[j][0] = b[j][0] - workspace->u2[j][0];
        workspace->r2[j][1] = b[j][1] - workspace->u2[j][1];
    }
    t_vops += MPI_Wtime( ) - t_start;

    /* pre-conditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        //Vector_Copy( workspace->u, workspace->r, system->n );
        for ( j = 0; j < system->n ; ++j )
        {
            workspace->u2[j][0] = workspace->r2[j][0];
            workspace->u2[j][1] = workspace->r2[j][1];
        }
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->u2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
            workspace->u2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->r2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        dual_Sparse_MatVec( workspace->H_app_inv, workspace->r2, workspace->u2, H->NT );
#else
        dual_Sparse_MatVec( workspace->H_app_inv, workspace->r2, workspace->u2, system->n );
#endif
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, workspace->u2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    dual_Sparse_MatVec( H, workspace->u2, workspace->w2, H->NT );
#else
    dual_Sparse_MatVec( H, workspace->u2, workspace->w2, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->w2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->w2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    //redux[0] = Dot_local( workspace->w, workspace->u, system->n );
    //redux[1] = Dot_local( workspace->r, workspace->u, system->n );
    //redux[2] = Dot_local( workspace->u, workspace->u, system->n );
    //redux[3] = Dot_local( b, b, system->n );
    for ( j = 0; j < 8; ++j )
    {
        redux[j] = 0.0;
    }
    for( j = 0; j < system->n; ++j )
    {
        redux[0] += workspace->w2[j][0] * workspace->u2[j][0];
        redux[1] += workspace->w2[j][1] * workspace->u2[j][1];

        redux[2] += workspace->r2[j][0] * workspace->u2[j][0];
        redux[3] += workspace->r2[j][1] * workspace->u2[j][1];

        redux[4] += workspace->u2[j][0] * workspace->u2[j][0];
        redux[5] += workspace->u2[j][1] * workspace->u2[j][1];

        redux[6] += b[j][0] * b[j][0];
        redux[7] += b[j][1] * b[j][1];
    }
    t_vops += MPI_Wtime( ) - t_start;

    MPI_Iallreduce( MPI_IN_PLACE, redux, 8, MPI_DOUBLE, MPI_SUM, mpi_data->world, &req );

    /* pre-conditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        //Vector_Copy( workspace->m, workspace->w, system->n );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->m2[j][0] = workspace->w2[j][0];
            workspace->m2[j][1] = workspace->w2[j][1];
        }
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->m2[j][0] = workspace->w2[j][0] * workspace->Hdia_inv[j];
            workspace->m2[j][1] = workspace->w2[j][1] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->w2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        dual_Sparse_MatVec( workspace->H_app_inv, workspace->w2, workspace->m2, H->NT );
#else
        dual_Sparse_MatVec( workspace->H_app_inv, workspace->w2, workspace->m2, system->n );
#endif
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, workspace->m2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    dual_Sparse_MatVec( H, workspace->m2, workspace->n2, H->NT );
#else
    dual_Sparse_MatVec( H, workspace->m2, workspace->n2, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->n2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->n2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    MPI_Wait( &req, MPI_STATUS_IGNORE );
    t_allreduce += MPI_Wtime( ) - t_start;
    delta[0] = redux[0];
    delta[1] = redux[1];
    gamma_new[0] = redux[2];
    gamma_new[1] = redux[3];
    norm[0] = sqrt( redux[4] );
    norm[1] = sqrt( redux[5] );
    b_norm[0] = sqrt( redux[6] );
    b_norm[1] = sqrt( redux[7] );

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( norm[0] / b_norm[0] <= tol || norm[1] / b_norm[1] <= tol )
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

        t_start = MPI_Wtime( );
        //Vector_Sum( workspace->z, 1.0, workspace->n, beta, workspace->z, system->n );
        //Vector_Sum( workspace->q, 1.0, workspace->m, beta, workspace->q, system->n );
        //Vector_Sum( workspace->p, 1.0, workspace->u, beta, workspace->p, system->n );
        //Vector_Sum( workspace->d, 1.0, workspace->w, beta, workspace->d, system->n );
        //Vector_Sum( x, 1.0, x, alpha, workspace->p, system->n );
        //Vector_Sum( workspace->u, 1.0, workspace->u, -alpha, workspace->q, system->n );
        //Vector_Sum( workspace->w, 1.0, workspace->w, -alpha, workspace->z, system->n );
        //Vector_Sum( workspace->r, 1.0, workspace->r, -alpha, workspace->d, system->n );
        //redux[0] = Dot_local( workspace->w, workspace->u, system->n );
        //redux[1] = Dot_local( workspace->r, workspace->u, system->n );
        //redux[2] = Dot_local( workspace->u, workspace->u, system->n );
        for ( j = 0; j < 6; ++j )
        {
            redux[j] = 0.0;
        }
        for ( j = 0; j < system->n; ++j )
        {
            workspace->z2[j][0] = workspace->n2[j][0] + beta[0] * workspace->z2[j][0];
            workspace->z2[j][1] = workspace->n2[j][1] + beta[1] * workspace->z2[j][1];

            workspace->q2[j][0] = workspace->m2[j][0] + beta[0] * workspace->q2[j][0];
            workspace->q2[j][1] = workspace->m2[j][1] + beta[1] * workspace->q2[j][1];

            workspace->p2[j][0] = workspace->u2[j][0] + beta[0] * workspace->p2[j][0];
            workspace->p2[j][1] = workspace->u2[j][1] + beta[1] * workspace->p2[j][1];

            workspace->d2[j][0] = workspace->w2[j][0] + beta[0] * workspace->d2[j][0];
            workspace->d2[j][1] = workspace->w2[j][1] + beta[1] * workspace->d2[j][1];

            x[j][0] += alpha[0] * workspace->p2[j][0];
            x[j][1] += alpha[1] * workspace->p2[j][1];

            workspace->u2[j][0] -= alpha[0] * workspace->q2[j][0];
            workspace->u2[j][1] -= alpha[1] * workspace->q2[j][1];

            workspace->w2[j][0] -= alpha[0] * workspace->z2[j][0];
            workspace->w2[j][1] -= alpha[1] * workspace->z2[j][1];

            workspace->r2[j][0] -= alpha[0] * workspace->d2[j][0];
            workspace->r2[j][1] -= alpha[1] * workspace->d2[j][1];

            redux[0] += workspace->w2[j][0] * workspace->u2[j][0];
            redux[1] += workspace->w2[j][1] * workspace->u2[j][1];
            
            redux[2] += workspace->r2[j][0] * workspace->u2[j][0];
            redux[3] += workspace->r2[j][1] * workspace->u2[j][1];
            
            redux[4] += workspace->u2[j][0] * workspace->u2[j][0];
            redux[5] += workspace->u2[j][1] * workspace->u2[j][1];

        }
        t_vops += MPI_Wtime( ) - t_start;

        MPI_Iallreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM, mpi_data->world, &req );

        /* pre-conditioning */
        if ( control->cm_solver_pre_comp_type == NONE_PC )
        {
            //Vector_Copy( workspace->m, workspace->w, system->n );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->m2[j][0] = workspace->w2[j][0];
                workspace->m2[j][1] = workspace->w2[j][1];
            }
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->m2[j][0] = workspace->w2[j][0] * workspace->Hdia_inv[j];
                workspace->m2[j][1] = workspace->w2[j][1] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime( );
            Dist( system, mpi_data, workspace->w2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
            t_comm += MPI_Wtime( ) - t_start;
            
            t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
            dual_Sparse_MatVec( workspace->H_app_inv, workspace->w2, workspace->m2, H->NT );
#else
            dual_Sparse_MatVec( workspace->H_app_inv, workspace->w2, workspace->m2, system->n );
#endif
            t_pa += MPI_Wtime( ) - t_start;
        }

        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->m2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2);
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        dual_Sparse_MatVec( H, workspace->m2, workspace->n2, H->NT );
#else
        dual_Sparse_MatVec( H, workspace->m2, workspace->n2, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->n2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->n2, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2);
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif

        gamma_old[0] = gamma_new[0];
        gamma_old[1] = gamma_new[1];

        t_start = MPI_Wtime( );
        MPI_Wait( &req, MPI_STATUS_IGNORE );
        t_allreduce += MPI_Wtime( ) - t_start;
        delta[0] = redux[0];
        delta[1] = redux[1];
        gamma_new[0] = redux[2];
        gamma_new[1] = redux[3];
        norm[0] = sqrt( redux[4] );
        norm[1] = sqrt( redux[5] );
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    // continue to solve the system that has not converged yet
    if ( norm[0] / b_norm[0] > tol )
    {
        for ( j = 0; j < system->n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        i += PIPECG( system, control, data, workspace,
                H, workspace->b_s, tol, workspace->s, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }
    }
    else if ( norm[1] / b_norm[1] > tol )
    {
        for ( j = 0; j < system->n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        i += PIPECG( system, control, data, workspace,
                H, workspace->b_t, tol, workspace->t, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
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
int PIPECG( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, real *b,
        real tol, real *x, mpi_datatypes* mpi_data )
{
    int i, j;
    real alpha, beta, delta, gamma_old, gamma_new, norm, b_norm;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[4];
    MPI_Request req;

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, x, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, x, workspace->u, H->NT );
#else
    Sparse_MatVec( H, x, workspace->u, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->u, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->u, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    Vector_Sum( workspace->r , 1.0,  b, -1.0, workspace->u, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    /* pre-conditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        Vector_Copy( workspace->u, workspace->r, system->n );
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->u[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->r, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->u, H->NT );
#else
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->u, system->n );
#endif
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, workspace->u, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, workspace->u, workspace->w, H->NT );
#else
    Sparse_MatVec( H, workspace->u, workspace->w, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->w, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->w, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    redux[0] = Dot_local( workspace->w, workspace->u, system->n );
    redux[1] = Dot_local( workspace->r, workspace->u, system->n );
    redux[2] = Dot_local( workspace->u, workspace->u, system->n );
    redux[3] = Dot_local( b, b, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    MPI_Iallreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM, mpi_data->world, &req );

    /* pre-conditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        Vector_Copy( workspace->m, workspace->w, system->n );
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->m[j] = workspace->w[j] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->w, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( workspace->H_app_inv, workspace->w, workspace->m, H->NT );
#else
        Sparse_MatVec( workspace->H_app_inv, workspace->w, workspace->m, system->n );
#endif
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, workspace->m, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, workspace->m, workspace->n, H->NT );
#else
    Sparse_MatVec( H, workspace->m, workspace->n, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->n, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->n, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    MPI_Wait( &req, MPI_STATUS_IGNORE );
    t_allreduce += MPI_Wtime( ) - t_start;
    delta = redux[0];
    gamma_new = redux[1];
    norm = sqrt( redux[2] );
    b_norm = sqrt( redux[3] );

    for ( i = 0; i < control->cm_solver_max_iters && norm / b_norm > tol; ++i )
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

        t_start = MPI_Wtime( );
        Vector_Sum( workspace->z, 1.0, workspace->n, beta, workspace->z, system->n );
        Vector_Sum( workspace->q, 1.0, workspace->m, beta, workspace->q, system->n );
        Vector_Sum( workspace->p, 1.0, workspace->u, beta, workspace->p, system->n );
        Vector_Sum( workspace->d, 1.0, workspace->w, beta, workspace->d, system->n );
        Vector_Sum( x, 1.0, x, alpha, workspace->p, system->n );
        Vector_Sum( workspace->u, 1.0, workspace->u, -alpha, workspace->q, system->n );
        Vector_Sum( workspace->w, 1.0, workspace->w, -alpha, workspace->z, system->n );
        Vector_Sum( workspace->r, 1.0, workspace->r, -alpha, workspace->d, system->n );
        redux[0] = Dot_local( workspace->w, workspace->u, system->n );
        redux[1] = Dot_local( workspace->r, workspace->u, system->n );
        redux[2] = Dot_local( workspace->u, workspace->u, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        MPI_Iallreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM, mpi_data->world, &req );

        /* pre-conditioning */
        if ( control->cm_solver_pre_comp_type == NONE_PC )
        {
            Vector_Copy( workspace->m, workspace->w, system->n );
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->m[j] = workspace->w[j] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime( );
            Dist( system, mpi_data, workspace->w, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
            
            t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
            Sparse_MatVec( workspace->H_app_inv, workspace->w, workspace->m, H->NT );
#else
            Sparse_MatVec( workspace->H_app_inv, workspace->w, workspace->m, system->n );
#endif
            t_pa += MPI_Wtime( ) - t_start;
        }

        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->m, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( H, workspace->m, workspace->n, H->NT );
#else
        Sparse_MatVec( H, workspace->m, workspace->n, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->n, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->n, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif

        gamma_old = gamma_new;

        t_start = MPI_Wtime( );
        MPI_Wait( &req, MPI_STATUS_IGNORE );
        t_allreduce += MPI_Wtime( ) - t_start;
        delta = redux[0];
        gamma_new = redux[1];
        norm = sqrt( redux[2] );
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] PIPECG convergence failed!\n" );
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
int PIPECR( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, real *b,
        real tol, real *x, mpi_datatypes* mpi_data )
{
    int i, j;
    real alpha, beta, delta, gamma_old, gamma_new, norm, b_norm;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    real timings[5], redux[3];
    MPI_Request req;

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime( );
    redux[0] = Dot_local( b, b, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    MPI_Iallreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE, MPI_SUM, mpi_data->world, &req );

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, x, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, x, workspace->u, H->NT );
#else
    Sparse_MatVec( H, x, workspace->u, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->u, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->u, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    Vector_Sum( workspace->r , 1.0,  b, -1.0, workspace->u, system->n );
    t_vops += MPI_Wtime( ) - t_start;

    /* pre-conditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        Vector_Copy( workspace->u, workspace->r, system->n );
    }
    else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        t_start = MPI_Wtime( );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->u[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime( ) - t_start;
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->r, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
        
        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->u, H->NT );
#else
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->u, system->n );
#endif
        t_pa += MPI_Wtime( ) - t_start;
    }

    t_start = MPI_Wtime( );
    Dist( system, mpi_data, workspace->u, REAL_PTR_TYPE, MPI_DOUBLE );
    t_comm += MPI_Wtime( ) - t_start;

    t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, workspace->u, workspace->w, H->NT );
#else
    Sparse_MatVec( H, workspace->u, workspace->w, system->N );
#endif
    t_spmv += MPI_Wtime( ) - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->w, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#if defined(NEUTRAL_TERRITORY)
    else
    {
        t_start = MPI_Wtime( );
        Coll( system, mpi_data, workspace->w, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;
    }
#endif

    t_start = MPI_Wtime( );
    MPI_Wait( &req, MPI_STATUS_IGNORE );
    t_allreduce += MPI_Wtime( ) - t_start;
    b_norm = sqrt( redux[0] );

    t_start =  MPI_Wtime( );
    norm = Parallel_Norm( workspace->u, system->n, mpi_data->world );
    t_allreduce += MPI_Wtime( ) - t_start;

    for ( i = 0; i < control->cm_solver_max_iters && norm / b_norm > tol; ++i )
    {
        /* pre-conditioning */
        if ( control->cm_solver_pre_comp_type == NONE_PC )
        {
            Vector_Copy( workspace->m, workspace->w, system->n );
        }
        else if ( control->cm_solver_pre_comp_type == JACOBI_PC )
        {
            t_start = MPI_Wtime( );
            for ( j = 0; j < system->n; ++j )
            {
                workspace->m[j] = workspace->w[j] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime( ) - t_start;
        }
        else if ( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime( );
            Dist( system, mpi_data, workspace->w, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
            
            t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
            Sparse_MatVec( workspace->H_app_inv, workspace->w, workspace->m, H->NT );
#else
            Sparse_MatVec( workspace->H_app_inv, workspace->w, workspace->m, system->n );
#endif
            t_pa += MPI_Wtime( ) - t_start;
        }

        t_start = MPI_Wtime( );
        redux[0] = Dot_local( workspace->w, workspace->u, system->n );
        redux[1] = Dot_local( workspace->m, workspace->w, system->n );
        redux[2] = Dot_local( workspace->u, workspace->u, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        MPI_Iallreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM, mpi_data->world, &req );

        t_start = MPI_Wtime( );
        Dist( system, mpi_data, workspace->m, REAL_PTR_TYPE, MPI_DOUBLE );
        t_comm += MPI_Wtime( ) - t_start;

        t_start = MPI_Wtime( );
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( H, workspace->m, workspace->n, H->NT );
#else
        Sparse_MatVec( H, workspace->m, workspace->n, system->N );
#endif
        t_spmv += MPI_Wtime( ) - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->n, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#if defined(NEUTRAL_TERRITORY)
        else
        {
            t_start = MPI_Wtime( );
            Coll( system, mpi_data, workspace->n, REAL_PTR_TYPE, MPI_DOUBLE );
            t_comm += MPI_Wtime( ) - t_start;
        }
#endif
        t_start = MPI_Wtime( );
        MPI_Wait( &req, MPI_STATUS_IGNORE );
        t_allreduce += MPI_Wtime( ) - t_start;
        gamma_new = redux[0];
        delta = redux[1];
        norm = sqrt( redux[2] );

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

        t_start = MPI_Wtime( );
        Vector_Sum( workspace->z, 1.0, workspace->n, beta, workspace->z, system->n );
        Vector_Sum( workspace->q, 1.0, workspace->m, beta, workspace->q, system->n );
        Vector_Sum( workspace->p, 1.0, workspace->u, beta, workspace->p, system->n );
        Vector_Sum( workspace->d, 1.0, workspace->w, beta, workspace->d, system->n );
        Vector_Sum( x, 1.0, x, alpha, workspace->p, system->n );
        Vector_Sum( workspace->u, 1.0, workspace->u, -alpha, workspace->q, system->n );
        Vector_Sum( workspace->w, 1.0, workspace->w, -alpha, workspace->z, system->n );
        Vector_Sum( workspace->r, 1.0, workspace->r, -alpha, workspace->d, system->n );
        t_vops += MPI_Wtime( ) - t_start;

        gamma_old = gamma_new;
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    if ( system->my_rank == MASTER_NODE )
    {
        MPI_Reduce( MPI_IN_PLACE, timings, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        data->timing.cm_solver_pre_app += timings[0] / control->nprocs;
        data->timing.cm_solver_spmv += timings[1] / control->nprocs;
        data->timing.cm_solver_vector_ops += timings[2] / control->nprocs;
        data->timing.cm_solver_comm += timings[3] / control->nprocs;
        data->timing.cm_solver_allreduce += timings[4] / control->nprocs;
    }
    else
    {
        MPI_Reduce( timings, NULL, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );
    }

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "[WARNING] PIPECR convergence failed!\n" );
        return i;
    }

    return i;
}
