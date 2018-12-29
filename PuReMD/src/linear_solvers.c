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
    pivotlist = smalloc( sizeof(real) *  (nprocs - 1),
           "setup_sparse_approx_inverse::pivotlist", MPI_COMM_WORLD );
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

#if defined(DEBUG)
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

#if defined(DEBUG)
                nnz++;
#endif
            }
        }
        (*A_spar_patt)->end[i] = size;
    }

#if defined(DEBUG)
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
    sfree( pivotlist, "setup_sparse_approx_inverse::pivotlist" );
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
    int cnt, scale;
    
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
    scale = sizeof(int) / sizeof(void);
    Dist_NT( system, mpi_data, row_nnz, MPI_INT, scale, int_packer );
    t_comm += MPI_Wtime() - t_start;
    fprintf( stdout,"SAI after Dist_NT call\n");
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

                fprintf( stdout,"Dist_NT communication receive phase direction %d will receive %d\n", d, cnt);
                fflush( stdout );
                t_start = MPI_Wtime();
                MPI_Irecv( j_recv + d, cnt, MPI_INT, nbr->receive_rank, d, comm, &(req[2*d]) );
                MPI_Irecv( val_recv + d, cnt, MPI_DOUBLE, nbr->receive_rank, d, comm, &(req[2*d+1]) );
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
            fprintf( stdout,"Dist_NT communication    send phase direction %d should  send %d\n", d, cnt);
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

                fprintf( stdout,"Dist_NT communication    send phase direction %d will    send %d\n", d, cnt );
                fflush( stdout );

                t_start = MPI_Wtime();
                MPI_Send( j_send, cnt, MPI_INT, nbr->rank, d, comm );
                fprintf( stdout,"Dist_NT communication send phase direction %d cnt = %d\n", d, cnt);
                fflush( stdout );
                MPI_Send( val_send, cnt, MPI_DOUBLE, nbr->rank, d, comm );
                fprintf( stdout,"Dist_NT communication send phase direction %d cnt = %d\n", d, cnt);
                fflush( stdout );
                t_comm += MPI_Wtime() - t_start;
            }
        }
    }
    fprintf( stdout," Dist_NT communication for sending row info before waitany\n");
    fflush( stdout );
    ///////////////////////
    for ( d = 0; d < count; ++d )
    {
        t_start = MPI_Wtime();
        MPI_Waitany( REAX_MAX_NT_NBRS, req, &index, stat);
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
    fprintf( stdout," wow wow wow, Dist_NT communication for sending row info worked\n");
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
    int cnt, scale;
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
    scale = sizeof(int) / sizeof(void);
    Dist( system, mpi_data, row_nnz, MPI_INT, scale, int_packer );
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
    q = smalloc( sizeof(int) * system->N * 2,
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


void dual_Sparse_MatVec( sparse_matrix *A, rvec2 *x, rvec2 *b, int N )
{
    int  i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i][0] = b[i][1] = 0;
    }

    if ( A->format == SYM_HALF_MATRIX )
    {
        /* perform multiplication */
        for ( i = 0; i < A->n; ++i )
        {
            si = A->start[i];

            b[i][0] += A->entries[si].val * x[i][0];
            b[i][1] += A->entries[si].val * x[i][1];

            for ( k = si + 1; k < A->end[i]; ++k )
            {
                j = A->entries[k].j;
                H = A->entries[k].val;

                b[i][0] += H * x[j][0];
                b[i][1] += H * x[j][1];

                // comment out for tryQEq
                //if( j < A->n ) {
                b[j][0] += H * x[i][0];
                b[j][1] += H * x[i][1];
                //}
            }
        }
    }
    else if ( A->format == SYM_FULL_MATRIX || A->format == FULL_MATRIX )
    {
        /* perform multiplication */
        for ( i = 0; i < A->n; ++i )
        {
            si = A->start[i];

            for ( k = si; k < A->end[i]; ++k )
            {
                j = A->entries[k].j;
                H = A->entries[k].val;

                b[i][0] += H * x[j][0];
                b[i][1] += H * x[j][1];
            }
        }
    }
}


/*int dual_CG( reax_system *system, storage *workspace, sparse_matrix *H,
        rvec2 *b, real tol, rvec2 *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, n, N, matvecs, scale;
    rvec2 tmp, alpha, beta;
    rvec2 my_sum, norm_sqr, b_norm, my_dot;
    rvec2 sig_old, sig_new;
    MPI_Comm comm;

    n = system->n;
    N = system->N;
    comm = mpi_data->world;
    matvecs = 0;
    scale = sizeof(rvec2) / sizeof(void);

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        matvecs = 0;
        t_start = matvec_time = dot_time = 0;
        t_start = MPI_Wtime();
    }
#endif

    Dist( system, mpi_data, x, mpi_data->mpi_rvec2, scale, rvec2_packer );
    dual_Sparse_MatVec( H, x, workspace->q2, N );
    if ( H->format == SYM_HALF_MATRIX )
    {
        Coll(system, mpi_data, workspace->q2, mpi_data->mpi_rvec2, scale, rvec2_unpacker);
    }

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        //Update_Timing_Info( &t_start, &matvec_time );
        real t_end = MPI_Wtime();
        matvec_time += t_end - t_start;
        t_start = t_end;
    }
#endif

    for ( j = 0; j < system->n; ++j )
    {
        // residual
        workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
        workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];
        // apply diagonal pre-conditioner
        workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
        workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
    }

    // norm of b
    my_sum[0] = my_sum[1] = 0;
    for ( j = 0; j < n; ++j )
    {
        my_sum[0] += SQR( b[j][0] );
        my_sum[1] += SQR( b[j][1] );
    }
    MPI_Allreduce( &my_sum, &norm_sqr, 2, MPI_DOUBLE, MPI_SUM, comm );
    b_norm[0] = sqrt( norm_sqr[0] );
    b_norm[1] = sqrt( norm_sqr[1] );
    //fprintf( stderr, "bnorm = %f %f\n", b_norm[0], b_norm[1] );

    // dot product: r.d
    my_dot[0] = my_dot[1] = 0;
    for ( j = 0; j < n; ++j )
    {
        my_dot[0] += workspace->r2[j][0] * workspace->d2[j][0];
        my_dot[1] += workspace->r2[j][1] * workspace->d2[j][1];
    }
    MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );
    //fprintf( stderr, "sig_new: %f %f\n", sig_new[0], sig_new[1] );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        //Update_Timing_Info( &t_start, &dot_time );
        real t_end = MPI_Wtime();
        dot_time += t_end - t_start;
        t_start = t_end;
    }
#endif

    for ( i = 1; i < 300; ++i )
    {
        Dist(system, mpi_data, workspace->d2, mpi_data->mpi_rvec2, scale, rvec2_packer);
        dual_Sparse_MatVec( H, workspace->d2, workspace->q2, N );
        if ( H->format == SYM_HALF_MATRIX )
        {
            Coll(system, mpi_data, workspace->q2, mpi_data->mpi_rvec2, scale, rvec2_unpacker);
        }

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            //Update_Timing_Info( &t_start, &matvec_time );
            real t_end = MPI_Wtime();
            matvec_time += t_end - t_start;
            t_start = t_end;
        }
#endif

        // dot product: d.q
        my_dot[0] = my_dot[1] = 0;
        for ( j = 0; j < n; ++j )
        {
            my_dot[0] += workspace->d2[j][0] * workspace->q2[j][0];
            my_dot[1] += workspace->d2[j][1] * workspace->q2[j][1];
        }
        MPI_Allreduce( &my_dot, &tmp, 2, MPI_DOUBLE, MPI_SUM, comm );
        //fprintf( stderr, "tmp: %f %f\n", tmp[0], tmp[1] );

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        my_dot[0] = my_dot[1] = 0;
        for ( j = 0; j < system->n; ++j )
        {
            // update x
            x[j][0] += alpha[0] * workspace->d2[j][0];
            x[j][1] += alpha[1] * workspace->d2[j][1];
            // update residual
            workspace->r2[j][0] -= alpha[0] * workspace->q2[j][0];
            workspace->r2[j][1] -= alpha[1] * workspace->q2[j][1];
            // apply diagonal pre-conditioner
            workspace->p2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
            workspace->p2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
            // dot product: r.p
            my_dot[0] += workspace->r2[j][0] * workspace->p2[j][0];
            my_dot[1] += workspace->r2[j][1] * workspace->p2[j][1];
        }
        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );
        //fprintf( stderr, "sig_new: %f %f\n", sig_new[0], sig_new[1] );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            //Update_Timing_Info( &t_start, &dot_time );
            real t_end = MPI_Wtime();
            dot_time += t_end - t_start;
            t_start = t_end;
        }
#endif

        if ( sqrt(sig_new[0]) / b_norm[0] <= tol || sqrt(sig_new[1]) / b_norm[1] <= tol )
        {
            break;
        }

        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        for ( j = 0; j < system->n; ++j )
        {
            // d = p + beta * d
            workspace->d2[j][0] = workspace->p2[j][0] + beta[0] * workspace->d2[j][0];
            workspace->d2[j][1] = workspace->p2[j][1] + beta[1] * workspace->d2[j][1];
        }
    }

    if ( sqrt(sig_new[0]) / b_norm[0] <= tol )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }
        matvecs = CG( system, workspace, H, workspace->b_t, tol,
                workspace->t,mpi_data, fout, -1 );
        for ( j = 0; j < n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
    }
    else if ( sqrt(sig_new[1]) / b_norm[1] <= tol )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }
        matvecs = CG( system, workspace, H, workspace->b_s, tol, workspace->s,
                mpi_data, fout, -1 );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }
    }

    if ( i >= 300 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
    }

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( fout, "QEq %d + %d iters. matvecs: %f  dot: %f\n", i + 1,
                matvecs, matvec_time, dot_time );
    }
#endif

    return (i + 1) + matvecs;
}*/


void Sparse_MatVec( sparse_matrix *A, real *x, real *b, int N )
{
    int i, j, k, si, num_rows;
    real val;

    for ( i = 0; i < N; ++i )
    {
        b[i] = 0;
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


int CG( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, sparse_matrix *H, real *b,
        real tol, real *x, mpi_datatypes* mpi_data, FILE *fout, int nprocs )
{
    int  i, j, scale;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new;
    real t_start, t_pa, t_spmv, t_vops, t_comm, t_allreduce;
    //real total_pa, total_spmv, total_vops, total_comm, total_allreduce;
    real timings[5], total_t[5];

    t_pa = 0.0;
    t_spmv = 0.0;
    t_vops = 0.0;
    t_comm = 0.0;
    t_allreduce = 0.0;

    t_start = MPI_Wtime();
    scale = sizeof(real) / sizeof(void);
#if defined(NEUTRAL_TERRITORY)
    Dist_NT( system, mpi_data, x, MPI_DOUBLE, scale, real_packer );
#else
    Dist( system, mpi_data, x, MPI_DOUBLE, scale, real_packer );
#endif
    t_comm += MPI_Wtime() - t_start;

    t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
    Sparse_MatVec( H, x, workspace->q, H->NT );
#else
    Sparse_MatVec( H, x, workspace->q, system->N );
#endif
    t_spmv += MPI_Wtime() - t_start;

    if ( H->format == SYM_HALF_MATRIX )
    {
        t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
        Coll_NT( system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker );
#else
        Coll( system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker );
#endif
        t_comm += MPI_Wtime() - t_start;
    }

    else
    {
        t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
        Coll_NT( system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker );
#endif
        t_comm += MPI_Wtime() - t_start;
    }

    t_start = MPI_Wtime();
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );
    t_vops += MPI_Wtime() - t_start;

    /* pre-conditioning */
    if( control->cm_solver_pre_comp_type == SAI_PC )
    {
        t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
        Dist_NT( system, mpi_data, workspace->r, MPI_DOUBLE, scale, real_packer );
#else
        Dist( system, mpi_data, workspace->r, MPI_DOUBLE, scale, real_packer );
#endif
        t_comm += MPI_Wtime() - t_start;
        
        t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, H->NT );
#else
        Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->d, system->n );
#endif
        t_pa += MPI_Wtime() - t_start;
    }

    else if ( control->cm_solver_pre_comp_type == JACOBI_PC)
    {
        t_start = MPI_Wtime();
        for ( j = 0; j < system->n; ++j )
        {
            workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }
        t_pa += MPI_Wtime() - t_start;
    }

    t_start = MPI_Wtime();
    b_norm = Parallel_Norm( b, system->n, mpi_data->world );
    sig_new = Parallel_Dot(workspace->r, workspace->d, system->n, mpi_data->world);
    t_allreduce += MPI_Wtime() - t_start;

    for ( i = 0; i < control->cm_solver_max_iters && sqrt(sig_new) / b_norm > tol; ++i )
    {
        t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
        Dist_NT( system, mpi_data, workspace->d, MPI_DOUBLE, scale, real_packer );
#else
        Dist( system, mpi_data, workspace->d, MPI_DOUBLE, scale, real_packer );
#endif
        t_comm += MPI_Wtime() - t_start;

        t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
        Sparse_MatVec( H, workspace->d, workspace->q, H->NT );
#else
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
#endif
        t_spmv += MPI_Wtime() - t_start;

        if ( H->format == SYM_HALF_MATRIX )
        {
            t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
            Coll_NT(system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker);
#else
            Coll(system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker);
#endif
            t_comm += MPI_Wtime() - t_start;
        }
        else
        {
            t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
            Coll_NT( system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker );
#endif
            t_comm += MPI_Wtime() - t_start;
        }

        t_start =  MPI_Wtime();
        tmp = Parallel_Dot(workspace->d, workspace->q, system->n, mpi_data->world);
        t_allreduce += MPI_Wtime() - t_start;

        t_start = MPI_Wtime();
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        t_vops += MPI_Wtime() - t_start;

        /* pre-conditioning */
        if( control->cm_solver_pre_comp_type == SAI_PC )
        {
            t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
            Dist_NT( system, mpi_data, workspace->r, MPI_DOUBLE, scale, real_packer );
#else
            Dist( system, mpi_data, workspace->r, MPI_DOUBLE, scale, real_packer );
#endif
            t_comm += MPI_Wtime() - t_start;

            t_start = MPI_Wtime();
#if defined(NEUTRAL_TERRITORY)
            Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->p, H->NT );
#else
            Sparse_MatVec( workspace->H_app_inv, workspace->r, workspace->p, system->n );
#endif
            t_pa += MPI_Wtime() - t_start;
        }

        else if ( control->cm_solver_pre_comp_type == JACOBI_PC)
        {
            t_start = MPI_Wtime();
            for ( j = 0; j < system->n; ++j )
            {
                workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
            }
            t_pa += MPI_Wtime() - t_start;
        }

        t_start = MPI_Wtime();
        sig_old = sig_new;
        sig_new = Parallel_Dot(workspace->r, workspace->p, system->n, mpi_data->world);
        t_allreduce += MPI_Wtime() - t_start;

        t_start = MPI_Wtime();
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );
        t_vops += MPI_Wtime() - t_start;
    }

    timings[0] = t_pa;
    timings[1] = t_spmv;
    timings[2] = t_vops;
    timings[3] = t_comm;
    timings[4] = t_allreduce;

    //MPI_Reduce(&t_pa, &total_pa, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);
    //MPI_Reduce(&t_spmv, &total_spmv, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);
    //MPI_Reduce(&t_vops, &total_vops, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);
    //MPI_Reduce(&t_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);
    //MPI_Reduce(&t_allreduce, &total_allreduce, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);
    MPI_Reduce(timings, total_t, 5, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);

    if ( system->my_rank == MASTER_NODE )
    {
        //data->timing.cm_solver_pre_app += total_pa / nprocs;
        //data->timing.cm_solver_spmv += total_spmv / nprocs;
        //data->timing.cm_solver_vector_ops += total_vops / nprocs;
        //data->timing.cm_solver_comm += total_comm / nprocs;
        //data->timing.cm_solver_allreduce += total_allreduce / nprocs;
        data->timing.cm_solver_pre_app += total_t[0] / nprocs;
        data->timing.cm_solver_spmv += total_t[1] / nprocs;
        data->timing.cm_solver_vector_ops += total_t[2] / nprocs;
        data->timing.cm_solver_comm += total_t[3] / nprocs;
        data->timing.cm_solver_allreduce += total_t[4] / nprocs;
    }

    MPI_Barrier(mpi_data->world);

    if ( i >= control->cm_solver_max_iters && system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}


/*int CG_test( reax_system *system, storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, scale;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new;

    scale = sizeof(real) / sizeof(void);
    b_norm = Parallel_Norm( b, system->n, mpi_data->world );
#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "n=%d, N=%d\n", system->n, system->N );
        fprintf( stderr, "p%d CGinit: b_norm=%24.15e\n", system->my_rank, b_norm );
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //Vector_Print( stderr, "q", workspace->q, system->N );
    }
    MPI_Barrier( mpi_data->world );
#endif

    Sparse_MatVec( H, x, workspace->q, system->N );
    //Coll( system, mpi_data, workspace->q, MPI_DOUBLE, real_unpacker );

    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );
    for ( j = 0; j < system->n; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j]; //pre-condition

    sig_new = Parallel_Dot( workspace->r, workspace->d, system->n,
            mpi_data->world );
#if defined(DEBUG)
    //if( system->my_rank == MASTER_NODE ) {
    fprintf( stderr, "p%d CG:sig_new=%24.15e,d_norm=%24.15e,q_norm=%24.15e\n",
            system->my_rank, sqrt(sig_new),
            Parallel_Norm(workspace->d, system->n, mpi_data->world),
            Parallel_Norm(workspace->q, system->n, mpi_data->world) );
    //Vector_Print( stderr, "d", workspace->d, system->N );
    //Vector_Print( stderr, "q", workspace->q, system->N );
    //}
    MPI_Barrier( mpi_data->world );
#endif

    for ( i = 1; i < 300 && sqrt(sig_new) / b_norm > tol; ++i )
    {
#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            t_start = MPI_Wtime();
#endif
        Dist( system, mpi_data, workspace->d, MPI_DOUBLE, scale, real_packer );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
        //tryQEq
        //Coll(system, mpi_data, workspace->q, MPI_DOUBLE, real_unpacker);
#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            t_elapsed = MPI_Wtime() - t_start;
            matvec_time += t_elapsed;
        }
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            t_start = MPI_Wtime();
#endif
        tmp = Parallel_Dot(workspace->d, workspace->q, system->n, mpi_data->world);
        alpha = sig_new / tmp;
#if defined(DEBUG)
        //if( system->my_rank == MASTER_NODE ){
        fprintf(stderr,
                "p%d CG iter%d:d_norm=%24.15e,q_norm=%24.15e,tmp = %24.15e\n",
                system->my_rank, i,
                //Parallel_Norm(workspace->d, system->n, mpi_data->world),
                //Parallel_Norm(workspace->q, system->n, mpi_data->world),
                Norm(workspace->d, system->n), Norm(workspace->q, system->n), tmp);
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //for( j = 0; j < system->N; ++j )
        //  fprintf( stdout, "%d  %24.15e\n",
        //     system->my_atoms[j].orig_id, workspace->q[j] );
        //fprintf( stdout, "\n" );
        //}
        MPI_Barrier( mpi_data->world );
#endif

        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        // pre-conditioning
        for ( j = 0; j < system->n; ++j )
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];

        sig_old = sig_new;
        sig_new = Parallel_Dot(workspace->r, workspace->p, system->n, mpi_data->world);
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );
#if defined(DEBUG)
        if ( system->my_rank == MASTER_NODE )
            fprintf(stderr, "p%d CG iter%d: sig_new = %24.15e\n",
                    system->my_rank, i, sqrt(sig_new) );
        MPI_Barrier( mpi_data->world );
#endif
#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            t_elapsed = MPI_Wtime() - t_start;
            dot_time += t_elapsed;
        }
#endif
    }

#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
        fprintf( stderr, "CG took %d iterations\n", i );
#endif
#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        fprintf( stderr, "%f  %f\n", matvec_time, dot_time );
#endif
    if ( i >= 300 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}*/


/*void Forward_Subs( sparse_matrix *L, real *b, real *y )
{
    int i, pj, j, si, ei;
    real val;

    for ( i = 0; i < L->n; ++i )
    {
        y[i] = b[i];
        si = L->start[i];
        ei = L->end[i];
        for ( pj = si; pj < ei - 1; ++pj )
        {
            j = L->entries[pj].j;
            val = L->entries[pj].val;
            y[i] -= val * y[j];
        }
        y[i] /= L->entries[pj].val;
    }
}*/


/*void Backward_Subs( sparse_matrix *U, real *y, real *x )
{
    int i, pj, j, si, ei;
    real val;

    for ( i = U->n - 1; i >= 0; --i )
    {
        x[i] = y[i];
        si = U->start[i];
        ei = U->end[i];
        for ( pj = si + 1; pj < ei; ++pj )
        {
            j = U->entries[pj].j;
            val = U->entries[pj].val;
            x[i] -= val * x[j];
        }
        x[i] /= U->entries[si].val;
    }
}*/


/*int PCG( reax_system *system, storage *workspace,
        sparse_matrix *H, real *b, real tol,
        sparse_matrix *L, sparse_matrix *U, real *x,
        mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, n, N, scale;
    real tmp, alpha, beta, b_norm, r_norm, sig_old, sig_new;
    MPI_Comm world;

#if defined(DEBUG_FOCUS)
    int me;
    me = system->my_rank;
#endif

    n = system->n;
    N = system->N;
    world = mpi_data->world;
    scale = sizeof(real) / sizeof(void);
    b_norm = Parallel_Norm( b, n, world );
#if defined(DEBUG_FOCUS)
    if ( me == MASTER_NODE )
    {
        fprintf( stderr, "init_PCG: n=%d, N=%d\n", n, N );
        fprintf( stderr, "init_PCG: |b|=%24.15e\n", b_norm );
    }
    MPI_Barrier( world );
#endif

    Sparse_MatVec( H, x, workspace->q, N );
    //Coll( system, workspace, mpi_data, workspace->q );
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, n );
    r_norm = Parallel_Norm( workspace->r, n, world );

    Forward_Subs( L, workspace->r, workspace->d );
    Backward_Subs( U, workspace->d, workspace->p );
    sig_new = Parallel_Dot( workspace->r, workspace->p, n, world );
#if defined(DEBUG_FOCUS)
    if ( me == MASTER_NODE )
    {
        fprintf( stderr, "init_PCG: sig_new=%.15e\n", r_norm );
        fprintf( stderr, "init_PCG: |d|=%.15e |q|=%.15e\n",
                Parallel_Norm(workspace->d, n, world),
                Parallel_Norm(workspace->q, n, world) );
    }
    MPI_Barrier( world );
#endif

    for ( i = 1; i < 100 && r_norm / b_norm > tol; ++i )
    {
        Dist( system, mpi_data, workspace->p, MPI_DOUBLE, scale, real_packer );
        Sparse_MatVec( H, workspace->p, workspace->q, N );
        // tryQEq
        //Coll(system,mpi_data,workspace->q, MPI_DOUBLE, real_unpacker);
        tmp = Parallel_Dot( workspace->q, workspace->p, n, world );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->p, n );
#if defined(DEBUG_FOCUS)
        if ( me == MASTER_NODE )
            fprintf(stderr, "iter%d: |p|=%.15e |q|=%.15e tmp=%.15e\n",
                    i, Parallel_Norm(workspace->p, n, world),
                    Parallel_Norm(workspace->q, n, world), tmp );
        MPI_Barrier( world );
#endif

        Vector_Add( workspace->r, -alpha, workspace->q, n );
        r_norm = Parallel_Norm( workspace->r, n, world );
#if defined(DEBUG_FOCUS)
        if ( me == MASTER_NODE )
            fprintf( stderr, "iter%d: res=%.15e\n", i, r_norm );
        MPI_Barrier( world );
#endif

        Forward_Subs( L, workspace->r, workspace->d );
        Backward_Subs( U, workspace->d, workspace->d );
        sig_old = sig_new;
        sig_new = Parallel_Dot( workspace->r, workspace->d, n, world );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->p, 1., workspace->d, beta, workspace->p, n );
    }

#if defined(DEBUG_FOCUS)
    if ( me == MASTER_NODE )
        fprintf( stderr, "PCG took %d iterations\n", i );
#endif
    if ( i >= 100 )
        fprintf( stderr, "PCG convergence failed!\n" );

    return i;
}*/


#if defined(OLD_STUFF)
/*int sCG( reax_system *system, storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new;

    b_norm = Norm( b, system->n );
#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "n=%d, N=%d\n", system->n, system->N );
        fprintf( stderr, "p%d CGinit: b_norm=%24.15e\n", system->my_rank, b_norm );
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //Vector_Print( stderr, "q", workspace->q, system->N );
    }
    MPI_Barrier( mpi_data->world );
#endif

    Sparse_MatVec( H, x, workspace->q, system->N );
    //Coll_Vector( system, workspace, mpi_data, workspace->q );

    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );
    for ( j = 0; j < system->n; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j]; //pre-condition

    sig_new = Dot( workspace->r, workspace->d, system->n );
#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "p%d CGinit:sig_new=%24.15e\n", system->my_rank, sig_new );
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //Vector_Print( stderr, "q", workspace->q, system->N );
    }
    MPI_Barrier( mpi_data->world );
#endif

    for ( i = 1; i < 100 && sqrt(sig_new) / b_norm > tol; ++i )
    {
        //Dist_Vector( system, mpi_data, workspace->d );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
        //Coll_Vector( system, workspace, mpi_data, workspace->q );

        tmp = Dot( workspace->d, workspace->q, system->n );
        alpha = sig_new / tmp;
#if defined(DEBUG)
        if ( system->my_rank == MASTER_NODE )
        {
            fprintf(stderr,
                    "p%d CG iter%d:d_norm=%24.15e,q_norm=%24.15e,tmp = %24.15e\n",
                    system->my_rank, i,
                    Parallel_Norm(workspace->d, system->n, mpi_data->world),
                    Parallel_Norm(workspace->q, system->n, mpi_data->world), tmp );
            //Vector_Print( stderr, "d", workspace->d, system->N );
            //Vector_Print( stderr, "q", workspace->q, system->N );
        }
        MPI_Barrier( mpi_data->world );
#endif

        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        // pre-conditioning
        for ( j = 0; j < system->n; ++j )
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];

        sig_old = sig_new;
        sig_new = Dot( workspace->r, workspace->p, system->n );

        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );
#if defined(DEBUG)
        if ( system->my_rank == MASTER_NODE )
            fprintf(stderr, "p%d CG iter%d: sig_new = %24.15e\n",
                    system->my_rank, i, sig_new );
        MPI_Barrier( mpi_data->world );
#endif
    }

#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
        fprintf( stderr, "CG took %d iterations\n", i );
#endif
    if ( i >= 100 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}*/


/*int GMRES( reax_system *system, storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;

    N = system->N;
    bnorm = Norm( b, N );

    // apply the diagonal pre-conditioner to rhs
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];

    // GMRES outer-loop
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        // calculate r0
        Sparse_MatVec( H, x, workspace->b_prm, N );
        for ( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; // pre-conditioner

        Vector_Sum( workspace->v[0],
                1.,  workspace->b_prc, -1., workspace->b_prm, N );
        workspace->g[0] = Norm( workspace->v[0], N );
        Vector_Scale( workspace->v[0],
                1. / workspace->g[0], workspace->v[0], N );

        // fprintf( stderr, "%10.6f\n", workspace->g[0] );

        // GMRES inner-loop
        for ( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            // matvec
            Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1], N );

            for ( k = 0; k < N; ++k )
                workspace->v[j + 1][k] *= workspace->Hdia_inv[k]; // pre-conditioner
            // fprintf( stderr, "%d-%d: matvec done.\n", itr, j );

            // apply modified Gram-Schmidt to orthogonalize the new residual
            for ( i = 0; i <= j; i++ )
            {
                workspace->h[i][j] = Dot(workspace->v[i], workspace->v[j + 1], N);
                Vector_Add( workspace->v[j + 1],
                        -workspace->h[i][j], workspace->v[i], N );
            }

            workspace->h[j + 1][j] = Norm( workspace->v[j + 1], N );
            Vector_Scale( workspace->v[j + 1],
                    1. / workspace->h[j + 1][j], workspace->v[j + 1], N );
            // fprintf(stderr, "%d-%d: orthogonalization completed.\n", itr, j);

            // Givens rotations on the H matrix to make it U
            for ( i = 0; i <= j; i++ )
            {
                if ( i == j )
                {
                    cc = sqrt(SQR(workspace->h[j][j]) + SQR(workspace->h[j + 1][j]));
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

            // apply Givens rotations to the rhs as well
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j + 1] = tmp2;

            // fprintf( stderr, "%10.6f\n", fabs(workspace->g[j+1]) );
        }

        // solve Hy = g.
        //   H is now upper-triangular, do back-substitution
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = workspace->g[i];
            for ( k = j - 1; k > i; k-- )
                temp -= workspace->h[i][k] * workspace->y[k];
            workspace->y[i] = temp / workspace->h[i][i];
        }

        // update x = x_0 + Vy
        for ( i = 0; i < j; i++ )
            Vector_Add( x, workspace->y[i], workspace->v[i], N );

        // stopping condition
        if ( fabs(workspace->g[j]) / bnorm <= tol )
            break;
    }

    //Sparse_MatVec( system, H, x, workspace->b_prm, mpi_data );
    //  for( i = 0; i < N; ++i )
    //  workspace->b_prm[i] *= workspace->Hdia_inv[i];

    //  fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
     // for( i = 0; i < N; ++i )
      //fprintf( fout, "%10.5f%15.12f%15.12f\n",
      //workspace->b_prc[i], workspace->b_prm[i], x[i] );

    fprintf( fout, "GMRES outer: %d, inner: %d - |rel residual| = %15.10f\n",
            itr, j, fabs( workspace->g[j] ) / bnorm );

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        return FAILURE;
    }

    return SUCCESS;
}*/


/*int GMRES_HouseHolder( reax_system *system, storage *workspace,
        sparse_matrix *H, real *b, real tol, real *x,
        mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[RESTART + 2][10000], w[RESTART + 2];
    real u[RESTART + 2][10000];

    N = system->N;
    bnorm = Norm( b, N );

    // apply the diagonal pre-conditioner to rhs
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];

    // GMRES outer-loop
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        // compute z = r0
        Sparse_MatVec( H, x, workspace->b_prm, N );

        for ( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; // pre-conditioner

        Vector_Sum( z[0], 1.,  workspace->b_prc, -1., workspace->b_prm, N );

        Vector_MakeZero( w, RESTART + 1 );
        w[0] = Norm( z[0], N );

        Vector_Copy( u[0], z[0], N );
        u[0][0] += ( u[0][0] < 0.0 ? -1 : 1 ) * w[0];
        Vector_Scale( u[0], 1 / Norm( u[0], N ), u[0], N );

        w[0]    *= ( u[0][0] < 0.0 ?  1 : -1 );
        // fprintf( stderr, "\n\n%12.6f\n", w[0] );

        // GMRES inner-loop
        for ( j = 0; j < RESTART && fabs( w[j] ) / bnorm > tol; j++ )
        {
            // compute v_j
            Vector_Scale( z[j], -2 * u[j][j], u[j], N );
            z[j][j] += 1.; // due to e_j

            for ( i = j - 1; i >= 0; --i )
                Vector_Add( z[j] + i, -2 * Dot( u[i] + i, z[j] + i, N - i ), u[i] + i, N - i );

            // matvec
            Sparse_MatVec( H, z[j], v, N );

            for ( k = 0; k < N; ++k )
                v[k] *= workspace->Hdia_inv[k]; // pre-conditioner

            for ( i = 0; i <= j; ++i )
                Vector_Add( v + i, -2 * Dot( u[i] + i, v + i, N - i ), u[i] + i, N - i );

            if ( !Vector_isZero( v + (j + 1), N - (j + 1) ) )
            {
                // compute the HouseHolder unit vector u_j+1
                for ( i = 0; i <= j; ++i )
                    u[j + 1][i] = 0;

                Vector_Copy( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) );

                u[j + 1][j + 1] +=
                    ( v[j + 1] < 0.0 ? -1 : 1 ) * Norm( v + (j + 1), N - (j + 1) );

                Vector_Scale( u[j + 1], 1 / Norm( u[j + 1], N ), u[j + 1], N );

                // overwrite v with P_m+1 * v
                v[j + 1] -=
                    2 * Dot( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) ) * u[j + 1][j + 1];
                Vector_MakeZero( v + (j + 2), N - (j + 2) );
            }


            // previous Givens rotations on H matrix to make it U
            for ( i = 0; i < j; i++ )
            {
                tmp1 =  workspace->hc[i] * v[i] + workspace->hs[i] * v[i + 1];
                tmp2 = -workspace->hs[i] * v[i] + workspace->hc[i] * v[i + 1];

                v[i]   = tmp1;
                v[i + 1] = tmp2;
            }

            // apply the new Givens rotation to H and right-hand side
            if ( fabs(v[j + 1]) >= ALMOST_ZERO )
            {
                cc = sqrt( SQR( v[j] ) + SQR( v[j + 1] ) );
                workspace->hc[j] = v[j] / cc;
                workspace->hs[j] = v[j + 1] / cc;

                tmp1 =  workspace->hc[j] * v[j] + workspace->hs[j] * v[j + 1];
                tmp2 = -workspace->hs[j] * v[j] + workspace->hc[j] * v[j + 1];

                v[j]   = tmp1;
                v[j + 1] = tmp2;

                / Givens rotations to rhs
                tmp1 =  workspace->hc[j] * w[j];
                tmp2 = -workspace->hs[j] * w[j];
                w[j]   = tmp1;
                w[j + 1] = tmp2;
            }

            // extend R
            for ( i = 0; i <= j; ++i )
                workspace->h[i][j] = v[i];


            // fprintf( stderr, "h:" );
            // for( i = 0; i <= j+1 ; ++i )
            // fprintf( stderr, "%.6f ", h[i][j] );
            // fprintf( stderr, "\n" );
            // fprintf( stderr, "%12.6f\n", w[j+1] );
        }


        // solve Hy = w.
        //   H is now upper-triangular, do back-substitution 
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = w[i];
            for ( k = j - 1; k > i; k-- )
                temp -= workspace->h[i][k] * workspace->y[k];

            workspace->y[i] = temp / workspace->h[i][i];
        }

        for ( i = j - 1; i >= 0; i-- )
            Vector_Add( x, workspace->y[i], z[i], N );

        // stopping condition
        if ( fabs( w[j] ) / bnorm <= tol )
            break;
    }

    // Sparse_MatVec( system, H, x, workspace->b_prm );
    // for( i = 0; i < N; ++i )
    // workspace->b_prm[i] *= workspace->Hdia_inv[i];

    // fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
    // for( i = 0; i < N; ++i )
    // fprintf( fout, "%10.5f%15.12f%15.12f\n",
    //          workspace->b_prc[i], workspace->b_prm[i], x[i] );

    fprintf( fout, "GMRES outer:%d  inner:%d iters, |rel residual| = %15.10f\n",
            itr, j, fabs( workspace->g[j] ) / bnorm );

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        return FAILURE;
    }

    return SUCCESS;
}*/
#endif
