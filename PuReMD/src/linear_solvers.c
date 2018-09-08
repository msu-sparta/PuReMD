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

#if defined(CG_PERFORMANCE)
real t_start, t_elapsed, matvec_time, dot_time;
#endif

int compare_dbls( const void* arg1, const void* arg2 )
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

void qsort_dbls( double *array, int array_len )
{
    qsort( array, (size_t)array_len, sizeof(double),
            compare_dbls );
}

int find_bucket( double *list, int len, double a )
{
    int s, e, m;

    if( len == 0 )
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

void setup_sparse_approx_inverse( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data, sparse_matrix *A, sparse_matrix **A_spar_patt,
        int nprocs, double filter )
{
    /* Print_Sparse_Matrix2( system, A, "Charge_Matrix_MPI_Step0.txt" ); */

    int i, bin, total, pos, push;
    int n, n_gather, m, s_local, s, n_local;
    int target_proc;
    int k;
    int pj, size;
    int left, right, p, turn;

    real threshold, pivot, tmp;
    real *input_array;
    real *samplelist_local, *samplelist;
    real *pivotlist;
    real *bucketlist_local, *bucketlist;
    real *local_entries;

    int *srecv, *sdispls;
    int *scounts_local, *scounts;
    int *dspls_local, *dspls;
    int *bin_elements;

    MPI_Comm comm;

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
    local_entries = NULL;

    comm = mpi_data->world;

    if ( *A_spar_patt == NULL )
    {
        //TODO
        Allocate_Matrix2(A_spar_patt, A->n, system->local_cap, A->m, comm );
    }

    else if ( (*A_spar_patt)->m < A->m )
    {
        //TODO
        Deallocate_Matrix( *A_spar_patt );
        Allocate_Matrix2( A_spar_patt, A->n, system->local_cap, A->m, comm );
        //Reallocate_Matrix( A_spar_patt, A->n, A->n_max, A->m );
    }

    m = 0;
    for( i = 0; i < A->n; ++i )
    {
        m += A->end[i] - A->start[i];
    }
    /* the sample ratio is 10% */
    /*n_local = m/10; */
    n_local = m;
    s_local = (int) (12.0 * log2(n_local*nprocs));
    MPI_Allreduce(&n_local, &n, 1, MPI_INT, MPI_SUM, comm);
    MPI_Reduce(&s_local, &s, 1, MPI_INT, MPI_SUM, MASTER_NODE, comm);

    /* count num. bin elements for each processor, uniform bin sizes */
    input_array = malloc( sizeof(real) * n_local );
    scounts_local = malloc( sizeof(int) * nprocs );
    scounts = malloc( sizeof(int) * nprocs );
    bin_elements = malloc( sizeof(int) * nprocs );
    dspls_local = malloc( sizeof(int) * nprocs );
    bucketlist_local = malloc( sizeof(real) * n_local  );
    dspls = malloc( sizeof(int) * nprocs );
    pivotlist = malloc( sizeof(real) *  (nprocs - 1) );
    samplelist_local = malloc( sizeof(real) * s_local );
    local_entries = malloc ( sizeof(real) * m );
    if ( system->my_rank == MASTER_NODE )
    {
        samplelist = malloc( sizeof(real) * s );
        srecv = malloc( sizeof(int) * nprocs );
        sdispls = malloc( sizeof(int) * nprocs );
    }

    push = 0;
    for( i = 0; i < A->n; ++i )
    {
        for( pj = A->start[i]; pj < A->end[i]; ++pj )
        {
            local_entries[push++] = A->entries[pj].val;
        }
    }

    srand( time(NULL) + system->my_rank );
    for ( i = 0; i < n_local ; i++ )
    {
        /* input_array[i] = local_entries[rand( ) % m]; */
        input_array[i] = local_entries[ i ];
    }

    for ( i = 0; i < s_local; i++)
    {
        /* samplelist_local[i] = input_array[rand( ) % n_local]; */
        samplelist_local[i] = input_array[ i ];
    }

    /* gather samples at the root process */
    MPI_Gather( &s_local, 1, MPI_INT, srecv, 1, MPI_INT, MASTER_NODE, comm );

    if( system->my_rank == MASTER_NODE )
    {
        sdispls[0] = 0;
        for ( i = 0; i < nprocs - 1; ++i )
        {
            sdispls[i + 1] = sdispls[i] + srecv[i];
        }
    }

    MPI_Gatherv( samplelist_local, s_local, MPI_DOUBLE,
            samplelist, srecv, sdispls, MPI_DOUBLE, MASTER_NODE, comm);

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
    MPI_Bcast( pivotlist, nprocs - 1, MPI_DOUBLE, MASTER_NODE, comm );

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
    MPI_Allreduce( MPI_IN_PLACE, scounts, nprocs, MPI_INT, MPI_SUM, comm );

    /* find the target process */
    target_proc = 0;
    total = 0;
    k = n*filter;
    for(i = nprocs - 1; i >= 0; --i )
    {
        if( total + scounts[i] >= k )
        {
            /* global k becomes local k*/
            k -= total;
            target_proc = i;
            break;
        }
        total += scounts[i];
    }

    n_gather = scounts[target_proc];
    if( system->my_rank == target_proc )
    {
        bucketlist = malloc( sizeof( real ) * n_gather );
    }

    /* send local buckets to target processor for quickselect */
    MPI_Gather( scounts_local + target_proc, 1, MPI_INT, scounts, 1, MPI_INT, target_proc, comm );
    if ( system->my_rank == target_proc )
    {
        dspls[0] = 0;
        for ( i = 0; i < nprocs - 1; ++i )
        {
            dspls[i + 1] = dspls[i] + scounts[i];
        }
    }

    MPI_Gatherv( bucketlist_local + dspls_local[target_proc], scounts_local[target_proc], MPI_DOUBLE,
            bucketlist, scounts, dspls, MPI_DOUBLE, target_proc, comm);

    /* apply quick select algorithm at the target process */
    if( system->my_rank == target_proc)
    {
        left = 0;
        right = n_gather-1;

        turn = 0;
        while( k ) {

            p  = left;
            turn = 1 - turn;

            /* alternating pivots in order to handle corner cases */
            if( turn == 1)
            {
                pivot = bucketlist[right];
            }
            else
            {
                pivot = bucketlist[left];
            }
            for( i = left + 1 - turn; i <= right-turn; ++i )
            {
                if( bucketlist[i] > pivot )
                {
                    tmp = bucketlist[i];
                    bucketlist[i] = bucketlist[p];
                    bucketlist[p] = tmp;
                    p++;
                }
            }
            if(turn == 1)
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

    /*if(system->my_rank == target_proc)
      fprintf(stdout,"threshold = %.15lf\n", threshold);*/
    /*broadcast the filtering value*/
    MPI_Bcast( &threshold, 1, MPI_DOUBLE, target_proc, comm );

    int nnz = 0;
    /*build entries of that pattern*/
    for ( i = 0; i < A->n; ++i )
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
                nnz++;
            }
        }
        (*A_spar_patt)->end[i] = size;
    }

    MPI_Allreduce( MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, comm );
    if( system->my_rank == MASTER_NODE )
    {
        fprintf(stdout,"total nnz in all sparsity patterns = %d\nthreshold = %.15lf\n", nnz, threshold);
        fflush(stdout);
    }

}

void sparse_approx_inverse(reax_system *system, storage *workspace, mpi_datatypes *mpi_data, 
        sparse_matrix *A, sparse_matrix *A_spar_patt, sparse_matrix **A_app_inv )
{
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

    int d;
    mpi_out_data *out_bufs;
    MPI_Comm comm;
    MPI_Request req1, req2, req3, req4;
    int flag1, flag2;
    MPI_Status stat1, stat2, stat3, stat4;
    const neighbor_proc *nbr1, *nbr2;
    int *j_send, *j_recv1, *j_recv2;
    real *val_send, *val_recv1, *val_recv2;

    real start;

    /* start = Get_Time( ); */

    comm = mpi_data->world;

    if ( *A_app_inv == NULL)
    {
        Allocate_Matrix2( A_app_inv, A_spar_patt->n, system->local_cap, A_spar_patt->m, comm );
    }

    else if ( (*A_app_inv)->m < A_spar_patt->m )
    {
        Deallocate_Matrix( *A_app_inv );
        Allocate_Matrix2( A_app_inv, A_spar_patt->n, system->local_cap, A_spar_patt->m, comm );
    }

    pos_x = NULL;
    X = NULL;

    row_nnz = NULL;
    j_list = NULL;
    val_list = NULL;

    j_send = NULL;
    val_send = NULL;
    j_recv1 = NULL;
    j_recv2 = NULL;
    val_recv1 = NULL;
    val_recv2 = NULL;

    row_nnz = (int *) malloc( sizeof(int) * system->total_cap );

    j_list = (int **) malloc( sizeof(int *) * system->N );
    val_list = (real **) malloc( sizeof(real *) * system->N );

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
    scale = sizeof(int) / sizeof(void);
    Dist( system, mpi_data, row_nnz, MPI_INT, scale, int_packer );

    comm = mpi_data->comm_mesh3D;
    out_bufs = mpi_data->out_buffers;

    /*  use a Dist-like approach to send the row information */
    for ( d = 0; d < 3; ++d)
    {
        flag1 = 0;
        flag2 = 0;
        cnt = 0;

        /* initiate recvs */
        nbr1 = &system->my_nbrs[2 * d];
        if ( nbr1->atoms_cnt )
        {
            /* calculate the total data that will be received */
            cnt = 0;
            for( i = nbr1->atoms_str; i < (nbr1->atoms_str + nbr1->atoms_cnt); ++i )
            {
                if( i >= A->n)
                {
                    cnt += row_nnz[i];
                }
            }

            /* initiate Irecv */
            if( cnt )
            {
                flag1 = 1;
                j_recv1 = (int *) malloc( sizeof(int) * cnt );
                val_recv1 = (real *) malloc( sizeof(real) * cnt );
                MPI_Irecv( j_recv1, cnt, MPI_INT, nbr1->rank, 2 * d + 1, comm, &req1 );
                MPI_Irecv( val_recv1, cnt, MPI_DOUBLE, nbr1->rank, 2 * d + 1, comm, &req2 );
            }
        }

        nbr2 = &system->my_nbrs[2 * d + 1];
        if ( nbr1->atoms_cnt )
        {
            /* calculate the total data that will be received */
            cnt = 0;
            for( i = nbr2->atoms_str; i < (nbr2->atoms_str + nbr2->atoms_cnt); ++i )
            {
                if( i >= A->n )
                {
                    cnt += row_nnz[i];
                }
            }

            /* initiate Irecv */
            if( cnt )
            {
                flag2 = 1;
                j_recv2 = (int *) malloc( sizeof(int) * cnt );
                val_recv2 = (real *) malloc( sizeof(real) * cnt );
                MPI_Irecv( j_recv2, cnt, MPI_INT, nbr2->rank, 2 * d, comm, &req3 );
                MPI_Irecv( val_recv2, cnt, MPI_DOUBLE, nbr2->rank, 2 * d, comm, &req4 );
            }
        }

        /* send both messages in dimension d */
        if( out_bufs[2 * d].cnt )
        {
            cnt = 0;
            for( i = 0; i < out_bufs[2 * d].cnt; ++i )
            {
                cnt += row_nnz[ out_bufs[2 * d].index[i] ];
            }

            if( cnt )
            {
                j_send = (int *) malloc( sizeof(int) * cnt );
                val_send = (real *) malloc( sizeof(real) * cnt );

                cnt = 0;
                for( i = 0; i < out_bufs[2 * d].cnt; ++i )
                {
                    if( out_bufs[2 * d].index[i] < A->n )
                    {
                        for( pj = A->start[ out_bufs[2 * d].index[i] ]; pj < A->end[ out_bufs[2 * d].index[i] ]; ++pj )
                        {
                            atom = &system->my_atoms[ A->entries[pj].j ];
                            j_send[cnt] = atom->orig_id;
                            val_send[cnt] = A->entries[pj].val;
                            cnt++;
                        }
                    }
                    else
                    {
                        for( pj = 0; pj < row_nnz[ out_bufs[2 * d].index[i] ]; ++pj )
                        {
                            j_send[cnt] = j_list[ out_bufs[2 * d].index[i] ][pj];
                            val_send[cnt] = val_list[ out_bufs[2 * d].index[i] ][pj];
                            cnt++;
                        }
                    }
                }

                MPI_Send( j_send, cnt, MPI_INT, nbr1->rank, 2 * d, comm );
                MPI_Send( val_send, cnt, MPI_DOUBLE, nbr1->rank, 2 * d, comm );
            }
        }

        if( out_bufs[2 * d + 1].cnt )
        {
            cnt = 0;
            for( i = 0; i < out_bufs[2 * d + 1].cnt; ++i )
            {
                cnt += row_nnz[ out_bufs[2 * d + 1].index[i] ];
            }

            if( cnt )
            {
                j_send = (int *) malloc( sizeof(int) * cnt );
                val_send = (real *) malloc( sizeof(real) * cnt );

                cnt = 0;
                for( i = 0; i < out_bufs[2 * d + 1].cnt; ++i )
                {
                    if( out_bufs[2 * d + 1].index[i] < A->n )
                    {
                        for( pj = A->start[ out_bufs[2 * d + 1].index[i] ]; pj < A->end[ out_bufs[2 * d + 1].index[i] ]; ++pj )
                        {
                            atom = &system->my_atoms[ A->entries[pj].j ];
                            j_send[cnt] = atom->orig_id;
                            val_send[cnt] = A->entries[pj].val;
                            cnt++;
                        }
                    }
                    else
                    {
                        for( pj = 0; pj < row_nnz[ out_bufs[2 * d + 1].index[i] ]; ++pj )
                        {
                            j_send[cnt] = j_list[ out_bufs[2 * d + 1].index[i] ][pj];
                            val_send[cnt] = val_list[ out_bufs[2 * d + 1].index[i] ][pj];
                            cnt++;
                        }
                    }
                }

                MPI_Send( j_send, cnt, MPI_INT, nbr1->rank, 2 * d + 1, comm );
                MPI_Send( val_send, cnt, MPI_DOUBLE, nbr1->rank, 2 * d + 1, comm );
            }

        }

        if( flag1 )
        {
            MPI_Wait( &req1, &stat1 );
            MPI_Wait( &req2, &stat2 );

            cnt = 0;
            for( i = nbr1->atoms_str; i < (nbr1->atoms_str + nbr1->atoms_cnt); ++i )
            {
                if( i >= A->n )
                {
                    j_list[i] = (int *) malloc( sizeof(int) *  row_nnz[i] );
                    val_list[i] = (real *) malloc( sizeof(real) * row_nnz[i] );

                    for( pj = 0; pj < row_nnz[i]; ++pj )
                    {
                        j_list[i][pj] = j_recv1[cnt];
                        val_list[i][pj] = val_recv1[cnt];
                        cnt++;
                    }
                }
            }
        }

        if( flag2 )
        {
            MPI_Wait( &req3, &stat3 );
            MPI_Wait( &req4, &stat4 );


            cnt = 0;
            for( i = nbr2->atoms_str; i < (nbr2->atoms_str + nbr2->atoms_cnt); ++i )
            {
                if( i >= A->n )
                {
                    j_list[i] = (int *) malloc( sizeof(int) *  row_nnz[i] );
                    val_list[i] = (real *) malloc( sizeof(real) * row_nnz[i] );

                    for( pj = 0; pj < row_nnz[i]; ++pj )
                    {
                        j_list[i][pj] = j_recv2[cnt];
                        val_list[i][pj] = val_recv2[cnt];
                        cnt++;
                    }
                }
            }
        }
    }

    X = (int *) malloc( sizeof(int) * (system->bigN + 1) );
    pos_x = (int *) malloc( sizeof(int) * (system->bigN + 1) );

    for ( i = 0; i < A_spar_patt->n; ++i )
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
             *              * search through the row of full A of that index */

            /* the case where the local matrix has that index's row */
            if( j_temp < A->n )
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
                fprintf( stdout, "THE LOCAL POSITION OF THE ATOM IS NOT VALID, STOP THE EXECUTION\n");
                fflush( stdout );
            }
            if( local_pos < A->n )
            {
                for ( d_i = A->start[local_pos]; d_i < A->end[local_pos]; ++d_i )
                {
                    atom = &system->my_atoms[ A->entries[d_i].j ];
                    if (pos_x[ atom->orig_id ] >= M || d_j >=  N )
                    {
                        fprintf( stdout, "CANNOT MAP IT TO THE DENSE MATRIX, STOP THE EXECUTION, orig_id = %d, i =  %d, j = %d, M = %d N = %d\n", atom->orig_id, pos_x[ atom->orig_id ], d_j, M, N );
                        fflush( stdout );
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
                        fprintf( stdout, "CANNOT MAP IT TO THE DENSE MATRIX, STOP THE EXECUTION, %d %d\n", pos_x[ j_list[local_pos][d_i] ], d_j);
                        fflush( stdout );
                    }
                    if ( X[ j_list[local_pos][d_i] ] == 1 )
                    {
                        dense_matrix[ pos_x[ j_list[local_pos][d_i] ] * N + d_j ] = val_list[local_pos][d_i];
                    }
                }
            }
        }

        /* create the right hand side of the linear equation
         *          * that is the full column of the identity matrix */
        e_j = (real *) malloc( sizeof(real) * M );

        for ( k = 0; k < M; ++k )
        {
            e_j[k] = 0.0;
        }
        e_j[identity_pos] = 1.0;

        /* Solve the overdetermined system AX = B through the least-squares problem:
         *          *          * min ||B - AX||_2 */
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

    /* return Get_Timing_Info( start ); */
}

void dual_Sparse_MatVec( sparse_matrix *A, rvec2 *x, rvec2 *b, int N )
{
    int  i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i][0] = b[i][1] = 0;
    }

    /* perform multiplication */
    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];
#if defined(HALF_LIST)
        b[i][0] += A->entries[si].val * x[i][0];
        b[i][1] += A->entries[si].val * x[i][1];
#endif

#if defined(HALF_LIST)
        for ( k = si + 1; k < A->end[i]; ++k )
#else
            for ( k = si; k < A->end[i]; ++k )
#endif
            {
                j = A->entries[k].j;
                H = A->entries[k].val;

                b[i][0] += H * x[j][0];
                b[i][1] += H * x[j][1];

#if defined(HALF_LIST)
                // comment out for tryQEq
                //if( j < A->n ) {
                b[j][0] += H * x[i][0];
                b[j][1] += H * x[i][1];
                //}
#endif
            }
    }
}


int dual_CG( reax_system *system, storage *workspace, sparse_matrix *H,
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
        t_start = Get_Time( );
    }
#endif

    Dist( system, mpi_data, x, mpi_data->mpi_rvec2, scale, rvec2_packer );
    dual_Sparse_MatVec( H, x, workspace->q2, N );
#if defined(HALF_LIST)
    // tryQEq
    Coll(system, mpi_data, workspace->q2, mpi_data->mpi_rvec2, scale, rvec2_unpacker);
#endif

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &matvec_time );
    }
#endif

    for ( j = 0; j < system->n; ++j )
    {
        /* residual */
        workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
        workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];
        /* apply diagonal pre-conditioner */
        workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
        workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
    }

    /* norm of b */
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

    /* dot product: r.d */
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
        Update_Timing_Info( &t_start, &dot_time );
#endif

    for ( i = 1; i < 300; ++i )
    {
        Dist(system, mpi_data, workspace->d2, mpi_data->mpi_rvec2, scale, rvec2_packer);
        dual_Sparse_MatVec( H, workspace->d2, workspace->q2, N );
#if defined(HALF_LIST)
        // tryQEq
        Coll(system, mpi_data, workspace->q2, mpi_data->mpi_rvec2, scale, rvec2_unpacker);
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        /* dot product: d.q */
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
            /* update x */
            x[j][0] += alpha[0] * workspace->d2[j][0];
            x[j][1] += alpha[1] * workspace->d2[j][1];
            /* update residual */
            workspace->r2[j][0] -= alpha[0] * workspace->q2[j][0];
            workspace->r2[j][1] -= alpha[1] * workspace->q2[j][1];
            /* apply diagonal pre-conditioner */
            workspace->p2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
            workspace->p2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
            /* dot product: r.p */
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
            Update_Timing_Info( &t_start, &dot_time );
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
            /* d = p + beta * d */
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
                workspace->t,mpi_data, fout );
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
                mpi_data, fout );
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
}


void Sparse_MatVec( sparse_matrix *A, real *x, real *b, int N )
{
    int  i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i] = 0;
    }

    /* perform multiplication */
    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];

#if defined(HALF_LIST)
        b[i] += A->entries[si].val * x[i];
#endif

#if defined(HALF_LIST)
        for ( k = si + 1; k < A->end[i]; ++k )
#else
            for ( k = si; k < A->end[i]; ++k )
#endif
            {
                j = A->entries[k].j;
                H = A->entries[k].val;

                b[i] += H * x[j];
#if defined(HALF_LIST)
                //if( j < A->n ) // comment out for tryQEq
                b[j] += H * x[i];
#endif
            }
    }
}


/* sparse matrix-vector product Ax = b
 * where:
 *   A: matrix, stored in CSR format
 *   x: vector
 *   b: vector (result) */
static void Sparse_MatVec_full( const sparse_matrix * const A,
        const real * const x, real * const b )
{
    //TODO: implement full SpMV in MPI
    //    int i, pj;
    //
    //    Vector_MakeZero( b, A->n );
    //
    //#ifdef _OPENMP
    //    #pragma omp for schedule(static)
    //#endif
    //    for ( i = 0; i < A->n; ++i )
    //    {
    //        for ( pj = A->start[i]; pj < A->start[i + 1]; ++pj )
    //        {
    //            b[i] += A->val[pj] * x[A->j[pj]];
    //        }
    //    }
}


int CG( reax_system *system, storage *workspace, sparse_matrix *H, real *b,
        real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, scale;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_start = matvec_time = dot_time = 0;
        t_start = Get_Time( );
    }
#endif

    scale = sizeof(real) / sizeof(void);
    Dist( system, mpi_data, x, MPI_DOUBLE, scale, real_packer );
    Sparse_MatVec( H, x, workspace->q, system->N );
#if defined(HALF_LIST)
    // tryQEq
    Coll( system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker );
#endif

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &matvec_time );
    }
#endif

    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );

    for ( j = 0; j < system->n; ++j )
    {
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j]; //pre-condition
    }
    //TODO: apply SAI preconditioner here, comment out diagonal preconditioning above
    //    Sparse_MatVec_full( workspace->H_app_inv, workspace->r, workspace->d );

    b_norm = Parallel_Norm( b, system->n, mpi_data->world );
    sig_new = Parallel_Dot(workspace->r, workspace->d, system->n, mpi_data->world);
    sig0 = sig_new;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < 300 && sqrt(sig_new) / b_norm > tol; ++i )
    {
        Dist( system, mpi_data, workspace->d, MPI_DOUBLE, scale, real_packer );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
#if defined(HALF_LIST)
        //tryQEq
        Coll(system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker);
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        tmp = Parallel_Dot(workspace->d, workspace->q, system->n, mpi_data->world);
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );

        /* pre-conditioning */
        for ( j = 0; j < system->n; ++j )
        {
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }
        //TODO: apply SAI preconditioner here, comment out diagonal preconditioning above
        //        Sparse_MatVec_full( workspace->H_app_inv, workspace->r, workspace->d );

        sig_old = sig_new;
        sig_new = Parallel_Dot(workspace->r, workspace->p, system->n, mpi_data->world);
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif
    }

    if ( i >= 300 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( fout, "QEq %d iters. matvecs: %f  dot: %f\n", i, matvec_time,
                dot_time );
    }
#endif

    return i;
}


int CG_test( reax_system *system, storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, scale;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

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
    sig0 = sig_new;
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
            t_start = Get_Time( );
#endif
        Dist( system, mpi_data, workspace->d, MPI_DOUBLE, scale, real_packer );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
        //tryQEq
        //Coll(system, mpi_data, workspace->q, MPI_DOUBLE, real_unpacker);
#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            t_elapsed = Get_Timing_Info( t_start );
            matvec_time += t_elapsed;
        }
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            t_start = Get_Time( );
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
        /* pre-conditioning */
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
            t_elapsed = Get_Timing_Info( t_start );
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
}


void Forward_Subs( sparse_matrix *L, real *b, real *y )
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
}


void Backward_Subs( sparse_matrix *U, real *y, real *x )
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
}


int PCG( reax_system *system, storage *workspace,
        sparse_matrix *H, real *b, real tol,
        sparse_matrix *L, sparse_matrix *U, real *x,
        mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, me, n, N, scale;
    real tmp, alpha, beta, b_norm, r_norm, sig_old, sig_new;
    MPI_Comm world;

    me = system->my_rank;
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
}


#if defined(OLD_STUFF)
int sCG( reax_system *system, storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

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
    sig0 = sig_new;
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
        /* pre-conditioning */
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
}


int GMRES( reax_system *system, storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;

    N = system->N;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* calculate r0 */
        Sparse_MatVec( H, x, workspace->b_prm, N );
        for ( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; // pre-conditioner

        Vector_Sum( workspace->v[0],
                1.,  workspace->b_prc, -1., workspace->b_prm, N );
        workspace->g[0] = Norm( workspace->v[0], N );
        Vector_Scale( workspace->v[0],
                1. / workspace->g[0], workspace->v[0], N );

        // fprintf( stderr, "%10.6f\n", workspace->g[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            /* matvec */
            Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1], N );

            for ( k = 0; k < N; ++k )
                workspace->v[j + 1][k] *= workspace->Hdia_inv[k]; // pre-conditioner
            // fprintf( stderr, "%d-%d: matvec done.\n", itr, j );

            /* apply modified Gram-Schmidt to orthogonalize the new residual */
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

            /* Givens rotations on the H matrix to make it U */
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

            /* apply Givens rotations to the rhs as well */
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j + 1] = tmp2;

            // fprintf( stderr, "%10.6f\n", fabs(workspace->g[j+1]) );
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

    /*Sparse_MatVec( system, H, x, workspace->b_prm, mpi_data );
      for( i = 0; i < N; ++i )
      workspace->b_prm[i] *= workspace->Hdia_inv[i];

      fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
      for( i = 0; i < N; ++i )
      fprintf( fout, "%10.5f%15.12f%15.12f\n",
      workspace->b_prc[i], workspace->b_prm[i], x[i] );*/

    fprintf( fout, "GMRES outer: %d, inner: %d - |rel residual| = %15.10f\n",
            itr, j, fabs( workspace->g[j] ) / bnorm );

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        return FAILURE;
    }

    return SUCCESS;
}


int GMRES_HouseHolder( reax_system *system, storage *workspace,
        sparse_matrix *H, real *b, real tol, real *x,
        mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[RESTART + 2][10000], w[RESTART + 2];
    real u[RESTART + 2][10000];

    N = system->N;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* compute z = r0 */
        Sparse_MatVec( H, x, workspace->b_prm, N );

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
            Sparse_MatVec( H, z[j], v, N );

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

                u[j + 1][j + 1] +=
                    ( v[j + 1] < 0.0 ? -1 : 1 ) * Norm( v + (j + 1), N - (j + 1) );

                Vector_Scale( u[j + 1], 1 / Norm( u[j + 1], N ), u[j + 1], N );

                /* overwrite v with P_m+1 * v */
                v[j + 1] -=
                    2 * Dot( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) ) * u[j + 1][j + 1];
                Vector_MakeZero( v + (j + 2), N - (j + 2) );
            }


            /* previous Givens rotations on H matrix to make it U */
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
                cc = sqrt( SQR( v[j] ) + SQR( v[j + 1] ) );
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

        for ( i = j - 1; i >= 0; i-- )
            Vector_Add( x, workspace->y[i], z[i], N );

        /* stopping condition */
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
}
#endif
