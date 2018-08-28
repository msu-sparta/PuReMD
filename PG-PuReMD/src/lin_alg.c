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

#include "reax_types.h"

#include "lin_alg.h"

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

#if defined(HAVE_CUDA) && defined(DEBUG)
#include "cuda/cuda_validation.h"
#endif

#if defined(CG_PERFORMANCE)
real t_start, t_elapsed, matvec_time, dot_time;
#endif

enum preconditioner_type
{
    LEFT = 0,
    RIGHT = 1,
};


static void Sparse_MatVec( const sparse_matrix * const A, const real * const x,
        real * const b, const int N )
{
    int i, j, k, si;
    real val;

    for ( i = 0; i < N; ++i )
    {
        b[i] = 0.0;
    }

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
            val = A->entries[k].val;

            b[i] += val * x[j];
#if defined(HALF_LIST)
            //if( j < A->n ) // comment out for tryQEq
            b[j] += val * x[i];
#endif
        }
    }
}


static void dual_Sparse_MatVec( const sparse_matrix * const A,
        const rvec2 * const x, rvec2 * const b, const int N )
{
    int  i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i][0] = 0.0;
        b[i][1] = 0.0;
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


/* Diagonal (Jacobi) preconditioner computation */
real diag_pre_comp( const reax_system * const system, real * const Hdia_inv )
{
    unsigned int i;
    real start;

    start = Get_Time( );

    for ( i = 0; i < system->n; ++i )
    {
        //        if ( H->entries[H->start[i + 1] - 1].val != 0.0 )
        //        {
        //            Hdia_inv[i] = 1.0 / H->entries[H->start[i + 1] - 1].val;
        Hdia_inv[i] = 1.0 / system->reax_param.sbp[ system->my_atoms[i].type ].eta;
        //        }
        //        else
        //        {
        //            Hdia_inv[i] = 1.0;
        //        }
    }

    return Get_Timing_Info( start );
}

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

void setup_sparse_approx_inverse( const reax_system * const system, storage * const workspace,
        const mpi_datatypes * const mpi_data, sparse_matrix *A, sparse_matrix *A_spar_patt,
        const int nprocs, const double filter )
{
    //Print_Sparse_Matrix2( system, A, "Charge_Matrix_MPI_Step0.txt" );

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

    if ( A_spar_patt->allocated == FALSE )
    {
        Allocate_Matrix(A_spar_patt, A->n, A->n_max, A->m );
    }

    else if ( (A_spar_patt->m) < (A->m) )
    {
        Reallocate_Matrix( A_spar_patt, A->n, A->n_max, A->m );
    }

    m = 0;
    for( i = 0; i < A->n; ++i )
    {
        m += A->end[i] - A->start[i];
    }
    /* the sample ratio is 10% */
    //n_local = m/10; 
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
        //input_array[i] = local_entries[rand( ) % m];
        input_array[i] = local_entries[ i ];
    }

    for ( i = 0; i < s_local; i++)
    {
        //samplelist_local[i] = input_array[rand( ) % n_local];
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
    
    /*find the target process*/
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

    /* send local buckets to target processor for quickselect*/
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

    /* apply quick select algorithm at the target process*/
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
        if(threshold < 1.000000)
        {                    
            threshold = 1.000001;                    
        }
    }

    /*broadcast the filtering value*/
    //threshold = 1.846110441744020;
    MPI_Bcast( &threshold, 1, MPI_DOUBLE, target_proc, comm );

    //int nnz = 0;
    /*build entries of that pattern*/
    for ( i = 0; i < A->n; ++i )
    {
        A_spar_patt->start[i] = A->start[i];
        size = A->start[i];

        for ( pj = A->start[i]; pj < A->end[i]; ++pj )
        {
            if ( ( A->entries[pj].val >= threshold )  || ( A->entries[pj].j == i ) ) 
            {
                A_spar_patt->entries[size].val = A->entries[pj].val;
                A_spar_patt->entries[size].j = A->entries[pj].j;
                size++;
                //nnz++;
            }
        }
        A_spar_patt->end[i] = size;
    }

    /*MPI_Allreduce( MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, comm );
    if( system->my_rank == MASTER_NODE )
    {
        fprintf(stdout,"total nnz in all sparsity patterns = %d\n", nnz);
        fflush(stdout);
    }*/
    
}


int sparse_approx_inverse(const reax_system * const system, storage * const workspace, 
        mpi_datatypes * const mpi_data, sparse_matrix * A, 
        sparse_matrix * A_spar_patt, sparse_matrix * A_app_inv )
{
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

    start = Get_Time( );
    
    if ( A_app_inv->allocated == FALSE)
    {
        Allocate_Matrix( A_app_inv, A_spar_patt->n, A_spar_patt->n_max, A_spar_patt->m );
    }

    else if ( (A_app_inv->m) < (A_spar_patt->m) )
    {
        Reallocate_Matrix( A_app_inv, A_spar_patt->n, A_spar_patt->n_max, A_spar_patt->m );
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
    Dist( system, mpi_data, row_nnz, INT_PTR_TYPE, MPI_INT );

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
                //if( i >= A->n)
                //{
                    cnt += row_nnz[i];
                //}
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
                //if( i >= A->n )
                //{
                    cnt += row_nnz[i];
                //}
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
                //if( i >= A->n )
                //{
                    j_list[i] = (int *) malloc( sizeof(int) *  row_nnz[i] );
                    val_list[i] = (real *) malloc( sizeof(real) * row_nnz[i] );

                    for( pj = 0; pj < row_nnz[i]; ++pj )
                    {
                        j_list[i][pj] = j_recv1[cnt];
                        val_list[i][pj] = val_recv1[cnt];
                        cnt++;
                    }
                //}
            }
        }

        if( flag2 )
        {
            MPI_Wait( &req3, &stat3 );
            MPI_Wait( &req4, &stat4 );


            cnt = 0;
            for( i = nbr2->atoms_str; i < (nbr2->atoms_str + nbr2->atoms_cnt); ++i )
            {
                //if( i >= A->n )
                //{
                    j_list[i] = (int *) malloc( sizeof(int) *  row_nnz[i] );
                    val_list[i] = (real *) malloc( sizeof(real) * row_nnz[i] );

                    for( pj = 0; pj < row_nnz[i]; ++pj )
                    {
                        j_list[i][pj] = j_recv2[cnt];
                        val_list[i][pj] = val_recv2[cnt];
                        cnt++;
                    }
                //}
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
             * search through the row of full A of that index */

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
        
        // printing dense matrix for a particular atom
        /*if(atom_pos == 3001)
        {
            int tcnt = 0;
            fprintf( stdout, "\nDENSE MATRIX IS %d by %d\n", M, N );
            for( d_i = 0; d_i < M; d_i++ )
            {
                for( d_j = 0; d_j < N; d_j++ )
                {
                    fprintf( stdout, "%.2lf ", dense_matrix[ d_i*N + d_j ]);
                    if( dense_matrix[d_i*N + d_j ] != 0.0 )
                        tcnt++;
                }
                fprintf( stdout, "\n" );
            }
            fprintf( stdout, "DENSE MATRIX NNZ = %d\n", tcnt );
            fflush( stdout );
        }*/
        
        /* create the right hand side of the linear equation
         * that is the full column of the identity matrix */
        e_j = (real *) malloc( sizeof(real) * M );

        for ( k = 0; k < M; ++k )
        {
            e_j[k] = 0.0;
        }
        e_j[identity_pos] = 1.0;

        /* Solve the overdetermined system AX = B through the least-squares problem:
         *          * min ||B - AX||_2 */
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
            A_app_inv->entries[k].j = A_spar_patt->entries[k].j;
            A_app_inv->entries[k].val = e_j[k - A_spar_patt->start[i]];
        }

        /* empty variables that will be used next iteration */
        free( dense_matrix );
        free( e_j );
    }

    free( pos_x);
    free( X );

    return Get_Timing_Info( start );
}

static void diag_pre_app( const real * const Hdia_inv, const real * const y,
        real * const x, const int N )
{
    unsigned int i;

    for ( i = 0; i < N; ++i )
    {
        x[i] = y[i] * Hdia_inv[i];
    }
}

static void apply_preconditioner( const reax_system * const system, const storage * const workspace, 
        const control_params * const control, const real * const y, real * const x, 
        const int fresh_pre, const int side )
{
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
                            case DIAG_PC:
                                //diag_pre_app( workspace->Hdia_inv, y, x, workspace->H->n );
                                diag_pre_app( workspace->Hdia_inv, y, x, system->n );
                                break;
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                  tri_solve( workspace->L, y, x, workspace->L->n, LOWER );
                                  break;*/
                            case SAI_PC:
                                Sparse_MatVec( &workspace->H_app_inv, y, x, system->n );
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
                                diag_pre_app( workspace->Hdia_inv, y, x, system->n );
                                break;
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                  tri_solve_level_sched( (static_storage *) workspace,
                                  workspace->L, y, x, workspace->L->n, LOWER, fresh_pre );
                                  break;*/
                            case SAI_PC:
                                Sparse_MatVec( &workspace->H_app_inv, y, x, system->n );
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
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                  for ( i = 0; i < workspace->H->n; ++i )
                                  {
                                  workspace->y_p[i] = y[i];
                                  }

                                  permute_vector( workspace, workspace->y_p, workspace->H->n, FALSE, LOWER );
                                  tri_solve_level_sched( (static_storage *) workspace,
                                  workspace->L, workspace->y_p, x, workspace->L->n, LOWER, fresh_pre );
                                  break;*/
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
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                // construct D^{-1}_L
                                if ( fresh_pre == TRUE )
                                {
                                for ( i = 0; i < workspace->L->n; ++i )
                                {
                                si = workspace->L->start[i + 1] - 1;
                                workspace->Dinv_L[i] = 1.0 / workspace->L->val[si];
                                }
                                }

                                jacobi_iter( workspace, workspace->L, workspace->Dinv_L,
                                y, x, LOWER, control->cm_solver_pre_app_jacobi_iters );
                                break;*/
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
                                    Vector_Copy( x, y, system->n );
                                }
                                break;
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                  tri_solve( workspace->U, y, x, workspace->U->n, UPPER );
                                  break;*/
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
                                    Vector_Copy( x, y, system->n );
                                }
                                break;
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                  tri_solve_level_sched( (static_storage *) workspace,
                                  workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
                                  break;*/
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
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                  tri_solve_level_sched( (static_storage *) workspace,
                                  workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
                                  permute_vector( workspace, x, workspace->H->n, TRUE, UPPER );
                                  break;*/
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
                                /*case ICHOLT_PC:
                                  case ILU_PAR_PC:
                                  case ILUT_PAR_PC:
                                  if ( fresh_pre == TRUE )
                                  {
                                  for ( i = 0; i < workspace->U->n; ++i )
                                  {
                                  si = workspace->U->start[i];
                                  workspace->Dinv_U[i] = 1.0 / workspace->U->val[si];
                                  }
                                  }

                                  jacobi_iter( workspace, workspace->U, workspace->Dinv_U,
                                  y, x, UPPER, control->cm_solver_pre_app_jacobi_iters );
                                  break;*/
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

int dual_CG( const reax_system * const system, const control_params * const control,
        const storage * const workspace, const simulation_data * const data,
        mpi_datatypes * const mpi_data,
        const sparse_matrix * const H, const rvec2 * const b,
        const real tol, rvec2 * const x, const int fresh_pre )
{
    int i, j, n, N, iters;
    rvec2 tmp, alpha, beta;
    rvec2 my_sum, norm_sqr, b_norm, my_dot;
    rvec2 sig_old, sig_new;
    MPI_Comm comm;

    real *d, *r, *p;


    n = system->n;
    N = system->N;
    comm = mpi_data->world;
    iters = 0;

    d = (real *) malloc( sizeof(real) * n);
    r = (real *) malloc( sizeof(real) * n);
    p = (real *) malloc( sizeof(real) * n);


#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        iters = 0;
        t_start = matvec_time = dot_time = 0;
        t_start = Get_Time( );
    }
#endif

#if defined(HAVE_CUDA) && defined(DEBUG)
    check_zeros_host( x, N, "x" );
#endif

    Dist( system, mpi_data, x, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(HAVE_CUDA) && defined(DEBUG)
    check_zeros_host( x, N, "x" );
#endif

    dual_Sparse_MatVec( H, x, workspace->q2, N );

#if defined(HALF_LIST)
    // tryQEq
    Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE,
            mpi_data->mpi_rvec2 );
#endif

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        Update_Timing_Info( &t_start, &matvec_time );
#endif

    for ( j = 0; j < n; ++j )
    {
        // residual
        workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
        workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];

        // apply diagonal pre-conditioner
        //workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
        //workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
    }

    // apply preconditioner for both systems
    for ( j = 0; j < n; ++j )
    {
        r[j] = workspace->r2[j][0];
        d[j] = workspace->d2[j][0];
        p[j] = workspace->p2[j][0];
    }

    apply_preconditioner( system, workspace, control, r, d, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, d, p, fresh_pre, RIGHT );

    for ( j = 0; j < n; ++j )
    {
        workspace->r2[j][0] = r[j];
        workspace->d2[j][0] = d[j];
        workspace->p2[j][0] = p[j];
    }


    for ( j = 0; j < n; ++j )
    {
        r[j] = workspace->r2[j][1];
        d[j] = workspace->d2[j][1];
        p[j] = workspace->p2[j][1];
    }

    apply_preconditioner( system, workspace, control, r, d, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, d, p, fresh_pre, RIGHT );

    for ( j = 0; j < n; ++j )
    {
        workspace->r2[j][1] = r[j];
        workspace->d2[j][1] = d[j];
        workspace->p2[j][1] = p[j];
    }

    //print_host_rvec2 (workspace->r2, n);

    // 2-norm of b
    my_sum[0] = 0.0;
    my_sum[1] = 0.0;
    for ( j = 0; j < n; ++j )
    {
        my_sum[0] += SQR( b[j][0] );
        my_sum[1] += SQR( b[j][1] );
    }
    MPI_Allreduce( &my_sum, &norm_sqr, 2, MPI_DOUBLE, MPI_SUM, comm );
    b_norm[0] = SQRT( norm_sqr[0] );
    b_norm[1] = SQRT( norm_sqr[1] );

    // inner product of r and d
    my_dot[0] = 0.0;
    my_dot[1] = 0.0;
    for ( j = 0; j < n; ++j )
    {
        my_dot[0] += workspace->r2[j][0] * workspace->d2[j][0];
        my_dot[1] += workspace->r2[j][1] * workspace->d2[j][1];
    }

    MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < control->cm_solver_max_iters; ++i )
    {
        Dist( system, mpi_data, workspace->d2, RVEC2_PTR_TYPE,
                mpi_data->mpi_rvec2 );

        dual_Sparse_MatVec( H, workspace->d2, workspace->q2, N );

#if defined(HALF_LIST)
        // tryQEq
        Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE,
                mpi_data->mpi_rvec2 );
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            Update_Timing_Info( &t_start, &matvec_time );
#endif

        // inner product of d and q
        my_dot[0] = 0.0;
        my_dot[1] = 0.0;
        for ( j = 0; j < n; ++j )
        {
            my_dot[0] += workspace->d2[j][0] * workspace->q2[j][0];
            my_dot[1] += workspace->d2[j][1] * workspace->q2[j][1];
        }
        MPI_Allreduce( &my_dot, &tmp, 2, MPI_DOUBLE, MPI_SUM, comm );

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        my_dot[0] = 0.0;
        my_dot[1] = 0.0;
        for ( j = 0; j < n; ++j )
        {
            // update x
            x[j][0] += alpha[0] * workspace->d2[j][0];
            x[j][1] += alpha[1] * workspace->d2[j][1];

            // update residual
            workspace->r2[j][0] -= alpha[0] * workspace->q2[j][0];
            workspace->r2[j][1] -= alpha[1] * workspace->q2[j][1];

            // apply diagonal pre-conditioner
            //workspace->p2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
            //workspace->p2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];

            // dot product: r.p
            //my_dot[0] += workspace->r2[j][0] * workspace->p2[j][0];
            //my_dot[1] += workspace->r2[j][1] * workspace->p2[j][1];
        }

        // apply preconditioner for both systems
        for ( j = 0; j < n; ++j )
        {
            r[j] = workspace->r2[j][0];
            d[j] = workspace->d2[j][0];
            p[j] = workspace->p2[j][0];
        }

        apply_preconditioner( system, workspace, control, r, d, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, d, p, FALSE, RIGHT );

        for ( j = 0; j < n; ++j )
        {
            workspace->r2[j][0] = r[j];
            workspace->d2[j][0] = d[j];
            workspace->p2[j][0] = p[j];
        }


        for ( j = 0; j < n; ++j )
        {
            r[j] = workspace->r2[j][1];
            d[j] = workspace->d2[j][1];
            p[j] = workspace->p2[j][1];
        }

        apply_preconditioner( system, workspace, control, r, d, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, d, p, FALSE, RIGHT );

        for ( j = 0; j < n; ++j )
        {
            workspace->r2[j][1] = r[j];
            workspace->d2[j][1] = d[j];
            workspace->p2[j][1] = p[j];
        }

        for ( j = 0; j < n; ++j )
        {
            // dot product: r.p
            my_dot[0] += workspace->r2[j][0] * workspace->p2[j][0];
            my_dot[1] += workspace->r2[j][1] * workspace->p2[j][1];
        }

        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            Update_Timing_Info( &t_start, &dot_time );
#endif

        if ( SQRT(sig_new[0]) / b_norm[0] <= tol || SQRT(sig_new[1]) / b_norm[1] <= tol )
        {
            break;
        }

        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        for ( j = 0; j < n; ++j )
        {
            // d = p + beta * d
            workspace->d2[j][0] = workspace->p2[j][0] + beta[0] * workspace->d2[j][0];
            workspace->d2[j][1] = workspace->p2[j][1] + beta[1] * workspace->d2[j][1];
        }
    }

    if ( SQRT(sig_new[0]) / b_norm[0] <= tol )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters = CG( system, control, workspace, data, mpi_data,
                H, workspace->b_t, tol, workspace->t, fresh_pre );

        for ( j = 0; j < n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
    }
    else if ( SQRT(sig_new[1]) / b_norm[1] <= tol )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters = CG( system, control, workspace, data, mpi_data, 
                H, workspace->b_s, tol, workspace->s, fresh_pre );

        for ( j = 0; j < n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] Dual CG convergence failed (%d iters)\n", i + 1 + iters );
    }
    
    return (i + 1) + iters;


    /*int j, iters1, iters2, n;

    n = system->n; 
    if( 1 )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters1 = CG( system, control, workspace, data, mpi_data,
                H, workspace->b_t, tol, workspace->t, fresh_pre );

        for ( j = 0; j < n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
    }
    if( 1 )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters2 = CG( system, control, workspace, data, mpi_data, 
                H, workspace->b_s, tol, workspace->s, fresh_pre );

        for ( j = 0; j < n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }
    }


    if ( iters1 >= control->cm_solver_max_iters  || iters2 >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] Dual CG convergence failed (%d iters)\n", iters1 + iters2 );
    }
    
    return iters1 + iters2;*/


}


int CG( const reax_system * const system, const control_params * const control,
        const storage * const workspace, const simulation_data * const data,
        mpi_datatypes * const mpi_data,
        const sparse_matrix * const H, const real * const b,
        const real tol, real * const x, const int fresh_pre )
{
    int i/*, j*/;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new/*, sig0*/;

    Dist( system, mpi_data, x, REAL_PTR_TYPE, MPI_DOUBLE );
    Sparse_MatVec( H, x, workspace->q, system->N );

#if defined(HALF_LIST)
    // tryQEq
    Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
#endif

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &matvec_time );
    }
#endif

    Vector_Sum( workspace->r , 1.0,  b, -1.0, workspace->q, system->n );

    apply_preconditioner( system, workspace, control, workspace->r, workspace->d, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, workspace->d, workspace->p, fresh_pre, RIGHT );

    /*preconditioner
    for ( j = 0; j < system->n; ++j )
    {
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
    }*/

    b_norm = Parallel_Norm( b, system->n, mpi_data->world );
    sig_new = Parallel_Dot( workspace->r, workspace->d, system->n, mpi_data->world );
    //sig0 = sig_new;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < control->cm_solver_max_iters && SQRT(sig_new) / b_norm > tol; ++i )
    {
        Dist( system, mpi_data, workspace->d, REAL_PTR_TYPE, MPI_DOUBLE );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );

#if defined(HALF_LIST)
        //tryQEq
        Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        tmp = Parallel_Dot( workspace->d, workspace->q, system->n, mpi_data->world );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );

        apply_preconditioner( system, workspace, control, workspace->r, workspace->d, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, workspace->d, workspace->p, FALSE, RIGHT );

        /* preconditioner
        for ( j = 0; j < system->n; ++j )
        {
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }*/

        sig_old = sig_new;
        sig_new = Parallel_Dot( workspace->r, workspace->p, system->n, mpi_data->world );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1.0, workspace->p, beta, workspace->d, system->n );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] CG convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual errors: %f\n",
                SQRT(sig_new) / b_norm );
        return i;
    }

    return i;
}
