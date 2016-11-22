/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
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

#include "QEq.h"
#include "allocate.h"
#include "GMRES.h"
#include "list.h"
#include "print_utils.h"
#include "index_utils.h"

#include "cuda_utils.h"
#include "cuda_init.h"
#include "cuda_copy.h"
#include "sort.h"
#include "validation.h"
#include "reduction.h"

#include "system_props.h"


HOST_DEVICE void swap(sparse_matrix_entry *array, int index1, int index2) 
{
    sparse_matrix_entry temp = array[index1];
    array[index1] = array[index2];
    array[index2] = temp;
}


HOST_DEVICE void quick_sort(sparse_matrix_entry *array, int start, int end)
{
    int i = start;
    int k = end; 

    if (end - start >= 1)  
    {  
        int pivot = array[start].j;

        while (k > i) 
        {  
            while ((array[i].j <= pivot) && (i <= end) && (k > i)) i++;
            while ((array[k].j > pivot) && (k >= start) && (k >= i)) k--;
            if (k > i) swap(array, i, k);
        }  
        swap(array, start, k);
        quick_sort(array, start, k - 1);
        quick_sort(array, k + 1, end);
    }  
}


int compare_matrix_entry(const void *v1, const void *v2)
{
    return ((sparse_matrix_entry *)v1)->j - ((sparse_matrix_entry *)v2)->j;
}


void Sort_Matrix_Rows( sparse_matrix *A )
{
    int i, si, ei;

    for( i = 0; i < A->n; ++i ) {
        si = A->start[i];
        ei = A->start[i+1];
        qsort( &(A->entries[si]), ei - si, 
                sizeof(sparse_matrix_entry), compare_matrix_entry );
    }
}


GLOBAL void Cuda_Sort_Matrix_Rows ( sparse_matrix A )
{
    int i;
    int si, ei;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= A.n ) return;

    si = A.start[i];
    ei = A.end [i];

    quick_sort( A.entries + si, 0, ei-si-1 );
}


void Calculate_Droptol( sparse_matrix *A, real *droptol, real dtol )
{
    int i, j, k;
    real val;

    /* init droptol to 0 */
    for( i = 0; i < A->n; ++i )
        droptol[i] = 0;

    /* calculate sqaure of the norm of each row */
    for( i = 0; i < A->n; ++i ) {
        for( k = A->start[i]; k < A->start[i+1]-1; ++k ) {
            j = A->entries[k].j;
            val = A->entries[k].val;

            droptol[i] += val*val;
            droptol[j] += val*val;
        }

        val = A->entries[k].val; // diagonal entry
        droptol[i] += val*val;
    }

    /* calculate local droptol for each row */
    //fprintf( stderr, "droptol: " );
    for( i = 0; i < A->n; ++i ) {
        //fprintf( stderr, "%f-->", droptol[i] );
        droptol[i] = SQRT( droptol[i] ) * dtol;
        //fprintf( stderr, "%f  ", droptol[i] );
    }
    //fprintf( stderr, "\n" );
}


GLOBAL void Cuda_Calculate_Droptol ( sparse_matrix p_A, real *droptol, real dtol )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, offset, x, diagonal;
    real val;
    sparse_matrix *A = &p_A;

    if ( i < A->n ) {
        droptol [i] = 0;

        for (k = A->start[i]; k < A->end[i]; ++k ) {
            val = A->entries[k].val;
            droptol [i] += val*val;
        }
    }

    __syncthreads ();
    if ( i < A->n ) {
        droptol [i] = SQRT (droptol[i]) * dtol;
    }

}


GLOBAL void Cuda_Calculate_Droptol_js ( sparse_matrix p_A, real *droptol, real dtol )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, offset, x, diagonal;
    real val;
    sparse_matrix *A = &p_A;

    for (x = 0; x < A->n; x ++)
    {
        if (i < (A->end[i]-1 - A->start[i])) {
            offset = A->start [i] + i;
            j = A->entries[offset].j;
            val = A->entries[offset].val;
            droptol [j] += val * val;
        }
        __syncthreads ();
    }
}


GLOBAL void Cuda_Calculate_Droptol_diagonal ( sparse_matrix p_A, real *droptol, real dtol )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, offset, x, diagonal;
    real val;
    sparse_matrix *A = &p_A;

    if ( i < A->n ) {
        //diagonal element
        diagonal = A->end[i]-1;
        val = A->entries [diagonal].val;
        droptol [i] += val*val;
    }

    /*calculate local droptol for each row*/
    if ( i < A->n )
        droptol [i] = SQRT (droptol[i]) * dtol;
}


int Estimate_LU_Fill( sparse_matrix *A, real *droptol )
{
    int i, j, pj;
    int fillin;
    real val;

    fillin = 0;

    //fprintf( stderr, "n: %d\n", A->n );
    for( i = 0; i < A->n; ++i )
        for( pj = A->start[i]; pj < A->start[i+1]-1; ++pj ){
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            //fprintf( stderr, "i: %d, j: %d", i, j );

            if( fabs(val) > droptol[i] )
                ++fillin;
        }

    return fillin + A->n;
}


GLOBAL void Cuda_Estimate_LU_Fill ( sparse_matrix p_A, real *droptol, int *fillin)
{
    int i, j, pj;
    real val;
    sparse_matrix *A = &p_A;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A->n) return;

    fillin [i] = 0;

    for (pj = A->start[i]; pj < A->end[i]-1; ++pj)
    {
        j = A->entries [pj].j;
        val = A->entries[pj].val;

        if (fabs (val) > droptol [i]) ++fillin [i];
    }
}


void ICHOLT( sparse_matrix *A, real *droptol, 
        sparse_matrix *L, sparse_matrix *U )
{
    sparse_matrix_entry tmp[1000];
    int i, j, pj, k1, k2, tmptop, Ltop;
    real val;
    int *Utop;

    Utop = (int*) malloc((A->n+1) * sizeof(int));

    // clear variables
    Ltop = 0;
    tmptop = 0;
    for( i = 0; i <= A->n; ++i )
        L->start[i] = U->start[i] = 0;

    for( i = 0; i < A->n; ++i )
        Utop[i] = 0;

    //fprintf( stderr, "n: %d\n", A->n );
    for( i = 0; i < A->n; ++i ){
        L->start[i] = Ltop;
        tmptop = 0;

        for( pj = A->start[i]; pj < A->start[i+1]-1; ++pj ){
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            //fprintf( stderr, "i: %d, j: %d", i, j );

            if( fabs(val) > droptol[i] ){
                k1 = 0;
                k2 = L->start[j];
                while( k1 < tmptop && k2 < L->start[j+1] ){
                    if( tmp[k1].j < L->entries[k2].j )
                        ++k1;
                    else if( tmp[k1].j > L->entries[k2].j )
                        ++k2;
                    else
                        val -= (tmp[k1++].val * L->entries[k2++].val);
                }

                // L matrix is lower triangular, 
                // so right before the start of next row comes jth diagonal
                val /= L->entries[L->start[j+1]-1].val;

                tmp[tmptop].j = j;
                tmp[tmptop].val = val;
                ++tmptop;
            }
            //fprintf( stderr, " -- done\n" );
        }

        // compute the ith diagonal in L
        // sanity check
        if( A->entries[pj].j != i ) {
            fprintf( stderr, "i=%d, badly built A matrix!\n", i );
            exit(999);
        }

        val = A->entries[pj].val;
        for( k1 = 0; k1 < tmptop; ++k1 )
            val -= (tmp[k1].val * tmp[k1].val);

        tmp[tmptop].j = i;
        tmp[tmptop].val = SQRT(val);

        // apply the dropping rule once again
        //fprintf( stderr, "row%d: tmptop: %d\n", i, tmptop );
        //for( k1 = 0; k1<= tmptop; ++k1 )
        //  fprintf( stderr, "%d(%f)  ", tmp[k1].j, tmp[k1].val );
        //fprintf( stderr, "\n" );
        //fprintf( stderr, "row(%d): droptol=%.4f\n", i+1, droptol[i] );
        for( k1 = 0; k1 < tmptop; ++k1 )
            if( fabs(tmp[k1].val) > droptol[i] / tmp[tmptop].val ){
                L->entries[Ltop].j = tmp[k1].j;
                L->entries[Ltop].val = tmp[k1].val;
                U->start[tmp[k1].j+1]++;
                ++Ltop;
                //fprintf( stderr, "%d(%.4f)  ", tmp[k1].j+1, tmp[k1].val );
            }
        // keep the diagonal in any case
        L->entries[Ltop].j = tmp[k1].j;
        L->entries[Ltop].val = tmp[k1].val;
        ++Ltop;
        //fprintf( stderr, "%d(%.4f)\n", tmp[k1].j+1,  tmp[k1].val );
    }

    L->start[i] = Ltop;
    //fprintf( stderr, "nnz(L): %d, max: %d\n", Ltop, L->n * 50 );

    for( i = 1; i <= U->n; ++i )
        Utop[i] = U->start[i] = U->start[i] + U->start[i-1] + 1;

    for( i = 0; i < L->n; ++i )
        for( pj = L->start[i]; pj < L->start[i+1]; ++pj ){
            j = L->entries[pj].j;
            U->entries[Utop[j]].j = i;
            U->entries[Utop[j]].val = L->entries[pj].val;
            Utop[j]++;
        }

    //fprintf( stderr, "nnz(U): %d, max: %d\n", Utop[U->n], U->n * 50 );
}


void Cuda_ICHOLT( sparse_matrix *A, real *droptol, 
        sparse_matrix *L, sparse_matrix *U )
{
    sparse_matrix_entry tmp[1000];
    int i, j, pj, k1, k2, tmptop, Ltop;
    real val;
    int *Utop;

    Utop = (int*) malloc((A->n+1) * sizeof(int));

    // clear variables
    Ltop = 0;
    tmptop = 0;
    for( i = 0; i <= A->n; ++i )
        L->start[i] = U->start[i] = 0;

    for( i = 0; i < A->n; ++i )
        Utop[i] = 0;

    //fprintf( stderr, "n: %d\n", A->n );
    for( i = 0; i < A->n; ++i ){
        L->start[i] = Ltop;
        tmptop = 0;

        for( pj = A->start[i]; pj < A->end[i]-1; ++pj ){
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            //fprintf( stderr, "i: %d, j: %d", i, j );

            //CHANGE ORIGINAL
            if (j >= i) break;
            //CHANGE ORIGINAL

            if( fabs(val) > droptol[i] ){
                k1 = 0;
                k2 = L->start[j];
                while( k1 < tmptop && k2 < L->start[j+1] ){
                    if( tmp[k1].j < L->entries[k2].j )
                        ++k1;
                    else if( tmp[k1].j > L->entries[k2].j )
                        ++k2;
                    else
                        val -= (tmp[k1++].val * L->entries[k2++].val);
                }

                // L matrix is lower triangular, 
                // so right before the start of next row comes jth diagonal
                val /= L->entries[L->start[j+1]-1].val;

                tmp[tmptop].j = j;
                tmp[tmptop].val = val;
                ++tmptop;
            }

            //fprintf( stderr, " -- done\n" );
        }

        // compute the ith diagonal in L
        // sanity check
        if( A->entries[pj].j != i ) {
            fprintf( stderr, "i=%d, badly built A matrix!\n", i );
            exit(999);
        }

        val = A->entries[pj].val;
        for( k1 = 0; k1 < tmptop; ++k1 )
            val -= (tmp[k1].val * tmp[k1].val);

        tmp[tmptop].j = i;
        tmp[tmptop].val = SQRT(val);

        // apply the dropping rule once again
        //fprintf( stderr, "row%d: tmptop: %d\n", i, tmptop );
        //for( k1 = 0; k1<= tmptop; ++k1 )
        //  fprintf( stderr, "%d(%f)  ", tmp[k1].j, tmp[k1].val );
        //fprintf( stderr, "\n" );
        //fprintf( stderr, "row(%d): droptol=%.4f\n", i+1, droptol[i] );
        for( k1 = 0; k1 < tmptop; ++k1 )
            if( fabs(tmp[k1].val) > droptol[i] / tmp[tmptop].val ){
                L->entries[Ltop].j = tmp[k1].j;
                L->entries[Ltop].val = tmp[k1].val;
                U->start[tmp[k1].j+1]++;
                ++Ltop;
                //fprintf( stderr, "%d(%.4f)  ", tmp[k1].j+1, tmp[k1].val );
            }
        // keep the diagonal in any case
        L->entries[Ltop].j = tmp[k1].j;
        L->entries[Ltop].val = tmp[k1].val;
        ++Ltop;
        //fprintf( stderr, "%d(%.4f)\n", tmp[k1].j+1,  tmp[k1].val );
    }

    L->start[i] = Ltop;
    //fprintf( stderr, "nnz(L): %d, max: %d\n", Ltop, L->n * 50 );

    for( i = 1; i <= U->n; ++i )
        Utop[i] = U->start[i] = U->start[i] + U->start[i-1] + 1;

    for( i = 0; i < L->n; ++i )
        for( pj = L->start[i]; pj < L->start[i+1]; ++pj ){
            j = L->entries[pj].j;
            U->entries[Utop[j]].j = i;
            U->entries[Utop[j]].val = L->entries[pj].val;
            Utop[j]++;
        }

    //fprintf( stderr, "nnz(U): %d, max: %d\n", Utop[U->n], U->n * 50 );
}



/*
//Parallel for each row
//Each kernel will run for 6540 number of times.
GLOBAL void Cuda_ICHOLT( reax_system *system, sparse_matrix p_A, real *droptol, 
sparse_matrix p_L, sparse_matrix p_U )
{
int start, end, count;
real tempvalue, val;
int i,pj,tmptop, offset;
int j, k1, k2;

sparse_matrix *A, *L, *U;
sparse_matrix_entry *tmp;

A = &p_A;
L = &p_L;
U = &p_U;

real *null_val;
null_val = 0;

extern __shared__ real tmp_val[];
extern __shared__ sparse_matrix_entry sh_tmp[];

int kid = blockIdx.x * blockDim.x + threadIdx.x;
tmp = (sparse_matrix_entry *) (tmp_val + blockDim.x);

offset = 0;
for( i = 0; i < 10; ++i )
{
//if (kid == 0) L->start[i] = i * system->max_sparse_matrix_entries;
if (kid == 0) L->start[i] = offset;
tmptop = 0;

start = A->start[i];
end = A->end[i]-1; //inclusive
count = end - start; //inclusive
tmp_val [kid] = 0;

if (kid < count) //diagonal not included
{
pj = start + kid;

j = A->entries[pj].j;
val = A->entries[pj].val;

if( fabs(val) > droptol[i] )
{
k1 = 0;
k2 = L->start[j];
while( k1 < tmptop && k2 < L->end[j] ){
if( tmp[k1].j < L->entries[k2].j )
++k1;
else if( tmp[k1].j > L->entries[k2].j )
++k2;
else
tmp_val[kid] = (tmp[k1++].val * L->entries[k2++].val);
}

//here read the shared memory of all the kernels 
if (kid == 0)
{
for (i = 0; i < count; i++)
tempvalue += tmp_val [i];

val -= tempvalue;

// L matrix is lower triangular, 
// so right before the start of next row comes jth diagonal
val /= L->entries[L->end[j]-1].val;

tmp[tmptop].j = j;
tmp[tmptop].val = val;
++tmptop;
}
}
}
__syncthreads ();


// compute the ith diagonal in L
// sanity check
if (kid == 0) 
{
    if( A->entries[end].j != i ) {
        //intentional core dump here for sanity sake
        *null_val = 1;
    }
}

//diagonal element
//val = A->entries[pj].val;
//for( k1 = 0; k1 < tmptop; ++k1 )
if (kid < count) 
    tmp_val[kid] = (tmp[kid].val * tmp[kid].val);

    __syncthreads ();

if (kid == 0)
{
    val = A->entries [end].val;
    for (i = 0; i < count; i++)
        tempvalue += tmp_val [i];

    val -= tempvalue;
    tmp[tmptop].j = i;
    tmp[tmptop].val = SQRT(val);
}
__syncthreads ();

//Fill in the LU entries
//for( k1 = 0; k1 < count; ++k1 )
if (kid < count )
{
    if( fabs(tmp[kid].val) > droptol[i] / tmp[tmptop].val ){
        L->entries[offset + kid].j = tmp[kid].j;
        L->entries[offset + kid].val = tmp[kid].val;
        U->start[tmp[kid].j+1]++;
    }
}
__syncthreads ();

if (kid == 0) {
    // keep the diagonal in any case
    offset += count;
    L->entries[offset].j = tmp[count].j;
    L->entries[offset].val = tmp[count].val;
    ++offset;
    L->end [i] = offset;
}
__syncthreads ();
} // end of main for loop
}

void Cuda_Fill_U    ( sparse_matrix *A, real *droptol, 
        sparse_matrix *L, sparse_matrix *U )
{
    int i, pj, j;

    for( i = 1; i <= U->n; ++i )
        Utop[i] = U->start[i] = U->start[i] + U->start[i-1] + 1;

    for( i = 0; i < L->n; ++i )
        for( pj = L->start[i]; pj < L->start[i+1]; ++pj ){
            j = L->entries[pj].j;
            U->entries[Utop[j]].j = i;
            U->entries[Utop[j]].val = L->entries[pj].val;
            Utop[j]++;
        }
}
*/


void Init_MatVec( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list *far_nbrs )
{
    int i, fillin;
    real s_tmp, t_tmp;
    //char fname[100];

    if(control->refactor > 0 && 
            ((data->step-data->prev_steps)%control->refactor==0 || workspace->L.entries==NULL))
    {
        //Print_Linear_System( system, control, workspace, data->step );
        Sort_Matrix_Rows( &workspace->H );

        //fprintf( stderr, "H matrix sorted\n" );

        Calculate_Droptol( &workspace->H, workspace->droptol, control->droptol ); 
        //fprintf( stderr, "drop tolerances calculated\n" );

        if( workspace->L.entries == NULL )
        {
            fillin = Estimate_LU_Fill( &workspace->H, workspace->droptol );

#ifdef __DEBUG_CUDA__
            fprintf( stderr, "fillin = %d\n", fillin );
#endif

            if( Allocate_Matrix( &(workspace->L), far_nbrs->n, fillin ) == 0 ||
                    Allocate_Matrix( &(workspace->U), far_nbrs->n, fillin ) == 0 )
            {
                fprintf( stderr, "not enough memory for LU matrices. terminating.\n" );
                exit(INSUFFICIENT_SPACE);
            }

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "fillin = %d\n", fillin );
            fprintf( stderr, "allocated memory: L = U = %ldMB\n",
                    fillin * sizeof(sparse_matrix_entry) / (1024*1024) );
#endif
        }

        ICHOLT( &workspace->H, workspace->droptol, &workspace->L, &workspace->U );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "icholt-" );
        //sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        //Print_Sparse_Matrix2( workspace->L, fname );
        //Print_Sparse_Matrix( U );
#endif
    }

    /* extrapolation for s & t */
    for( i = 0; i < system->N; ++i ) {
        // no extrapolation
        //s_tmp = workspace->s[0][i];
        //t_tmp = workspace->t[0][i];

        // linear
        //s_tmp = 2 * workspace->s[0][i] - workspace->s[1][i];
        //t_tmp = 2 * workspace->t[0][i] - workspace->t[1][i];

        // quadratic
        //s_tmp = workspace->s[2][i] + 3 * (workspace->s[0][i]-workspace->s[1][i]);
        t_tmp = workspace->t[index_wkspace_sys(2,i,system)] + 3*(workspace->t[index_wkspace_sys(0,i,system)]-workspace->t[index_wkspace_sys(1,i,system)]);

        // cubic
        s_tmp = 4 * (workspace->s[index_wkspace_sys(0,i,system)] + workspace->s[index_wkspace_sys(2,i,system)]) - 
            (6 * workspace->s[index_wkspace_sys(1,i,system)] + workspace->s[index_wkspace_sys(3,i,system)] );
        //t_tmp = 4 * (workspace->t[0][i] + workspace->t[2][i]) - 
        //  (6 * workspace->t[1][i] + workspace->t[3][i] );

        // 4th order
        //s_tmp = 5 * (workspace->s[0][i] - workspace->s[3][i]) + 
        //  10 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];
        //t_tmp = 5 * (workspace->t[0][i] - workspace->t[3][i]) + 
        //  10 * (-workspace->t[1][i] + workspace->t[2][i] ) + workspace->t[4][i];

        workspace->s[index_wkspace_sys(4,i,system)] = workspace->s[index_wkspace_sys(3,i,system)];
        workspace->s[index_wkspace_sys(3,i,system)] = workspace->s[index_wkspace_sys(2,i,system)]; 
        workspace->s[index_wkspace_sys(2,i,system)] = workspace->s[index_wkspace_sys(1,i,system)];
        workspace->s[index_wkspace_sys(1,i,system)] = workspace->s[index_wkspace_sys(0,i,system)];
        workspace->s[index_wkspace_sys(0,i,system)] = s_tmp;

        workspace->t[index_wkspace_sys(4,i,system)] = workspace->t[index_wkspace_sys(3,i,system)];
        workspace->t[index_wkspace_sys(3,i,system)] = workspace->t[index_wkspace_sys(2,i,system)]; 
        workspace->t[index_wkspace_sys(2,i,system)] = workspace->t[index_wkspace_sys(1,i,system)];
        workspace->t[index_wkspace_sys(1,i,system)] = workspace->t[index_wkspace_sys(0,i,system)];
        workspace->t[index_wkspace_sys(0,i,system)] = t_tmp;
    }
}


void Cuda_Init_MatVec(reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, list *far_nbrs )
{
    int i, fillin;
    real s_tmp, t_tmp;
    int *spad = (int *)scratch;
    real start = 0, end = 0;

    if( control->refactor > 0 && 
            ((data->step-data->prev_steps)%control->refactor==0 ||
             dev_workspace->L.entries==NULL) )
    {
        Cuda_Sort_Matrix_Rows<<< BLOCKS, BLOCK_SIZE >>>
            ( dev_workspace->H );
        cudaThreadSynchronize();
        cudaCheckError();

#ifdef __DEBUG_CUDA__
        fprintf (stderr, "Sorting done... \n");
#endif

        Cuda_Calculate_Droptol<<<BLOCKS, BLOCK_SIZE >>>
            ( dev_workspace->H, dev_workspace->droptol, control->droptol );
        cudaThreadSynchronize();
        cudaCheckError();

#ifdef __DEBUG_CUDA__
        fprintf (stderr, "Droptol done... \n");
#endif

        if( dev_workspace->L.entries == NULL )
        {
            cuda_memset ( spad, 0, 2 * INT_SIZE * system->N, RES_SCRATCH );
            Cuda_Estimate_LU_Fill <<< BLOCKS, BLOCK_SIZE >>>
                ( dev_workspace->H, dev_workspace->droptol, spad );
            cudaThreadSynchronize();
            cudaCheckError();

            //Reduction for fill in 
            Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, INT_SIZE * BLOCK_SIZE >>>  
                (spad, spad + system->N,  system->N);
            cudaThreadSynchronize ();
            cudaCheckError ();

            Cuda_reduction <<<1, BLOCKS_POW_2, INT_SIZE * BLOCKS_POW_2>>> 
                (spad + system->N, spad + system->N + BLOCKS_POW_2, BLOCKS_POW_2); 
            cudaThreadSynchronize ();
            cudaCheckError ();

            copy_host_device (&fillin, spad + system->N + BLOCKS_POW_2, INT_SIZE, cudaMemcpyDeviceToHost, RES_SCRATCH );
            fillin += dev_workspace->H.n;

#ifdef __DEBUG_CUDA__
            fprintf (stderr, "Calculated value of the fill in is --> %d \n ", fillin );
#endif

            dev_workspace->L.n = far_nbrs->n;
            dev_workspace->L.m = fillin;
            Cuda_Init_Sparse_Matrix( &dev_workspace->L, fillin, far_nbrs->n );

            dev_workspace->U.n = far_nbrs->n;
            dev_workspace->U.m = fillin;
            Cuda_Init_Sparse_Matrix( &dev_workspace->U, fillin, far_nbrs->n );
        }

#ifdef __DEBUG_CUDA__
        fprintf (stderr, "LU matrix done...\n");
#endif

        //TODO -- This is the ILU Factorization of the H Matrix. 
        //This is present in the CUDA 5.0 compilation which is not working currently. 
        //Fix this when CUDA 5.0 is correctly setup. 
        //TODO
        //shared memory is per block
        // here we have only one block - 
        /*
           fprintf (stderr, "max sparse matrix entries %d \n", system->max_sparse_matrix_entries );
           Cuda_ICHOLT <<<1, system->max_sparse_matrix_entries, 
           system->max_sparse_matrix_entries *(REAL_SIZE + SPARSE_MATRIX_ENTRY_SIZE)   >>>
           ( system, dev_workspace->H, 
           dev_workspace->droptol, 
           dev_workspace->L, 
           dev_workspace->U );
           cudaThreadSynchronize ();
           fprintf (stderr, "Cuda_ICHOLT .. done ...-> %d\n ", cudaGetLastError ());
         */

        //1. copy the H matrix from device to host
        //2. Allocate the L/U matrices on the host and device. 
        //3. Compute the L/U on the host
        //4. copy the results to the device
        //5. Continue the computation.
        sparse_matrix t_H, t_L, t_U;
        real *t_droptol;

        t_droptol = (real *) malloc (REAL_SIZE * system->N);

#ifdef __DEBUG_CUDA__
        fprintf (stderr, " Allocation temp matrices count %d entries %d \n", dev_workspace->H.n, dev_workspace->H.m );
#endif

        start = Get_Time ();
        if(!Allocate_Matrix(&t_H, dev_workspace->H.n, dev_workspace->H.m))
        {
            fprintf(stderr, "No space for H matrix \n");
            exit(0);
        }
        if(!Allocate_Matrix(&t_L, far_nbrs->n, dev_workspace->L.m))
        {
            fprintf(stderr, "No space for L matrix \n");
            exit(0);
        }
        if(!Allocate_Matrix(&t_U, far_nbrs->n, dev_workspace->U.m))
        {
            fprintf(stderr, "No space for U matrix \n");
            exit(0);
        }

        copy_host_device ( t_H.start, dev_workspace->H.start, INT_SIZE *
                (dev_workspace->H.n + 1), cudaMemcpyDeviceToHost,
                RES_SPARSE_MATRIX_INDEX );
        copy_host_device ( t_H.end, dev_workspace->H.end, INT_SIZE *
                (dev_workspace->H.n + 1), cudaMemcpyDeviceToHost,
                RES_SPARSE_MATRIX_INDEX );
        copy_host_device ( t_H.entries, dev_workspace->H.entries,
                SPARSE_MATRIX_ENTRY_SIZE * dev_workspace->H.m,
                cudaMemcpyDeviceToHost, RES_SPARSE_MATRIX_ENTRY );

        copy_host_device ( t_droptol, dev_workspace->droptol, REAL_SIZE *
                system->N, cudaMemcpyDeviceToHost, RES_STORAGE_DROPTOL );

        //fprintf (stderr, " Done copying LUH .. \n");
        Cuda_ICHOLT (&t_H, t_droptol, &t_L, &t_U);

        Sync_Host_Device (&t_L, &t_U, cudaMemcpyHostToDevice);
        end += Get_Timing_Info (start);

        /*
           fprintf (stderr, "Done syncing .... \n");
           free (t_droptol);
           fprintf (stderr, "Freed droptol ... \n");
           Deallocate_Matrix (&t_H);
           fprintf (stderr, "Freed H ... \n");
           Deallocate_Matrix (&t_L);
           fprintf (stderr, "Freed l ... \n");
           Deallocate_Matrix (&t_U);
           fprintf (stderr, "Freed u ... \n");
         */

        //#ifdef __DEBUG_CUDA__
        fprintf (stderr, "Done copying the L/U matrices to the device ---> %f \n", end);
        //#endif

        //#ifdef __BUILD_DEBUG__
        //        validate_lu (workspace);
        //#endif
    }
}


GLOBAL void Init_MatVec_Postprocess (static_storage p_workspace, int N )
{

    static_storage *workspace = &p_workspace;
    real s_tmp, t_tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;
    // no extrapolation
    //s_tmp = workspace->s[0][i];
    //t_tmp = workspace->t[0][i];

    // linear
    //s_tmp = 2 * workspace->s[0][i] - workspace->s[1][i];
    //t_tmp = 2 * workspace->t[0][i] - workspace->t[1][i];

    // quadratic
    //s_tmp = workspace->s[2][i] + 3 * (workspace->s[0][i]-workspace->s[1][i]);
    t_tmp = workspace->t[index_wkspace_sys(2,i,N)] + 3*(workspace->t[index_wkspace_sys(0,i,N)]-workspace->t[index_wkspace_sys(1,i,N)]);

    // cubic
    s_tmp = 4 * (workspace->s[index_wkspace_sys(0,i,N)] + workspace->s[index_wkspace_sys(2,i,N)]) - 
        (6 * workspace->s[index_wkspace_sys(1,i,N)] + workspace->s[index_wkspace_sys(3,i,N)] );
    //t_tmp = 4 * (workspace->t[0][i] + workspace->t[2][i]) - 
    //  (6 * workspace->t[1][i] + workspace->t[3][i] );

    // 4th order
    //s_tmp = 5 * (workspace->s[0][i] - workspace->s[3][i]) + 
    //  10 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];
    //t_tmp = 5 * (workspace->t[0][i] - workspace->t[3][i]) + 
    //  10 * (-workspace->t[1][i] + workspace->t[2][i] ) + workspace->t[4][i];

    workspace->s[index_wkspace_sys(4,i,N)] = workspace->s[index_wkspace_sys(3,i,N)];
    workspace->s[index_wkspace_sys(3,i,N)] = workspace->s[index_wkspace_sys(2,i,N)]; 
    workspace->s[index_wkspace_sys(2,i,N)] = workspace->s[index_wkspace_sys(1,i,N)];
    workspace->s[index_wkspace_sys(1,i,N)] = workspace->s[index_wkspace_sys(0,i,N)];
    workspace->s[index_wkspace_sys(0,i,N)] = s_tmp;

    workspace->t[index_wkspace_sys(4,i,N)] = workspace->t[index_wkspace_sys(3,i,N)];
    workspace->t[index_wkspace_sys(3,i,N)] = workspace->t[index_wkspace_sys(2,i,N)]; 
    workspace->t[index_wkspace_sys(2,i,N)] = workspace->t[index_wkspace_sys(1,i,N)];
    workspace->t[index_wkspace_sys(1,i,N)] = workspace->t[index_wkspace_sys(0,i,N)];
    workspace->t[index_wkspace_sys(0,i,N)] = t_tmp;
}


void Calculate_Charges( reax_system *system, static_storage *workspace )
{
    int i;
    real u, s_sum, t_sum;

    s_sum = t_sum = 0.;
    for( i = 0; i < system->N; ++i ) {
        s_sum += workspace->s[index_wkspace_sys(0,i,system)];
        t_sum += workspace->t[index_wkspace_sys(0,i,system)];
    }

    u = s_sum / t_sum;

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Host --->s %13.2f, t %13.f, u %13.2f \n", s_sum, t_sum, u );
#endif

    for( i = 0; i < system->N; ++i )
    {
        system->atoms[i].q = workspace->s[index_wkspace_sys(0,i,system)] - u * workspace->t[index_wkspace_sys(0,i,system)];
    }
}


GLOBAL void Cuda_Update_Atoms_q ( reax_atom *atoms, real *s, real u, real *t, int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= N)
    {
        return;
    }

    atoms[i].q = s[index_wkspace_sys(0,i,N)] - u * t[index_wkspace_sys(0,i,N)];
}


void Cuda_Calculate_Charges (reax_system *system, static_storage *workspace)
{
    real *spad = (real *) scratch;
    real u, s_sum, t_sum;

    cuda_memset (spad, 0, (BLOCKS_POW_2 * 2 * REAL_SIZE), RES_SCRATCH );

    //s_sum 
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>>  
        (&dev_workspace->s [index_wkspace_sys (0, 0,system->N)], spad,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> 
        (spad, spad+BLOCKS_POW_2, BLOCKS_POW_2); 
    cudaThreadSynchronize ();
    cudaCheckError ();

    copy_host_device (&s_sum, spad+BLOCKS_POW_2, REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);

    //t_sum
    cuda_memset (spad, 0, (BLOCKS_POW_2 * 2 * REAL_SIZE), RES_SCRATCH );
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>>  
        (&dev_workspace->t [index_wkspace_sys (0, 0,system->N)], spad,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> 
        (spad, spad+BLOCKS_POW_2, BLOCKS_POW_2); 
    cudaThreadSynchronize ();
    cudaCheckError ();

    copy_host_device (&t_sum, spad+BLOCKS_POW_2, REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);

    //fraction here
    u = s_sum / t_sum;

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "DEVICE ---> s %13.2f, t %13.f, u %13.2f \n", s_sum, t_sum, u );
#endif

    Cuda_Update_Atoms_q <<< BLOCKS, BLOCK_SIZE >>>
        ( (reax_atom *)system->d_atoms, dev_workspace->s, u, dev_workspace->t, system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();
}


void QEq( reax_system *system, control_params *control, simulation_data *data, 
        static_storage *workspace, list *far_nbrs, 
        output_controls *out_control )
{
    int matvecs;

    //real t_start, t_elapsed;

    //t_start = Get_Time ();
    Init_MatVec( system, control, data, workspace, far_nbrs );
    //t_elapsed = Get_Timing_Info ( t_start );

    //fprintf (stderr, " CPU Init_MatVec timing ----> %f \n", t_elapsed );

    //if( data->step % 10 == 0 )
    //  Print_Linear_System( system, control, workspace, far_nbrs, data->step );

    //t_start = Get_Time ( );
    matvecs = GMRES( workspace, &workspace->H, 
            workspace->b_s, control->q_err, &workspace->s[0], out_control->log, system );
    matvecs += GMRES( workspace, &workspace->H, 
            workspace->b_t, control->q_err, &workspace->t[0], out_control->log, system );
    //t_elapsed = Get_Timing_Info ( t_start );

    //fprintf (stderr, " CPU GMRES timing ---> %f \n", t_elapsed );

    //matvecs = GMRES_HouseHolder( workspace, workspace->H, 
    //    workspace->b_s, control->q_err, workspace->s[0], out_control->log );
    //matvecs += GMRES_HouseHolder( workspace, workspace->H,  
    //    workspace->b_t, control->q_err, workspace->t[0], out_control->log );

    //matvecs = PGMRES( workspace, &workspace->H, workspace->b_s, control->q_err,
    //  &workspace->L, &workspace->U, &workspace->s[index_wkspace_sys(0,0,system)], out_control->log, system );
    //matvecs += PGMRES( workspace, &workspace->H, workspace->b_t, control->q_err,
    //  &workspace->L, &workspace->U, &workspace->t[index_wkspace_sys(0,0,system)], out_control->log, system );

    //matvecs=PCG( workspace, workspace->H, workspace->b_s, control->q_err, 
    //      workspace->L, workspace->U, workspace->s[0], out_control->log ) + 1;
    ///matvecs+=PCG( workspace, workspace->H, workspace->b_t, control->q_err, 
    //     workspace->L, workspace->U, workspace->t[0], out_control->log ) + 1;

    //matvecs = CG( workspace, workspace->H, 
    // workspace->b_s, control->q_err, workspace->s[0], out_control->log ) + 1;
    //matvecs += CG( workspace, workspace->H, 
    // workspace->b_t, control->q_err, workspace->t[0], out_control->log ) + 1;

    //matvecs = SDM( workspace, workspace->H, 
    // workspace->b_s, control->q_err, workspace->s[0], out_control->log ) + 1;
    //matvecs += SDM( workspace, workspace->H, 
    // workspace->b_t, control->q_err, workspace->t[0], out_control->log ) + 1;

    //fprintf (stderr, " GMRES done with iterations %d \n", matvecs );

    data->timing.matvecs += matvecs;
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "linsolve-" );
#endif

    Calculate_Charges( system, workspace );
    //fprintf( stderr, "%d %.9f %.9f %.9f %.9f %.9f %.9f\n", 
    //   data->step, 
    //   workspace->s[0][0], workspace->t[0][0], 
    //   workspace->s[0][1], workspace->t[0][1], 
    //   workspace->s[0][2], workspace->t[0][2] );
    // if( data->step == control->nsteps )
    //Print_Charges( system, control, workspace, data->step );
}


void Cuda_QEq( reax_system *system, control_params *control, simulation_data *data, 
        static_storage *workspace, list *far_nbrs, 
        output_controls *out_control )
{
    int matvecs = 0;
    real t_start, t_elapsed;

#ifdef __DEBUG_CUDA__
    t_start = Get_Time ();
#endif

    /*
    //Cuda_Init_MatVec( system, control, data, workspace, far_nbrs );

    Cuda_Sort_Matrix_Rows <<< BLOCKS, BLOCK_SIZE >>>
    ( dev_workspace->H );
    cudaThreadSynchronize ();
    cudaCheckError ();

    t_elapsed = Get_Timing_Info (t_start);
    fprintf (stderr, "Sorting done...tming --> %f \n", t_elapsed);
     */
    Init_MatVec_Postprocess <<< BLOCKS, BLOCK_SIZE >>>
        (*dev_workspace, system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info (t_start);
    fprintf (stderr, "Done with post processing of init_matvec --> %d  with time ---> %f \n", cudaGetLastError (), t_elapsed);
#endif

    //Here goes the GMRES part of the program ()
    //#ifdef __DEBUG_CUDA__
    t_start = Get_Time ();
    //#endif

    //matvecs = Cuda_GMRES( dev_workspace, dev_workspace->b_s, control->q_err, dev_workspace->s );
    //matvecs += Cuda_GMRES( dev_workspace, dev_workspace->b_t, control->q_err, dev_workspace->t );

    matvecs = Cublas_GMRES( system, dev_workspace, dev_workspace->b_s, control->q_err, dev_workspace->s );
    matvecs += Cublas_GMRES( system, dev_workspace, dev_workspace->b_t, control->q_err, dev_workspace->t );

    d_timing.matvecs += matvecs;

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info ( t_start );
    fprintf (stderr, " Cuda_GMRES done with iterations %d with timing ---> %f \n", matvecs, t_elapsed );
#endif

    //Here cuda calculate charges
    Cuda_Calculate_Charges (system, workspace);
}
