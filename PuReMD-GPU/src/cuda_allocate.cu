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

#include "cuda_allocate.h"

#include "list.h"

#include "cuda_utils.h"
#include "cuda_reduction.h"


GLOBAL void Init_HBond_Indexes ( int *, int *, list , int  );
GLOBAL void Init_Bond_Indexes ( int *, list , int  );


void Cuda_Reallocate_Neighbor_List( list *far_nbrs, int n, int num_intrs )
{
    Delete_List( far_nbrs, TYP_DEVICE );
    if(!Make_List( n, num_intrs, TYP_FAR_NEIGHBOR, far_nbrs, TYP_DEVICE )){
        fprintf(stderr, "Problem in initializing far nbrs list. Terminating!\n");
        exit( INIT_ERR );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "num_far = %d, far_nbrs = %d -> reallocating!\n",
            num_intrs, far_nbrs->num_intrs );  
    fprintf( stderr, "memory allocated: far_nbrs = %ldMB\n", 
            num_intrs * sizeof(far_neighbor_data) / (1024*1024) );
#endif
}


int Cuda_Allocate_Matrix( sparse_matrix *H, int n, int m )
{
    H->n = n;
    H->m = m;

    cuda_malloc ((void **) &H->start, INT_SIZE * (n+1), 0, RES_SPARSE_MATRIX_INDEX );
    cuda_malloc ((void **) &H->end, INT_SIZE *(n+1), 0, RES_SPARSE_MATRIX_INDEX );
    cuda_malloc ((void **) &H->entries, SPARSE_MATRIX_ENTRY_SIZE * m, 0, RES_SPARSE_MATRIX_ENTRY );

    return 1;
}


void Cuda_Deallocate_Matrix( sparse_matrix *H )
{
    cuda_free(H->start, RES_SPARSE_MATRIX_INDEX);
    cuda_free(H->end, RES_SPARSE_MATRIX_INDEX);
    cuda_free(H->entries, RES_SPARSE_MATRIX_ENTRY);

    H->start = NULL;
    H->end = NULL;
    H->entries = NULL;
}


int Cuda_Reallocate_Matrix( sparse_matrix *H, int n, int m, char *name )
{
    Cuda_Deallocate_Matrix( H );

    if( !Cuda_Allocate_Matrix( H, n, m ) ) {
        fprintf(stderr, "not enough space for %s matrix on GPU . terminating!\n", name);
        exit( 1 );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "reallocating %s matrix, n = %d, m = %d\n",
            name, n, m );
    fprintf( stderr, "memory allocated: %s = %ldMB\n", 
            name, m * sizeof(sparse_matrix_entry) / (1024*1024) );
#endif
    return 1;
}


int Cuda_Allocate_HBond_List( int n, int num_h, int *h_index, int *hb_top, list *hbonds )
{
    int i, num_hbonds;
    int blocks, block_size;
    int *d_hb_top;
    num_hbonds = 0;

    /* find starting indexes for each H and the total number of hbonds */
    for( i = 1; i < n; ++i )
        hb_top[i] += hb_top[i-1];
    num_hbonds = hb_top[n-1];

    if( !Make_List(num_h, num_hbonds, TYP_HBOND, hbonds , TYP_DEVICE) ) {
        fprintf( stderr, "not enough space for hbonds list. terminating!\n" );
        exit( INIT_ERR );
    }

    //cuda_malloc ((void **) &d_hb_top, INT_SIZE * (n), 1, __LINE__);
    d_hb_top = (int *) scratch;
    cuda_memset ( d_hb_top, 0, INT_SIZE * n, RES_SCRATCH );
    copy_host_device (hb_top, (d_hb_top), INT_SIZE * n, cudaMemcpyHostToDevice, __LINE__);

    Init_HBond_Indexes <<< BLOCKS, BLOCK_SIZE >>>
        ( h_index, d_hb_top, *hbonds, n);
    cudaThreadSynchronize ();

#ifdef __DEBUG_CUDA__
    fprintf( stderr, "Done with allocating hbonds - num_hbonds: %d\n", num_hbonds );
#endif

    return 1;
}


int Cuda_Reallocate_HBonds_List(  int n, int num_h, int *h_index, list *hbonds )
{
    int i;
    int *hb_top;
    int *hb_start;
    int *hb_end;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "reallocating hbonds\n" );
#endif
    hb_top = (int *)calloc( n, sizeof(int) );
    hb_start = (int *) calloc (hbonds->n, sizeof (int));
    hb_end = (int *) calloc (hbonds->n, sizeof (int));

    copy_host_device (hb_start, hbonds->index, sizeof (int) * hbonds->n, 
            cudaMemcpyDeviceToHost, LIST_INDEX);
    copy_host_device (hb_end , hbonds->end_index, sizeof (int) * hbonds->n, 
            cudaMemcpyDeviceToHost, LIST_END_INDEX);

    for( i = 0; i < n; ++i )
        //if( h_index[i] >= 0 )
        hb_top[i] = MAX((hb_end [i] - hb_start[i])*SAFE_HBONDS, MIN_HBONDS);

    Delete_List( hbonds, TYP_DEVICE );

    Cuda_Allocate_HBond_List( n, num_h, h_index, hb_top, hbonds );

    free( hb_top );
    free( hb_start );
    free( hb_end );

    return 1;
}


int Cuda_Allocate_Bond_List( int num_b, int *b_top, list *bonds )
{
    int i, num_bonds;
    int *d_b_top = (int *) scratch;
    num_bonds = 0;

    /* find starting indexes for each H and the total number of hbonds */
    for( i = 1; i < num_b; ++i )
        b_top[i] += b_top[i-1];
    num_bonds = b_top[num_b-1];

    if( !Make_List(num_b, num_bonds, TYP_BOND, bonds, TYP_DEVICE) ) {
        fprintf( stderr, "not enough space for bonds list. terminating!\n" );
        exit( INIT_ERR );
    }

    //cuda_malloc ((void **) &d_b_top, INT_SIZE * num_b, 1, __LINE__);
    cuda_memset ( d_b_top, 0, INT_SIZE * num_b, RES_SCRATCH );
    copy_host_device (b_top, d_b_top, INT_SIZE * num_b, cudaMemcpyHostToDevice, __LINE__);

    Init_Bond_Indexes <<< BLOCKS, BLOCK_SIZE>>>
        ( d_b_top, *bonds, num_b);
    cudaThreadSynchronize ();
    cudaCheckError ();

    return 1;
}


int Cuda_Reallocate_Bonds_List( int n, list *bonds, int *num_3body )
{
    int i;
    int *b_top;
    int *b_start;
    int *b_end;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "reallocating bonds\n" );
#endif
    b_top = (int *)calloc( n, sizeof(int) );
    b_start = (int *) calloc (bonds->n, sizeof (int));
    b_end = (int *) calloc (bonds->n, sizeof (int));

    copy_host_device (b_start, bonds->index, sizeof (int) * bonds->n, 
            cudaMemcpyDeviceToHost, LIST_INDEX);
    copy_host_device (b_end , bonds->end_index, sizeof (int) * bonds->n, 
            cudaMemcpyDeviceToHost, LIST_END_INDEX);

    for( i = 0; i < n; ++i ) {
        *num_3body += SQR (b_end[i] - b_start[i]);
        b_top[i] = MAX((b_end [i] - b_start[i])*2, MIN_BONDS);
    }

    Delete_List( bonds, TYP_DEVICE );

    Cuda_Allocate_Bond_List(n, b_top, bonds );

    i = b_top[ n-1 ];

    free( b_top );
    free( b_start );
    free( b_end );

    return i;
}


int Cuda_Reallocate_ThreeBody_List ( list *thblist, int count )
{
    int i;
    int thb_total = 0;
    int *thb_start;
    int *thb_end;

    int new_total, new_count;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "reallocating bonds\n" );
#endif
    thb_start = (int *) calloc (thblist->n, sizeof (int));
    thb_end = (int *) calloc (thblist->n, sizeof (int));

    copy_host_device (thb_start, thblist->index, sizeof (int) * thblist->n, 
            cudaMemcpyDeviceToHost, LIST_INDEX);
    copy_host_device (thb_end , thblist->end_index, sizeof (int) * thblist->n, 
            cudaMemcpyDeviceToHost, LIST_END_INDEX);

    for( i = 0; i < thblist->n; ++i )
        thb_total += (thb_end[i] - thb_start[i]) * SAFE_ZONE;

    //new_total = MAX( thb_total, thblist->num_intrs );
    //new_count = MAX( num_3body, thblist->n );

    new_total = thb_total;
    new_count = count;

    Delete_List( thblist, TYP_DEVICE );

    /*Allocate the list */
    if(!Make_List( new_count, new_total, TYP_THREE_BODY, thblist, TYP_DEVICE )){
        fprintf(stderr, "Problem in reallocating three-body list. Terminating!\n");
        exit( INIT_ERR );
    }

#if defined(__CUDA_MEM__)
    fprintf( stderr, "reallocating 3 bodies - \n" );
    fprintf( stderr, "num_bonds: %d ", new_count);
    fprintf( stderr, "num_3body: %d ", new_total);
    fprintf( stderr, "3body memory: %ldMB\n", 
            new_total * sizeof(three_body_interaction_data)/
            (1024*1024) );
#endif

    free( thb_start );
    free( thb_end );

    return 1;
}


/*
   int Cuda_Reallocate_Bonds_List( int n, list *bonds, int *num_bonds, int *est_3body )
   {
   int i;
   int *bond_top, *d_bond_top;

#if defined(DEBUG_FOCUS)
fprintf( stderr, "reallocating bonds on the device \n" );
#endif
bond_top = (int *)calloc( n, sizeof(int) );
d_bond_top = (int *) scratch;
cuda_memset (d_bond_top, 0, (n+BLOCKS_POW_2+1) * INT_SIZE, RES_SCRATCH );

 *est_3body = 0;

 Calculate_Bond_Indexes <<< BLOCKS_POW_2, BLOCK_SIZE, INT_SIZE * BLOCK_SIZE >>>
 (d_bond_top, *bonds, d_bond_top + n, n);
 cudaThreadSynchronize ();
 cudaCheckError ();

 Cuda_reduction <<<1, BLOCKS_POW_2, INT_SIZE * BLOCKS_POW_2>>> 
 (d_bond_top + n, d_bond_top + n + BLOCKS_POW_2, BLOCKS_POW_2); 
 cudaThreadSynchronize ();

 copy_host_device (bond_top, d_bond_top, n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );
 copy_host_device (est_3body, d_bond_top + n + BLOCKS_POW_2, INT_SIZE, cudaMemcpyDeviceToHost, __LINE__);

 Delete_List( bonds, TYP_DEVICE );

 Cuda_Allocate_Bond_List( n, bond_top, bonds );
 *num_bonds = bond_top[n-1];

 free( bond_top );

 return 1;
 }
 */


void Cuda_Reallocate( reax_system *system, static_storage *workspace, list *lists, 
        int nbr_flag, int step )
{
    int num_bonds, est_3body;
    int old_count = 0;
    reallocate_data *realloc;
    grid *g;

    realloc = &(workspace->realloc);
    g = &(system->d_g);

    if( realloc->num_far > 0 && nbr_flag ) {

#ifdef __CUDA_MEM__
        fprintf (stderr, " Reallocating Neighbors: step: %d, old_count: %d new_count: %d size: %d (MB)\n", 
                step, (dev_lists+FAR_NBRS)->num_intrs, (int)(realloc->num_far * SAFE_ZONE), 
                (int)(sizeof (far_neighbor_data) * realloc->num_far * SAFE_ZONE)/(1024*1024));
#endif
        Cuda_Reallocate_Neighbor_List( lists+FAR_NBRS, 
                system->N, realloc->num_far * SAFE_ZONE );

        realloc->num_far = -1;
        realloc->estimate_nbrs = 1;
    }

    if( realloc->Htop > 0 ){

#ifdef __CUDA_MEM__
        fprintf (stderr, " Reallocating Matrix : step: %d, old_count: %d new_count: %d size: %d (MB)\n", 
                step, dev_workspace->H.m, (int)(realloc->Htop * system->N * SAFE_ZONE), 
                (int) (sizeof (sparse_matrix_entry) * (realloc->Htop * system->N * SAFE_ZONE))/(1024 * 1024));
#endif
        //Cuda_Reallocate_Matrix(&(workspace->H), system->N, realloc->Htop*SAFE_ZONE,"H");
        Cuda_Reallocate_Matrix(&(workspace->H), system->N, realloc->Htop * system->N * SAFE_ZONE,"H");
        system->max_sparse_matrix_entries = realloc->Htop * SAFE_ZONE;
        realloc->Htop = -1;

        /*
           Cuda_Deallocate_Matrix( &workspace->L );
           fprintf (stderr, "Done deallocating the L ower matrix \n");
           Cuda_Deallocate_Matrix( &workspace->U );
           fprintf (stderr, "Done deallocating the Upper  matrix \n");
         */
    }

    if( realloc->hbonds > 0 ){

        old_count = (dev_lists+HBONDS)->num_intrs;

        Cuda_Reallocate_HBonds_List(system->N, workspace->num_H, workspace->hbond_index,
                dev_lists+HBONDS );

#ifdef __CUDA_MEM__
        fprintf (stderr, " Reallocating HBonds: step: %d, old_count: %d, new_count: %d, size: %d (MB)\n", 
                step, old_count,(dev_lists+HBONDS)->num_intrs, 
                (int) sizeof (hbond_data) * (dev_lists+HBONDS)->num_intrs / (1024 * 1024));
#endif
        realloc->hbonds = -1;
    }

    num_bonds = est_3body = -1;
    if( realloc->bonds > 0 ){

        old_count = (dev_lists+BONDS)->num_intrs;
        num_bonds = Cuda_Reallocate_Bonds_List( system->N, dev_lists+BONDS, &est_3body );

#ifdef __CUDA_MEM__
        fprintf (stderr, " Reallocating Bonds: step: %d, old_count: %d, new_count: %d, size: %d (MB) \n", 
                step, old_count,(dev_lists+BONDS)->num_intrs, 
                (int) sizeof (bond_data) * (dev_lists+BONDS)->num_intrs / (1024 * 1024));
#endif

        realloc->bonds = -1;
        realloc->num_3body = 1;//MAX( realloc->num_3body, est_3body );
    }

    /*
       if( realloc->num_3body > 0 ) {

       if (num_bonds < 0)
       num_bonds = (dev_lists+BONDS)->num_intrs;

       Cuda_Reallocate_ThreeBody_List (dev_lists + THREE_BODIES, num_bonds);
       realloc->num_3body = -1;
       }
     */

    if( realloc->gcell_atoms > -1 ){
#if defined(DEBUG_FOCUS)
        fprintf(stderr, "reallocating gcell: g->max_atoms: %d\n", g->max_atoms);
#endif

#ifdef __CUDA_MEM__
        fprintf (stderr, "Reallocating the atoms in the grid ---> %d \n", workspace->realloc.gcell_atoms );
#endif

        free (g->atoms);
        g->atoms = (int *) calloc ( g->ncell[0]*g->ncell[1]*g->ncell[2],
                sizeof (int) * workspace->realloc.gcell_atoms);

        cuda_free (g->atoms, RES_GRID_ATOMS);
        cuda_malloc ((void **) &g->atoms, INT_SIZE * workspace->realloc.gcell_atoms * g->ncell[0]*g->ncell[1]*g->ncell[2], 1, RES_GRID_ATOMS );
        realloc->gcell_atoms = -1;
    }
}


GLOBAL void Init_HBond_Indexes ( int *h_index, int *hb_top, list hbonds, int N )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) return;

    if( h_index[index] == 0 ){
        Set_Start_Index( 0, 0, &hbonds ); 
        Set_End_Index( 0, 0, &hbonds ); 
    }
    else if( h_index[index] > 0 ){
        Set_Start_Index( h_index[index], hb_top[index-1], &hbonds ); 
        Set_End_Index( h_index[index], hb_top[index-1], &hbonds ); 
    }
}


GLOBAL void Init_Bond_Indexes ( int *b_top, list bonds, int N )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) return;

    if( index == 0 ){
        Set_Start_Index( 0, 0, &bonds ); 
        Set_End_Index( 0, 0, &bonds ); 
    }
    else if( index > 0 ){
        Set_Start_Index( index, b_top[index-1], &bonds ); 
        Set_End_Index( index, b_top[index-1], &bonds ); 
    }
}


void GLOBAL Calculate_Bond_Indexes (int *bond_top, list bonds, int *per_block_results, int n)
{
    extern __shared__ int sh_input[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = SQR (Num_Entries( i, &bonds ) );
        bond_top[i] = MAX( Num_Entries( i, &bonds ) * 2, MIN_BONDS );
    }
    sh_input[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {   
            sh_input[threadIdx.x] += sh_input[threadIdx.x + offset];
        }   

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sh_input[0];
    }
}
