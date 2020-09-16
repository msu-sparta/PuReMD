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

#include "cuda_neighbors.h"

#include "cuda_list.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "../index_utils.h"
#include "../tool_box.h"
#include "../vector.h"


CUDA_DEVICE real Cuda_DistSqr_to_Special_Point( rvec cp, rvec x ) 
{
    int  i;  
    real d_sqr = 0.0;

    for( i = 0; i < 3; ++i )
    {
        if( cp[i] > NEG_INF )
        {
            d_sqr += SQR( cp[i] - x[i] );
        }
    }

    return d_sqr;
}


/* Generate far neighbor lists by scanning the atoms list and applying cutoffs */
CUDA_GLOBAL void k_generate_neighbor_lists( reax_atom *my_atoms, 
        simulation_box my_ext_box, grid g, reax_list far_nbr_list,
        int n, int N, int *far_nbrs, int *max_far_nbrs, int *realloc_far_nbrs )
{
    int i, j, k, l, m, itr, num_far, my_num_far;
    real d, cutoff;
    ivec c, nbrs_x;
    rvec dvec;
    reax_atom *atom1, *atom2;

    l = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( l >= N )
    {
        return;
    }

    atom1 = &my_atoms[l];
    num_far = Start_Index( l, &far_nbr_list );

    /* get the coordinates of the atom and compute the grid cell */
    if ( l < n )
    {
        for ( i = 0; i < 3; i++ )
        {
            c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);   
            if ( c[i] >= g.native_end[i] )
            {
                c[i] = g.native_end[i] - 1;
            }
            else if ( c[i] < g.native_str[i] )
            {
                c[i] = g.native_str[i];
            }
        }
    }
    else
    {
        for ( i = 0; i < 3; i++ )
        {
            c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);
            if ( c[i] < 0 )
            {
                c[i] = 0;
            }
            else if ( c[i] >= g.ncells[i] )
            {
                c[i] = g.ncells[i] - 1;
            }
        }
    }

    i = c[0];
    j = c[1];
    k = c[2];

    cutoff = SQR( g.cutoff[index_grid_3d(i, j, k, &g)] );

    /* scan neighboring grid cells within cutoff */
    itr = 0;
    while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
    { 
        ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );

        /* if neighboring grid cell is further in the "positive" direction AND within cutoff */
        if ( g.str[index_grid_3d(i, j, k, &g)] <= g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]
                && Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff )
        {
            /* pick up another atom from the neighbor cell */
            for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                    m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
            {
                /* prevent recounting same pairs within a gcell */
                if ( l < m )
                {
                    atom2 = &my_atoms[m];
                    dvec[0] = atom2->x[0] - atom1->x[0];
                    dvec[1] = atom2->x[1] - atom1->x[1];
                    dvec[2] = atom2->x[2] - atom1->x[2];
                    d = rvec_Norm_Sqr( dvec );

                    if ( d <= cutoff )
                    { 
                        /* commit far neighbor to list */
                        far_nbr_list.far_nbr_list.nbr[num_far] = m;
                        far_nbr_list.far_nbr_list.d[num_far] = SQRT( d );
                        rvec_Copy( far_nbr_list.far_nbr_list.dvec[num_far], dvec );
                        ivec_ScaledSum( far_nbr_list.far_nbr_list.rel_box[num_far],
                                1, g.rel_box[ index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ], 
                                -1, g.rel_box[ index_grid_3d(i, j, k, &g) ] );

                        ++num_far;
                    }
                }
            }
        }

        ++itr;
    }   

    /* scan neighboring grid cells within cutoff */
    itr = 0;
    while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
    { 
        ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );
        cutoff = SQR( g.cutoff[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] );

        /* if neighboring grid cell is further in the "negative" direction AND within cutoff */
        if ( g.str[index_grid_3d(i, j, k, &g)] >= g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] &&  
                Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff )
        {
            /* pick up another atom from the neighbor cell */
            for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                    m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
            {
                /* prevent recounting same pairs within a gcell */
                if ( l > m )
                {
                    atom2 = &my_atoms[m];
                    dvec[0] = atom1->x[0] - atom2->x[0];
                    dvec[1] = atom1->x[1] - atom2->x[1];
                    dvec[2] = atom1->x[2] - atom2->x[2];
                    d = rvec_Norm_Sqr( dvec );

                    if ( d <= cutoff )
                    {
                        /* commit far neighbor to list */
                        far_nbr_list.far_nbr_list.nbr[num_far] = m;
                        far_nbr_list.far_nbr_list.d[num_far] = SQRT( d );
                        rvec_Copy( far_nbr_list.far_nbr_list.dvec[num_far], dvec );
                        ivec_ScaledSum( far_nbr_list.far_nbr_list.rel_box[num_far],
                                1, g.rel_box[ index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ], 
                                -1, g.rel_box[ index_grid_3d(i, j, k, &g) ] );

                        ++num_far;
                    }
                }   
            }
        }

        ++itr;
    }   

    Set_End_Index( l, num_far, &far_nbr_list );

    /* reallocation check */
    my_num_far = num_far - Start_Index( l, &far_nbr_list );
    if ( my_num_far > max_far_nbrs[l] )
    {
        *realloc_far_nbrs = TRUE;
    }
}


//CUDA_GLOBAL void __launch_bounds__ (1024) k_mt_generate_neighbor_lists( reax_atom *my_atoms, 
CUDA_GLOBAL void k_mt_generate_neighbor_lists( reax_atom *my_atoms, 
        simulation_box my_ext_box, grid g, reax_list far_nbr_list, int n, int N )
{
    extern __shared__ int __nbr[];
    bool nbrgen;
    int __THREADS_PER_ATOM__, thread_id, group_id, lane_id, my_bucket;
    int *tnbr, *nbrssofar;
    int max, leader, loopcount, iterations;
    int i, j, k, l, m, itr, num_far, ll;
    real d, cutoff, cutoff_ji;
    ivec c, nbrs_x;
    rvec dvec;
    reax_atom *atom1, *atom2;

    __THREADS_PER_ATOM__ = NB_KER_THREADS_PER_ATOM;
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    group_id = thread_id / __THREADS_PER_ATOM__;

    if ( group_id >= N )
    {
        return;
    }

    lane_id = thread_id & (__THREADS_PER_ATOM__ - 1); 
    my_bucket = threadIdx.x / __THREADS_PER_ATOM__;
    tnbr = __nbr;
    nbrssofar = __nbr + blockDim.x;
    l = group_id;
    atom1 = &my_atoms[l];
    num_far = Start_Index( l, &far_nbr_list );

    //get the coordinates of the atom and 
    //compute the grid cell
    if ( l < n )
    {
        for ( i = 0; i < 3; i++ )
        {
            c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);
            if ( c[i] >= g.native_end[i] )
            {
                c[i] = g.native_end[i] - 1;
            }
            else if ( c[i] < g.native_str[i] )
            {
                c[i] = g.native_str[i];
            }
        }
    }
    else
    {
        for ( i = 0; i < 3; i++ )
        {
            c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);
            if ( c[i] < 0 )
            {
                c[i] = 0;
            }
            else if ( c[i] >= g.ncells[i] )
            {
                c[i] = g.ncells[i] - 1;
            }
        }
    }

    i = c[0];
    j = c[1];
    k = c[2];

    tnbr[threadIdx.x] = 0;
    if ( lane_id == 0 )
    {
        nbrssofar[my_bucket] = 0;
    }
    __syncthreads( );

    itr = 0;
    while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
    { 
        tnbr[threadIdx.x] = 0;
        nbrgen = false;

        ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );

        cutoff = SQR( g.cutoff[index_grid_3d(i, j, k, &g)] );
        cutoff_ji = SQR( g.cutoff[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] );

        if ( (g.str[index_grid_3d(i, j, k, &g)] <= g.str[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] 
                && Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff) 
                || (g.str[index_grid_3d(i, j, k, &g)] >= g.str[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] 
                && Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff_ji) )
        {
            max = g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]
                    - g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)];
            tnbr[threadIdx.x] = 0;
            nbrgen = false;
            m = lane_id  + g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; //0-31
            loopcount = max / __THREADS_PER_ATOM__ + ((max % __THREADS_PER_ATOM__) == 0 ? 0 : 1);
            iterations = 0;

            // pick up another atom from the neighbor cell
            while ( iterations < loopcount )
            {
                tnbr[threadIdx.x] = 0;
                nbrgen = false;

                // prevent recounting same pairs within a gcell 
                if ( l < m && m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] )
                {
                    atom2 = &my_atoms[m];
                    dvec[0] = atom2->x[0] - atom1->x[0];
                    dvec[1] = atom2->x[1] - atom1->x[1];
                    dvec[2] = atom2->x[2] - atom1->x[2];
                    d = rvec_Norm_Sqr( dvec );

                    if ( d <= cutoff )
                    { 
                        tnbr [threadIdx.x] = 1;
                        nbrgen = true;
                    }
                }

                if ( l > m && m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] )
                {
                    atom2 = &my_atoms[m];
                    dvec[0] = atom1->x[0] - atom2->x[0];
                    dvec[1] = atom1->x[1] - atom2->x[1];
                    dvec[2] = atom1->x[2] - atom2->x[2];
                    d = rvec_Norm_Sqr( dvec );

                    if ( d <= cutoff_ji )
                    {
                        tnbr [threadIdx.x] = 1;
                        nbrgen = true;
                    }
                } 

                //is neighbor generated
                if ( nbrgen )
                {
                    //do leader selection here
                    leader = -1;
                    for ( ll = my_bucket *__THREADS_PER_ATOM__;
                            ll < (my_bucket) * __THREADS_PER_ATOM__ + __THREADS_PER_ATOM__; ll++ )
                    {
                        if ( tnbr[ll] )
                        {
                            leader = ll;
                            break;
                        }
                    }

                    //do the reduction;
                    if ( threadIdx.x == leader )
                    {
                        for ( ll = 1; ll < __THREADS_PER_ATOM__; ll++ )
                        {
                            tnbr[my_bucket * __THREADS_PER_ATOM__ + ll]
                                    += tnbr[my_bucket * __THREADS_PER_ATOM__ + (ll-1)];
                        }
                    }
                }

                if ( nbrgen )
                {
                    far_nbr_list.far_nbr_list.nbr[
                        num_far + nbrssofar[my_bucket] + tnbr[threadIdx.x] - 1 ] = m;

                    if ( l < m )
                    {
                        dvec[0] = atom2->x[0] - atom1->x[0];
                        dvec[1] = atom2->x[1] - atom1->x[1];
                        dvec[2] = atom2->x[2] - atom1->x[2];
                        d = rvec_Norm_Sqr( dvec );
                        far_nbr_list.far_nbr_list.d[num_far] = SQRT( d );
                        rvec_Copy( far_nbr_list.far_nbr_list.dvec[num_far], dvec );
                        ivec_ScaledSum( far_nbr_list.far_nbr_list.rel_box[num_far],
                                1, g.rel_box[ index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ], 
                                -1, g.rel_box[ index_grid_3d(i, j, k, &g) ] );
                    } 
                    else
                    {
                        dvec[0] = atom1->x[0] - atom2->x[0];
                        dvec[1] = atom1->x[1] - atom2->x[1];
                        dvec[2] = atom1->x[2] - atom2->x[2];
                        d = rvec_Norm_Sqr( dvec );
                        far_nbr_list.far_nbr_list.d[num_far] = SQRT( d );
                        rvec_Copy( far_nbr_list.far_nbr_list.dvec[num_far], dvec );
                        /* CHANGE ORIGINAL
                         * This is a bug in the original code */
//                        ivec_ScaledSum( far_nbr_list.far_nbr_list.rel_box[num_far],
//                                1, g.rel_box[ index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ], 
//                                -1, g.rel_box[ index_grid_3d( i, j, k, &g) ] );
                        ivec_ScaledSum( far_nbr_list.far_nbr_list.rel_box[num_far],
                                -1, g.rel_box[ index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ], 
                                1, g.rel_box[ index_grid_3d(i, j, k, &g) ] );
                    }

                    if ( threadIdx.x == leader )
                    {
                        nbrssofar[my_bucket] += tnbr[my_bucket *__THREADS_PER_ATOM__ + (__THREADS_PER_ATOM__ - 1)];
                    }
                }

                m += __THREADS_PER_ATOM__;
                iterations++;

                //cleanup
                nbrgen = false;
                tnbr[threadIdx.x] = 0;
            }
        }
        ++itr;
    }

    if ( lane_id == 0 )
    {
        Set_End_Index( l, num_far + nbrssofar[my_bucket], &far_nbr_list );
    }
}


/* Estimate the number of far neighbors per atom (GPU) */
CUDA_GLOBAL void k_estimate_neighbors( reax_atom *my_atoms, 
        simulation_box my_ext_box, grid g, int n, int N, int total_cap,
        int *far_nbrs, int *max_far_nbrs )
{
    int i, j, k, l, m, itr, num_far;
    real d, cutoff;
    ivec c, nbrs_x;
    rvec dvec;
    reax_atom *atom1, *atom2;

    l = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( l >= total_cap )
    {
        return;
    }

    if ( l < N )
    {
        num_far = 0;
        atom1 = &my_atoms[l];

        /* get the coordinates of the atom and compute the grid cell
         * if atom is locally owned by processor AND not ghost atom */
        if ( l < n )
        {
            for ( i = 0; i < 3; i++ )
            {
                c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);   
                if ( c[i] >= g.native_end[i] )
                {
                    c[i] = g.native_end[i] - 1;
                }
                else if ( c[i] < g.native_str[i] )
                {
                    c[i] = g.native_str[i];
                }
            }
        }
        /* same as above, but for ghost atoms */
        else
        {
            for ( i = 0; i < 3; i++ )
            {
                c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);
                if ( c[i] < 0 )
                {
                    c[i] = 0;
                }
                else if ( c[i] >= g.ncells[i] )
                {
                    c[i] = g.ncells[i] - 1;
                }
            }
        }

        i = c[0];
        j = c[1];
        k = c[2];

        cutoff = SQR( g.cutoff[ index_grid_3d(i, j, k, &g) ] );

        itr = 0;
        while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
        { 
            ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );

            if ( //(g.str[index_grid_3d(i, j, k, &g)] <= g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]) &&  
                    Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff ) 
            {
                /* pick up another atom from the neighbor cell */
                for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                        m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
                {
                    /* prevent recounting same pairs within a gcell */
                    if ( l < m )
                    {
                        atom2 = &my_atoms[m];
                        dvec[0] = atom2->x[0] - atom1->x[0];
                        dvec[1] = atom2->x[1] - atom1->x[1];
                        dvec[2] = atom2->x[2] - atom1->x[2];
                        d = rvec_Norm_Sqr( dvec );

                        if( d <= cutoff )
                        { 
                            num_far++;
                        }
                    }   
                }
            }
            ++itr;

        }   

        itr = 0;
        while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
        {
            ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );
            cutoff = SQR( g.cutoff[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] );

            if ( g.str[index_grid_3d(i, j, k, &g)] >= g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] &&  
                    Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)],atom1->x) <= cutoff ) 
            {
                /* pick up another atom from the neighbor cell */
                for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                        m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
                {
                    /* prevent recounting same pairs within a gcell */
                    if ( l > m )
                    {
                        atom2 = &my_atoms[m];
                        dvec[0] = atom2->x[0] - atom1->x[0];
                        dvec[1] = atom2->x[1] - atom1->x[1];
                        dvec[2] = atom2->x[2] - atom1->x[2];
                        d = rvec_Norm_Sqr( dvec );

                        if ( d <= cutoff )
                        { 
                            num_far++;
                        }
                    }   
                }
            }
            ++itr;
        }   
    }
    else
    {
        /* used to trigger assignment of max_far_nbrs below */
        num_far = MIN_NBRS;
    }

    far_nbrs[l] = num_far;
    max_far_nbrs[l] = MAX( (int)(num_far * SAFE_ZONE), MIN_NBRS );
}


extern "C" int Cuda_Generate_Neighbor_Lists( reax_system *system,
        simulation_data *data, storage *workspace, reax_list **lists )
{
    int blocks, ret, ret_far_nbr;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;
    cudaEvent_t time_event[2];
    
    for ( int i = 0; i < 2; ++i )
    {
        cudaEventCreate( &time_event[i] );
    }
#endif

    /* reset reallocation flag on device */
    /* careful: this wrapper around cudaMemset(...) performs a byte-wide assignment
     * to the provided literal */
    cuda_memset( system->d_realloc_far_nbrs, FALSE, sizeof(int), 
            "Cuda_Generate_Neighbor_Lists::d_realloc_far_nbrs" );

    /* one thread per atom implementation */
    blocks = (system->N / NBRS_BLOCK_SIZE) +
        ((system->N % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( time_event[0] );
#endif

    k_generate_neighbor_lists <<< blocks, NBRS_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->my_ext_box,
          system->d_my_grid, *(lists[FAR_NBRS]),
          system->n, system->N,
          system->d_far_nbrs, system->d_max_far_nbrs, system->d_realloc_far_nbrs );
    cudaCheckError( );

    /* multiple threads per atom implementation */
//    blocks = ((system->N * NB_KER_THREADS_PER_ATOM) / NBRS_BLOCK_SIZE) + 
//        (((system->N * NB_KER_THREADS_PER_ATOM) % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
//    k_mt_generate_neighbor_lists <<< blocks, NBRS_BLOCK_SIZE, 
//        //sizeof(int) * (NBRS_BLOCK_SIZE + NBRS_BLOCK_SIZE / NB_KER_THREADS_PER_ATOM) >>>
//        sizeof(int) * 2 * NBRS_BLOCK_SIZE >>>
//            ( system->d_my_atoms, system->my_ext_box, system->d_my_grid,
//              *(lists[FAR_NBRS]), system->n, system->N );
//    cudaCheckError( );

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( time_event[1] );
#endif

    /* check reallocation flag on device */
    copy_host_device( &ret_far_nbr, system->d_realloc_far_nbrs, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Generate_Neighbor_Lists::d_realloc_far_nbrs" );

    ret = (ret_far_nbr == FALSE) ? SUCCESS : FAILURE;
    workspace->d_workspace->realloc.far_nbrs = ret_far_nbr;

#if defined(LOG_PERFORMANCE)
    if ( cudaEventQuery( time_event[0] ) != cudaSuccess ) 
    {
        cudaEventSynchronize( time_event[0] );
    }

    if ( cudaEventQuery( time_event[1] ) != cudaSuccess ) 
    {
        cudaEventSynchronize( time_event[1] );
    }

    cudaEventElapsedTime( &time_elapsed, time_event[0], time_event[1] ); 
    data->timing.nbrs += (real) (time_elapsed / 1000.0);
#endif

    return ret;
}


/* Estimate the number of far neighbors for each atoms 
 *
 * system: atomic system info */
void Cuda_Estimate_Num_Neighbors( reax_system *system, simulation_data *data )
{
    int blocks;
#if defined(LOG_PERFORMANCE)
    double time;
    
    time = Get_Time( );
#endif

    blocks = system->total_cap / DEF_BLOCK_SIZE
        + (system->total_cap % DEF_BLOCK_SIZE == 0 ? 0 : 1);

    k_estimate_neighbors <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->my_ext_box, system->d_my_grid,
          system->n, system->N, system->total_cap,
          system->d_far_nbrs, system->d_max_far_nbrs );
    cudaCheckError( );

    Cuda_Reduction_Sum( system->d_max_far_nbrs, system->d_total_far_nbrs,
            system->total_cap );
    copy_host_device( &system->total_far_nbrs, system->d_total_far_nbrs, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Estimate_Neighbors::d_total_far_nbrs" );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.nbrs );
#endif
}
