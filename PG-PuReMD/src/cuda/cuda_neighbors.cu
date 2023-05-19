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

#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>


#define FULL_WARP_MASK (0xFFFFFFFF)


GPU_DEVICE static inline real Cuda_DistSqr_to_Special_Point( rvec cp, rvec x ) 
{
    int i;  
    real d_sqr = 0.0;

    for ( i = 0; i < 3; ++i )
    {
        if ( cp[i] > NEG_INF )
        {
            d_sqr += SQR( cp[i] - x[i] );
        }
    }

    return d_sqr;
}


/* Generate far neighbor lists in full format
 * by scanning the atoms list and applying cutoffs */
GPU_GLOBAL void k_generate_neighbor_lists_full( reax_atom const * const my_atoms, 
        simulation_box my_ext_box, grid g, reax_list far_nbr_list,
        int n, int N, int * const far_nbrs, int * const max_far_nbrs,
        int * const realloc_far_nbrs, real cutoff2 )
{
    int i, j, k, l, m, itr, num_far, my_num_far, flag;
    real d, cutoff;
    ivec c, nbrs_x, rel_box, rel_box2;
    rvec x, dvec;

    l = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( l >= N )
    {
        return;
    }

    rvec_Copy( x, my_atoms[l].x );
    num_far = Start_Index( l, &far_nbr_list );

    /* map this atom to its grid cell indices */
    if ( l < n )
    {
        for ( i = 0; i < 3; ++i )
        {
            c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);   

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
        for ( i = 0; i < 3; ++i )
        {
            c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);

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
    ivec_Copy( rel_box, g.rel_box[index_grid_3d(i, j, k, &g)] );
    cutoff = SQR( g.cutoff[index_grid_3d(i, j, k, &g)] );

    /* scan neighboring grid cells within cutoff */
    itr = 0;
    while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
    { 
        ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );

        /* if neighboring grid cell is within cutoff */
        if ( Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], x) <= cutoff )
        {
            ivec_Copy( rel_box2, g.rel_box[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] );

            /* pick up another atom from the neighbor cell */
            for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                    m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
            {
                /* prevent recounting same pairs within a gcell */
                if ( l != m )
                {
                    dvec[0] = my_atoms[m].x[0] - x[0];
                    dvec[1] = my_atoms[m].x[1] - x[1];
                    dvec[2] = my_atoms[m].x[2] - x[2];
                    d = rvec_Norm_Sqr( dvec );

                    /* further restrict ghost-ghost atom interactions
                     * to bond cut-off distance */
                    if ( l >= N && m >= N )
                    {
                        flag = (d <= cutoff2 ? TRUE : FALSE);
                    }
                    else
                    {
                        flag = TRUE;
                    }

                    if ( d <= cutoff && flag == TRUE )
                    { 
                        /* commit far neighbor to list */
                        far_nbr_list.far_nbr_list.nbr[num_far] = m;
                        far_nbr_list.far_nbr_list.d[num_far] = SQRT( d );
                        rvec_Copy( far_nbr_list.far_nbr_list.dvec[num_far], dvec );
                        ivec_ScaledSum( far_nbr_list.far_nbr_list.rel_box[num_far],
                                1, rel_box2, -1, rel_box );

                        ++num_far;
                    }
                }
            }
        }

        ++itr;
    }   

    Set_End_Index( l, num_far, &far_nbr_list );

    my_num_far = num_far - Start_Index( l, &far_nbr_list );

    /* reallocation check */
    if ( my_num_far > max_far_nbrs[l] )
    {
        *realloc_far_nbrs = TRUE;
    }
}


/* Generate far neighbor lists in full format
 * by scanning the atoms list and applying cutoffs */
GPU_GLOBAL void k_generate_neighbor_lists_full_opt( reax_atom const * const my_atoms, 
        simulation_box my_ext_box, grid g, reax_list far_nbr_list,
        int n, int N, int * const far_nbrs, int * const max_far_nbrs,
        int * const realloc_far_nbrs, real cutoff2 )
{
    extern __shared__ cub::WarpScan<int>::TempStorage temp1[];
    int i, j, k, l, m, itr, num_far, my_num_far, lane_id, itr2;
    int start, end, offset, flag;
    real d, cutoff;
    ivec c, nbrs_x, rel_box, rel_box2;
    rvec x, dvec;

    l = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( l >= N )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize; 

    rvec_Copy( x, my_atoms[l].x );
    num_far = Start_Index( l, &far_nbr_list );

    /* map this atom to its grid cell indices */
    if ( l < n )
    {
        for ( i = 0; i < 3; ++i )
        {
            c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);   

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
        for ( i = 0; i < 3; ++i )
        {
            c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);

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
    ivec_Copy( rel_box, g.rel_box[index_grid_3d(i, j, k, &g)] );
    cutoff = SQR( g.cutoff[index_grid_3d(i, j, k, &g)] );

    /* scan neighboring grid cells within cutoff */
    itr = 0;
    while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
    { 
        ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );

        /* if neighboring grid cell is within cutoff */
        if ( Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], x) <= cutoff )
        {
            start = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)];
            end = g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)];
            ivec_Copy( rel_box2, g.rel_box[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] );

            /* pick up another atom from the neighbor cell */
            for ( itr2 = 0, m = start + lane_id;
                    itr2 < (end - start + warpSize - 1); ++itr2 )
            {
                /* prevent recounting same pairs within a gcell */
                if ( m < end && l != m )
                {
                    dvec[0] = my_atoms[m].x[0] - x[0];
                    dvec[1] = my_atoms[m].x[1] - x[1];
                    dvec[2] = my_atoms[m].x[2] - x[2];
                    d = rvec_Norm_Sqr( dvec );
                }

                /* further restrict ghost-ghost atom interactions
                 * to bond cut-off distance */
                if ( l >= N && m >= N )
                {
                    flag = (d <= cutoff2 ? TRUE : FALSE);
                }
                else
                {
                    flag = TRUE;
                }

                offset = (m < end && l != m && d <= cutoff && flag == TRUE) ? 1 : 0;
                flag = (offset == 1) ? TRUE : FALSE;
                cub::WarpScan<int>(temp1[threadIdx.x / warpSize]).ExclusiveSum(offset, offset);

                if ( flag == TRUE )
                { 
                    /* commit far neighbor to list */
                    far_nbr_list.far_nbr_list.nbr[num_far + offset] = m;
                    far_nbr_list.far_nbr_list.d[num_far + offset] = SQRT( d );
                    rvec_Copy( far_nbr_list.far_nbr_list.dvec[num_far + offset], dvec );
                    ivec_ScaledSum( far_nbr_list.far_nbr_list.rel_box[num_far + offset],
                            1, rel_box2, -1, rel_box );
                }

                /* get num_far from thread in last lane */
                num_far = num_far + offset + (flag == TRUE ? 1 : 0);
                num_far = cub::ShuffleIndex<WARP_SIZE>( num_far, warpSize - 1, FULL_WARP_MASK );

                m += warpSize;
            }
        }

        ++itr;
    }   

    if ( lane_id == 0 )
    {
        Set_End_Index( l, num_far, &far_nbr_list );

        my_num_far = num_far - Start_Index( l, &far_nbr_list );

        /* reallocation check */
        if ( my_num_far > max_far_nbrs[l] )
        {
            *realloc_far_nbrs = TRUE;
        }
    }
}


/* Estimate the number of entries in the far neighbors list in full format
 * using one thread per atom */
GPU_GLOBAL void k_estimate_neighbors_full( reax_atom const * const my_atoms, 
        simulation_box my_ext_box, grid g, int n, int N, int total_cap,
        int * const far_nbrs, int * const max_far_nbrs, real cutoff2 )
{
    int i, j, k, l, m, itr, num_far, flag;
    real d, cutoff;
    ivec c, nbrs_x;
    rvec dvec, x;

    l = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( l >= total_cap )
    {
        return;
    }

    if ( l < N )
    {
        num_far = 0;
        rvec_Copy( x, my_atoms[l].x );

        /* get the coordinates of the atom and compute the grid cell
         * if atom is locally owned by processor AND not ghost atom */
        if ( l < n )
        {
            for ( i = 0; i < 3; i++ )
            {
                c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);   

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
                c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);

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

            if ( Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], x) <= cutoff ) 
            {
                /* pick up another atom from the neighbor cell */
                for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                        m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
                {
                    /* prevent recounting same pairs within a gcell */
                    if ( l != m )
                    {
                        dvec[0] = my_atoms[m].x[0] - x[0];
                        dvec[1] = my_atoms[m].x[1] - x[1];
                        dvec[2] = my_atoms[m].x[2] - x[2];
                        d = rvec_Norm_Sqr( dvec );

                        /* further restrict ghost-ghost atom interactions
                         * to bond cut-off distance */
                        if ( l >= N && m >= N )
                        {
                            flag = (d <= cutoff2 ? TRUE : FALSE);
                        }
                        else
                        {
                            flag = TRUE;
                        }

                        if ( d <= cutoff && flag == TRUE )
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
    /* round up to the nearest multiple of WARP_SIZE to ensure that reads along
     * rows can be coalesced */
    max_far_nbrs[l] = MAX( ((int) CEIL( num_far * SAFE_ZONE )
                + warpSize - 1) / warpSize * warpSize, MIN_NBRS );
}


/* Estimate the number of entries in the far neighbors list in full format
 * using one warp of threads per atom */
GPU_GLOBAL void k_estimate_neighbors_full_opt( reax_atom const * const my_atoms, 
        simulation_box my_ext_box, grid g, int n, int N, int total_cap,
        int * const far_nbrs, int * const max_far_nbrs, real cutoff2 )
{
    extern __shared__ cub::WarpReduce<int>::TempStorage temp2[];
    int i, j, k, l, m, itr, num_far, lane_id, flag;
    real d, cutoff;
    ivec c, nbrs_x;
    rvec dvec, x;

    l = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( l >= total_cap )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize; 

    if ( l < N )
    {
        num_far = 0;
        rvec_Copy( x, my_atoms[l].x );

        /* get the coordinates of the atom and compute the grid cell
         * if atom is locally owned by processor AND not ghost atom */
        if ( l < n )
        {
            for ( i = 0; i < 3; i++ )
            {
                c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);   

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
                c[i] = (int) ((x[i] - my_ext_box.min[i]) * g.inv_len[i]);

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

        itr = 0;
        while ( g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)][0] >= 0 )
        { 
            ivec_Copy( nbrs_x, g.nbrs_x[index_grid_nbrs(i, j, k, itr, &g)] );

            if ( Cuda_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], x) <= cutoff ) 
            {
                /* pick up another atom from the neighbor cell */
                for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] + lane_id; 
                        m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; m += warpSize )
                {
                    /* prevent recounting same pairs within a gcell */
                    if ( l != m )
                    {
                        dvec[0] = my_atoms[m].x[0] - x[0];
                        dvec[1] = my_atoms[m].x[1] - x[1];
                        dvec[2] = my_atoms[m].x[2] - x[2];
                        d = rvec_Norm_Sqr( dvec );

                        /* further restrict ghost-ghost atom interactions
                         * to bond cut-off distance */
                        if ( l >= N && m >= N )
                        {
                            flag = (d <= cutoff2 ? TRUE : FALSE);
                        }
                        else
                        {
                            flag = TRUE;
                        }

                        if ( d <= cutoff && flag == TRUE )
                        { 
                            num_far++;
                        }
                    }   
                }
            }

            ++itr;
        }

        num_far = cub::WarpReduce<int>(temp2[threadIdx.x / warpSize]).Sum(num_far);
    }
    else
    {
        /* used to trigger assignment of max_far_nbrs below */
        num_far = MIN_NBRS;
    }

    if ( lane_id == 0 )
    {
        far_nbrs[l] = num_far;
        /* round up to the nearest multiple of WARP_SIZE to ensure that reads along
         * rows can be coalesced */
        max_far_nbrs[l] = MAX( ((int) CEIL( num_far * SAFE_ZONE )
                    + warpSize - 1) / warpSize * warpSize, MIN_NBRS );
    }
}


extern "C" int Cuda_Generate_Neighbor_Lists( reax_system *system,
        control_params *control, simulation_data *data, storage *workspace,
        reax_list **lists )
{
    int blocks, ret;
#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->cuda_time_events[TE_NBRS_START], control->cuda_streams[0] );
#endif

    /* reset reallocation flag on device */
    /* careful: this wrapper around cudaMemset(...) performs a byte-wide assignment
     * to the provided literal */
    sCudaMemsetAsync( system->d_realloc_far_nbrs, FALSE, sizeof(int), 
            control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

//    blocks = (system->N / NBRS_BLOCK_SIZE) +
//        ((system->N % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
    blocks = (system->N * WARP_SIZE / NBRS_BLOCK_SIZE) +
        ((system->N * WARP_SIZE % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);

//    k_generate_neighbor_lists_full <<< blocks, NBRS_BLOCK_SIZE, 0, control->cuda_streams[0] >>>
//        ( system->d_my_atoms, system->my_ext_box,
//          system->d_my_grid, *(lists[FAR_NBRS]),
//          system->n, system->N,
//          system->d_far_nbrs, system->d_max_far_nbrs,
//          system->d_realloc_far_nbrs, SQR( control->bond_cut ) );
    k_generate_neighbor_lists_full_opt <<< blocks, NBRS_BLOCK_SIZE,
                                       sizeof(cub::WarpScan<int>::TempStorage) * (DEF_BLOCK_SIZE / WARP_SIZE),
                                       control->cuda_streams[0] >>>
        ( system->d_my_atoms, system->my_ext_box,
          system->d_my_grid, *(lists[FAR_NBRS]),
          system->n, system->N,
          system->d_far_nbrs, system->d_max_far_nbrs, system->d_realloc_far_nbrs,
          SQR( control->bond_cut ) );
    cudaCheckError( );

    /* check reallocation flag on device */
    sCudaMemcpyAsync( &workspace->d_workspace->realloc->far_nbrs,
            system->d_realloc_far_nbrs, sizeof(int), 
            cudaMemcpyDeviceToHost, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    ret = (workspace->d_workspace->realloc->far_nbrs == FALSE) ? SUCCESS : FAILURE;

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->cuda_time_events[TE_NBRS_STOP], control->cuda_streams[0] );
#endif

    return ret;
}


/* Estimate the number of far neighbors for each atoms 
 *
 * system: atomic system info */
void Cuda_Estimate_Num_Neighbors( reax_system *system, control_params *control,
        simulation_data *data )
{
    int blocks;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;

    cudaEventRecord( control->cuda_time_events[TE_NBRS_START], control->cuda_streams[0] );
#endif

//    blocks = system->total_cap / DEF_BLOCK_SIZE
//        + (system->total_cap % DEF_BLOCK_SIZE == 0 ? 0 : 1);
    blocks = system->total_cap * WARP_SIZE / DEF_BLOCK_SIZE
        + (system->total_cap * WARP_SIZE % DEF_BLOCK_SIZE == 0 ? 0 : 1);

//    k_estimate_neighbors_full <<< blocks, DEF_BLOCK_SIZE, 0, control->cuda_streams[0] >>>
//        ( system->d_my_atoms, system->my_ext_box, system->d_my_grid,
//          system->n, system->N, system->total_cap,
//          system->d_far_nbrs, system->d_max_far_nbrs, SQR( control->bond_cut ) );
    k_estimate_neighbors_full_opt <<< blocks, DEF_BLOCK_SIZE,
                                  sizeof(cub::WarpReduce<int>::TempStorage) * (DEF_BLOCK_SIZE / WARP_SIZE),
                                  control->cuda_streams[0] >>>
        ( system->d_my_atoms, system->my_ext_box, system->d_my_grid,
          system->n, system->N, system->total_cap,
          system->d_far_nbrs, system->d_max_far_nbrs, SQR( control->bond_cut ) );
    cudaCheckError( );

    Cuda_Reduction_Sum( system->d_max_far_nbrs, system->d_total_far_nbrs,
            system->total_cap, 0, control->cuda_streams[0] );
    sCudaMemcpyAsync( &system->total_far_nbrs, system->d_total_far_nbrs, sizeof(int), 
            cudaMemcpyDeviceToHost, control->cuda_streams[0], __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->cuda_time_events[TE_NBRS_STOP], control->cuda_streams[0] );
#endif

    cudaStreamSynchronize( control->cuda_streams[0] );

#if defined(LOG_PERFORMANCE)
    cudaEventElapsedTime( &time_elapsed, control->cuda_time_events[TE_NBRS_START],
            control->cuda_time_events[TE_NBRS_STOP] ); 
    data->timing.nbrs += (real) (time_elapsed / 1000.0);
#endif
}
