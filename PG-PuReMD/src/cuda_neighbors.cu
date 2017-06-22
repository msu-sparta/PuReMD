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

#include "reax_types.h"

#include "cuda_list.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "vector.h"
#include "index_utils.h"
#include "tool_box.h"


CUDA_DEVICE real Dev_DistSqr_to_Special_Point( rvec cp, rvec x ) 
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
        simulation_box my_ext_box, grid g, reax_list far_nbrs, int n, int N )
{
    int i, j, k, l, m, itr, num_far;
    real d, cutoff;
    ivec c, nbrs_x;
    rvec dvec;
    far_neighbor_data *nbr_data;//, *my_start;
    reax_atom *atom1, *atom2;

    l = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( l >= N )
    {
        return;
    }

    atom1 = &(my_atoms[l]);
    num_far = Dev_Start_Index( l, &far_nbrs );

    //get the coordinates of the atom and 
    //compute the grid cell
    if ( l < n )
    {
        for ( i = 0; i < 3; i++ )
        {
            c[i] = (int)((my_atoms[l].x[i]- my_ext_box.min[i])*g.inv_len[i]);   
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
        if ( g.str[index_grid_3d(i, j, k, &g)] <= g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] &&  
                Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs (i, j, k, itr, &g)], atom1->x) <= cutoff )
        {
            /* pick up another atom from the neighbor cell */
            for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                    m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
            {
                /* prevent recounting same pairs within a gcell */
                if ( l < m )
                {
                    atom2 = &(my_atoms[m]);
                    dvec[0] = atom2->x[0] - atom1->x[0];
                    dvec[1] = atom2->x[1] - atom1->x[1];
                    dvec[2] = atom2->x[2] - atom1->x[2];
                    d = rvec_Norm_Sqr( dvec );

                    if ( d <= cutoff )
                    { 
                        /* commit far neighbor to list */
                        nbr_data = &(far_nbrs.select.far_nbr_list[num_far]);
                        nbr_data->nbr = m;
                        nbr_data->d = SQRT( d );
                        rvec_Copy( nbr_data->dvec, dvec );
                        ivec_ScaledSum( nbr_data->rel_box, 1,
                                g.rel_box[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
                                -1, g.rel_box[index_grid_3d(i, j, k, &g)] );

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
                Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff )
        {
            /* pick up another atom from the neighbor cell */
            for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                    m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
            {
                /* prevent recounting same pairs within a gcell */
                if ( l > m )
                {
                    atom2 = &(my_atoms[m]);
                    dvec[0] = atom1->x[0] - atom2->x[0];
                    dvec[1] = atom1->x[1] - atom2->x[1];
                    dvec[2] = atom1->x[2] - atom2->x[2];
                    d = rvec_Norm_Sqr( dvec );

                    if ( d <= cutoff )
                    {
                        /* commit far neighbor to list */
                        nbr_data = &(far_nbrs.select.far_nbr_list[num_far]);
                        nbr_data->nbr = m;
                        nbr_data->d = SQRT( d );
                        rvec_Copy( nbr_data->dvec, dvec );
                        ivec_ScaledSum( nbr_data->rel_box, 1,
                                g.rel_box[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
                                -1, g.rel_box[index_grid_3d(i, j, k, &g)] );

                        ++num_far;
                    }
                }   
            }
        }

        ++itr;
    }   

    Dev_Set_End_Index( l, num_far, &far_nbrs );
}


//CUDA_GLOBAL void __launch_bounds__ (1024) k_mt_generate_neighbor_lists( reax_atom *my_atoms, 
CUDA_GLOBAL void k_mt_generate_neighbor_lists( reax_atom *my_atoms, 
        simulation_box my_ext_box, grid g, reax_list far_nbrs, int n, int N )
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
    far_neighbor_data *nbr_data, *my_start;
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
    atom1 = &(my_atoms[l]);
    num_far = Dev_Start_Index( l, &far_nbrs );
    my_start = &( far_nbrs.select.far_nbr_list[num_far] );

    //get the coordinates of the atom and 
    //compute the grid cell
    if ( l < n )
    {
        for ( i = 0; i < 3; i++ )
        {
            c[i] = (int)((my_atoms[l].x[i]- my_ext_box.min[i])*g.inv_len[i]);   
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
                && Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff) 
                || (g.str[index_grid_3d(i, j, k, &g)] >= g.str[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] 
                && Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff_ji) )
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
                if ( l < m  && m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] )
                {
                    atom2 = &(my_atoms[m]);
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

                if ( l > m  && m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] )
                {
                    atom2 = &(my_atoms[m]);
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
                    //got the indices
                    nbr_data = my_start + nbrssofar[my_bucket] + tnbr[threadIdx.x] - 1;
                    nbr_data->nbr = m;
                    if ( l < m )
                    {
                        dvec[0] = atom2->x[0] - atom1->x[0];
                        dvec[1] = atom2->x[1] - atom1->x[1];
                        dvec[2] = atom2->x[2] - atom1->x[2];
                        d = rvec_Norm_Sqr( dvec );
                        nbr_data->d = SQRT(d);
                        rvec_Copy( nbr_data->dvec, dvec );
                        ivec_ScaledSum( nbr_data->rel_box, 1, g.rel_box[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
                                -1, g.rel_box[index_grid_3d(i, j, k, &g)] );
                    } 
                    else
                    {
                        dvec[0] = atom1->x[0] - atom2->x[0];
                        dvec[1] = atom1->x[1] - atom2->x[1];
                        dvec[2] = atom1->x[2] - atom2->x[2];
                        d = rvec_Norm_Sqr( dvec );
                        nbr_data->d = SQRT(d);
                        rvec_Copy( nbr_data->dvec, dvec );
                        /*
                           CHANGE ORIGINAL
                           This is a bug in the original code 
                        ivec_ScaledSum( nbr_data->rel_box, 1, g.rel_box[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
                                -1, g.rel_box[index_grid_3d( i, j, k, &g)] );
                         */
                        ivec_ScaledSum( nbr_data->rel_box, -1, g.rel_box[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
                                1, g.rel_box[index_grid_3d(i, j, k, &g)] );
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
        Dev_Set_End_Index( l, num_far + nbrssofar[my_bucket], &far_nbrs );
    }
}


void Cuda_Generate_Neighbor_Lists( reax_system *system, simulation_data *data, 
        storage *workspace, reax_list **lists )
{
    int i, blocks;
#if defined(LOG_PERFORMANCE)
    real t_start = 0, t_elapsed = 0;

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
#endif

    /* one thread per atom implementation */
//    blocks = (system->N / NBRS_BLOCK_SIZE) +
//        ((system->N % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
//    k_generate_neighbor_lists <<< blocks, NBRS_BLOCK_SIZE >>>
//        ( system->d_my_atoms, system->my_ext_box, system->d_my_grid,
//          *(*dev_lists + FAR_NBRS), system->n, system->N );
//    cudaThreadSynchronize( );
//    cudaCheckError( );

    /* multiple threads per atom implementation */
    blocks = ((system->N * NB_KER_THREADS_PER_ATOM) / NBRS_BLOCK_SIZE) + 
        (((system->N * NB_KER_THREADS_PER_ATOM) % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
    k_mt_generate_neighbor_lists <<< blocks, NBRS_BLOCK_SIZE, 
        //sizeof(int) * (NBRS_BLOCK_SIZE + NBRS_BLOCK_SIZE / NB_KER_THREADS_PER_ATOM) >>>
        sizeof(int) * 2 * NBRS_BLOCK_SIZE >>>
            ( system->d_my_atoms, system->my_ext_box, system->d_my_grid,
              *(*dev_lists + FAR_NBRS), system->n, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );

#if defined(LOG_PERFORMANCE)
    if( system->my_rank == MASTER_NODE )
    {
        t_elapsed = Get_Timing_Info( t_start );
        data->timing.nbrs += t_elapsed;
    }
#endif

#if defined(DEBUG_FOCUS)  
    fprintf( stderr, "p%d @ step%d: nbrs done\n", 
            system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}


/* Estimate the number of far neighbors per atom (GPU) */
CUDA_GLOBAL void k_estimate_neighbors( reax_atom *my_atoms, 
        simulation_box my_ext_box, grid g, int n, int N, int total_cap,
        int *far_nbrs, int *max_far_nbrs, int *realloc_far_nbrs )
{
    int i, j, k, l, m, itr, num_far;
    real d, cutoff;
    ivec c, nbrs_x;
    rvec dvec;
    far_neighbor_data *nbr_data;
    reax_atom *atom1, *atom2;

    l = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( l >= total_cap )
    {
        return;
    }

    if ( l < N )
    {
        num_far = 0;
        atom1 = &(my_atoms[l]);

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
                    Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)], atom1->x) <= cutoff ) 
            {
                /* pick up another atom from the neighbor cell */
                for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                        m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
                {
                    /* prevent recounting same pairs within a gcell */
                    if ( l < m )
                    {
                        atom2 = &(my_atoms[m]);
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
                    Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs(i, j, k, itr, &g)],atom1->x) <= cutoff ) 
            {
                /* pick up another atom from the neighbor cell */
                for ( m = g.str[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
                        m < g.end[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
                {
                    /* prevent recounting same pairs within a gcell */
                    if ( l > m )
                    {
                        atom2 = &(my_atoms[m]);
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

    if ( num_far > max_far_nbrs[l] )
    {
        max_far_nbrs[l] = MAX( (int)(num_far * SAFE_ZONE), MIN_NBRS );
        *realloc_far_nbrs = TRUE;
    }

    far_nbrs[l] = num_far;
}


/* Estimate the number of far neighbors for each atoms 
 *
 * system: atomic system info
 * returns: SUCCESS if reallocation of the far neighbors list is necessary
 *  based on current per-atom far neighbor limits, FAILURE otherwise */
int Cuda_Estimate_Neighbors( reax_system *system, int step )
{
    int blocks, ret, ret_far_nbr;
    reax_list *far_nbrs;

    ret = SUCCESS;

    /* careful: this wrapper around cudaMemset(...) performs a byte-wide assignment
     * to the provided literal */
    cuda_memset( system->d_realloc_far_nbrs, FALSE, sizeof(int), 
            "Cuda_Estimate_Neighbors::d_realloc_far_nbrs" );

    blocks = system->total_cap / DEF_BLOCK_SIZE + 
        ((system->total_cap % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_estimate_neighbors <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->my_ext_box, system->d_my_grid,
          system->n, system->N, system->total_cap,
          system->d_far_nbrs, system->d_max_far_nbrs, system->d_realloc_far_nbrs );
    cudaThreadSynchronize( );
    cudaCheckError( );

    /* check reallocation flag on device */
    copy_host_device( &ret_far_nbr, system->d_realloc_far_nbrs, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Estimate_Neighbors::d_realloc_far_nbrs" );

    if ( ret_far_nbr == TRUE )
    {
        Cuda_Reduction_Sum( system->d_max_far_nbrs, system->d_total_far_nbrs,
                system->total_cap );

        copy_host_device( &(system->total_far_nbrs), system->d_total_far_nbrs, sizeof(int), 
                cudaMemcpyDeviceToHost, "Cuda_Estimate_Neighbors::d_total_far_nbrs" );

        if ( step > 0 )
        {
            dev_workspace->realloc.far_nbrs = TRUE;
        }
        ret = FAILURE;
    }

    return ret;
}


CUDA_GLOBAL void k_init_end_index( int * intr_cnt, int *indices, int *end_indices, int N )
{
    int i;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    end_indices[i] = indices[i] + intr_cnt[i];
}


CUDA_GLOBAL void k_setup_hindex( reax_atom *my_atoms, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    my_atoms[i].Hindex = i;
}


CUDA_GLOBAL void k_setup_hindex_part1( reax_atom *my_atoms, int * hindex, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    hindex[i] = my_atoms[i].Hindex;
}


CUDA_GLOBAL void k_setup_hindex_part2( reax_atom *my_atoms, int * hindex, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    if ( hindex[i + 1] - hindex[i] > 0 )
    {
        my_atoms[i].Hindex = hindex[i];
    }
    else
    {
        my_atoms[i].Hindex = -1;
    }
}


CUDA_GLOBAL void k_init_hbond_indices( reax_atom * atoms, single_body_parameters *sbp,
        int *hbonds, int *max_hbonds, int *indices, int *end_indices, int N )
{
    int i, hindex, my_hbonds;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    hindex = atoms[i].Hindex;

    if ( sbp[ atoms[i].type ].p_hbond == H_ATOM || 
            sbp[ atoms[i].type ].p_hbond == H_BONDING_ATOM )
    {
        my_hbonds = hbonds[i];
        indices[hindex] = max_hbonds[i];
        end_indices[hindex] = indices[hindex] + hbonds[i];
    }
    else
    {
        my_hbonds = 0;
        indices[hindex] = 0;
        end_indices[hindex] = 0;
    }
    atoms[i].num_hbonds = my_hbonds;

//    hindex = atoms[i].Hindex;
//
//    if ( hindex >= 0 )
//    {
//        my_hbonds = hbonds[i];
//        indices[hindex] = max_hbonds[i];
//        end_indices[hindex] = indices[hindex] + hbonds[i];
//    }
//    else
//    {
//        my_hbonds = 0;
//    }
//    atoms[i].num_hbonds = my_hbonds;
}


/* Initialize indices for far neighbors list post reallocation
 *
 * system: atomic system info. */
void Cuda_Init_Neighbor_Indices( reax_system *system )
{
    int blocks;
    reax_list *far_nbrs = *dev_lists + FAR_NBRS;

    /* init indices */
    Cuda_Scan_Excl_Sum( system->d_max_far_nbrs, far_nbrs->index, system->total_cap );

    /* init end_indices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_far_nbrs, far_nbrs->index, far_nbrs->end_index, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


void Cuda_Init_HBond_Indices( reax_system *system )
{
    int blocks;
    int *temp;
    reax_list *hbonds = *dev_lists + HBONDS;

    temp = (int *) scratch;
//    cuda_memset( temp, 0, 2 * (system->N + 1) * sizeof(int), 
//            "Cuda_Init_HBond_Indices::temp" );

    /* init Hindices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_setup_hindex <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );

//    blocks = system->n / DEF_BLOCK_SIZE + 
//        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);
//
//    k_setup_hindex_part1 <<< blocks, DEF_BLOCK_SIZE >>>
//        ( system->d_my_atoms, temp, system->n );
//    cudaThreadSynchronize( );
//    cudaCheckError( );
//
//    Cuda_Scan_Excl_Sum( temp, temp + system->n + 1, system->n + 1 );
//
//    k_setup_hindex_part2 <<< blocks, DEF_BLOCK_SIZE >>>
//        ( system->d_my_atoms, temp + system->n + 1, system->n );
//    cudaThreadSynchronize( );
//    cudaCheckError( );

    /* init indices and end_indices */
    Cuda_Scan_Excl_Sum( system->d_max_hbonds, temp, system->total_cap );

    k_init_hbond_indices <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, system->d_hbonds, temp, 
          hbonds->index, hbonds->end_index, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


void Cuda_Init_Bond_Indices( reax_system *system )
{
    int blocks;
    reax_list *bonds = *dev_lists + BONDS;

    /* init indices */
    Cuda_Scan_Excl_Sum( system->d_max_bonds, bonds->index, system->total_cap );

    /* init end_indices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_bonds, bonds->index, bonds->end_index, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


void Cuda_Init_Three_Body_Indices( int *indices, int entries )
{
    reax_list *thbody = *dev_lists + THREE_BODIES;

    Cuda_Scan_Excl_Sum( indices, thbody->index, entries );
}
