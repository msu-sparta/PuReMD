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
#include "dev_list.h"
#include "vector.h"

#include "index_utils.h"
#include "cuda_reax_constants.h"
#include "cuda_utils.h"
#include "tool_box.h"

//extern "C" real Get_Time( );
//extern "C" real Get_Timing_Info( real );

CUDA_DEVICE real Dev_DistSqr_to_Special_Point( rvec cp, rvec x ) 
{
  int  i;  
  real d_sqr = 0;

  for( i = 0; i < 3; ++i )
    if( cp[i] > NEG_INF )
      d_sqr += SQR( cp[i] - x[i] );

  return d_sqr;
}


CUDA_GLOBAL void ker_generate_neighbor_lists (	reax_atom *my_atoms, 
																simulation_box my_ext_box,
																grid g,
																reax_list far_nbrs, 
																int n, int N )
{
	int  i, j, k, l, m, itr, num_far;
	real d, cutoff;
	ivec c, nbrs_x;
	rvec dvec;
	grid_cell *gci, *gcj;
	far_neighbor_data *nbr_data;//, *my_start;
	reax_atom *atom1, *atom2;

	l = blockIdx.x * blockDim.x  + threadIdx.x;
	if (l >= N) return;

   atom1 = &(my_atoms[l]);
	num_far = Dev_Start_Index (l, &far_nbrs);

	//get the coordinates of the atom and 
	//compute the grid cell
	/*
	i = (int) (my_atoms[ l ].x[0] * g.inv_len[0]);
	j = (int) (my_atoms[ l ].x[1] * g.inv_len[1]);
	k = (int) (my_atoms[ l ].x[2] * g.inv_len[2]);
	*/
	if (l < n) {
		for (i = 0; i < 3; i++)
		{
   		c[i] = (int)((my_atoms[l].x[i]- my_ext_box.min[i])*g.inv_len[i]);   
			if( c[i] >= g.native_end[i] )
				c[i] = g.native_end[i] - 1;
			else if( c[i] < g.native_str[i] )
				c[i] = g.native_str[i];
		}
	} else {
		for (i = 0; i < 3; i++)
		{
			c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);
			if( c[i] < 0 ) c[i] = 0;
			else if( c[i] >= g.ncells[i] ) c[i] = g.ncells[i] - 1;
		}
	}

	i = c[0];
	j = c[1];
	k = c[2];

	//gci = &( g.cells[ index_grid_3d (i, j, k, &g) ] );
	cutoff = SQR(g.cutoff[index_grid_3d (i, j, k, &g)]);

   itr = 0;
   while( (g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)][0]) >= 0 ) { 

		ivec_Copy (nbrs_x, g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)] );
		//gcj =  &( g.cells [ index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ]);

   	if( g.str[index_grid_3d (i, j, k, &g)] <= g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] &&  
      (Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs (i, j, k, itr, &g)],atom1->x)<=cutoff) )
         /* pick up another atom from the neighbor cell */
         for( m = g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
					m < g.end[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m ) {
      		if(( l < m )) { // prevent recounting same pairs within a gcell 
      		  atom2 = &(my_atoms[m]);
      		  dvec[0] = atom2->x[0] - atom1->x[0];
      		  dvec[1] = atom2->x[1] - atom1->x[1];
      		  dvec[2] = atom2->x[2] - atom1->x[2];
      		  d = rvec_Norm_Sqr( dvec );
      		  if( d <= cutoff ) { 
      		    nbr_data = &(far_nbrs.select.far_nbr_list[num_far]);
      		    nbr_data->nbr = m;
      		    nbr_data->d = SQRT(d);
      		    rvec_Copy( nbr_data->dvec, dvec );
      		    //ivec_ScaledSum( nbr_data->rel_box, 1, gcj->rel_box, -1, gci->rel_box );
      		    ivec_ScaledSum( nbr_data->rel_box, 1, g.rel_box[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
					 												-1, g.rel_box[index_grid_3d (i, j, k, &g)] );
      		    ++num_far;
      		  }
				}
/*
      		if(( l > m )) { // prevent recounting same pairs within a gcell 
      		  atom2 = &(my_atoms[m]);
      		  dvec[0] = atom1->x[0] - atom2->x[0];
      	     dvec[1] = atom1->x[1] - atom2->x[1];
      		  dvec[2] = atom1->x[2] - atom2->x[2];
      		  d = rvec_Norm_Sqr( dvec );
      		  if( d <= cutoff ) { 
      		    nbr_data = &(far_nbrs.select.far_nbr_list[num_far]);
      		    nbr_data->nbr = m;
      		    nbr_data->d = SQRT(d);
      		    rvec_Copy( nbr_data->dvec, dvec );
      		    ivec_ScaledSum( nbr_data->rel_box, 
      		          -1, gcj->rel_box, 1, gci->rel_box );
      		    ++num_far;
      		  }
      		}   
*/
			}
       ++itr;
   }   

   itr = 0;
   while( (g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)][0]) >= 0 ) { 
		ivec_Copy (nbrs_x, g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)] );
		//gcj =  &( g.cells [ index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ]);
		cutoff = SQR(g.cutoff[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]);

   	if( g.str[index_grid_3d (i, j, k, &g)] >= g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] &&  
      (Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs (i, j, k, itr, &g)],atom1->x)<=cutoff) )
         for( m = g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
					m < g.end[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m ) {
      		if(( l > m )) {
      		  atom2 = &(my_atoms[m]);
      		  dvec[0] = atom1->x[0] - atom2->x[0];
      	     dvec[1] = atom1->x[1] - atom2->x[1];
      		  dvec[2] = atom1->x[2] - atom2->x[2];
      		  d = rvec_Norm_Sqr( dvec );
      		  if( d <= cutoff ) { 
      		    nbr_data = &(far_nbrs.select.far_nbr_list[num_far]);
      		    nbr_data->nbr = m;
      		    nbr_data->d = SQRT(d);
      		    rvec_Copy( nbr_data->dvec, dvec );
      		    //ivec_ScaledSum( nbr_data->rel_box, -1, gcj->rel_box, 1, gci->rel_box );
      		    ivec_ScaledSum( nbr_data->rel_box, 1, g.rel_box[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
					 												-1, g.rel_box[index_grid_3d (i, j, k, &g)] );
      		    ++num_far;
      		  }
      		}   
			}
       ++itr;
   }   

   Dev_Set_End_Index( l, num_far, &far_nbrs );
}


CUDA_GLOBAL void ker_mt_generate_neighbor_lists (	reax_atom *my_atoms, 
//CUDA_GLOBAL void __launch_bounds__ (1024) ker_mt_generate_neighbor_lists (	reax_atom *my_atoms, 
																simulation_box my_ext_box,
																grid g,
																reax_list far_nbrs, 
																int n, int N )
{

extern __shared__ int __nbr[];
extern __shared__ int __sofar [];
bool  nbrgen;

int __THREADS_PER_ATOM__ = NB_KER_THREADS_PER_ATOM;

int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
int warp_id = thread_id / __THREADS_PER_ATOM__;
int lane_id = thread_id & (__THREADS_PER_ATOM__ -1); 
int my_bucket = threadIdx.x / __THREADS_PER_ATOM__;

if (warp_id >= N ) return;

      int *tnbr = __nbr;
      int *nbrssofar = __nbr + blockDim.x;
		int max, leader;

	int  i, j, k, l, m, itr, num_far, ll;
	real d, cutoff, cutoff_ji;
	ivec c, nbrs_x;
	rvec dvec;
	grid_cell *gci, *gcj;
	far_neighbor_data *nbr_data, *my_start;
	reax_atom *atom1, *atom2;

	//l = blockIdx.x * blockDim.x  + threadIdx.x;
	//if (l >= N) return;

l = warp_id;

   atom1 = &(my_atoms[l]);
	num_far = Dev_Start_Index (l, &far_nbrs);

	my_start = &( far_nbrs.select.far_nbr_list [num_far] );

	//get the coordinates of the atom and 
	//compute the grid cell
	if (l < n) {
		for (i = 0; i < 3; i++)
		{
   		c[i] = (int)((my_atoms[l].x[i]- my_ext_box.min[i])*g.inv_len[i]);   
			if( c[i] >= g.native_end[i] )
				c[i] = g.native_end[i] - 1;
			else if( c[i] < g.native_str[i] )
				c[i] = g.native_str[i];
		}
	} else {
		for (i = 0; i < 3; i++)
		{
			c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);
			if( c[i] < 0 ) c[i] = 0;
			else if( c[i] >= g.ncells[i] ) c[i] = g.ncells[i] - 1;
		}
	}

	i = c[0];
	j = c[1];
	k = c[2];

	//gci = &( g.cells[ index_grid_3d (i, j, k, &g) ] );


tnbr[threadIdx.x] = 0;
      if (lane_id == 0) {
         nbrssofar [my_bucket] = 0;
      }
      __syncthreads ();

   itr = 0;
   //while( (gci->nbrs_x[itr][0]) >= 0 ) { 
   while( (g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)][0]) >= 0 ) { 

tnbr[threadIdx.x] = 0;
nbrgen = false;

		//ivec_Copy (nbrs_x, gci->nbrs_x[itr] );
		ivec_Copy (nbrs_x, g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)] );
		//gcj =  &( g.cells [ index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ]);

		//cutoff = SQR(gci->cutoff);
		cutoff = SQR (g.cutoff [index_grid_3d (i, j, k, &g)]);
		//cutoff_ji = SQR(gcj->cutoff);
		cutoff_ji = SQR(g.cutoff[ index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ]);
   	//if( ((gci->str <= gcj->str) && (Dev_DistSqr_to_Special_Point(gci->nbrs_cp[itr],atom1->x)<=cutoff)) 
		//	 || ((gci->str >= gcj->str) && (Dev_DistSqr_to_Special_Point(gci->nbrs_cp[itr],atom1->x)<=cutoff_ji)))
   	if( ((g.str[index_grid_3d (i, j, k, &g)] <= g.str[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]) 
					&& (Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs (i, j, k, itr, &g)],atom1->x)<=cutoff)) 
			 || ((g.str[index_grid_3d (i, j, k, &g)] >= g.str[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]) 
			 		&& (Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs (i, j, k, itr, &g)],atom1->x)<=cutoff_ji)))
		{
//max = gcj->end - gcj->str;
max = g.end[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] - g.str[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)];
tnbr[threadIdx.x] = 0;
nbrgen = false;
//m = lane_id  + gcj->str; //0-31
m = lane_id  + g.str[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; //0-31
int loopcount = max / __THREADS_PER_ATOM__ + ((max % __THREADS_PER_ATOM__) == 0 ? 0 : 1);
int iterations = 0;

         // pick up another atom from the neighbor cell
         //for( m = gcj->str; m < gcj->end; ++m ) 
			while (iterations < loopcount) {
tnbr [threadIdx.x] = 0;
nbrgen = false;

      		//if(( l < m ) && (m < gcj->end)) { // prevent recounting same pairs within a gcell 
      		if(( l < m ) && (m < g.end [index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)])) { // prevent recounting same pairs within a gcell 
      		  atom2 = &(my_atoms[m]);
      		  dvec[0] = atom2->x[0] - atom1->x[0];
      		  dvec[1] = atom2->x[1] - atom1->x[1];
      		  dvec[2] = atom2->x[2] - atom1->x[2];
      		  d = rvec_Norm_Sqr( dvec );
      		  if( d <= cutoff ) { 
tnbr [threadIdx.x] = 1;
nbrgen = true;
      		  }
				}

      		//if(( l > m ) && (m < gcj->end)) {
      		if(( l > m ) && (m < g.end[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)])) {
      		  atom2 = &(my_atoms[m]);
      		  dvec[0] = atom1->x[0] - atom2->x[0];
      	     dvec[1] = atom1->x[1] - atom2->x[1];
      		  dvec[2] = atom1->x[2] - atom2->x[2];
      		  d = rvec_Norm_Sqr( dvec );
      		  if( d <= cutoff_ji ) { 
tnbr [threadIdx.x] = 1;
nbrgen = true;
      		  }
      		} 

				//is neighbor generated
            if (nbrgen)
            {
               //do leader selection here
            	leader = -1;
               for (ll = my_bucket *__THREADS_PER_ATOM__; ll < (my_bucket)*__THREADS_PER_ATOM__ + __THREADS_PER_ATOM__; ll++)
               	if (tnbr[ll]){
                  	leader = ll;
                     break;
                  }

               //do the reduction;
               if (threadIdx.x == leader)
               	for (ll = 1; ll < __THREADS_PER_ATOM__; ll++)
                  	tnbr [my_bucket * __THREADS_PER_ATOM__ + ll] += tnbr [my_bucket * __THREADS_PER_ATOM__ + (ll-1)];
            }

            if (nbrgen)
            {
            	//got the indices
               nbr_data = my_start + nbrssofar[my_bucket] + tnbr [threadIdx.x] - 1;
               nbr_data->nbr = m;
					if (l < m) {
      		  		dvec[0] = atom2->x[0] - atom1->x[0];
      		  		dvec[1] = atom2->x[1] - atom1->x[1];
      		  		dvec[2] = atom2->x[2] - atom1->x[2];
      		  		d = rvec_Norm_Sqr( dvec );
						nbr_data->d = SQRT (d);
      		    	rvec_Copy( nbr_data->dvec, dvec );
      		    	//ivec_ScaledSum( nbr_data->rel_box, 1, gcj->rel_box, -1, gci->rel_box );
      		    	ivec_ScaledSum( nbr_data->rel_box, 1, g.rel_box[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
																		-1, g.rel_box[index_grid_3d( i, j, k, &g)] );
					} 
					else {
      		  		dvec[0] = atom1->x[0] - atom2->x[0];
      	     		dvec[1] = atom1->x[1] - atom2->x[1];
      		  		dvec[2] = atom1->x[2] - atom2->x[2];
      		  		d = rvec_Norm_Sqr( dvec );
      		    	nbr_data->d = SQRT(d);
      		    	rvec_Copy( nbr_data->dvec, dvec );
      		    	//ivec_ScaledSum( nbr_data->rel_box, -1, gcj->rel_box, 1, gci->rel_box );
						/*
						CHANGE ORIGINAL
						This is a bug in the original code 
      		    	ivec_ScaledSum( nbr_data->rel_box, 1, g.rel_box[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
																		-1, g.rel_box[index_grid_3d( i, j, k, &g)] );
						*/
      		    	ivec_ScaledSum( nbr_data->rel_box, -1, g.rel_box[index_grid_3d( nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)], 
																		1, g.rel_box[index_grid_3d( i, j, k, &g)] );
					}

               if (threadIdx.x == leader)
               	nbrssofar[my_bucket] += tnbr[my_bucket *__THREADS_PER_ATOM__ + (__THREADS_PER_ATOM__ - 1)];
            }

            m += __THREADS_PER_ATOM__;
            iterations ++;

            //cleanup
            nbrgen = false;
            tnbr [threadIdx.x] = 0;
			}
		}
      ++itr;
   }   

	if (lane_id == 0)
		Dev_Set_End_Index (l, num_far + nbrssofar[my_bucket], &far_nbrs);
   	//Dev_Set_End_Index( l, num_far, &far_nbrs );
}



CUDA_GLOBAL void ker_count_total_nbrs (reax_list far_nbrs, int N, int *result)
{
	//strided access
	extern __shared__ int count[];
	unsigned int i = threadIdx.x;
	int my_count = 0;
	count[i] = 0;

	for (i = threadIdx.x; i < N; i += threadIdx.x + blockDim.x)
		count[threadIdx.x] += Dev_Num_Entries (i, &far_nbrs);

	__syncthreads ();

	for (int offset = blockDim.x/2; offset > 0; offset >>=1 )
		if(threadIdx.x < offset)
			count [threadIdx.x] += count [threadIdx.x + offset];
	
	__syncthreads ();

	if (threadIdx.x == 0)
		*result = count [threadIdx.x];
}

extern "C" void Cuda_Generate_Neighbor_Lists( reax_system *system, simulation_data *data, 
			      storage *workspace, reax_list **lists )
{
	int blocks, num_far;
	int *d_num_far = (int *) scratch;
#if defined(LOG_PERFORMANCE)
  real t_start=0, t_elapsed=0;
  
  if( system->my_rank == MASTER_NODE )
    t_start = Get_Time( );
#endif

	cuda_memset (d_num_far, 0, sizeof (int), "num_far");

	//invoke the kernel here
	//one thread per atom implementation
	/*
	blocks = (system->N / NBRS_BLOCK_SIZE) + 
				((system->N % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
	ker_generate_neighbor_lists <<<blocks, NBRS_BLOCK_SIZE>>>
		(system->d_my_atoms, system->my_ext_box, system->d_my_grid,
		*(*dev_lists + FAR_NBRS), system->n, system->N);
	cudaThreadSynchronize ();
	cudaCheckError ();
	*/

	//Multiple threads per atom implementation
	blocks = ((system->N * NB_KER_THREADS_PER_ATOM) / NBRS_BLOCK_SIZE) + 
				(((system->N * NB_KER_THREADS_PER_ATOM) % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
	ker_mt_generate_neighbor_lists <<<blocks, NBRS_BLOCK_SIZE, 
			//sizeof (int) * (NBRS_BLOCK_SIZE + (NBRS_BLOCK_SIZE / NB_KER_THREADS_PER_ATOM)) >>>
			sizeof (int) *  2 * (NBRS_BLOCK_SIZE) >>>
		(system->d_my_atoms, system->my_ext_box, system->d_my_grid,
		*(*dev_lists + FAR_NBRS), system->n, system->N);
	cudaThreadSynchronize ();
	cudaCheckError ();

	/*
	ker_count_total_nbrs  <<<1, NBRS_BLOCK_SIZE, sizeof (int) * NBRS_BLOCK_SIZE>>>
		(*(*dev_lists + FAR_NBRS), system->N, d_num_far);
	cudaThreadSynchronize ();
	cudaCheckError ();
	copy_host_device (&num_far, d_num_far, sizeof (int), cudaMemcpyDeviceToHost, "num_far");
	*/

	int *index = (int *) host_scratch;
	memset (index , 0, 2 * sizeof (int) * system->N);
	int *end_index = index + system->N;

	copy_host_device (index, (*dev_lists + FAR_NBRS)->index, 
									sizeof (int) * (*dev_lists + FAR_NBRS)->n, cudaMemcpyDeviceToHost, "nbrs:index");
	copy_host_device (end_index, (*dev_lists + FAR_NBRS)->end_index, 
									sizeof (int) * (*dev_lists + FAR_NBRS)->n, cudaMemcpyDeviceToHost, "nbrs:end_index");

	num_far = 0;
	for (int i = 0; i < system->N; i++)
		num_far = end_index[i] - index[i];

   dev_workspace->realloc.num_far = num_far;

#if defined(LOG_PERFORMANCE)
  if( system->my_rank == MASTER_NODE ) {
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.nbrs += t_elapsed;
  }
#endif

#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "p%d @ step%d: nbrs done - num_far=%d\n", 
	   system->my_rank, data->step, num_far );
  MPI_Barrier( MPI_COMM_WORLD );
#endif
}

CUDA_GLOBAL void ker_estimate_neighbors (	reax_atom *my_atoms, 
														simulation_box my_ext_box,
														grid g,
														int n,
														int N, 
														int *indices)
{
	int  i, j, k, l, m, itr, num_far;
	real d, cutoff;
	rvec dvec, c;
	ivec nbrs_x;
	grid_cell *gci, *gcj;
	far_neighbor_data *nbr_data;//, *my_start;
	reax_atom *atom1, *atom2;

	l = blockIdx.x * blockDim.x  + threadIdx.x;
	if (l >= N) return;

	num_far = 0;
   atom1 = &(my_atoms[l]);
	indices [l] = 0;

	//if (atom1->orig_id < 0) return;

	//get the coordinates of the atom and 
	//compute the grid cell
	if (l < n) {
		for (i = 0; i < 3; i++)
		{
   		c[i] = (int)((my_atoms[l].x[i]- my_ext_box.min[i])*g.inv_len[i]);   
			if( c[i] >= g.native_end[i] )
				c[i] = g.native_end[i] - 1;
			else if( c[i] < g.native_str[i] )
				c[i] = g.native_str[i];
		}
	} else {
		for (i = 0; i < 3; i++)
		{
			c[i] = (int)((my_atoms[l].x[i] - my_ext_box.min[i]) * g.inv_len[i]);
			if( c[i] < 0 ) c[i] = 0;
			else if( c[i] >= g.ncells[i] ) c[i] = g.ncells[i] - 1;
		}
	}

	i = c[0];
	j = c[1];
	k = c[2];

	//gci = &( g.cells[ index_grid_3d (i, j, k, &g) ] );
	//cutoff = SQR(gci->cutoff);
	cutoff = SQR(g.cutoff [index_grid_3d (i, j, k, &g) ]);

   itr = 0;
   while( (g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)][0]) >= 0) { 
		ivec_Copy (nbrs_x, g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)] );
		//gcj =  &( g.cells [ index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ]);

   	if( //(g.str[index_grid_3d (i, j, k, &g)] <= g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]) &&  
      (Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs (i, j, k, itr, &g)],atom1->x)<=cutoff) ) 
		{
         // pick up another atom from the neighbor cell 
         for( m = g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
					m < g.end[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
			{
      		if( l < m ) { // prevent recounting same pairs within a gcell 
      		  atom2 = &(my_atoms[m]);
      		  dvec[0] = atom2->x[0] - atom1->x[0];
      		  dvec[1] = atom2->x[1] - atom1->x[1];
      		  dvec[2] = atom2->x[2] - atom1->x[2];
      		  d = rvec_Norm_Sqr( dvec );
      		  if( d <= cutoff ) { 
				  		num_far ++;
      		  }
      		}   
			}
		}
      ++itr;

   }   

   itr = 0;
   while( (g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)][0]) >= 0) { 
		ivec_Copy (nbrs_x, g.nbrs_x[index_grid_nbrs (i, j, k, itr, &g)] );
		//gcj =  &( g.cells [ index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g) ]);
		cutoff = SQR(g.cutoff[index_grid_3d(nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]);

   	if( g.str[index_grid_3d (i, j, k, &g)] >= g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)] &&  
      (Dev_DistSqr_to_Special_Point(g.nbrs_cp[index_grid_nbrs (i, j, k, itr, &g)],atom1->x)<=cutoff) ) 
		{
         // pick up another atom from the neighbor cell 
         for( m = g.str[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; 
					m < g.end[index_grid_3d (nbrs_x[0], nbrs_x[1], nbrs_x[2], &g)]; ++m )
			{
      		if( l > m ) { // prevent recounting same pairs within a gcell 
      		  atom2 = &(my_atoms[m]);
      		  dvec[0] = atom2->x[0] - atom1->x[0];
      		  dvec[1] = atom2->x[1] - atom1->x[1];
      		  dvec[2] = atom2->x[2] - atom1->x[2];
      		  d = rvec_Norm_Sqr( dvec );
      		  if( d <= cutoff ) { 
				  		num_far ++;
      		  }
      		}   
			}
		}
      ++itr;
   }   

	indices [l] = num_far;// * SAFE_ZONE;
}

void Cuda_Estimate_Neighbors( reax_system *system, int *nbr_indices )
{
	int blocks, num_nbrs;
	int *indices = (int *) scratch;
	reax_list *far_nbrs;

	cuda_memset (indices, 0, sizeof (int) * system->total_cap, 
					"neighbors:indices");

	blocks = system->N / DEF_BLOCK_SIZE + 
				((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);
	ker_estimate_neighbors <<< blocks, DEF_BLOCK_SIZE >>>
		(system->d_my_atoms, (system->my_ext_box), system->d_my_grid, 
			system->n, system->N, indices);
	cudaThreadSynchronize ();
	cudaCheckError ();

	copy_host_device (nbr_indices, indices, sizeof (int) * system->total_cap, 
							cudaMemcpyDeviceToHost, "nbrs:indices");
}

void Cuda_Init_Neighbors_Indices (int *indices, int entries)
{
	reax_list *far_nbrs = *dev_lists + FAR_NBRS;

	copy_host_device (indices, (far_nbrs->index + 1), (entries -1) * sizeof (int), 
							cudaMemcpyHostToDevice, "nbrs:index");
	copy_host_device (indices, (far_nbrs->end_index + 1), (entries-1) * sizeof (int), 
							cudaMemcpyHostToDevice, "nbrs:end_index");
}

void Cuda_Init_HBond_Indices (int *indices, int entries)
{
	reax_list *hbonds = *dev_lists + HBONDS;

	for (int i = 1 ; i < entries; i++)
		indices [i] += indices [i-1];

	copy_host_device (indices, hbonds->index + 1, (entries-1) * sizeof (int), 
							cudaMemcpyHostToDevice, "hbonds:index");
	copy_host_device (indices, hbonds->end_index + 1, (entries-1) * sizeof (int), 
							cudaMemcpyHostToDevice, "hbonds:end_index");
}

void Cuda_Init_Bond_Indices (int *indices, int entries, int num_intrs)
{
	reax_list *bonds = *dev_lists + BONDS;

	indices[0] = MAX( indices[0]*2, MIN_BONDS);
	for (int i = 1 ; i < entries; i++) {
		indices[i] = MAX( indices[i]*2, MIN_BONDS);
	}

	for (int i = 1 ; i < entries; i++) {
		indices[i] += indices[i-1];
	}

	copy_host_device (indices, (bonds->index + 1), (entries - 1) * sizeof (int), 
							cudaMemcpyHostToDevice, "bonds:index");
	copy_host_device (indices, (bonds->end_index + 1), (entries - 1) * sizeof (int), 
							cudaMemcpyHostToDevice, "bonds:end_index");
	
	for (int i = 1 ; i < entries; i++)
		if (indices [i] > num_intrs) {
			fprintf (stderr, "We have a problem here ==> %d index: %d, num_intrs: %d \n", 
									i, indices[i], num_intrs);	
			exit (-1);
		}
}

/*

CUDA_GLOBAL void ker_validate_neighbors (reax_atom *my_atoms, 
													reax_list far_nbrs, 
													int N)
{
	int i, j, pj;
	far_neighbor_data *nbr_pj;
	reax_atom *atom_i;
	int start_i, end_i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	
	atom_i = &( my_atoms[i] );
	start_i = Dev_Start_Index (i, &far_nbrs );
	end_i = Dev_End_Index (i, &far_nbrs );

    for( pj = start_i; pj < end_i; ++pj ) {
      nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
      j = nbr_pj->nbr;
		nbr_pj->d = 0;
		rvec_MakeZero (nbr_pj->dvec);
	}
}

void validate_neighbors (reax_system *system)
{
	int blocks;
	blocks = (system->N / NBRS_BLOCK_SIZE) + 
				((system->N % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
	ker_validate_neighbors <<< blocks, NBRS_BLOCK_SIZE>>>
		(system->d_my_atoms, *(*dev_lists + FAR_NBRS), system->N);
	cudaThreadSynchronize ();
	cudaCheckError ();

	fprintf (stderr, " Neighbors validated and is fine... \n");
}

*/
