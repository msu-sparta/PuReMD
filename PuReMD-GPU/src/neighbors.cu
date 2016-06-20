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

#include "neighbors.h"
#include "box.h"
#include "grid.h"
#include "list.h"
#include "reset_utils.h"
#include "system_props.h"
#include "vector.h"
#include "index_utils.h"
#include "cuda_utils.h"

extern inline DEVICE int index_grid (int blocksize)
{
	return blockIdx.x * gridDim.y * gridDim.z * blocksize +  
		blockIdx.y * gridDim.z * blocksize +  
		blockIdx.z * blocksize ;
}

extern inline HOST_DEVICE int index_grid_debug (int x, int y, int z, int blocksize)
{
	return x * 8 * 8 * blocksize +  
		y * 8 * blocksize +  
		z * blocksize ;
}

inline HOST_DEVICE real DistSqr_to_CP( rvec cp, rvec x )
{
	int  i;
	real d_sqr = 0;

	for( i = 0; i < 3; ++i )
		if( cp[i] > NEG_INF )
			d_sqr += SQR( cp[i] - x[i] );

	return d_sqr;
}

HOST_DEVICE int Are_Far_Neighbors( rvec x1, rvec x2, simulation_box *box, 
		real cutoff, far_neighbor_data *data )
{
	real norm_sqr, d, tmp;
	int i;

	norm_sqr = 0;

	for( i = 0; i < 3; i++ ) { 
		d = x2[i] - x1[i];
		tmp = SQR(d);

		if( tmp >= SQR( box->box_norms[i] / 2.0 ) ) {    
			if( x2[i] > x1[i] ) { 
				d -= box->box_norms[i];
				data->rel_box[i] = -1; 
			}   
			else {
				d += box->box_norms[i];
				data->rel_box[i] = +1; 
			}   

			data->dvec[i] = d;
			norm_sqr += SQR(d);
		}   
		else {
			data->dvec[i] = d;
			norm_sqr += tmp;
			data->rel_box[i] = 0;
		}   
	}

	if( norm_sqr <= SQR(cutoff) ){
		data->d = sqrt(norm_sqr);
		return 1;
	}

	return 0;
}

void Generate_Neighbor_Lists( reax_system *system, control_params *control, 
		simulation_data *data, static_storage *workspace,
		list **lists, output_controls *out_control )
{
	int  i, j, k, l, m, itr;
	int  x, y, z;
	int  atom1, atom2, max;
	int  num_far;
	int  *nbr_atoms;
	ivec *nbrs;
	rvec *nbrs_cp;
	grid *g;
	list *far_nbrs;
	far_neighbor_data *nbr_data;
	real t_start, t_elapsed;

	// fprintf( stderr, "\n\tentered nbrs - " );
	g = &( system->g );
	far_nbrs = (*lists) + FAR_NBRS;
	Bin_Atoms( system, workspace );

	t_start = Get_Time( );

	// fprintf( stderr, "atoms sorted - " );
	num_far = 0;

	/* first pick up a cell in the grid */
	for( i = 0; i < g->ncell[0]; i++ )
		for( j = 0; j < g->ncell[1]; j++ )
			for( k = 0; k < g->ncell[2]; k++ ) {
				nbrs = &g->nbrs[ index_grid_nbrs (i,j,k,0,g) ];
				nbrs_cp = &g->nbrs_cp[ index_grid_nbrs (i,j,k,0,g) ];
				//fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

				/* pick up an atom from the current cell */
				for(l = 0; l < g->top[ index_grid_3d (i,j,k,g) ]; ++l ){
					atom1 = g->atoms[ index_grid_atoms (i,j,k,l,g) ];
					Set_Start_Index( atom1, num_far, far_nbrs );
					//fprintf( stderr, "\tatom %d\n", atom1 );

					itr = 0;
					while( nbrs[itr][0] >= 0 ){
						x = nbrs[itr][0];
						y = nbrs[itr][1];
						z = nbrs[itr][2];
						//fprintf( stderr, "\t\tgridcell %d %d %d\n", x, y, z );

						if( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <= 
								SQR(control->vlist_cut) ) { 	
							nbr_atoms = &g->atoms[ index_grid_atoms (x,y,z,0,g) ];
							max = g->top[ index_grid_3d (x,y,z,g) ];
							//fprintf( stderr, "\t\tmax: %d\n", max );

							/* pick up another atom from the neighbor cell */
							for( m = 0; m < max; ++m ) {
								atom2 = nbr_atoms[m];
								if( atom1 > atom2 ) {
									nbr_data = &(far_nbrs->select.far_nbr_list[num_far]);
									if(Are_Far_Neighbors(system->atoms[atom1].x,
												system->atoms[atom2].x, 
												&(system->box), control->vlist_cut, 
												nbr_data)) {
										nbr_data->nbr = atom2;

										++num_far;
									}
								}
							}
						}

						++itr;
					}

					Set_End_Index( atom1, num_far, far_nbrs );
					//fprintf(stderr, "i:%d, start: %d, end: %d - itr: %d\n", 
					//  atom1,Start_Index(atom1,far_nbrs),End_Index(atom1,far_nbrs),
					//  itr); 
				}
			}

	fprintf (stderr, " TOTAL HOST NEIGHBORS : %d \n", num_far);

	if( num_far > far_nbrs->num_intrs * DANGER_ZONE ) {
		workspace->realloc.num_far = num_far;
		if( num_far > far_nbrs->num_intrs ){
			fprintf( stderr, "step%d-ran out of space on far_nbrs: top=%d, max=%d",
					data->step, num_far, far_nbrs->num_intrs );
			exit( INSUFFICIENT_SPACE );
		}
	}

	t_elapsed = Get_Timing_Info( t_start );
	data->timing.nbrs += t_elapsed;

#if defined(DEBUG)
	for( i = 0; i < system->N; ++i ) {
		qsort( &(far_nbrs->select.far_nbr_list[ Start_Index(i, far_nbrs) ]), 
				Num_Entries(i, far_nbrs), sizeof(far_neighbor_data), 
				compare_far_nbrs ); 
	}
#endif
#if defined(DEBUG_FOCUS)  
	//fprintf( stderr, "nbrs - ");
	//fprintf( stderr, "nbrs done, num_far: %d\n", num_far );
#endif
#if defined(TEST_ENERGY)
	//Print_Far_Neighbors( system, control, workspace, lists );
#endif
}


int Estimate_NumNeighbors( reax_system *system, control_params *control, 
		static_storage *workspace, list **lists )
{
	int  i, j, k, l, m, itr;
	int  x, y, z;
	int  atom1, atom2, max;
	int  num_far;
	int  *nbr_atoms;
	ivec *nbrs;
	rvec *nbrs_cp;
	grid *g;
	far_neighbor_data nbr_data;

	int 	start = 0, finish = 0;

	// fprintf( stderr, "\n\tentered nbrs - " );
	g = &( system->g );
	Bin_Atoms( system, workspace );
	// fprintf( stderr, "atoms sorted - " );
	num_far = 0;
	g->max_cuda_nbrs = 0;

	/* first pick up a cell in the grid */
	for( i = 0; i < g->ncell[0]; i++ )
		for( j = 0; j < g->ncell[1]; j++ )
			for( k = 0; k < g->ncell[2]; k++ ) {
				nbrs = &g->nbrs[index_grid_nbrs (i,j,k,0,g) ];
				nbrs_cp = &g->nbrs_cp[index_grid_nbrs (i,j,k,0,g) ];
				//fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

				/* pick up an atom from the current cell */
				for(l = 0; l < g->top[index_grid_3d (i,j,k,g) ]; ++l ){
					atom1 = g->atoms[index_grid_atoms (i,j,k,l,g) ];
					start = num_far;

					itr = 0;
					while( nbrs[itr][0] >= 0 ){
						x = nbrs[itr][0];
						y = nbrs[itr][1];
						z = nbrs[itr][2];
						//fprintf( stderr, "\t\tgridcell %d %d %d\n", x, y, z );

						if( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <= 
								SQR(control->vlist_cut) ) { 	
							nbr_atoms = &g->atoms[index_grid_atoms (x,y,z,0,g) ];
							max = g->top[index_grid_3d (x,y,z,g) ];
							//fprintf( stderr, "\t\tmax: %d\n", max );

							/* pick up another atom from the neighbor cell -
							   we have to compare atom1 with its own periodic images as well, 
							   that's why there is also equality in the if stmt below */
							for( m = 0; m < max; ++m ) {
								atom2 = nbr_atoms[m];
								//if( nbrs[itr+1][0] >= 0 || atom1 > atom2 ) {
								if( atom1 > atom2 ) {
									if(Are_Far_Neighbors(system->atoms[atom1].x,
												system->atoms[atom2].x, 
												&(system->box), control->vlist_cut, 
												&nbr_data))
										++num_far;
								}
							}
							}

							++itr;
						}

						// finish note
						finish = num_far;
						if (g->max_cuda_nbrs <= (finish - start)){
							g->max_cuda_nbrs	= finish - start;
						}
					}
				}

#if defined(DEBUG_FOCUS)  
				fprintf( stderr, "estimate nbrs done, num_far: %d\n", num_far );
#endif
				return num_far * SAFE_ZONE;
			}

	GLOBAL void Estimate_NumNeighbors ( reax_atom *sys_atoms,
			grid g,
			simulation_box *box,
			control_params *control,
			int *indices)
	{
		int *atoms = g.atoms;
		int *top = g.top;
		ivec *nbrs = g.nbrs; 
		rvec *nbrs_cp = g.nbrs_cp;

		int *nbr_atoms;
		int atom1, atom2, l, iter, max, m, num_far;
		far_neighbor_data nbr_data;
		int x, y, z, i;

		if (threadIdx.x >= *(top + index_grid(1))){
			return;
		} 

		nbrs = nbrs + index_grid (g.max_nbrs);
		nbrs_cp = nbrs_cp + index_grid (g.max_nbrs);
		atom1 = atoms [ index_grid (g.max_atoms) + threadIdx.x];

		num_far = 0;
		iter = 0;

		while (nbrs[iter][0] >= 0) {
			x = nbrs[iter][0];
			y = nbrs[iter][1];
			z = nbrs[iter][2];

			//condition check for cutoff here
			if (DistSqr_to_CP (nbrs_cp[iter], sys_atoms[atom1].x) <= 
					SQR (control->vlist_cut)) 
			{
				nbr_atoms = &(atoms [index_grid_atoms (x, y, z, 0, &g) ]);
				max = top [index_grid_3d(x, y, z, &g)];
				for (m = 0; m < max; m++) {
					atom2 = nbr_atoms[m];

					//CHANGE ORIGINAL
					/*
					   if (atom1 > atom2) {
					   if (Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
					   control->vlist_cut, &nbr_data)){
					   ++num_far;
					   }
					   }
					 */
					if (atom1 > atom2) {
						if (Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
									control->vlist_cut, &nbr_data)){
							++num_far;
						}
					}
					else if (atom1 < atom2) {
						if (Are_Far_Neighbors (sys_atoms[atom2].x, sys_atoms[atom1].x, box, 
									control->vlist_cut, &nbr_data)){
							++num_far;
						}
					}
					//CHANGE ORIGINAL
				}
			}
			++iter;
		}

		//indices[ atom1 ] = num_far;// * SAFE_ZONE;
		indices[ atom1 ] = num_far * SAFE_ZONE;
	}

	/*One thread per atom Implementation */
	GLOBAL void New_Estimate_NumNeighbors ( 	reax_atom *sys_atoms,
			grid g,
			simulation_box *box,
			control_params* control, 
			int N, int *indices)
	{
		int *atoms = g.atoms;
		int *top = g.top;
		ivec *nbrs = g.nbrs; 
		rvec *nbrs_cp = g.nbrs_cp;

		int 	*nbr_atoms;
		int   atom1, atom2, iter, max, m, num_far;
		int 	x, y, z, i;
		int atom_x, atom_y, atom_z;
		far_neighbor_data temp;
		rvec atom1_x;

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index > N) return;

		atom_x = (int)(sys_atoms[index].x[0] * g.inv_len[0]);
		atom_y = (int)(sys_atoms[index].x[1] * g.inv_len[1]);
		atom_z = (int)(sys_atoms[index].x[2] * g.inv_len[2]);

#ifdef __BNVT_FIX__
		if (atom_x >= g.ncell[0]) atom_x = g.ncell[0]-1;
		if (atom_y >= g.ncell[1]) atom_y = g.ncell[1]-1;
		if (atom_z >= g.ncell[2]) atom_z = g.ncell[2]-1;
#endif

		nbrs = nbrs + index_grid_nbrs (atom_x, atom_y, atom_z, 0, &g);
		nbrs_cp = nbrs_cp + index_grid_nbrs (atom_x, atom_y, atom_z, 0, &g);
		atom1 = index;

		rvec_Copy (atom1_x, sys_atoms [atom1].x );

		num_far = 0;
		iter = 0;

		while (nbrs[iter][0] >= 0) {
			x = nbrs[iter][0];
			y = nbrs[iter][1];
			z = nbrs[iter][2];

			if (DistSqr_to_CP (nbrs_cp[iter], atom1_x) <= 
					SQR (control->vlist_cut)) 
			{
				nbr_atoms = &(atoms [index_grid_atoms (x, y, z, 0, &g) ]);
				max = top [index_grid_3d(x, y, z, &g)];

				for (m = 0; m < max; m++) 
				{
					atom2 = nbr_atoms[m];
					if (atom1 > atom2) {
						if (Are_Far_Neighbors (atom1_x, sys_atoms[atom2].x, box, 
									control->vlist_cut, &temp)){
							num_far++;
						}
					}
					else if (atom1 < atom2) {
						if (Are_Far_Neighbors (sys_atoms[atom2].x, atom1_x, box, 
									control->vlist_cut, &temp)){
							num_far ++;
						}
					}
				}
			}
			++iter;
		}
		indices [atom1] = num_far * SAFE_ZONE;
	}



	/*One thread per entry in the gcell implementation */
	GLOBAL void Generate_Neighbor_Lists ( 	reax_atom *sys_atoms,
			grid g,
			simulation_box *box,
			control_params* control, 
			list far_nbrs)
	{
		int *atoms = g.atoms;
		int *top = g.top;
		ivec *nbrs = g.nbrs; 
		rvec *nbrs_cp = g.nbrs_cp;

		int 	*nbr_atoms;
		int   atom1, atom2, l, iter, max, m, num_far;
		int 	x, y, z, i;
		far_neighbor_data *nbr_data;
		far_neighbor_data temp;

		if (threadIdx.x >= *(top + index_grid(1))){
			return;
		} 

		nbrs = nbrs + index_grid (g.max_nbrs);
		nbrs_cp = nbrs_cp + index_grid (g.max_nbrs);
		atom1 = atoms [ index_grid (g.max_atoms) + threadIdx.x];

		num_far = Start_Index (atom1, &far_nbrs);
		//Set_Start_Index (atom1, 0, &far_nbrs);
		//num_far =  0;
		iter = 0;

		while (nbrs[iter][0] >= 0) {
			x = nbrs[iter][0];
			y = nbrs[iter][1];
			z = nbrs[iter][2];

			//condition check for cutoff here
			if (DistSqr_to_CP (nbrs_cp[iter], sys_atoms[atom1].x) <= 
					SQR (control->vlist_cut)) 
			{
				nbr_atoms = &(atoms [index_grid_atoms (x, y, z, 0, &g) ]);
				max = top [index_grid_3d(x, y, z, &g)];

				for (m = 0; m < max; m++) {
					atom2 = nbr_atoms[m];

					//nbr_data = & ( far_nbrs.select.far_nbr_list[atom1 * g.max_cuda_nbrs + num_far] );

					//CHANGE ORIGINAL
					/*
					   if (atom1 > atom2) {
					   if (Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
					   control->vlist_cut, &temp)){

					   nbr_data = & ( far_nbrs.select.far_nbr_list[num_far] );
					   nbr_data->nbr = atom2;
					   nbr_data->rel_box[0] = temp.rel_box[0];
					   nbr_data->rel_box[1] = temp.rel_box[1];
					   nbr_data->rel_box[2] = temp.rel_box[2];

					   nbr_data->d = temp.d;
					   nbr_data->dvec[0] = temp.dvec[0];
					   nbr_data->dvec[1] = temp.dvec[1];
					   nbr_data->dvec[2] = temp.dvec[2];
					   ++num_far;
					   }
					   }
					 */
					if (atom1 > atom2) {
						if (Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
									control->vlist_cut, &temp)){
							nbr_data = & ( far_nbrs.select.far_nbr_list[num_far] );
							nbr_data->nbr = atom2;
							nbr_data->rel_box[0] = temp.rel_box[0];
							nbr_data->rel_box[1] = temp.rel_box[1];
							nbr_data->rel_box[2] = temp.rel_box[2];

							nbr_data->d = temp.d;
							nbr_data->dvec[0] = temp.dvec[0];
							nbr_data->dvec[1] = temp.dvec[1];
							nbr_data->dvec[2] = temp.dvec[2];
							++num_far;
						}
					}
					else if (atom1 < atom2) {
						if (Are_Far_Neighbors (sys_atoms[atom2].x, sys_atoms[atom1].x, box, 
									control->vlist_cut, &temp)){
							nbr_data = & ( far_nbrs.select.far_nbr_list[num_far] );
							nbr_data->nbr = atom2;
							nbr_data->rel_box[0] = temp.rel_box[0];
							nbr_data->rel_box[1] = temp.rel_box[1];
							nbr_data->rel_box[2] = temp.rel_box[2];

							nbr_data->d = temp.d;
							nbr_data->dvec[0] = temp.dvec[0];
							nbr_data->dvec[1] = temp.dvec[1];
							nbr_data->dvec[2] = temp.dvec[2];
							++num_far;
						}
					}
					//CHANGE ORIGINAL
				}
			}
			++iter;
		}

		//end the far_neighbor list here
		Set_End_Index (atom1, num_far, &far_nbrs);
	}


	/*One thread per atom Implementation */
	GLOBAL void New_Generate_Neighbor_Lists ( 	reax_atom *sys_atoms,
			grid g,
			simulation_box *box,
			control_params* control, 
			list far_nbrs, int N)
	{
		int *atoms = g.atoms;
		int *top = g.top;
		ivec *nbrs = g.nbrs; 
		rvec *nbrs_cp = g.nbrs_cp;

		int 	*nbr_atoms;
		int   atom1, atom2, l, iter, max, m, num_far;
		int 	x, y, z, i;
		far_neighbor_data *nbr_data, *my_start;
		far_neighbor_data temp;
		int atom_x, atom_y, atom_z;
		rvec atom1_x;

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index > N) return;

		atom_x = (int)(sys_atoms[index].x[0] * g.inv_len[0]);
		atom_y = (int)(sys_atoms[index].x[1] * g.inv_len[1]);
		atom_z = (int)(sys_atoms[index].x[2] * g.inv_len[2]);

#ifdef __BNVT_FIX__
		if (atom_x >= g.ncell[0]) atom_x = g.ncell[0]-1;
		if (atom_y >= g.ncell[1]) atom_y = g.ncell[1]-1;
		if (atom_z >= g.ncell[2]) atom_z = g.ncell[2]-1;
#endif

		nbrs = nbrs + index_grid_nbrs (atom_x, atom_y, atom_z, 0, &g);
		nbrs_cp = nbrs_cp + index_grid_nbrs (atom_x, atom_y, atom_z, 0, &g);
		atom1 = index;

		rvec_Copy (atom1_x, sys_atoms [atom1].x );

		num_far = Start_Index (atom1, &far_nbrs);
		my_start = & (far_nbrs.select.far_nbr_list [num_far] );

		//Set_Start_Index (atom1, 0, &far_nbrs);
		//num_far =  0;
		iter = 0;

		while (nbrs[iter][0] >= 0) {
			x = nbrs[iter][0];
			y = nbrs[iter][1];
			z = nbrs[iter][2];

			//condition check for cutoff here
			//if (DistSqr_to_CP (nbrs_cp[iter], sys_atoms[atom1].x) <= 
			if (DistSqr_to_CP (nbrs_cp[iter], atom1_x) <= 
					SQR (control->vlist_cut)) 
			{
				nbr_atoms = &(atoms [index_grid_atoms (x, y, z, 0, &g) ]);
				max = top [index_grid_3d(x, y, z, &g)];

				for (m = 0; m < max; m++) 
				{
					atom2 = nbr_atoms[m];
					if (atom1 > atom2) {
						if (Are_Far_Neighbors (atom1_x, sys_atoms[atom2].x, box, 
									control->vlist_cut, &temp)){
							//nbr_data = & ( far_nbrs.select.far_nbr_list[num_far] );
							nbr_data = my_start;
							nbr_data->nbr = atom2;
							nbr_data->rel_box[0] = temp.rel_box[0];
							nbr_data->rel_box[1] = temp.rel_box[1];
							nbr_data->rel_box[2] = temp.rel_box[2];

							nbr_data->d = temp.d;
							nbr_data->dvec[0] = temp.dvec[0];
							nbr_data->dvec[1] = temp.dvec[1];
							nbr_data->dvec[2] = temp.dvec[2];
							num_far++;
							my_start ++;
						}
					}
					else if (atom1 < atom2) {
						if (Are_Far_Neighbors (sys_atoms[atom2].x, atom1_x, box, 
									control->vlist_cut, &temp)){
							//nbr_data = & ( far_nbrs.select.far_nbr_list[num_far] );
							nbr_data = my_start;
							nbr_data->nbr = atom2;
							nbr_data->rel_box[0] = temp.rel_box[0];
							nbr_data->rel_box[1] = temp.rel_box[1];
							nbr_data->rel_box[2] = temp.rel_box[2];

							nbr_data->d = temp.d;
							nbr_data->dvec[0] = temp.dvec[0];
							nbr_data->dvec[1] = temp.dvec[1];
							nbr_data->dvec[2] = temp.dvec[2];
							num_far ++;
							my_start ++;
						}
					}
					//CHANGE ORIGINAL
				}
			}
			++iter;
		}

		//end the far_neighbor list here
		Set_End_Index (atom1, num_far, &far_nbrs);
	}

	/*Multiple threads per atom Implementation */
	GLOBAL void Test_Generate_Neighbor_Lists ( 	reax_atom *sys_atoms,
			grid g,
			simulation_box *box,
			control_params* control, 
			list far_nbrs, int N )
	{

		extern __shared__ int __nbr[];
		extern __shared__ int __sofar [];
		bool	nbrgen;

		int __THREADS_PER_ATOM__ = NBRS_THREADS_PER_ATOM;

		int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
		int warp_id = thread_id / __THREADS_PER_ATOM__;
		int lane_id = thread_id & (__THREADS_PER_ATOM__ -1);
		int my_bucket = threadIdx.x / __THREADS_PER_ATOM__;

		if (warp_id >= N ) return;

		int *tnbr = __nbr;
		//int *nbrssofar = __nbr + __THREADS_PER_ATOM__;
		int *nbrssofar = __nbr + blockDim.x;

		int *atoms = g.atoms;
		int *top = g.top;
		ivec *nbrs = g.nbrs; 
		rvec *nbrs_cp = g.nbrs_cp;

		int 	*nbr_atoms;
		int   atom1, atom2, l, iter, max, m, num_far;
		int leader = -10;
		int 	x, y, z, i;
		far_neighbor_data *nbr_data, *my_start;
		far_neighbor_data temp;
		int atom_x, atom_y, atom_z;


		atom1 = warp_id;
		atom_x = (int)(sys_atoms[atom1].x[0] * g.inv_len[0]);
		atom_y = (int)(sys_atoms[atom1].x[1] * g.inv_len[1]);
		atom_z = (int)(sys_atoms[atom1].x[2] * g.inv_len[2]);

#ifdef __BNVT_FIX__
		if (atom_x >= g.ncell[0]) atom_x = g.ncell[0]-1;
		if (atom_y >= g.ncell[1]) atom_y = g.ncell[1]-1;
		if (atom_z >= g.ncell[2]) atom_z = g.ncell[2]-1;
#endif

		nbrs = nbrs + index_grid_nbrs (atom_x, atom_y, atom_z, 0, &g);
		nbrs_cp = nbrs_cp + index_grid_nbrs (atom_x, atom_y, atom_z, 0, &g);

		num_far = Start_Index (atom1, &far_nbrs);
		my_start = & (far_nbrs.select.far_nbr_list [num_far] );

		iter = 0;
		tnbr[threadIdx.x] = 0;

		if (lane_id == 0) {
			//nbrssofar [threadIdx.x /__THREADS_PER_ATOM__] = 0;
			nbrssofar [my_bucket] = 0;
		}

		__syncthreads ();

		while ((nbrs[iter][0] >= 0)) {
			x = nbrs[iter][0];
			y = nbrs[iter][1];
			z = nbrs[iter][2];

			tnbr[threadIdx.x] = 0;
			nbrgen = false;

			if (DistSqr_to_CP (nbrs_cp[iter], sys_atoms [atom1].x) <= 
					SQR (control->vlist_cut)) 
			{
				nbr_atoms = &(atoms [index_grid_atoms (x, y, z, 0, &g) ]);
				max = top [index_grid_3d(x, y, z, &g)];

				tnbr[threadIdx.x] = 0;
				nbrgen = false;
				m = lane_id ; //0-31
				int loopcount = max / __THREADS_PER_ATOM__ + ((max % __THREADS_PER_ATOM__) == 0 ? 0 : 1);
				int iterations = 0;
				//while (m < max)
				while (iterations < loopcount)
				{
					tnbr [threadIdx.x] = 0;
					nbrgen = false;

					if (m < max) {
						atom2 = nbr_atoms[m];
						if (atom1 > atom2) {
							if (Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
										control->vlist_cut, &temp))
							{
								tnbr [threadIdx.x] = 1;
								nbrgen = true;
							}
						}
						else if (atom1 < atom2) {
							if (Are_Far_Neighbors (sys_atoms[atom2].x, sys_atoms[atom1].x, box, 
										control->vlist_cut, &temp)){
								tnbr [threadIdx.x] = 1;
								nbrgen = true;
							}
						}
					}

					if (nbrgen)
					{
						//do leader selection here
						leader = -1;
						//for (l = threadIdx.x / __THREADS_PER_ATOM__; l < threadIdx.x / __THREADS_PER_ATOM__ + __THREADS_PER_ATOM__; l++)
						for (l = my_bucket *__THREADS_PER_ATOM__; l < (my_bucket)*__THREADS_PER_ATOM__ + __THREADS_PER_ATOM__; l++)
							if (tnbr[l]){
								leader = l;
								break;
							}

						//do the reduction;
						if (threadIdx.x == leader) 
							for (l = 1; l < __THREADS_PER_ATOM__; l++)
								//tnbr [(threadIdx.x / __THREADS_PER_ATOM__) * __THREADS_PER_ATOM__ + l] += tnbr [(threadIdx.x / __THREADS_PER_ATOM__) * __THREADS_PER_ATOM__ + (l-1)];	
								tnbr [my_bucket * __THREADS_PER_ATOM__ + l] += tnbr [my_bucket * __THREADS_PER_ATOM__ + (l-1)];	
					}

					//__syncthreads ();
					//atomicAdd ( &warp_sync [threadIdx.x / __THREADS_PER_ATOM__ ], 1);
					//while ( warp_sync [threadIdx.x / __THREADS_PER_ATOM__ ] < __THREADS_PER_ATOM__ ) ;

					if (nbrgen)
					{
						//got the indices
						//nbr_data = my_start + nbrssofar[threadIdx.x / __THREADS_PER_ATOM__] + tnbr [threadIdx.x] - 1;
						nbr_data = my_start + nbrssofar[my_bucket] + tnbr [threadIdx.x] - 1;
						nbr_data->nbr = atom2;
						nbr_data->rel_box[0] = temp.rel_box[0];
						nbr_data->rel_box[1] = temp.rel_box[1];
						nbr_data->rel_box[2] = temp.rel_box[2];

						nbr_data->d = temp.d;
						nbr_data->dvec[0] = temp.dvec[0];
						nbr_data->dvec[1] = temp.dvec[1];
						nbr_data->dvec[2] = temp.dvec[2];

						if (threadIdx.x == leader)
							//nbrssofar[threadIdx.x / __THREADS_PER_ATOM__] += tnbr[(threadIdx.x / __THREADS_PER_ATOM__)*__THREADS_PER_ATOM__ + (__THREADS_PER_ATOM__ - 1)];
							nbrssofar[my_bucket] += tnbr[my_bucket *__THREADS_PER_ATOM__ + (__THREADS_PER_ATOM__ - 1)];
					}

					m += __THREADS_PER_ATOM__;
					iterations ++;

					//cleanup
					nbrgen = false;
					tnbr [threadIdx.x] = 0;
				}
			}
			++iter;
		}

		__syncthreads ();

		//end the far_neighbor list here
		if (lane_id == 0)
			Set_End_Index (atom1, num_far + nbrssofar[my_bucket], &far_nbrs);
		//Set_End_Index (atom1, num_far + tnbr[63], &far_nbrs);
	}

	void Cuda_Generate_Neighbor_Lists (reax_system *system, static_storage *workspace, control_params *control, bool estimate)
	{
		real t_start, t_elapsed;
		real t_1, t_2;

		list *far_nbrs = dev_lists + FAR_NBRS;

		int *d_indices = (int *) scratch;
		int *nbrs_start, *nbrs_end;
		int i, max_nbrs = 0;
		int nbs;

		t_start = Get_Time (); 

		Cuda_Bin_Atoms (system, workspace);
		Cuda_Bin_Atoms_Sync ( system );

		if (dev_workspace->realloc.estimate_nbrs > -1) {

			/*reset the re-neighbor condition */
			dev_workspace->realloc.estimate_nbrs = -1;

			//#ifdef __DEBUG_CUDA__
			fprintf (stderr, "Recomputing the neighbors estimate.... \n");
			//#endif
			cuda_memset (d_indices, 0, INT_SIZE * system->N, RES_SCRATCH );
			/*
			   dim3 blockspergrid (system->g.ncell[0], system->g.ncell[1], system->g.ncell[2]);
			   dim3 threadsperblock (system->g.max_atoms);

			   Estimate_NumNeighbors <<<blockspergrid, threadsperblock >>>
			   (system->d_atoms, system->d_g, system->d_box, 
			   (control_params *)control->d_control, d_indices);
			   cudaThreadSynchronize ();
			   cudaCheckError ();
			 */
			nbs = (system->N / NBRS_BLOCK_SIZE) + (((system->N) % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
			New_Estimate_NumNeighbors <<<nbs, NBRS_BLOCK_SIZE>>> 
				( 	system->d_atoms, system->d_g,
					system->d_box, (control_params *)control->d_control,
					system->N, d_indices);
			cudaThreadSynchronize ();
			cudaCheckError ();


			int *nbrs_indices = NULL;
			nbrs_indices = (int *) malloc( INT_SIZE * (system->N+1) );
			if (nbrs_indices == NULL) 
			{
				fprintf (stderr, "Malloc failed for nbrs indices .... \n");
				exit (1);
			}
			memset (nbrs_indices , 0, INT_SIZE * (system->N+1) ); 

			copy_host_device (nbrs_indices+1, d_indices, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__); 
			for (int i = 1; i <= system->N; i++) 
				nbrs_indices [i] += nbrs_indices [i-1];

			copy_host_device (nbrs_indices, (far_nbrs->index), INT_SIZE * (system->N), cudaMemcpyHostToDevice, __LINE__ );
			copy_host_device (nbrs_indices, (far_nbrs->end_index), INT_SIZE * (system->N), cudaMemcpyHostToDevice, __LINE__ );

			free (nbrs_indices);
		}

		/*
		   One thread per atom Implementation
		   Generate_Neighbor_Lists <<<blockspergrid, threadsperblock >>> 
		   (system->d_atoms, system->d_g, system->d_box, 
		   (control_params *)control->d_control, *far_nbrs);
		 */
		nbs = (system->N * NBRS_THREADS_PER_ATOM/ NBRS_BLOCK_SIZE) + 
			(((system->N *NBRS_THREADS_PER_ATOM) % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);

		/* Multiple threads per atom Implementation */
		Test_Generate_Neighbor_Lists <<<nbs, NBRS_BLOCK_SIZE, 
					     INT_SIZE * (NBRS_BLOCK_SIZE+ NBRS_BLOCK_SIZE/NBRS_THREADS_PER_ATOM) >>> 
						     (system->d_atoms, system->d_g, system->d_box, 
						      (control_params *)control->d_control, *far_nbrs, system->N );
		cudaThreadSynchronize (); 
		cudaCheckError (); 

		t_elapsed = Get_Timing_Info (t_start);
		d_timing.nbrs += t_elapsed;

#ifdef __DEBUG_CUDA__
		fprintf (stderr, "Done with neighbor generation ---> %f \n", t_elapsed);
#endif

		/*validate neighbors list*/
		nbrs_start = (int *) calloc (system->N, INT_SIZE);
		nbrs_end = (int *) calloc (system->N, INT_SIZE);

		copy_host_device (nbrs_start, far_nbrs->index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__ );
		copy_host_device (nbrs_end, far_nbrs->end_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__ );

		int device_nbrs = 0;
		for(i = 0; i < system->N; i++)
		{
			if ((nbrs_end[i] - nbrs_start[i]) > max_nbrs)
				max_nbrs = nbrs_end[i] - nbrs_start[i];

			device_nbrs += nbrs_end[i] - nbrs_start[i]; 
		}
#ifdef __CUDA_TEST__
		//fprintf (stderr, " New Device count is : %d \n", device_nbrs);
		//dev_workspace->realloc.num_far = device_nbrs;
#endif

#ifdef __DEBUG_CUDA__
		fprintf (stderr, "Max neighbors is ---> %d \n", max_nbrs );
		fprintf (stderr, "DEVICE NEIGHBORS ---> %d \n", device_nbrs);
#endif

		//validate check here
		//get the num_far from the list here
		for (i = 0; i < system->N-1; i++)
		{
			if ((nbrs_end[i] - nbrs_start[i]) > (nbrs_start[i+1] - nbrs_start[i]) * DANGER_ZONE )
			{
				dev_workspace->realloc.num_far = device_nbrs;
				//#ifdef __CUDA_MEM__
				//fprintf (stderr, "Need to reallocate the neighbors ----> %d \n", dev_workspace->realloc.num_far);
				//fprintf (stderr, "Reaching the limits of neighbors for index ----> %d (%d %d %d) \n", 
				//							i, nbrs_start[i], nbrs_end[i], nbrs_start[i+1]);
				//#endif
			}

			if (nbrs_end[i] > nbrs_start[i+1]) {
				fprintf( stderr, "**ran out of space on far_nbrs: start[i] = %d, end[i]=%d, start[i+1]=%d, end[i+1] = %d",
						nbrs_start[i], nbrs_end[i], nbrs_start[i+1], nbrs_end[i+1]);
				exit( INSUFFICIENT_SPACE );
			}
		}

		if ((nbrs_end[i] - nbrs_start[i]) > (far_nbrs->num_intrs - nbrs_start[i]) * DANGER_ZONE ) {
			dev_workspace->realloc.num_far = device_nbrs;
			//#ifdef __CUDA_MEM__
			//fprintf (stderr, "Need to reallocate the neighbors ----> %d \n", dev_workspace->realloc.num_far);
			//fprintf (stderr, "Reaching the limits of neighbors for index ----> %d start: %d, end: %d, count: %d\n"
			//					, i, nbrs_start[i], nbrs_end[i], far_nbrs->num_intrs);
			//#endif
		}
		if (nbrs_end[i] > far_nbrs->num_intrs) {
			fprintf( stderr, "**ran out of space on far_nbrs: top=%d, max=%d",
					nbrs_end[i], far_nbrs->num_intrs );
			exit( INSUFFICIENT_SPACE );
		}

		free (nbrs_start);
		free (nbrs_end);
	}

	//Code not used anymore
#if defined DONE

	void Choose_Neighbor_Finder( reax_system *system, control_params *control, 
			get_far_neighbors_function *Get_Far_Neighbors )
	{
		if( control->periodic_boundaries )
		{
			if( system->box.box_norms[0] > 2.0 * control->vlist_cut &&
					system->box.box_norms[1] > 2.0 * control->vlist_cut &&
					system->box.box_norms[2] > 2.0 * control->vlist_cut )
				(*Get_Far_Neighbors) = Get_Periodic_Far_Neighbors_Big_Box;
			else  (*Get_Far_Neighbors) = Get_Periodic_Far_Neighbors_Small_Box;
		}
		else
			(*Get_Far_Neighbors) = Get_NonPeriodic_Far_Neighbors;
	}


	int compare_near_nbrs(const void *v1, const void *v2)
	{
		return ((*(near_neighbor_data *)v1).nbr - (*(near_neighbor_data *)v2).nbr);
	}


	int compare_far_nbrs(const void *v1, const void *v2)
	{
		return ((*(far_neighbor_data *)v1).nbr - (*(far_neighbor_data *)v2).nbr);
	}


	inline void Set_Far_Neighbor( far_neighbor_data *dest, int nbr, real d, real C,
			rvec dvec, ivec rel_box/*, rvec ext_factor*/ )
	{
		dest->nbr = nbr;
		dest->d = d;
		rvec_Scale( dest->dvec, C, dvec );
		ivec_Copy( dest->rel_box, rel_box );
		// rvec_Scale( dest->ext_factor, C, ext_factor );
	}


	inline void Set_Near_Neighbor(near_neighbor_data *dest, int nbr, real d, real C,
			rvec dvec, ivec rel_box/*, rvec ext_factor*/)
	{
		dest->nbr = nbr;
		dest->d = d;
		rvec_Scale( dest->dvec, C, dvec );
		ivec_Scale( dest->rel_box, C, rel_box );
		// rvec_Scale( dest->ext_factor, C, ext_factor );
	}


	/* In case bond restrictions are applied, this method checks if
	   atom1 and atom2 are allowed to bond with each other */
	inline int can_Bond( static_storage *workspace, int atom1, int atom2 )
	{
		int i;

		// fprintf( stderr, "can bond %6d %6d?\n", atom1, atom2 );

		if( !workspace->restricted[ atom1 ] && !workspace->restricted[ atom2 ] )
			return 1;

		for( i = 0; i < workspace->restricted[ atom1 ]; ++i )
			if( workspace->restricted_list[ atom1 ][i] == atom2 )
				return 1;

		for( i = 0; i < workspace->restricted[ atom2 ]; ++i )
			if( workspace->restricted_list[ atom2 ][i] == atom1 )
				return 1;

		return 0;
	}


	/* check if atom2 is on atom1's near neighbor list */
	inline int is_Near_Neighbor( list *near_nbrs, int atom1, int atom2 )
	{
		int i;

		for( i=Start_Index(atom1,near_nbrs); i<End_Index(atom1,near_nbrs); ++i )
			if( near_nbrs->select.near_nbr_list[i].nbr == atom2 )
			{
				// fprintf( stderr, "near neighbors %6d %6d\n", atom1, atom2 );
				return 1;
			}

		return 0;
	}

	void Generate_Neighbor_Lists( reax_system *system, control_params *control, 
			simulation_data *data, static_storage *workspace,
			list **lists, output_controls *out_control )
	{
		int  i, j, k;
		int  x, y, z;
		int  *nbr_atoms;
		int  atom1, atom2, max;
		int   num_far;
		int   c, count;
		int   grid_top;
		grid *g = &( system->g );  
		list *far_nbrs = (*lists) + FAR_NBRS;
		//int   hb_type1, hb_type2;
		//list *hbonds = (*lists) + HBOND;
		//int   top_hbond1, top_hbond2;
		get_far_neighbors_function Get_Far_Neighbors;
		far_neighbor_data new_nbrs[125];
#ifndef REORDER_ATOMS
		int   l, m;
#endif

		// fprintf( stderr, "\n\tentered nbrs - " );
		if( control->ensemble == iNPT || control->ensemble == sNPT || 
				control->ensemble == NPT )
			Update_Grid( system );
		// fprintf( stderr, "grid updated - " );

		Bin_Atoms( system, out_control );
		// fprintf( stderr, "atoms sorted - " );

#ifdef REORDER_ATOMS
		Cluster_Atoms( system, workspace );
		// fprintf( stderr, "atoms clustered - " );
#endif

		Choose_Neighbor_Finder( system, control, &Get_Far_Neighbors );
		// fprintf( stderr, "function chosen - " );  

		Reset_Neighbor_Lists( system, workspace, lists );  
		// fprintf( stderr, "lists cleared - " );

		num_far = 0;
		num_near = 0;
		c = 0;

		/* first pick up a cell in the grid */
		for( i = 0; i < g->ncell[0]; i++ )
			for( j = 0; j < g->ncell[1]; j++ )
				for( k = 0; k < g->ncell[2]; k++ ) {
					nbrs = g->nbrs[i][j][k];
					nbrs_cp = g->nbrs_cp[i][j][k];

					/* pick up an atom from the current cell */
					//#ifdef REORDER_ATOMS
					//  for(atom1 = g->start[i][j][k]; atom1 < g->end[i][j][k]; atom1++)
					//#else
					for(l = 0; l < g->top[i][j][k]; ++l ){
						atom1 = g->atoms[i][j][k][l];
						Set_End_Index( atom1, num_far, far_nbrs );
						// fprintf( stderr, "atom %d:\n", atom1 );

						itr = 0;
						while( nbrs[itr][0] > 0 ){
							x = nbrs[itr][0];
							y = nbrs[itr][1];
							z = nbrs[itr][2];

							// if( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <= 
							//     SQR(control->r_cut)) 	
							nbr_atoms = g->atoms[x][y][z];
							max_atoms = g->top[x][y][z];

							/* pick up another atom from the neighbor cell -
							   we have to compare atom1 with its own periodic images as well, 
							   that's why there is also equality in the if stmt below */
							//#ifdef REORDER_ATOMS
							//for(atom2=g->start[x][y][z]; atom2<g->end[x][y][z]; atom2++)
							//#else
							for( m = 0, atom2=nbr_atoms[m]; m < max; ++m, atom2=nbr_atoms[m] )
								if( atom1 >= atom2 ) {
									//fprintf( stderr, "\tatom2 %d", atom2 );
									//top_near1 = End_Index( atom1, near_nbrs );
									//Set_Start_Index( atom1, num_far, far_nbrs );
									//hb_type1=system->reaxprm.sbp[system->atoms[atom1].type].p_hbond;
									Get_Far_Neighbors( system->atoms[atom1].x,
											system->atoms[atom2].x, 
											&(system->box), control, new_nbrs, &count );
									fprintf( stderr, "\t%d count:%d\n", atom2, count );

									for( c = 0; c < count; ++c )
										if(atom1 != atom2 || (atom1 == atom2 && new_nbrs[c].d>=0.1)){
											Set_Far_Neighbor(&(far_nbrs->select.far_nbr_list[num_far]),
													atom2, new_nbrs[c].d, 1.0, 
													new_nbrs[c].dvec, new_nbrs[c].rel_box );
											++num_far;

											/*fprintf(stderr,"FARNBR:%6d%6d%8.3f[%8.3f%8.3f%8.3f]\n",
											  atom1, atom2, new_nbrs[c].d, 
											  new_nbrs[c].dvec[0], new_nbrs[c].dvec[1], 
											  new_nbrs[c].dvec[2] ); */


											/* hydrogen bond lists */ 
											/*if( control->hb_cut > 0.1 && 
											  new_nbrs[c].d <= control->hb_cut ) {
											// fprintf( stderr, "%d %d\n", atom1, atom2 );
											hb_type2=system->reaxprm.sbp[system->atoms[atom2].type].p_hbond;
											if( hb_type1 == 1 && hb_type2 == 2 ) {
											top_hbond1=End_Index(workspace->hbond_index[atom1],hbonds);
											Set_Near_Neighbor(&(hbonds->select.hbond_list[top_hbond1]),
											atom2, new_nbrs[c].d, 1.0, new_nbrs[c].dvec,
											new_nbrs[c].rel_box );
											Set_End_Index( workspace->hbond_index[atom1], 
											top_hbond1 + 1, hbonds );
											}
											else if( hb_type1 == 2 && hb_type2 == 1 ) {
											top_hbond2 = End_Index( workspace->hbond_index[atom2], hbonds );
											Set_Near_Neighbor(&(hbonds->select.hbond_list[top_hbond2]),
											atom1, new_nbrs[c].d, -1.0, new_nbrs[c].dvec, 
											new_nbrs[c].rel_box );
											Set_End_Index( workspace->hbond_index[atom2], 
											top_hbond2 + 1, hbonds );
											}*/
										}
										}
								}

							Set_End_Index( atom1, top_far1, far_nbrs );
						}
					}


					fprintf( stderr, "nbrs done-" );


					/* apply restrictions on near neighbors only */
					if( (data->step - data->prev_steps) < control->restrict_bonds ) {
						for( atom1 = 0; atom1 < system->N; ++atom1 )
							if( workspace->restricted[ atom1 ] ) {
								// fprintf( stderr, "atom1: %d\n", atom1 );

								top_near1 = End_Index( atom1, near_nbrs );

								for( j = 0; j < workspace->restricted[ atom1 ]; ++j )
									if(!is_Near_Neighbor(near_nbrs, atom1, 
												atom2 = workspace->restricted_list[atom1][j])) {
										fprintf( stderr, "%3d-%3d: added bond by applying restrictions!\n",
												atom1, atom2 );

										top_near2 = End_Index( atom2, near_nbrs );		  

										/* we just would like to get the nearest image, so a call to 
										   Get_Periodic_Far_Neighbors_Big_Box is good enough. */
										Get_Periodic_Far_Neighbors_Big_Box( system->atoms[ atom1 ].x, 
												system->atoms[ atom2 ].x, 
												&(system->box), control, 
												new_nbrs, &count );

										Set_Near_Neighbor( &(near_nbrs->select.near_nbr_list[ top_near1 ]),
												atom2, new_nbrs[c].d, 1.0, 
												new_nbrs[c].dvec, new_nbrs[c].rel_box );
										++top_near1;

										Set_Near_Neighbor( &(near_nbrs->select.near_nbr_list[ top_near2 ]),
												atom1, new_nbrs[c].d, -1.0, 
												new_nbrs[c].dvec, new_nbrs[c].rel_box );
										Set_End_Index( atom2, top_near2+1, near_nbrs );
									}

								Set_End_Index( atom1, top_near1, near_nbrs );
							}
					}
					// fprintf( stderr, "restrictions applied-" );


					/* verify nbrlists, count num_intrs, sort nearnbrs */
					near_nbrs->num_intrs = 0;
					far_nbrs->num_intrs = 0;
					for( i = 0; i < system->N-1; ++i ) {
						if( End_Index(i, near_nbrs) > Start_Index(i+1, near_nbrs) ) {
							fprintf( stderr, 
									"step%3d: nearnbr list of atom%d is overwritten by atom%d\n",
									data->step, i+1, i );
							exit( 1 );
						}

						near_nbrs->num_intrs += Num_Entries(i, near_nbrs);

						if( End_Index(i, far_nbrs) > Start_Index(i+1, far_nbrs) ) {
							fprintf( stderr, 
									"step%3d: farnbr list of atom%d is overwritten by atom%d\n", 
									data->step, i+1, i );
							exit( 1 );
						}

						far_nbrs->num_intrs += Num_Entries(i, far_nbrs);
					}

					for( i = 0; i < system->N; ++i ) {
						qsort( &(near_nbrs->select.near_nbr_list[ Start_Index(i, near_nbrs) ]),
								Num_Entries(i, near_nbrs), sizeof(near_neighbor_data), 
								compare_near_nbrs );
					}
					// fprintf( stderr, "near nbrs sorted\n" );


#ifdef TEST_ENERGY
					/* for( i = 0; i < system->N; ++i ) {
					   qsort( &(far_nbrs->select.far_nbr_list[ Start_Index(i, far_nbrs) ]), 
					   Num_Entries(i, far_nbrs), sizeof(far_neighbor_data), 
					   compare_far_nbrs ); 
					   } */

					fprintf( stderr, "Near neighbors/atom: %d (compare to 150)\n", 
							num_near / system->N );
					fprintf( stderr, "Far neighbors per atom: %d (compare to %d)\n", 
							num_far / system->N, control->max_far_nbrs );
#endif

					//fprintf( stderr, "step%d: num of nearnbrs = %6d   num of farnbrs: %6d\n",
					//       data->step, num_near, num_far );

					//fprintf( stderr, "\talloc nearnbrs = %6d   alloc farnbrs: %6d\n", 
					//   system->N * near_nbrs->intrs_per_unit, 
					//   system->N * far_nbrs->intrs_per_unit );
				}



		void Generate_Neighbor_Lists( reax_system *system, control_params *control, 
				simulation_data *data, static_storage *workspace,
				list **lists, output_controls *out_control )
		{
			int  i, j, k, l, m, itr;
			int  x, y, z;
			int  atom1, atom2, max;
			int  num_far, c, count;
			int  *nbr_atoms;
			ivec *nbrs;
			rvec *nbrs_cp;
			grid *g;
			list *far_nbrs;
			get_far_neighbors_function Get_Far_Neighbors;
			far_neighbor_data new_nbrs[125];

			g = &( system->g );
			far_nbrs = (*lists) + FAR_NBRS;

			// fprintf( stderr, "\n\tentered nbrs - " );
			if( control->ensemble == iNPT || 
					control->ensemble == sNPT || 
					control->ensemble == NPT )
				Update_Grid( system );
			// fprintf( stderr, "grid updated - " );

			Bin_Atoms( system, out_control );
			// fprintf( stderr, "atoms sorted - " );
			Choose_Neighbor_Finder( system, control, &Get_Far_Neighbors );
			// fprintf( stderr, "function chosen - " );  
			Reset_Neighbor_Lists( system, workspace, lists );  
			// fprintf( stderr, "lists cleared - " );

			num_far = 0;
			c = 0;

			/* first pick up a cell in the grid */
			for( i = 0; i < g->ncell[0]; i++ )
				for( j = 0; j < g->ncell[1]; j++ )
					for( k = 0; k < g->ncell[2]; k++ ) {
						nbrs = g->nbrs[i][j][k];
						nbrs_cp = g->nbrs_cp[i][j][k];
						fprintf( stderr, "gridcell %d %d %d\n", i, j, k );

						/* pick up an atom from the current cell */
						for(l = 0; l < g->top[i][j][k]; ++l ){
							atom1 = g->atoms[i][j][k][l];
							Set_Start_Index( atom1, num_far, far_nbrs );
							fprintf( stderr, "\tatom %d\n", atom1 );

							itr = 0;
							while( nbrs[itr][0] > 0 ){
								x = nbrs[itr][0];
								y = nbrs[itr][1];
								z = nbrs[itr][2];
								fprintf( stderr, "\t\tgridcell %d %d %d\n", x, y, z );

								// if( DistSqr_to_CP(nbrs_cp[itr], system->atoms[atom1].x ) <= 
								//     SQR(control->r_cut)) 	
								nbr_atoms = g->atoms[x][y][z];
								max = g->top[x][y][z];
								fprintf( stderr, "\t\tmax: %d\n", max );


								/* pick up another atom from the neighbor cell -
								   we have to compare atom1 with its own periodic images as well, 
								   that's why there is also equality in the if stmt below */
								for( m = 0, atom2=nbr_atoms[m]; m < max; ++m, atom2=nbr_atoms[m] )
									if( atom1 >= atom2 ) {
										Get_Far_Neighbors( system->atoms[atom1].x,
												system->atoms[atom2].x, 
												&(system->box), control, new_nbrs, &count );
										fprintf( stderr, "\t\t\t%d count:%d\n", atom2, count );

										for( c = 0; c < count; ++c )
											if(atom1 != atom2 || (atom1 == atom2 && new_nbrs[c].d>=0.1)){
												Set_Far_Neighbor(&(far_nbrs->select.far_nbr_list[num_far]),
														atom2, new_nbrs[c].d, 1.0, 
														new_nbrs[c].dvec, new_nbrs[c].rel_box );
												++num_far;

												/*fprintf(stderr,"FARNBR:%6d%6d%8.3f[%8.3f%8.3f%8.3f]\n",
												  atom1, atom2, new_nbrs[c].d, 
												  new_nbrs[c].dvec[0], new_nbrs[c].dvec[1], 
												  new_nbrs[c].dvec[2] ); */
											}
									}

								++itr;
							}

							Set_End_Index( atom1, num_far, far_nbrs );
						}
					}

			far_nbrs->num_intrs = num_far;  
			fprintf( stderr, "nbrs done, num_far: %d\n", num_far );

#if defined(DEBUG)
			for( i = 0; i < system->N; ++i ) {
				qsort( &(far_nbrs->select.far_nbr_list[ Start_Index(i, far_nbrs) ]), 
						Num_Entries(i, far_nbrs), sizeof(far_neighbor_data), 
						compare_far_nbrs ); 
			}

			fprintf( stderr, "step%d: num of farnbrs=%6d\n", data->step, num_far );
			fprintf( stderr, "\tallocated farnbrs: %6d\n", 
					system->N * far_nbrs->intrs_per_unit );
#endif
		}



#endif
