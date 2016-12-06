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

#include "cuda_neighbors.h"

#include "box.h"
#include "grid.h"
#include "list.h"
#include "neighbors.h"
#include "reset_utils.h"
#include "system_props.h"
#include "vector.h"
#include "index_utils.h"

#include "cuda_utils.h"
#include "cuda_grid.h"


extern inline DEVICE int index_grid (int blocksize)
{
    return blockIdx.x * gridDim.y * gridDim.z * blocksize +  
        blockIdx.y * gridDim.z * blocksize +  
        blockIdx.z * blocksize ;
}


DEVICE int d_Are_Far_Neighbors( rvec x1, rvec x2, simulation_box *box, 
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


GLOBAL void k_Estimate_NumNeighbors( reax_atom *sys_atoms,
        grid g, simulation_box *box, control_params *control, int *indices )
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
                   if (d_Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
                   control->vlist_cut, &nbr_data)){
                   ++num_far;
                   }
                   }
                 */
                if (atom1 > atom2) {
                    if (d_Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
                                control->vlist_cut, &nbr_data)){
                        ++num_far;
                    }
                }
                else if (atom1 < atom2) {
                    if (d_Are_Far_Neighbors (sys_atoms[atom2].x, sys_atoms[atom1].x, box, 
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
GLOBAL void k_New_Estimate_NumNeighbors( reax_atom *sys_atoms,
        grid g, simulation_box *box, control_params* control, int N, int *indices )
{
    int *atoms = g.atoms;
    int *top = g.top;
    ivec *nbrs = g.nbrs; 
    rvec *nbrs_cp = g.nbrs_cp;

    int     *nbr_atoms;
    int   atom1, atom2, iter, max, m, num_far;
    int     x, y, z, i;
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
                    if (d_Are_Far_Neighbors (atom1_x, sys_atoms[atom2].x, box, 
                                control->vlist_cut, &temp)){
                        num_far++;
                    }
                }
                else if (atom1 < atom2) {
                    if (d_Are_Far_Neighbors (sys_atoms[atom2].x, atom1_x, box, 
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
GLOBAL void k_Generate_Neighbor_Lists ( reax_atom *sys_atoms,
        grid g, simulation_box *box, control_params* control, 
        list far_nbrs )
{
    int *atoms = g.atoms;
    int *top = g.top;
    ivec *nbrs = g.nbrs; 
    rvec *nbrs_cp = g.nbrs_cp;

    int     *nbr_atoms;
    int   atom1, atom2, l, iter, max, m, num_far;
    int     x, y, z, i;
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
                   if (d_Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
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
                    if (d_Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
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
                    if (d_Are_Far_Neighbors (sys_atoms[atom2].x, sys_atoms[atom1].x, box, 
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
GLOBAL void k_New_Generate_Neighbor_Lists( reax_atom *sys_atoms,
        grid g, simulation_box *box, control_params* control, 
        list far_nbrs, int N )
{
    int *atoms = g.atoms;
    int *top = g.top;
    ivec *nbrs = g.nbrs; 
    rvec *nbrs_cp = g.nbrs_cp;

    int     *nbr_atoms;
    int   atom1, atom2, l, iter, max, m, num_far;
    int     x, y, z, i;
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
                    if (d_Are_Far_Neighbors (atom1_x, sys_atoms[atom2].x, box, 
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
                    if (d_Are_Far_Neighbors (sys_atoms[atom2].x, atom1_x, box, 
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
GLOBAL void Test_Generate_Neighbor_Lists( reax_atom *sys_atoms,
        grid g, simulation_box *box, control_params* control, 
        list far_nbrs, int N )
{

    extern __shared__ int __nbr[];
    extern __shared__ int __sofar [];
    int nbrgen;

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

    int     *nbr_atoms;
    int   atom1, atom2, l, iter, max, m, num_far;
    int leader = -10;
    int     x, y, z, i;
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
        nbrgen = FALSE;

        if (DistSqr_to_CP (nbrs_cp[iter], sys_atoms [atom1].x) <= 
                SQR (control->vlist_cut)) 
        {
            nbr_atoms = &(atoms [index_grid_atoms (x, y, z, 0, &g) ]);
            max = top [index_grid_3d(x, y, z, &g)];

            tnbr[threadIdx.x] = 0;
            nbrgen = FALSE;
            m = lane_id ; //0-31
            int loopcount = max / __THREADS_PER_ATOM__ + ((max % __THREADS_PER_ATOM__) == 0 ? 0 : 1);
            int iterations = 0;
            //while (m < max)
            while (iterations < loopcount)
            {
                tnbr [threadIdx.x] = 0;
                nbrgen = FALSE;

                if (m < max) {
                    atom2 = nbr_atoms[m];
                    if (atom1 > atom2) {
                        if (d_Are_Far_Neighbors (sys_atoms[atom1].x, sys_atoms[atom2].x, box, 
                                    control->vlist_cut, &temp))
                        {
                            tnbr [threadIdx.x] = 1;
                            nbrgen = TRUE;
                        }
                    }
                    else if (atom1 < atom2) {
                        if (d_Are_Far_Neighbors (sys_atoms[atom2].x, sys_atoms[atom1].x, box, 
                                    control->vlist_cut, &temp)){
                            tnbr [threadIdx.x] = 1;
                            nbrgen = TRUE;
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
                //MYATOMICADD( &warp_sync [threadIdx.x / __THREADS_PER_ATOM__ ], 1);
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
                nbrgen = FALSE;
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


void Cuda_Generate_Neighbor_Lists (reax_system *system, static_storage *workspace, control_params *control, int estimate)
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

           k_Estimate_NumNeighbors <<<blockspergrid, threadsperblock >>>
           (system->d_atoms, system->d_g, system->d_box, 
           (control_params *)control->d_control, d_indices);
           cudaThreadSynchronize ();
           cudaCheckError ();
         */
        nbs = (system->N / NBRS_BLOCK_SIZE) + (((system->N) % NBRS_BLOCK_SIZE) == 0 ? 0 : 1);
        k_New_Estimate_NumNeighbors <<<nbs, NBRS_BLOCK_SIZE>>> 
            (     system->d_atoms, system->d_g,
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
            //                            i, nbrs_start[i], nbrs_end[i], nbrs_start[i+1]);
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
        //                    , i, nbrs_start[i], nbrs_end[i], far_nbrs->num_intrs);
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
