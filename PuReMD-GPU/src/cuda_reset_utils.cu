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

#include "cuda_reset_utils.h"

#include "list.h"
#include "reset_utils.h"
#include "vector.h"

#include "cuda_utils.h"
#include "cuda_copy.h"


GLOBAL void Reset_Atoms (reax_atom *atoms, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    atoms[i].f[0] = 0.0;
    atoms[i].f[1] = 0.0;
    atoms[i].f[2] = 0.0;
}


void Cuda_Reset_Atoms (reax_system *system )
{
    Reset_Atoms <<<BLOCKS, BLOCK_SIZE>>>
        (system->d_atoms, system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();
}


void Cuda_Sync_Simulation_Data (simulation_data *data)
{
    //copy_host_device (&data->E_BE, &((simulation_data *)data->d_simulation_data)->E_BE, 
    //                                        REAL_SIZE * 12, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );
    cuda_memset (&((simulation_data *)data->d_simulation_data)->E_BE, 0, REAL_SIZE * 12, RES_SIMULATION_DATA );

    //copy_host_device (&data->E_Kin, &((simulation_data *)data->d_simulation_data)->E_Kin, 
    //                                        REAL_SIZE, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );
    cuda_memset (&((simulation_data *)data->d_simulation_data)->E_Kin, 0, REAL_SIZE, RES_SIMULATION_DATA );

}


void Cuda_Reset_Workspace( reax_system *system, static_storage *workspace )
{
    cuda_memset( workspace->total_bond_order, 0, system->N * REAL_SIZE, RES_STORAGE_TOTAL_BOND_ORDER );
    cuda_memset( workspace->dDeltap_self, 0, system->N * RVEC_SIZE, RES_STORAGE_DDELTAP_SELF );
    cuda_memset( workspace->CdDelta, 0, system->N * REAL_SIZE, RES_STORAGE_CDDELTA );
}


GLOBAL void Reset_Neighbor_Lists (single_body_parameters *sbp, reax_atom *atoms, 
        list bonds, list hbonds, control_params *control, 
        static_storage workspace, int N)
{
    int tmp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) return;

    tmp = Start_Index (index, &bonds);
    Set_End_Index (index, tmp, &bonds);


    if (control->hb_cut > 0) {
        if ((sbp[ atoms[index].type ].p_hbond == 1) || 
                (sbp[ atoms[index].type ].p_hbond == 2)) {
            tmp = Start_Index ( workspace.hbond_index[index], &hbonds );
            Set_End_Index ( workspace.hbond_index[index], tmp, &hbonds );
        }
    }
}


void Cuda_Reset_Neighbor_Lists (reax_system *system, control_params *control, 
        static_storage *workspace, list **lists ) 
{
    Reset_Neighbor_Lists <<<BLOCKS, BLOCK_SIZE>>>
        ( system->reaxprm.d_sbp, system->d_atoms, *(dev_lists + BONDS), *(dev_lists + HBONDS), 
          (control_params *)control->d_control, *dev_workspace, system->N );
    cudaThreadSynchronize ();
    cudaCheckError ();

    //reset here
    list *bonds = (dev_lists + BONDS );
    //TODO - check if this is needed.
    cuda_memset (bonds->select.bond_list, 0, BOND_DATA_SIZE * bonds->num_intrs, LIST_BOND_DATA );
}


GLOBAL void Reset_Far_Neighbors_List (list far_nbrs, int N)
{
    int tmp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) return;

    tmp = Start_Index (index, &far_nbrs);
    Set_End_Index (index, tmp, &far_nbrs);
}


void Cuda_Reset_Far_Neighbors_List ( reax_system *system )
{
    Reset_Far_Neighbors_List <<<BLOCKS, BLOCK_SIZE>>>
        (*(dev_lists + FAR_NBRS), system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();
}


void Cuda_Reset_Sparse_Matrix (reax_system *system, static_storage *workspace)
{
    cuda_memset (workspace->H.j, 0, (system->N + 1) * INT_SIZE, RES_SPARSE_MATRIX_INDEX );
    cuda_memset (workspace->H.val, 0, (system->N * system->max_sparse_matrix_entries) * INT_SIZE, RES_SPARSE_MATRIX_INDEX );
}


void Cuda_Reset( reax_system *system, control_params *control,  
        simulation_data *data, static_storage *workspace, list **lists  )
{
    Cuda_Reset_Atoms( system );

    //Reset_Simulation_Data( data );
    Cuda_Sync_Simulation_Data ( data );
    //Sync_Host_Device_Data( data, (simulation_data *)data->d_simulation_data, cudaMemcpyHostToDevice );

    if( control->ensemble == NPT || control->ensemble == sNPT || 
            control->ensemble == iNPT )
        Reset_Pressures( data );

    Cuda_Reset_Workspace( system, dev_workspace );  

    Cuda_Reset_Neighbor_Lists( system, control, workspace, lists );

    Cuda_Reset_Far_Neighbors_List (system);

    Cuda_Reset_Sparse_Matrix (system, dev_workspace);

}


void Cuda_Reset_Grid (grid *g)
{
    cuda_memset (g->top, 0, INT_SIZE * g->ncell[0]*g->ncell[1]*g->ncell[2], RES_GRID_TOP);
}
