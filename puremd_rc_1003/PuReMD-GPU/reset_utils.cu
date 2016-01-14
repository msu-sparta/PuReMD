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

#include "reset_utils.h"
#include "list.h"
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

void Reset_Atoms( reax_system* system )
{
  int i;

  for( i = 0; i < system->N; ++i )
    memset( system->atoms[i].f, 0.0, RVEC_SIZE );
}


void Reset_Pressures( simulation_data *data )
{
  rtensor_MakeZero( data->flex_bar.P );  
  data->iso_bar.P = 0;
  rvec_MakeZero( data->int_press );
  rvec_MakeZero( data->ext_press );
  /* fprintf( stderr, "reset: ext_press (%12.6f %12.6f %12.6f)\n", 
     data->ext_press[0], data->ext_press[1], data->ext_press[2] ); */
}


void Reset_Simulation_Data( simulation_data* data )
{
  data->E_BE = 0;
  data->E_Ov = 0;
  data->E_Un = 0;
  data->E_Lp = 0;
  data->E_Ang = 0;
  data->E_Pen = 0;
  data->E_Coa = 0;
  data->E_HB = 0;
  data->E_Tor = 0;
  data->E_Con = 0;
  data->E_vdW = 0;
  data->E_Ele = 0;
  data->E_Kin = 0;
}

void Cuda_Sync_Simulation_Data (simulation_data *data)
{
	//copy_host_device (&data->E_BE, &((simulation_data *)data->d_simulation_data)->E_BE, 
	//										REAL_SIZE * 12, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );
	cuda_memset (&((simulation_data *)data->d_simulation_data)->E_BE, 0, REAL_SIZE * 12, RES_SIMULATION_DATA );

	//copy_host_device (&data->E_Kin, &((simulation_data *)data->d_simulation_data)->E_Kin, 
	//										REAL_SIZE, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );
	cuda_memset (&((simulation_data *)data->d_simulation_data)->E_Kin, 0, REAL_SIZE, RES_SIMULATION_DATA );

}


#ifdef TEST_FORCES
void Reset_Test_Forces( reax_system *system, static_storage *workspace )
{
  memset( workspace->f_ele, 0, system->N * sizeof(rvec) );
  memset( workspace->f_vdw, 0, system->N * sizeof(rvec) );
  memset( workspace->f_bo, 0, system->N * sizeof(rvec) );
  memset( workspace->f_be, 0, system->N * sizeof(rvec) );
  memset( workspace->f_lp, 0, system->N * sizeof(rvec) );
  memset( workspace->f_ov, 0, system->N * sizeof(rvec) );
  memset( workspace->f_un, 0, system->N * sizeof(rvec) );
  memset( workspace->f_ang, 0, system->N * sizeof(rvec) );
  memset( workspace->f_coa, 0, system->N * sizeof(rvec) );
  memset( workspace->f_pen, 0, system->N * sizeof(rvec) );
  memset( workspace->f_hb, 0, system->N * sizeof(rvec) );
  memset( workspace->f_tor, 0, system->N * sizeof(rvec) );
  memset( workspace->f_con, 0, system->N * sizeof(rvec) );
}
#endif


void Reset_Workspace( reax_system *system, static_storage *workspace )
{
  memset( workspace->total_bond_order, 0, system->N * sizeof( real ) );
  memset( workspace->dDeltap_self, 0, system->N * sizeof( rvec ) );

  memset( workspace->CdDelta, 0, system->N * sizeof( real ) );
  //memset( workspace->virial_forces, 0, system->N * sizeof( rvec ) );

#ifdef TEST_FORCES
  memset( workspace->dDelta, 0, sizeof(rvec) * system->N );
  Reset_Test_Forces( system, workspace );
#endif
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

void Reset_Neighbor_Lists( reax_system *system, control_params *control, 
			   static_storage *workspace, list **lists )
{
  int i, tmp;
  list *bonds = (*lists) + BONDS;
  list *hbonds = (*lists) + HBONDS;

  for( i = 0; i < system->N; ++i ) {
    tmp = Start_Index( i, bonds );
    Set_End_Index( i, tmp, bonds );
  }

  //TODO check if this is needed
  memset (bonds->select.bond_list, 0, BOND_DATA_SIZE * bonds->num_intrs );

  if( control->hb_cut > 0 )
    for( i = 0; i < system->N; ++i )
      if( system->reaxprm.sbp[system->atoms[i].type].p_hbond == 1) {
	tmp = Start_Index( workspace->hbond_index[i], hbonds );
	Set_End_Index( workspace->hbond_index[i], tmp, hbonds );
	/* fprintf( stderr, "i:%d, hbond: %d-%d\n", 
	   i, Start_Index( workspace->hbond_index[i], hbonds ), 
	   End_Index( workspace->hbond_index[i], hbonds ) );*/
      }
}


void Reset( reax_system *system, control_params *control,  
	    simulation_data *data, static_storage *workspace, list **lists  )
{
  Reset_Atoms( system );
  
  Reset_Simulation_Data( data );

  if( control->ensemble == NPT || control->ensemble == sNPT || 
      control->ensemble == iNPT )
    Reset_Pressures( data );

  Reset_Workspace( system, workspace );  

  Reset_Neighbor_Lists( system, control, workspace, lists );

#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "reset - ");
#endif
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
  //Sync_Host_Device (data, (simulation_data *)data->d_simulation_data, cudaMemcpyHostToDevice);

  if( control->ensemble == NPT || control->ensemble == sNPT || 
      control->ensemble == iNPT )
    Reset_Pressures( data );

  Cuda_Reset_Workspace( system, dev_workspace );  

  Cuda_Reset_Neighbor_Lists( system, control, workspace, lists );

  Cuda_Reset_Far_Neighbors_List (system);

  Cuda_Reset_Sparse_Matrix (system, dev_workspace);

}


void Reset_Grid( grid *g )
{
	memset (g->top, 0, INT_SIZE * g->ncell[0]*g->ncell[1]*g->ncell[2]);
}

void Cuda_Reset_Grid (grid *g)
{
	cuda_memset (g->top, 0, INT_SIZE * g->ncell[0]*g->ncell[1]*g->ncell[2], RES_GRID_TOP);
}


void Reset_Marks( grid *g, ivec *grid_stack, int grid_top )
{
  int i;
  
  for( i = 0; i < grid_top; ++i )
    g->mark[grid_stack[i][0] * g->ncell[1]*g->ncell[2] + 
			grid_stack[i][1] * g->ncell[2] + 
			grid_stack[i][2]] = 0;
}
